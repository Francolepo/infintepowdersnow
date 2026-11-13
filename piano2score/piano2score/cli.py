"""Command line interface:  python -m piano2score <youtube-url | video file> [options]"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _time(s: str) -> float:
    """'90', '1:30' or '1:02:03' -> seconds."""
    parts = [float(p) for p in s.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", s, flags=re.U).strip().replace(" ", "_")
    return (s or "score")[:60]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="piano2score",
        description="Turn a falling-notes piano tutorial video (YouTube link or local file) into sheet music.")
    p.add_argument("source", help="YouTube/URL or path to a video file")
    p.add_argument("-o", "--out-dir", default="output", help="folder for the results (default: ./output)")
    p.add_argument("--title", help="title printed on the score (default: video title / file name)")
    g = p.add_argument_group("what part of the video")
    g.add_argument("--start", type=_time, default=0.0, help="start time, e.g. 15 or 0:15 (skip an intro)")
    g.add_argument("--end", type=_time, default=None, help="end time, e.g. 3:20")
    g.add_argument("--max-height", type=int, default=1080, help="download quality limit (default 1080)")
    g = p.add_argument_group("reading the video")
    g.add_argument("--calibrate-only", action="store_true",
                   help="only find the keyboard and save debug/calibration.png to check it, then stop")
    g.add_argument("--strip-offset", type=float, default=3.3, metavar="KEYS",
                   help="where notes are read, in key-widths above the keyboard (default 3.3). Notes that "
                        "burn up in a glow right above the keys need this above the glow; plain falling "
                        "bars can use a small value")
    g.add_argument("--keyboard-top", type=float, metavar="ROW",
                   help="row (in debug/calibration.png) where the notes land, if auto-detection is off")
    g.add_argument("--first-note", help="lowest key on screen, e.g. A0 (if the keyboard is not 88 keys)")
    g.add_argument("--last-note", help="highest key on screen, e.g. C8")
    g.add_argument("--threshold", type=float, help="brightness difference counted as 'a note is here' (0-255)")
    g.add_argument("--min-note-ms", type=float, default=70.0, help="ignore blips shorter than this")
    g.add_argument("--max-fps", type=float, default=60.0, help="analyse at most this many frames/second")
    g = p.add_argument_group("music")
    g.add_argument("--bpm", type=float, help="quarter-note tempo (default: estimated)")
    g.add_argument("--time-sig", default="4/4", help="time signature (default 4/4)")
    g.add_argument("--subdivision", type=int, default=4, choices=(2, 3, 4),
                   help="grid slots per quarter note: 4 = 16ths (default), 2 = 8ths, 3 = 8th-note triplets")
    g.add_argument("--pickup", type=float, default=0.0, metavar="BEATS",
                   help="length of a pick-up (anacrusis) in beats; default: first note starts bar 1")
    g = p.add_argument_group("left / right hand")
    g.add_argument("--hands", choices=("auto", "video", "pitch"), default="auto",
                   help="how notes are given to the hands: 'video' tracks the real hands in the video, "
                        "'pitch' splits at --split, 'auto' (default) uses the video when hands are visible "
                        "and falls back to pitch otherwise")
    g.add_argument("--hand-fps", type=float, default=15.0, help="hand-tracking frames per second (default 15)")
    g.add_argument("--split", default="C4", help="pitch split (and fallback): this note and above is the "
                                                  "right hand (default C4)")
    g = p.add_argument_group("more music options")
    g.add_argument("--track", choices=("auto", "on", "off"), default="auto",
                   help="follow tempo changes (auto: only when the playing drifts)")
    g.add_argument("--no-pdf", action="store_true", help="skip the PDF (MusicXML and MIDI are still written)")
    return p


def _sanity_check(det, kb, y0: int, offset: float) -> list[str]:
    """Cheap checks that hint at a badly placed reading band."""
    out: list[str] = []
    if y0 <= 2:
        out.append("the reading band is at the very top of the picture: use a smaller --strip-offset")
    if len(det.notes) >= 20:
        dur = np.array([n.duration for n in det.notes])
        if np.mean(dur < 0.12) > 0.35:
            out.append("many very short notes were found; the reading band may be inside the glow/sparks "
                       "that surround the keys. Try a larger --strip-offset (e.g. 4.5), or check "
                       "debug/calibration.png")
        span = max(det.times[-1] - det.times[0], 1.0)
        if len(det.notes) / span > 14:
            out.append("an unusually large number of notes per second was found; see the hint above")
    return out


def _second_band(kb, offset: float):
    """A second reading band 1.5 key-widths above (or, if there is no room, below) the first."""
    for off in (offset + 1.5, offset - 1.5):
        if off < 0.8:
            continue
        rows = kb.strip_rows(off)
        if rows[0] >= 3:
            return off, rows
    return None


def _hand_stage(args, info, kb, det, det_b, band_b, band_a, dbg: Path) -> str:
    """Track the hands and give every detected note to the left or right hand."""
    from . import hands, keyboard

    ca = kb.y_top - (band_a[0] + band_a[1]) / 2.0
    timing = None
    if det_b is not None and band_b is not None:
        cb = kb.y_top - (band_b[1][0] + band_b[1][1]) / 2.0
        timing = hands.estimate_fall_delay(det.notes, det_b.notes, ca, cb)
    if timing:
        print(f"Bars fall at ~{timing.speed / kb.white_w:.1f} key-widths per second: a note is pressed "
              f"{timing.delay:.2f} s after it enters the reading band")
    else:
        print("! Could not measure how fast the bars fall; matching hands to notes over a wider time window")

    last = [-1.0]

    def progress(frac: float) -> None:
        if frac - last[0] >= 0.05:
            last[0] = frac
            print(f"\rTracking hands ... {max(0.0, min(frac, 1.0)) * 100:3.0f}%", end="", flush=True)

    t = time.time()
    track = hands.track_hands(info, kb, args.start, args.end, args.hand_fps, progress)
    cov = track.coverage()
    print(f"\rTracking hands ... done ({len(track.frames)} frames, {time.time() - t:.0f} s; "
          f"left hand seen in {cov['L'] * 100:.0f}% of frames, right hand in {cov['R'] * 100:.0f}%)")
    if track.debug_image is not None:
        cv2.imwrite(str(dbg / "hands.png"), track.debug_image)
    if max(cov.values()) < 0.2:
        if args.hands == "video":
            print("! Hardly any hands were found in the video; the result will mostly follow pitch")
        else:
            print("! Hardly any hands were found in the video, so the hands are split by pitch")
            return f"no hands visible in the video: split by pitch at {args.split}"
    stats = hands.assign_hands(det.notes, kb, track, timing.delay if timing else None,
                               split_midi=keyboard.parse_note(args.split))
    return stats.summary()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from . import detect, download, keyboard, rhythm, score, video

    out_dir = Path(args.out_dir)
    dbg = out_dir / "debug"
    dbg.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. get the video
    title = args.title
    if download.is_url(args.source):
        print(f"Downloading {args.source} ...")
        path, vtitle = download.download(args.source, out_dir / "video", args.max_height)
        title = title or vtitle
    else:
        path = Path(args.source)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            return 2
        title = title or path.stem
    path = download.ensure_decodable(path, out_dir / "video")
    info = video.probe(str(path))
    name = _slug(title)
    print(f"Video: {info.width}x{info.height}, {info.fps:.1f} fps, {info.duration:.0f} s")

    # ------------------------------------------------------------------ 2. find the keyboard
    frames = video.sample_frames(info, 48, args.start, args.end)
    kb = keyboard.calibrate(
        frames,
        first_note=keyboard.parse_note(args.first_note) if args.first_note else None,
        last_note=keyboard.parse_note(args.last_note) if args.last_note else None,
        y_top=args.keyboard_top)
    y0, y1 = kb.strip_rows(args.strip_offset)
    cv2.imwrite(str(dbg / "calibration.png"), keyboard.draw_overlay(frames[len(frames) // 2], kb, (y0, y1)))
    print(f"Keyboard: {kb.method}; keys {keyboard.note_name(int(kb.midis[0]))}-"
          f"{keyboard.note_name(int(kb.midis[-1]))}; notes are read at row {y0}")
    for n in kb.notes:
        print(f"  - {n}")
    print(f"  (check {dbg / 'calibration.png'}: red line = where notes land, cyan band = where they are read)")
    if args.calibrate_only:
        return 0

    # ------------------------------------------------------------------ 3. read the notes
    stride = max(1, int(round(info.fps / args.max_fps)))
    last = [-1.0]

    def progress(frac: float) -> None:
        if frac - last[0] >= 0.05:
            last[0] = frac
            print(f"\rReading video ... {frac * 100:3.0f}%", end="", flush=True)

    # a second band at another height measures how fast the bars fall (needed to know when each
    # key is really pressed, for the hand assignment)
    band_b = _second_band(kb, args.strip_offset) if args.hands != "pitch" else None
    bands = [(y0, y1)] + ([band_b[1]] if band_b else [])
    t = time.time()
    times, strips = video.read_strips(info, bands, stride, args.start, args.end, progress)
    print(f"\rReading video ... done ({len(times)} frames, {time.time() - t:.0f} s)")
    det = detect.detect_notes(times, strips[0], kb, threshold=args.threshold, min_ms=args.min_note_ms)
    print(f"Detected {len(det.notes)} notes (brightness threshold {det.threshold:.0f})")
    reading_warnings = _sanity_check(det, kb, y0, args.strip_offset)
    for w in reading_warnings:
        print(f"! {w}")
    if not det.notes:
        print("No notes found. Open debug/calibration.png: the cyan band should sit on the falling "
              "notes' path just above the keyboard. Try --strip-offset, --threshold, --keyboard-top.",
              file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ 4. which hand plays what
    hand_report = "split by pitch at " + args.split
    if args.hands != "pitch":
        det_b = detect.detect_notes(times, strips[1], kb, threshold=args.threshold,
                                    min_ms=args.min_note_ms) if band_b else None
        try:
            hand_report = _hand_stage(args, info, kb, det, det_b, band_b, (y0, y1), dbg)
        except RuntimeError as e:
            if args.hands == "video":
                print(f"Hand tracking failed: {e}", file=sys.stderr)
                return 1
            print(f"! Hand tracking unavailable ({e}); splitting hands by pitch instead")
    cv2.imwrite(str(dbg / "piano_roll.png"), detect.render_piano_roll(det, kb))

    # ------------------------------------------------------------------ 5. rhythm and score
    num, den = (int(x) for x in args.time_sig.split("/"))
    r = rhythm.quantize(det.notes, sub=args.subdivision, bpm=args.bpm, track=args.track)
    sc, sinfo = score.build_score(r, title=title, time_sig=(num, den), split=keyboard.parse_note(args.split),
                                  pickup_beats=args.pickup)
    xml_path, mid_path, pdf_path = (out_dir / f"{name}.{ext}" for ext in ("musicxml", "mid", "pdf"))
    score.write_musicxml(sc, xml_path)
    score.write_midi(det.notes, mid_path)
    reason = None
    if not args.no_pdf:
        reason = score.render_pdf(xml_path, pdf_path)

    print()
    print(f"Hands: {hand_report}")
    print(f"Tempo ~{sinfo.bpm:.0f} bpm{' (follows tempo changes)' if r.tracked else ''}, "
          f"{args.time_sig}, key {sinfo.key}, {sinfo.bars} bars, "
          f"{sinfo.notes_rh} right-hand and {sinfo.notes_lh} left-hand notes")
    for w in sinfo.warnings:
        print(f"! {w}")
    print("Written:")
    print(f"  {xml_path}   (open in MuseScore / Finale / Sibelius / Dorico to edit)")
    print(f"  {mid_path}   (the performance as played in the video)")
    if not args.no_pdf:
        print(f"  {pdf_path}" if reason is None else f"  (PDF skipped: {reason})")
    print(f"  {dbg}/   (calibration.png, piano_roll.png, hands.png: check these if the result looks wrong)")
    return 0


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
