"""Build engraved-quality notation (MusicXML), MIDI and a PDF from quantised notes."""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np

from .detect import RawNote
from .rhythm import QNote, Rhythm

# ----------------------------------------------------------------------------- key estimation
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJOR_SHARPS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 9, 10, 11}     # natural + harmonic + melodic minor


def estimate_key(notes: list[QNote]) -> tuple[int, str, int]:
    """Krumhansl-Schmuckler key finding, penalising keys the music leaves.

    Returns (tonic pitch class, mode, number of sharps; negative = flats).
    """
    hist = np.zeros(12)       # weighted by duration (what the ear hears as tonal weight)
    count = np.zeros(12)      # how many notes (a short chromatic note is still an accidental)
    for n in notes:
        hist[n.midi % 12] += n.dur
        count[n.midi % 12] += 1
    if hist.sum() == 0:
        return 0, "major", 0
    best = (-9.0, 0, "major")
    for tonic in range(12):
        for mode, prof, scale in (("major", _MAJOR, _MAJOR_SCALE), ("minor", _MINOR, _MINOR_SCALE)):
            corr = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            outside = sum(count[pc] for pc in range(12) if (pc - tonic) % 12 not in scale) / count.sum()
            score = corr - 3.0 * outside
            if score > best[0]:
                best = (score, tonic, mode)
    _, tonic, mode = best
    rel_major = tonic if mode == "major" else (tonic + 3) % 12
    return tonic, mode, _MAJOR_SHARPS[rel_major]


def key_name(tonic: int, mode: str) -> str:
    return f"{_NAMES[tonic]} {mode}"


# ----------------------------------------------------------------------------- note values
# Allowed single-note lengths (in grid units), and the grid position they must start on.
_SIZES = {
    4: [(16, 16), (12, 4), (8, 4), (6, 2), (4, 2), (3, 1), (2, 1), (1, 1)],
    2: [(8, 8), (6, 2), (4, 2), (3, 1), (2, 1), (1, 1)],
    3: [(12, 12), (9, 3), (6, 3), (3, 3), (2, 1), (1, 1)],
}


def decompose(pos: int, length: int, sub: int) -> list[int]:
    """Split a duration into notatable pieces that respect the beat structure."""
    table = _SIZES.get(sub, _SIZES[4])
    out: list[int] = []
    while length > 0:
        for size, align in table:
            if size <= length and pos % align == 0:
                out.append(size)
                pos += size
                length -= size
                break
        else:                       # pragma: no cover - the table always ends with (1, 1)
            out.append(1)
            pos += 1
            length -= 1
    return out


# ----------------------------------------------------------------------------- per-hand preparation
@dataclass
class Event:
    start: int
    dur: int
    midis: tuple[int, ...]

    @property
    def end(self) -> int:
        return self.start + self.dur


def _chord_events(notes: list[QNote]) -> list[Event]:
    """Notes starting together become chords; near-equal lengths are made equal."""
    by_start: dict[int, list[QNote]] = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    events: list[Event] = []
    for start in sorted(by_start):
        group = sorted(by_start[start], key=lambda n: n.dur)
        clusters: list[list[QNote]] = [[group[0]]]
        for n in group[1:]:
            ref = clusters[-1][-1].dur
            if n.dur - ref <= max(1, int(round(0.2 * ref))):
                clusters[-1].append(n)
            else:
                clusters.append([n])
        for c in clusters:
            dur = int(round(float(np.median([n.dur for n in c]))))
            events.append(Event(start, max(1, dur), tuple(sorted({n.midi for n in c}))))
    return events


def _tidy(events: list[Event], overlap_tol: int, legato_fill: int, sub: int) -> None:
    """Remove sloppiness from real playing.

    * a note that overlaps the next one by a hair is cut at the next note;
    * a note that stops a hair before the next note is stretched to it;
    * a note that stops a hair before a beat (and nothing else starts sooner) is stretched
      to that beat, so a half note played a bit short is not written as a tied 7/16.
    """
    starts = sorted({e.start for e in events})
    nxt = {s: starts[i + 1] for i, s in enumerate(starts[:-1])}
    for e in events:
        s2 = nxt.get(e.start)
        if s2 is not None:
            if e.end > s2 and e.end - s2 <= overlap_tol:
                e.dur = s2 - e.start
                continue
            if e.end < s2 and s2 - e.end <= legato_fill:
                e.dur = s2 - e.start
                continue
        beat_end = -(-e.end // sub) * sub
        tol = max(legato_fill, int(round(0.15 * e.dur)))      # long notes are released a bit earlier
        if e.end % sub and beat_end - e.end <= tol and (s2 is None or s2 >= beat_end):
            e.dur = beat_end - e.start


def _voices(events: list[Event], max_voices: int = 4) -> list[list[Event]]:
    """Greedy voice allocation: the upper line first, held notes go to lower voices."""
    voices: list[list[Event]] = []
    for e in sorted(events, key=lambda e: (e.start, -max(e.midis))):
        for v in voices:
            if v[-1].end <= e.start:
                v.append(e)
                break
        else:
            if len(voices) < max_voices:
                voices.append([e])
            else:                                   # too dense: cut the voice that ends soonest
                v = min(voices, key=lambda v: v[-1].end)
                if v[-1].start < e.start:
                    v[-1].dur = e.start - v[-1].start
                    v.append(e)
    return voices


# ----------------------------------------------------------------------------- score assembly
@dataclass
class ScoreInfo:
    bars: int = 0
    key: str = "C major"
    bpm: float = 100.0
    notes_rh: int = 0
    notes_lh: int = 0
    voices_rh: int = 1
    voices_lh: int = 1
    warnings: list[str] = field(default_factory=list)


def _spell(midi: int, flats: bool):
    from music21 import pitch
    p = pitch.Pitch(midi)
    # music21 picks its own spelling for black keys (e.g. E-flat but C-sharp): force the
    # spelling that matches the key signature's direction.
    if p.accidental is not None and (p.accidental.alter < 0) != flats:
        p = p.getEnharmonic()
    return p


def _make_item(kind: str, size: int, sub: int, midis: tuple[int, ...], flats: bool,
               tie_prev: bool, tie_next: bool, full_bar: bool = False):
    from music21 import chord, duration, note, tie
    ql = Fraction(size, sub)
    if kind == "rest":
        r = note.Rest()
        r.duration = duration.Duration(ql)
        if full_bar:
            r.fullMeasure = "always"
        return r
    pitches = [_spell(m, flats) for m in midis]
    obj = note.Note(pitches[0]) if len(pitches) == 1 else chord.Chord(pitches)
    obj.duration = duration.Duration(ql)
    tkind = "continue" if tie_prev and tie_next else "stop" if tie_prev else "start" if tie_next else None
    if tkind:
        for n in ([obj] if len(pitches) == 1 else list(obj.notes)):
            n.tie = tie.Tie(tkind)
    return obj


def _apply_accidentals(timeline: list[tuple[int, object]], key_alter: dict[str, int]) -> None:
    """Show an accidental only where the pitch differs from what the key signature (or an
    earlier note of the same pitch in this bar) already says."""
    from music21 import chord, pitch
    state: dict[tuple[str, int], int] = {}
    for _, obj in sorted(timeline, key=lambda x: x[0]):
        if not isinstance(obj, (chord.Chord,)) and not hasattr(obj, "pitch"):
            continue
        notes = list(obj.notes) if isinstance(obj, chord.Chord) else [obj]
        for n in notes:
            p = n.pitch
            alter = int(p.accidental.alter) if p.accidental is not None else 0
            cur = state.get((p.step, p.octave), key_alter.get(p.step, 0))
            tied_on = n.tie is not None and n.tie.type in ("stop", "continue")
            show = alter != cur and not tied_on
            if p.accidental is None and show:
                p.accidental = pitch.Accidental("natural")
            if p.accidental is not None:
                p.accidental.displayStatus = bool(show)
            state[(p.step, p.octave)] = alter


def _build_part(voices: list[list[Event]], n_bars: int, bar_units: int, sub: int, flats: bool,
                clef, key_sig, time_sig, tempo_mark):
    from music21 import stream
    part = stream.PartStaff()
    n_voice = max(len(voices), 1)
    key_alter = {p.step: int(p.accidental.alter) for p in key_sig.alteredPitches}
    # events split at barlines: per bar, per voice, list of (pos, size, midis, tie_prev, tie_next)
    grid: list[list[list[tuple]]] = [[[] for _ in range(n_voice)] for _ in range(n_bars)]
    for vi, v in enumerate(voices):
        for e in v:
            s, remaining, first = e.start, e.dur, True
            while remaining > 0:
                bar, pos = divmod(s, bar_units)
                take = min(remaining, bar_units - pos)
                last = take == remaining
                sizes = decompose(pos, take, sub)
                p = pos
                for k, size in enumerate(sizes):
                    tie_prev = (not first) or k > 0
                    tie_next = (not last) or k < len(sizes) - 1
                    grid[bar][vi].append((p, size, e.midis, tie_prev, tie_next))
                    p += size
                s += take
                remaining -= take
                first = False
    for b in range(n_bars):
        m = stream.Measure(number=b + 1)
        if b == 0:
            m.insert(0, clef)
            m.insert(0, key_sig)
            m.insert(0, time_sig)
            if tempo_mark is not None:
                m.insert(0, tempo_mark)
        voice_streams = []
        timeline: list[tuple[int, object]] = []
        for vi in range(n_voice):
            items, cursor = [], 0
            for pos, size, midis, tp, tn in sorted(grid[b][vi], key=lambda x: x[0]):
                if pos > cursor:
                    for rs in decompose(cursor, pos - cursor, sub):
                        items.append(_make_item("rest", rs, sub, (), flats, False, False))
                obj = _make_item("note", size, sub, midis, flats, tp, tn)
                items.append(obj)
                timeline.append((pos, obj))
                cursor = pos + size
            if cursor < bar_units:
                gap = bar_units - cursor
                if cursor == 0:
                    items.append(_make_item("rest", gap, sub, (), flats, False, False, full_bar=True))
                else:
                    for rs in decompose(cursor, gap, sub):
                        items.append(_make_item("rest", rs, sub, (), flats, False, False))
            voice_streams.append(items)
        _apply_accidentals(timeline, key_alter)
        if n_voice == 1:
            for it in voice_streams[0]:
                m.append(it)
        else:
            for vi, items in enumerate(voice_streams):
                v = stream.Voice()
                v.id = str(vi + 1)
                for it in items:
                    v.append(it)
                m.insert(0, v)
        part.append(m)
    return part


def build_score(r: Rhythm, title: str = "Transcription", time_sig: tuple[int, int] = (4, 4),
                split: int = 60, pickup_beats: float = 0.0, key: Optional[tuple[int, str, int]] = None,
                overlap_tol: int = 1, legato_fill: int = 1):
    """Return (music21 Score, ScoreInfo)."""
    from music21 import clef, key as m21key, layout, metadata, meter, stream, tempo, note

    sub = r.sub
    num, den = time_sig
    bar_units = int(round(num * 4 / den * sub))
    lead = (bar_units - int(round(pickup_beats * sub))) % bar_units if pickup_beats else 0
    notes = [QNote(n.midi, n.start + lead, n.dur) for n in r.notes]
    tonic, mode, sharps = key or estimate_key(notes)
    flats = sharps < 0
    info = ScoreInfo(key=key_name(tonic, mode), bpm=r.bpm)
    info.warnings.extend(r.warnings)

    hands = {"R": [n for n in notes if n.midi >= split], "L": [n for n in notes if n.midi < split]}
    info.notes_rh, info.notes_lh = len(hands["R"]), len(hands["L"])
    end = max((n.start + n.dur for n in notes), default=bar_units)
    n_bars = max(1, int(np.ceil(end / bar_units)))
    info.bars = n_bars

    parts = {}
    for hand in ("R", "L"):
        ev = _chord_events(hands[hand])
        _tidy(ev, overlap_tol, legato_fill, sub)
        voices = _voices(ev)
        if hand == "R":
            info.voices_rh = max(len(voices), 1)
        else:
            info.voices_lh = max(len(voices), 1)
        mm = tempo.MetronomeMark(number=int(round(r.bpm)), referent=note.Note(type="quarter")) \
            if hand == "R" else None
        parts[hand] = _build_part(
            voices, n_bars, bar_units, sub, flats,
            clef.TrebleClef() if hand == "R" else clef.BassClef(),
            m21key.KeySignature(sharps), meter.TimeSignature(f"{num}/{den}"), mm)

    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = "Transcribed from video"
    for p in (parts["R"], parts["L"]):
        score.insert(0, p)
    score.insert(0, layout.StaffGroup([parts["R"], parts["L"]], name="Piano", abbreviation="Pno.",
                                      symbol="brace", barTogether=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in (parts["R"], parts["L"]):
            try:
                p.makeBeams(inPlace=True)
            except Exception as e:            # beaming/accidentals are cosmetic
                info.warnings.append(f"note: could not finish engraving details ({e})")
    return score, info


# ----------------------------------------------------------------------------- output files
def write_musicxml(score, path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score.write("musicxml", fp=str(path))


def write_midi(notes: list[RawNote], path: Path, tempo_bpm: float = 120.0) -> None:
    """Performance MIDI with the real (unquantised) timing of the detected notes."""
    from music21 import note, stream, tempo
    s = stream.Part()
    s.insert(0, tempo.MetronomeMark(number=tempo_bpm))
    qps = tempo_bpm / 60.0
    t0 = min((n.onset for n in notes), default=0.0)
    for n in notes:
        nn = note.Note(n.midi)
        nn.quarterLength = max(0.05, n.duration * qps)
        nn.volume.velocity = 80
        s.insert((n.onset - t0) * qps, nn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.write("midi", fp=str(path))


def _plain_text_glyphs(svg: str) -> str:
    """Verovio writes the note symbol of a tempo mark as a character of its music font, which a
    PDF converter without that font shows as a missing glyph.  Use plain Unicode notes instead."""
    import re

    def fix(m):
        ch = m.group(2)
        sym = "\u266a" if ch in ("\ue1d7", "\ue1d8", "\ue1d9", "\ue1da") else "\u2669"
        return f'<tspan font-family="DejaVu Sans, sans-serif" font-size="{m.group(1)}px">{sym}</tspan>'

    return re.sub(r'<tspan font-family="Leipzig" font-size="(\d+)px">([^<])</tspan>', fix, svg)


def render_pdf(musicxml: Path, pdf_path: Path) -> Optional[str]:
    """Render the MusicXML to PDF.  Returns None on success, or a reason it was skipped."""
    import shutil
    import subprocess
    for exe in ("mscore", "musescore3", "musescore", "MuseScore4", "mscore4portable"):
        path = shutil.which(exe)
        if path:                                    # MuseScore gives the nicest result
            try:
                subprocess.run([path, "-o", str(pdf_path), str(musicxml)], check=True,
                               capture_output=True, timeout=300)
                if pdf_path.exists():
                    return None
            except Exception:
                pass
    try:
        import verovio
        import cairosvg
        from pypdf import PdfWriter, PdfReader
    except ImportError:
        return "install verovio, cairosvg and pypdf (pip install verovio cairosvg pypdf) for PDF output"
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 45, "pageMarginLeft": 80,
                   "pageMarginRight": 80, "pageMarginTop": 100, "pageMarginBottom": 80,
                   "footer": "none", "breaks": "auto", "svgViewBox": True})
    if not tk.loadFile(str(musicxml)):
        return "verovio could not read the MusicXML"
    writer = PdfWriter()
    for page in range(1, tk.getPageCount() + 1):
        svg = _plain_text_glyphs(tk.renderToSVG(page))
        pdf_bytes = cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), output_width=595, output_height=842)
        for pg in PdfReader(io.BytesIO(pdf_bytes)).pages:
            writer.add_page(pg)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return None
