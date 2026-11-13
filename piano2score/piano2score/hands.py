"""Decide which hand plays each note, using the hands visible in the video.

How it works
------------
1. A hand-landmark model (MediaPipe) finds both hands in the video and gives the position of
   every fingertip.  Left and right are decided by position (the left hand is the left one),
   with continuity from frame to frame when only one hand is visible.
2. Notes are detected a few key-widths *above* the keys, so a note is seen a moment before
   the finger presses it.  The speed of the falling bars is measured by reading two bands at
   different heights, which gives that delay exactly.
3. At the moment a key is pressed, the note goes to the hand whose fingertips are closest to
   that key.  Notes where the two hands are about equally close (hands overlapping or
   crossing, a hand out of view) are decided from the neighbouring notes instead:
   a hand plays notes near the ones it just played.
4. Anything still undecided falls back to a split by pitch.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .detect import RawNote
from .keyboard import Keyboard

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
_TIPS = (4, 8, 12, 16, 20)                  # thumb ... little finger tips
_PALM = (0, 5, 9, 13, 17)                   # wrist and the knuckles: a stable "centre of the hand"


# ----------------------------------------------------------------------------- the model
def model_path() -> Path:
    env = os.environ.get("PIANO2SCORE_HAND_MODEL")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "piano2score" / "hand_landmarker.task"


def ensure_model() -> Path:
    """Return the hand-landmark model, downloading it (7.8 MB, once) if necessary."""
    path = model_path()
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the hand-tracking model to {path} (one time, 8 MB) ...")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, \
                tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(r.read())
        os.replace(tmp.name, path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the hand-tracking model ({e}). Download it manually from\n  {MODEL_URL}\n"
            f"and save it as {path} (or set PIANO2SCORE_HAND_MODEL to its location).") from e
    return path


# ----------------------------------------------------------------------------- data
@dataclass
class Hand:
    side: str                 # 'L' or 'R'
    tips_x: np.ndarray        # (5,) fingertip x positions (processing-resolution pixels)
    tips_y: np.ndarray
    palm_x: float
    palm_y: float
    label: str = ""           # what the model called it ("Left"/"Right"), kept for diagnostics
    score: float = 1.0


@dataclass
class HandTrack:
    times: np.ndarray                     # (T,) seconds
    frames: list[list[Hand]]              # hands found in each frame
    dt: float                             # time between tracked frames
    debug_image: Optional[np.ndarray] = None

    def at(self, t: float) -> list[Hand]:
        """Hands in the tracked frame nearest to ``t`` (nothing if that frame is too far away)."""
        if len(self.times) == 0:
            return []
        i = int(np.searchsorted(self.times, t))
        cand = [j for j in (i - 1, i) if 0 <= j < len(self.times)]
        j = min(cand, key=lambda j: abs(self.times[j] - t))
        return self.frames[j] if abs(self.times[j] - t) <= 1.5 * self.dt else []

    def between(self, t0: float, t1: float) -> list[list[Hand]]:
        i0 = int(np.searchsorted(self.times, t0 - 0.5 * self.dt))
        i1 = int(np.searchsorted(self.times, t1 + 0.5 * self.dt))
        return self.frames[i0:i1]

    def coverage(self) -> dict[str, float]:
        """Share of frames in which each hand was really seen (not interpolated)."""
        n = max(len(self.frames), 1)

        def seen(side: str) -> float:
            return sum(any(h.side == side and h.label not in ("filled", "held") for h in f)
                       for f in self.frames) / n
        return {"L": seen("L"), "R": seen("R")}


# ----------------------------------------------------------------------------- detection
class HandDetector:
    """Thin wrapper around MediaPipe's HandLandmarker (video mode, two hands)."""

    def __init__(self, model: Optional[Path] = None, confidence: float = 0.3):
        os.environ.setdefault("GLOG_minloglevel", "2")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mpp
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise RuntimeError("hand tracking needs mediapipe: pip install mediapipe") from e
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(model or ensure_model())),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=confidence, min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence)
        self._lm = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def close(self) -> None:
        """Release the model now (otherwise MediaPipe complains while Python shuts down)."""
        if self._lm is not None:
            self._lm.close()
            self._lm = None

    def detect(self, bgr: np.ndarray, t_seconds: float) -> list[dict]:
        """Hands in ``bgr`` as dicts of pixel coordinates (relative to ``bgr``)."""
        ms = max(int(t_seconds * 1000), self._last_ms + 1)      # video mode needs rising timestamps
        self._last_ms = ms
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._lm.detect_for_video(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms)
        h, w = bgr.shape[:2]
        out = []
        for lm, hd in zip(res.hand_landmarks, res.handedness):
            xs = np.array([p.x * w for p in lm])
            ys = np.array([p.y * h for p in lm])
            out.append({"tips_x": xs[list(_TIPS)], "tips_y": ys[list(_TIPS)],
                        "palm_x": float(xs[list(_PALM)].mean()), "palm_y": float(ys[list(_PALM)].mean()),
                        "label": hd[0].category_name, "score": float(hd[0].score)})
        return out


# ----------------------------------------------------------------------------- left / right
def resolve_sides(raw_frames: list[list[dict]], white_w: float) -> list[list[Hand]]:
    """Name each detected hand left or right.

    With two hands the left one is the left hand.  With one hand, continuity from the last
    frames decides, and the model's own label breaks ties (the model's labelling convention is
    checked against the two-hand frames, so it also works if it is mirrored for this view).
    """
    # drop a hand that was found twice
    cleaned: list[list[dict]] = []
    for hands in raw_frames:
        hs = sorted(hands, key=lambda h: -h["score"])
        if len(hs) == 2 and abs(hs[0]["palm_x"] - hs[1]["palm_x"]) < white_w and \
                abs(hs[0]["palm_y"] - hs[1]["palm_y"]) < white_w:
            hs = hs[:1]
        cleaned.append(hs)

    agree = disagree = 0
    for hs in cleaned:
        if len(hs) == 2 and abs(hs[0]["palm_x"] - hs[1]["palm_x"]) > 3 * white_w:
            left = min(hs, key=lambda h: h["palm_x"])
            if left["label"] == "Left":
                agree += 1
            else:
                disagree += 1
    flip = disagree > agree

    def label_side(h: dict) -> str:
        side = "L" if h["label"] == "Left" else "R"
        return ("R" if side == "L" else "L") if flip else side

    last: dict[str, Optional[float]] = {"L": None, "R": None}
    out: list[list[Hand]] = []
    for hs in cleaned:
        sides: list[str] = []
        if len(hs) == 2:
            order = sorted(range(2), key=lambda i: hs[i]["palm_x"])
            sides = ["L" if i == order[0] else "R" for i in range(2)]
        elif len(hs) == 1:
            h = hs[0]
            dl = abs(h["palm_x"] - last["L"]) if last["L"] is not None else None
            dr = abs(h["palm_x"] - last["R"]) if last["R"] is not None else None
            if dl is not None and dr is not None and abs(dl - dr) > 2 * white_w:
                sides = ["L" if dl < dr else "R"]
            elif dl is not None and dr is None and dl < 8 * white_w:
                sides = ["L"]
            elif dr is not None and dl is None and dr < 8 * white_w:
                sides = ["R"]
            else:
                sides = [label_side(h)]
        frame_hands = []
        for h, sd in zip(hs, sides):
            last[sd] = h["palm_x"]
            frame_hands.append(Hand(sd, h["tips_x"], h["tips_y"], h["palm_x"], h["palm_y"], h["label"], h["score"]))
        out.append(frame_hands)
    return out


def fill_gaps(frames: list[list[Hand]], times: np.ndarray, max_gap: float = 0.7,
              hold: float = 0.3) -> list[list[Hand]]:
    """Fill short dropouts of a hand (motion blur, a finger hiding it).

    A hand missing for at most ``max_gap`` seconds between two sightings is interpolated; one
    that disappears for good is assumed to stay where it was for ``hold`` seconds.
    """
    out = [list(f) for f in frames]
    for side in ("L", "R"):
        seen = [i for i, f in enumerate(frames) if any(h.side == side for h in f)]
        by_i = {i: next(h for h in frames[i] if h.side == side) for i in seen}
        for a, b in zip(seen, seen[1:] + [None]):
            ha = by_i[a]
            lo = a + 1
            hi = b if b is not None else len(frames)
            for i in range(lo, hi):
                dt_a = times[i] - times[a]
                if b is not None and times[b] - times[a] <= max_gap:
                    f = (times[i] - times[a]) / (times[b] - times[a])
                    hb = by_i[b]
                    out[i].append(Hand(side, ha.tips_x + f * (hb.tips_x - ha.tips_x),
                                       ha.tips_y + f * (hb.tips_y - ha.tips_y),
                                       ha.palm_x + f * (hb.palm_x - ha.palm_x),
                                       ha.palm_y + f * (hb.palm_y - ha.palm_y), "filled", 0.5))
                elif b is None and dt_a <= hold:
                    out[i].append(Hand(side, ha.tips_x, ha.tips_y, ha.palm_x, ha.palm_y, "held", 0.5))
    return out


def draw_hands(img: np.ndarray, hands: list[Hand]) -> np.ndarray:
    vis = img.copy()
    for h in hands:
        col = (60, 170, 255) if h.side == "R" else (255, 140, 60)
        for x, y in zip(h.tips_x, h.tips_y):
            cv2.circle(vis, (int(x), int(y)), 8, col, 2, cv2.LINE_AA)
        cv2.putText(vis, "right hand" if h.side == "R" else "left hand", (int(h.palm_x) - 50, int(h.palm_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    return vis


def track_hands(info, kb: Keyboard, t_start: float = 0.0, t_end: Optional[float] = None,
                fps: float = 15.0, progress: Optional[Callable[[float], None]] = None,
                detector: Optional[HandDetector] = None) -> HandTrack:
    """Run the hand detector over the video and name the hands left/right."""
    own = detector is None
    det = detector or HandDetector()
    try:
        return _track(info, kb, det, t_start, t_end, fps, progress)
    finally:
        if own:
            det.close()


def _track(info, kb: Keyboard, det: HandDetector, t_start: float, t_end: Optional[float], fps: float,
           progress: Optional[Callable[[float], None]]) -> HandTrack:
    from . import video
    y0 = int(max(0, kb.y_top - 2.0 * kb.white_w))             # hands are at/below the keys
    t_end_eff = info.duration if t_end is None else min(t_end, info.duration)
    raw: list[list[dict]] = []
    times: list[float] = []
    for t, img in video.iter_frames(info, fps, t_start, t_end, rows=(y0, info.proc_size[1])):
        hs = det.detect(img, t)
        for h in hs:                                           # back to full-frame coordinates
            h["tips_y"] = h["tips_y"] + y0
            h["palm_y"] = h["palm_y"] + y0
        raw.append(hs)
        times.append(t)
        if progress:
            progress((t - t_start) / max(t_end_eff - t_start, 1e-6))
    if not times:
        return HandTrack(np.array([]), [], 1.0 / fps)
    frames = resolve_sides(raw, kb.white_w)
    track_times = np.asarray(times)
    frames = fill_gaps(frames, track_times)
    track = HandTrack(track_times, frames, float(np.median(np.diff(times))) if len(times) > 1 else 1.0 / fps)
    # debug picture: the first frame (after 30 % of the video) in which both hands were found
    for t, fr in zip(times, frames):
        if len(fr) == 2 and t >= t_start + 0.3 * (t_end_eff - t_start):
            full = None
            for _, img in video.iter_frames(info, info.fps, t, t + 1.5 / info.fps):
                full = img
                break
            if full is not None:
                track.debug_image = draw_hands(full, fr)
            break
    return track


# ----------------------------------------------------------------------------- timing offset
@dataclass
class FallTiming:
    delay: float          # seconds from "seen in the reading band" to "key pressed"
    speed: float          # bar speed in pixels per second
    matches: int


def estimate_fall_delay(notes_a: list[RawNote], notes_b: list[RawNote], dist_a: float,
                        dist_b: float) -> Optional[FallTiming]:
    """Delay between a note appearing in band A and its key being pressed.

    ``dist_a`` / ``dist_b`` are the distances (pixels) of the two reading bands above the keys.
    The same bar passes both bands, one after the other; the time between gives its speed.
    """
    if abs(dist_b - dist_a) < 1:
        return None
    sign = 1.0 if dist_b > dist_a else -1.0
    by_pitch: dict[int, list[RawNote]] = {}
    for n in notes_a:
        by_pitch.setdefault(n.midi, []).append(n)
    lags = []
    for nb in notes_b:
        best = None
        for na in by_pitch.get(nb.midi, []):
            lag = (na.onset - nb.onset) * sign          # positive: B (the higher band) sees it first
            if 0.01 < lag < 4.0 and abs(na.duration - nb.duration) <= max(0.08, 0.25 * nb.duration):
                if best is None or lag < best:
                    best = lag
        if best is not None:
            lags.append(best)
    if len(lags) < max(5, 0.2 * min(len(notes_a), len(notes_b))):
        return None
    lags_a = np.array(lags)
    med = float(np.median(lags_a))
    if float(np.median(np.abs(lags_a - med))) > 0.05 + 0.1 * med:      # matches disagree: do not trust
        return None
    speed = abs(dist_b - dist_a) / med
    return FallTiming(delay=dist_a / speed, speed=speed, matches=len(lags))


# ----------------------------------------------------------------------------- assignment
@dataclass
class AssignStats:
    total: int = 0
    from_video: int = 0
    from_neighbours: int = 0
    from_pitch: int = 0

    def summary(self) -> str:
        if not self.total:
            return "no notes"
        f = lambda n: f"{100.0 * n / self.total:.0f}%"        # noqa: E731
        return (f"{f(self.from_video)} decided by the hands in the video, "
                f"{f(self.from_neighbours)} by neighbouring notes, {f(self.from_pitch)} by pitch")


def _hand_cost(hand: Hand, kx: float, w: float) -> float:
    """How far (in key widths) a hand is from a key: nearest fingertip, plus a little palm."""
    return float(np.min(np.abs(hand.tips_x - kx))) / w + 0.1 * abs(hand.palm_x - kx) / w


def assign_hands(notes: list[RawNote], kb: Keyboard, track: HandTrack, delay: Optional[float],
                 split_midi: int = 60, confident_margin: float = 1.0) -> AssignStats:
    """Set ``note.hand`` ('L'/'R') and ``note.hand_source`` for every note."""
    w = kb.white_w
    xmap = {int(m): float(x) for m, x in zip(kb.midis, kb.x_center)}
    stats = AssignStats(total=len(notes))
    verdict: list[Optional[str]] = [None] * len(notes)       # confident answers
    tentative: list[Optional[str]] = [None] * len(notes)     # weak answers

    for i, n in enumerate(notes):
        kx = xmap.get(n.midi)
        if kx is None:
            continue
        if delay is not None:
            t_press = n.onset + delay
            frames = track.between(t_press - 0.08, t_press + min(0.20, 0.5 * n.duration))
        else:                                                # timing offset unknown: look ahead
            frames = track.between(n.onset, n.onset + 1.2)
        cost: dict[str, Optional[float]] = {"L": None, "R": None}
        for side in ("L", "R"):
            cs = [_hand_cost(h, kx, w) for fr in frames for h in fr if h.side == side]
            if cs:
                cost[side] = float(np.mean(cs)) if delay is not None else float(np.min(cs))
        cl, cr = cost["L"], cost["R"]
        if cl is not None and cr is not None:
            win = "L" if cl < cr else "R"
            if abs(cl - cr) >= confident_margin:
                verdict[i] = win
            else:
                tentative[i] = win
        elif cl is not None or cr is not None:
            side, c = ("L", cl) if cl is not None else ("R", cr)
            other = "R" if side == "L" else "L"
            if c <= 2.5:
                verdict[i] = side
            elif c >= 4.5:
                tentative[i] = other                         # the hand we see is far: the other one plays it

    # second pass: undecided / weak notes follow the nearby notes of each hand
    order = np.argsort([n.onset for n in notes])
    onsets = np.array([notes[j].onset for j in order])
    for i, n in enumerate(notes):
        if verdict[i] is not None:
            n.hand, n.hand_source = verdict[i], "video"
            stats.from_video += 1
            continue
        lo = int(np.searchsorted(onsets, n.onset - 1.5))
        hi = int(np.searchsorted(onsets, n.onset + 1.5))
        dist = {"L": 99.0, "R": 99.0}                       # semitones to the nearest sure note of each hand
        for j in order[lo:hi]:
            if verdict[j] is not None and j != i:
                dist[verdict[j]] = min(dist[verdict[j]], abs(notes[j].midi - n.midi))
        both = dist["L"] < 99 and dist["R"] < 99
        guess: Optional[str] = tentative[i]
        if both:
            nearer = "L" if dist["L"] < dist["R"] else "R"
            gap = abs(dist["L"] - dist["R"])
            if guess is None and gap >= 1:
                guess = nearer
            elif guess is not None and guess != nearer and gap >= 5:
                guess = nearer                               # neighbours clearly disagree with a weak video hint
        elif guess is None:
            only = "L" if dist["L"] < 99 else "R" if dist["R"] < 99 else None
            if only is not None and dist[only] <= 7:         # close to one hand's recent notes: same hand
                guess = only
        if guess is not None:
            n.hand, n.hand_source = guess, "neighbours"
            stats.from_neighbours += 1
        else:
            n.hand, n.hand_source = ("R" if n.midi >= split_midi else "L"), "pitch"
            stats.from_pitch += 1
    return stats
