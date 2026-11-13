"""Synthetic "falling notes" piano video, used for tests and demos.

It imitates the look of tutorials like the one this project was built for: a keyboard
at the bottom, gem-like note bars falling onto it, a glowing beam/sparks where a note
is being played, soft glow that bleeds onto neighbouring keys, hands over the keys
and some video noise.  Because the video is generated from a known note list we can
measure exactly how well the transcriber recovers it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from .keyboard import is_black, white_index

# black keys sit slightly off the white-key boundary, like on a real piano
_BLACK_OFFSET = {1: -0.10, 3: +0.10, 6: -0.14, 8: 0.0, 10: +0.14}


@dataclass
class SynthNote:
    midi: int
    t_on: float
    t_off: float
    beat_on: float = 0.0     # ideal position in quarter notes (ground truth for the score)
    beat_off: float = 0.0
    hand: str = ""           # 'L' / 'R': which hand plays it (ground truth), if known


# --------------------------------------------------------------------------- the demo piece
def demo_piece(bpm: float = 92.0, articulation: float = 0.9, jitter: float = 0.0,
               drift: float = 0.0, seed: int = 1) -> list[SynthNote]:
    """Ode to Joy (public domain) in D major, 4/4, with a simple left hand.

    ``jitter`` (seconds, std-dev) emulates human timing, ``drift`` (fraction) a slow
    tempo change over the piece.
    """
    rng = random.Random(seed)
    # right hand (pitch, beats)
    mel_a = [(66, 1), (66, 1), (67, 1), (69, 1), (69, 1), (67, 1), (66, 1), (64, 1),
             (62, 1), (62, 1), (64, 1), (66, 1)]
    end1 = [(66, 1.5), (64, .5), (64, 2)]
    end2 = [(64, 1.5), (62, .5), (62, 2)]
    rh: list[tuple[int, float]] = mel_a + end1 + mel_a + end2
    rh += [(62, 1), (62, 1), (64, 1), (66, 1)]            # 2 bars of a closing figure
    rh += [(64, 2), (62, 2)]
    # left hand: roots in half notes, with a held 2-note chord in bar 1
    lh_chords = [([50, 57], 4), ([45, 52], 4), ([50, 57], 4), ([45, 52], 4)] * 2
    lh_chords += [([50, 57], 4), ([45, 52], 4)]

    def to_time(beat: float) -> float:
        t = beat * 60.0 / bpm
        return t * (1.0 + drift * (beat / 40.0))  # slows down (drift>0) progressively

    notes: list[SynthNote] = []
    b = 0.0
    for p, dur in rh:
        on, off = to_time(b), to_time(b + dur * articulation)
        j = rng.gauss(0, jitter) if jitter else 0.0
        notes.append(SynthNote(p, on + j, off + j, b, b + dur))
        b += dur
    b = 0.0
    for pitches, dur in lh_chords:
        for p in pitches:
            j = rng.gauss(0, jitter) if jitter else 0.0
            notes.append(SynthNote(p, to_time(b) + j, to_time(b + dur * 0.98) + j, b, b + dur))
        b += dur
    return sorted(notes, key=lambda n: (n.t_on, n.midi))


def stress_piece(bpm: float = 80.0, jitter: float = 0.0, drift: float = 0.0, seed: int = 1,
                 articulation: float = 0.9) -> list[SynthNote]:
    """A harder 4/4 test: adjacent-key chords, repeated notes, a fast run, black-key notes."""
    rng = random.Random(seed)
    # (beat, [midis], beats)
    ev = [
        (0.0, [60, 62], 1), (1.0, [64, 65], 1), (2.0, [59, 60], 1), (3.0, [61, 63], 1),   # dyads on adjacent keys
        (4.0, [67], .5), (4.5, [67], .5), (5.0, [67], .5), (5.5, [67], .5),                # repeated notes
        (6.0, [72], 1), (7.0, [61], 1),                                                    # isolated black key
        (8.0, [60], .25), (8.25, [62], .25), (8.5, [64], .25), (8.75, [65], .25),          # fast run of 16ths
        (9.0, [67], .25), (9.25, [69], .25), (9.5, [71], .25), (9.75, [72], .25),
        (10.0, [66, 70], 2), (12.0, [48, 55, 64], 1), (13.0, [49, 56], 1),                # triad, black-key chords
        (14.0, [60, 72], 1.5), (15.5, [61], .5), (16.0, [62, 66, 69], 4),
    ]
    def to_time(beat: float) -> float:
        return beat * 60.0 / bpm * (1.0 + drift * beat / 40.0)
    notes = []
    for beat, pitches, dur in ev:
        for p in pitches:
            j = rng.gauss(0, jitter) if jitter else 0.0
            notes.append(SynthNote(p, to_time(beat) + j, to_time(beat + dur * articulation) + j, beat, beat + dur))
    return sorted(notes, key=lambda n: (n.t_on, n.midi))


def hands_piece(bpm: float = 90.0, jitter: float = 0.0, seed: int = 1, articulation: float = 0.9) -> list[SynthNote]:
    """12 bars where a split at middle C gets the hands wrong, but the hands never cross.

    bars 1-4:  ordinary: right hand melody above C4, left hand bass below.
    bars 5-8:  the right hand plays low (down to G3), the left hand plays low bass.
    bars 9-12: the left hand takes the tune *above* middle C while the right hand holds
               chords still higher.
    """
    rng = random.Random(seed)
    ev: list[tuple[float, int, float, str]] = []           # (beat, midi, beats, hand)

    def add(beat, midis, dur, hand):
        for m in (midis if isinstance(midis, (list, tuple)) else [midis]):
            ev.append((beat, m, dur, hand))

    for bar, (mel, bass) in enumerate([([64, 65, 67, 69], 48), ([67, 65, 64, 62], 43),
                                       ([60, 62, 64, 65], 48), ([64, 62, 60, 64], 43)]):
        for i, m in enumerate(mel):
            add(bar * 4 + i, m, 1, "R")
        add(bar * 4, [bass, bass + 7], 4, "L")
    for bar, (mel, bass) in enumerate([([59, 60, 62, 60], 45), ([59, 57, 55, 57], 43),
                                       ([59, 60, 62, 64], 45), ([62, 60, 59, 55], 43)]):
        for i, m in enumerate(mel):
            add(16 + bar * 4 + i, m, 1, "R")
        for i in range(2):
            add(16 + bar * 4 + 2 * i, bass - 5 * (i % 2), 2, "L")
    for bar, mel in enumerate([[60, 62, 64, 62], [60, 59, 57, 55], [62, 64, 65, 64], [62, 60, 60, 60]]):
        for i, m in enumerate(mel):
            add(32 + bar * 4 + i, m, 1, "L")
        add(32 + bar * 4, [72, 76], 4, "R")

    def to_time(beat: float) -> float:
        return beat * 60.0 / bpm
    notes = []
    for beat, m, dur, hand in ev:
        j = rng.gauss(0, jitter) if jitter else 0.0
        notes.append(SynthNote(m, to_time(beat) + j, to_time(beat + dur * articulation) + j,
                               beat, beat + dur, hand))
    return sorted(notes, key=lambda n: (n.t_on, n.midi))


# --------------------------------------------------------------------------- rendering
class _Layout:
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.w = W / 52.0
        self.top = int(H * 0.46)
        self.kh = int(H * 0.23)
        self.bottom = self.top + self.kh

    def x(self, m: int) -> float:
        base = white_index(21)
        if not is_black(m):
            return (white_index(m) - base + 0.5) * self.w
        return (white_index(m - 1) + 1 - base) * self.w + _BLACK_OFFSET[m % 12] * self.w


def _keyboard_image(L: _Layout) -> np.ndarray:
    img = np.zeros((L.H, L.W, 3), np.float32)
    y0, y1 = L.top, L.bottom
    img[y0:y1] = (222, 218, 212)
    for i in range(53):                                 # gaps between white keys
        x = int(round(i * L.w))
        img[y0:y1, max(0, x - 1):x + 1] = (150, 150, 150)
    for m in range(21, 109):
        if is_black(m):
            xc, bw = L.x(m), 0.58 * L.w
            img[y0:y0 + int(0.62 * L.kh), int(xc - bw / 2):int(xc + bw / 2)] = (14, 14, 14)
    fade = np.clip((y1 - np.arange(y0, y1)) / (0.2 * L.kh), 0, 1)[:, None, None]
    vign = np.clip(0.55 + 0.45 * np.minimum(np.arange(L.W) / (0.15 * L.W), 1), 0, 1)[None, :, None]
    img[y0:y1] *= fade * vign
    img[y0 - 2:y0] = (205, 215, 215)                    # bright edge where notes land
    return img


def _hand(canvas: np.ndarray, L: _Layout, xc: float, spread: float) -> None:
    skin, dark = (70, 130, 205), (40, 85, 150)
    y = L.top + 0.52 * L.kh
    w = L.w
    cv2.fillConvexPoly(canvas, np.array([[xc - 1.3 * w, L.H], [xc + 1.3 * w, L.H],
                                         [xc + 1.5 * w, y + 1.2 * w], [xc - 1.5 * w, y + 1.2 * w]], np.int32), dark)
    cv2.ellipse(canvas, (int(xc), int(y + 0.6 * w)), (int(1.7 * w), int(1.3 * w)), 0, 0, 360, skin, -1)
    for k in range(5):
        ang = np.deg2rad(-90 + (k - 2) * 17 * spread)
        fx = xc + np.sin(ang) * 2.6 * w + (k - 2) * 0.2 * w
        fy = y - np.cos(ang) * (1.7 + 0.15 * (2 - abs(k - 2))) * w + 0.5 * w
        cv2.line(canvas, (int(xc + (k - 2) * 0.55 * w), int(y + 0.4 * w)), (int(fx), int(fy)), skin,
                 max(2, int(0.62 * w)), cv2.LINE_AA)
        cv2.circle(canvas, (int(fx), int(fy)), max(1, int(0.34 * w)), (90, 150, 220), -1, cv2.LINE_AA)


def _paste_sprite(img: np.ndarray, sprite: dict, x_center: float, y_top: float) -> None:
    """Alpha-blend a hand picture onto ``img``.  ``sprite``: {'img' BGR, 'alpha' 0..1, 'cx'}."""
    pic, alpha = sprite["img"], sprite["alpha"]
    h, w = alpha.shape
    x0 = int(round(x_center - sprite["cx"]))
    y0 = int(round(y_top))
    xs0, ys0 = max(0, -x0), max(0, -y0)
    xs1, ys1 = min(w, img.shape[1] - x0), min(h, img.shape[0] - y0)
    if xs1 <= xs0 or ys1 <= ys0:
        return
    a = alpha[ys0:ys1, xs0:xs1, None]
    dst = img[y0 + ys0:y0 + ys1, x0 + xs0:x0 + xs1]
    dst[:] = dst * (1 - a) + pic[ys0:ys1, xs0:xs1].astype(np.float32) * a


def render_video(notes: list[SynthNote], path: str, width: int = 1280, height: int = 720,
                 fps: float = 30.0, seconds_visible: float = 3.2, style: str = "burn",
                 tail: float = 1.5, lead: float = 3.0, seed: int = 0, noise: float = 2.0,
                 sprites: dict | None = None) -> float:
    """Render ``notes`` to an mp4.  Returns the video duration in seconds.

    ``sprites={'L': {...}, 'R': {...}}`` replaces the cartoon hands by pictures (see
    ``_paste_sprite``; each also has 'top', the sprite's top edge relative to the keyboard top in
    pixels).  The hands then follow the notes of their own hand when the notes carry a ``hand``
    label, otherwise the notes below/above middle C.

    ``style='burn'`` consumes each bar in a sparkle beam about 2.4 key-widths above the
    keys (like the reference video); ``'classic'`` lets bars fall right onto the keys.
    """
    rng = np.random.default_rng(seed)
    L = _Layout(width, height)
    speed = (L.top - 0.04 * height) / seconds_visible            # px per second
    y_burn = L.top - (2.4 * L.w if style == "burn" else 0.0)
    shift = lead                                                 # video time = note time + lead
    duration = max(n.t_off for n in notes) + shift + tail

    bg = np.zeros((height, width, 3), np.float32)
    bg += rng.normal(0, 1.5, bg.shape).clip(0, None)
    for k in range(1, 8):                                        # dim octave guide lines
        bg[:L.top, int(round((7 * k - 5) * L.w))] = (45, 45, 45)
    kb = _keyboard_image(L)
    base = np.maximum(bg, 0)
    base[L.top - 2:L.bottom] = kb[L.top - 2:L.bottom]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError("could not open video writer")
    hx = {"L": L.x(48), "R": L.x(66)}
    n_frames = int(duration * fps)
    for f in range(n_frames):
        t = f / fps - shift
        img = base.copy()
        light = np.zeros((height, width, 3), np.float32)     # glow, blurred below
        sharp = np.zeros((height, width, 3), np.float32)     # crisp gems / sparks
        active: list[int] = []
        for n in notes:
            bottom = L.top - (n.t_on - t) * speed
            top = L.top - (n.t_off - t) * speed
            if bottom < 0 or top > L.top:
                continue
            xc = L.x(n.midi)
            gw = 0.62 * L.w if is_black(n.midi) else 0.92 * L.w
            x0, x1 = int(xc - gw / 2), int(xc + gw / 2) + 1
            sounding = n.t_on <= t < n.t_off
            ye = min(bottom, y_burn)
            ys = max(top, 0.0)
            if ys < ye:
                prof = 1.0 - 0.55 * np.abs(np.linspace(-1, 1, x1 - x0)) ** 2
                col = np.array([40, 205, 240], np.float32)         # gold (BGR)
                y_a, y_b = int(ys), max(int(ye), int(ys) + 1)
                sharp[y_a:y_b, x0:x1] = col[None, None, :] * prof[None, :, None]
                sharp[y_a:y_a + 2, x0:x1] = (170, 240, 255)         # bevel highlight
                light[y_a:y_b, max(0, x0 - 2):x1 + 2] += (25, 70, 90)
            if sounding:
                active.append(n.midi)
                by = int(min(ye, L.top))
                h = max(L.top - by, 1)
                bw = int(2.0 * L.w)
                ramp = np.linspace(0.35, 1.0, h)[:, None, None]
                light[by:L.top, int(xc - bw / 2):int(xc + bw / 2)] += ramp * np.array([220, 240, 250], np.float32)
                for _ in range(int(h / 2.5)):                       # sparkles in the beam
                    sx = xc + rng.normal(0, 0.55 * L.w)
                    sy = by + rng.random() * h
                    cv2.circle(sharp, (int(sx), int(sy)), 1, (210, 235, 250), -1)
                fy = int(max(0, by - 3 * L.w))                        # dim flames above the gem
                light[fy:by, int(xc - 0.7 * L.w):int(xc + 0.7 * L.w)] += np.array([15, 45, 90], np.float32) * rng.random()
        light = cv2.GaussianBlur(light, (0, 0), 0.6 * L.w)
        img = img + sharp + light * 1.0
        for m in active:                                             # keys light up
            xc = L.x(m)
            kw = 0.5 * L.w
            y_a = L.top
            img[y_a:L.bottom, int(xc - kw / 2):int(xc + kw / 2)] += np.array([20, 70, 60], np.float32) * 0.9
        if any(n.hand for n in notes):               # hands follow the notes they really play
            near = {"L": [], "R": []}
            for n in notes:
                if n.hand and n.t_on - 0.35 <= t < n.t_off:
                    near[n.hand].append(L.x(n.midi))
            lo, hi = near["L"], near["R"]
        else:
            lo = [L.x(m) for m in active if m < 60]
            hi = [L.x(m) for m in active if m >= 60]
        hx["L"] += 0.25 * ((np.mean(lo) if lo else hx["L"]) - hx["L"])
        hx["R"] += 0.25 * ((np.mean(hi) if hi else hx["R"]) - hx["R"])
        if sprites:
            for side in ("L", "R"):
                _paste_sprite(img, sprites[side], hx[side], L.top + sprites[side]["top"])
        else:
            _hand(img, L, hx["L"], 1.0)
            _hand(img, L, hx["R"], 1.0)
        img[:int(0.04 * height)] = 0                                   # letterbox bars
        img[-int(0.04 * height):] = 0
        img += rng.normal(0, noise, img.shape)
        vw.write(np.clip(img, 0, 255).astype(np.uint8))
    vw.release()
    return duration
