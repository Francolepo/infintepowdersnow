"""Find the piano keyboard in a video and map every key (MIDI number) to an x position.

Strategy
--------
1. Take a few dozen frames spread over the video.  The *median* of them removes
   falling notes and (mostly) hands, leaving the static keyboard.  The keyboard
   is the tall bright band, so its top/bottom rows are easy to find.
2. Black keys are located on a row that crosses them, using the 20th percentile
   over time (a finger can hide a black key in a few frames, but not in most).
3. The detected black-key centres are aligned with the real piano pattern
   (2 black keys, gap, 3 black keys, gap ...) which tells us the pitch of every
   key, and gives a very accurate white-key width.
4. If anything fails, fall back to "88 keys spread over the keyboard width".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

BLACK_PCS = (1, 3, 6, 8, 10)
WHITE_PCS = (0, 2, 4, 5, 7, 9, 11)
_WHITE_IDX = {pc: i for i, pc in enumerate(WHITE_PCS)}
# centre of each pitch class inside an octave, in white-key widths from the left edge of C
_POS = {0: .5, 1: 1., 2: 1.5, 3: 2., 4: 2.5, 5: 3.5, 6: 4., 7: 4.5, 8: 5., 9: 5.5, 10: 6., 11: 6.5}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (lowest MIDI, highest MIDI) of the common keyboard sizes: 88, 76, 61, 49, 37, 25 keys
LAYOUTS = [(21, 108), (28, 103), (36, 96), (36, 84), (41, 77), (48, 72)]


def is_black(m: int) -> bool:
    return m % 12 in BLACK_PCS


def key_pos(m: int) -> float:
    """Centre of key ``m`` in white-key widths (absolute, C-1 left edge = 0)."""
    return 7 * (m // 12) + _POS[m % 12]


def white_index(m: int) -> int:
    """Index of white key ``m`` counted over the whole MIDI range."""
    return 7 * (m // 12) + _WHITE_IDX[m % 12]


def white_midi(k: int) -> int:
    return 12 * (k // 7) + WHITE_PCS[k % 7]


def note_name(m: int) -> str:
    return f"{_NAMES[m % 12]}{m // 12 - 1}"


def parse_note(s: str) -> int:
    mt = re.fullmatch(r"\s*([A-Ga-g])([#b]?)(-?\d+)\s*", s)
    if not mt:
        raise ValueError(f"Not a note name: {s!r} (examples: A0, C4, F#2)")
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[mt.group(1).upper()]
    pc += {"#": 1, "b": -1, "": 0}[mt.group(2)]
    return 12 * (int(mt.group(3)) + 1) + pc


@dataclass
class Keyboard:
    y_top: float                 # row where the notes "land" (top edge of the keys)
    y_bottom: float
    white_w: float               # white-key width in pixels (processing resolution)
    midis: np.ndarray            # (K,) MIDI number of each key
    x_center: np.ndarray         # (K,) x of each key centre
    proc_w: int
    proc_h: int
    method: str = "black-key pattern"
    notes: list = field(default_factory=list)   # human-readable diagnostics
    white_half: float = 0.22     # half-width of the sampling window, in white-key widths
    black_half: float = 0.12
    white_gem_half: float = 0.46  # half-width of a white key's falling bar
    black_gem_half: float = 0.34  # half-width of a black key's falling bar

    @property
    def black_mask(self) -> np.ndarray:
        return np.array([is_black(int(m)) for m in self.midis])

    def windows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sampling window [x0, x1) of each key and a mask of keys that are on screen.

        A black key's bar is narrower than a white key, and it overlaps the neighbouring
        white keys' columns.  So a white key's window stops short of any adjacent black
        key's bar, otherwise a black note would also trigger the white keys beside it.
        """
        w = self.white_w
        mid_index = {int(m): i for i, m in enumerate(self.midis)}
        lo = np.empty(len(self.midis))
        hi = np.empty(len(self.midis))
        for i, m in enumerate(self.midis):
            m = int(m)
            c = self.x_center[i]
            if is_black(m):
                lo[i], hi[i] = c - self.black_half * w, c + self.black_half * w
                continue
            l, h = c - self.white_half * w, c + self.white_half * w
            for nb, side in ((m - 1, -1), (m + 1, +1)):
                j = mid_index.get(nb)
                if j is not None and is_black(nb):
                    edge = self.x_center[j] - side * self.black_gem_half * w
                    if side > 0:
                        h = min(h, edge)
                    else:
                        l = max(l, edge)
            if h - l < 0.12 * w:                       # keep a usable window
                l, h = c - 0.06 * w, c + 0.06 * w
            lo[i], hi[i] = l, h
        x0 = np.floor(lo).astype(int)
        x1 = np.ceil(hi).astype(int) + 1
        valid = (x1 > 0) & (x0 < self.proc_w)
        x0 = np.clip(x0, 0, self.proc_w - 1)
        x1 = np.clip(x1, x0 + 1, self.proc_w)
        return x0, x1, valid

    def footprints(self) -> tuple[np.ndarray, np.ndarray]:
        """Left/right pixel edge of the falling bar each key would produce."""
        half = np.where(self.black_mask, self.black_gem_half, self.white_gem_half) * self.white_w
        return self.x_center - half, self.x_center + half

    def strip_rows(self, offset_keys: float = 1.0, height_px: Optional[float] = None) -> tuple[int, int]:
        """Rows (y0, y1) of the thin band used to detect notes: just above the keys.

        A note is visible in this band exactly while its key is sounding.  The band
        sits ``offset_keys`` white-key widths above the keyboard to stay clear of the
        glow/sparks that many tutorials draw where the notes land.
        """
        h = height_px if height_px else max(2.0, 0.08 * self.white_w)
        y1 = self.y_top - offset_keys * self.white_w
        y0 = y1 - h
        y0 = int(max(0, round(y0)))
        y1 = int(max(y0 + 1, round(y1)))
        return y0, y1


# --------------------------------------------------------------------------- helpers
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[0::2].tolist(), d[1::2].tolist()))


def _keyboard_rows(med: np.ndarray) -> tuple[int, int]:
    h, w = med.shape
    band = med[:, int(w * .1):int(w * .9)]
    rowm = cv2.blur(band.mean(axis=1).astype(np.float32).reshape(-1, 1), (1, 3)).ravel()
    ref = np.percentile(rowm, 97)
    if ref < 40:
        raise RuntimeError("Could not find a bright keyboard in the video.")
    runs = _runs(rowm > 0.4 * ref)
    runs = [r for r in runs if r[1] - r[0] >= 0.03 * h]
    if not runs:
        raise RuntimeError("Could not find the keyboard (no tall bright band).")
    s, e = max(runs, key=lambda r: float(rowm[r[0]:r[1]].sum()))
    # Refine both edges to the sharpest brightness step.  This ignores the soft glow that
    # some tutorials draw above the keys (it can survive in the median image).
    step = 4
    d = np.zeros_like(rowm)
    d[step:-step] = rowm[2 * step:] - rowm[:-2 * step]      # d[y] ~ rowm[y+step] - rowm[y-step]
    span = e - s
    def edge(lo: int, hi: int, sign: int) -> int:
        """Row of the sharpest brightness step, from the leading/trailing side of the gradient
        peak (a bright rim line at the keyboard edge would otherwise shift the peak)."""
        seg = sign * d[lo:hi + 1]
        above = np.flatnonzero(seg >= 0.6 * seg.max())
        if sign > 0:
            return lo + int(above[0]) + step
        return lo + int(above[-1]) - (step - 1)

    top = edge(max(step, s - step), min(h - step - 1, s + int(0.6 * span)), +1)
    bot = edge(max(top + 1, e - int(0.4 * span)), min(h - step - 1, e + step), -1)
    if bot - top < 0.03 * h:
        return s, e
    return top, bot


def _black_key_centres(row: np.ndarray, x_left: int, x_right: int) -> tuple[np.ndarray, float]:
    r = cv2.blur(row.astype(np.float32).reshape(1, -1), (3, 1)).ravel()
    k = max(15, len(r) // 25)
    local_max = cv2.dilate(r.reshape(1, -1), np.ones((1, k), np.uint8)).ravel()
    dark = (r / np.maximum(local_max, 1.0)) < 0.5     # "darker than the white keys around it"
    dark[:x_left] = False
    dark[x_right:] = False
    runs = _runs(dark)
    if len(runs) < 4:
        return np.array([]), 0.0
    widths = np.array([e - s for s, e in runs], dtype=float)
    ref = np.percentile(widths, 85)
    keep = [(s, e) for (s, e), wd in zip(runs, widths)
            if 0.55 * ref <= wd <= 1.5 * ref and s > x_left + 1 and e < x_right - 1]
    if not keep:
        return np.array([]), 0.0
    return np.array([(s + e - 1) / 2.0 for s, e in keep]), float(ref)


def _estimate_white_width(centres: np.ndarray) -> Optional[float]:
    """White-key width from black-key spacing.

    Black keys come in a repeating cycle of gaps (small, large, small, small, large) that
    spans exactly 7 white keys, whatever the exact black-key offsets of the piano are:
    ``w = (3 * small + 2 * large) / 7``.
    """
    gaps = np.diff(centres)
    gaps = gaps[gaps < 2.2 * np.median(gaps)]          # drop gaps where keys were missed
    if len(gaps) < 6:
        return None
    lo, hi = gaps.min(), gaps.max()
    for _ in range(20):                                # 1-D 2-means
        big = np.abs(gaps - hi) < np.abs(gaps - lo)
        if big.all() or (~big).all():
            return None
        lo, hi = gaps[~big].mean(), gaps[big].mean()
    if not 1.25 < hi / lo < 2.4:
        return None
    return float((3 * lo + 2 * hi) / 7.0)


def _fit_black_pattern(centres: np.ndarray, w_est: float):
    """Align detected black keys with the real piano pattern.

    Returns (a, b, midi_rel, ok) such that ``x = a + b * key_pos(midi_rel)`` for the
    keys flagged in ``ok``.  ``midi_rel`` is only known up to whole octaves.
    """
    lat_m = np.array([m for m in range(0, 128) if is_black(m)])
    lat_p = np.array([key_pos(int(m)) for m in lat_m])
    n = len(centres)
    best = None
    for pc0 in BLACK_PCS:
        a, b = centres[0] - w_est * key_pos(48 + pc0), w_est
        for horizon in range(6, n + 6, 3):          # grow the fitted span gradually
            idx = np.arange(min(horizon, n))
            pos = (centres[idx] - a) / b
            j = np.abs(pos[:, None] - lat_p[None, :]).argmin(axis=1)
            ok = np.abs(pos - lat_p[j]) < 0.3
            if ok.sum() >= 3:
                b, a = np.polyfit(lat_p[j][ok], centres[idx][ok], 1)
        for _ in range(2):
            pos = (centres - a) / b
            j = np.abs(pos[:, None] - lat_p[None, :]).argmin(axis=1)
            ok = np.abs(pos - lat_p[j]) < 0.3
            if ok.sum() >= 3:
                b, a = np.polyfit(lat_p[j][ok], centres[ok], 1)
        pos = (centres - a) / b
        j = np.abs(pos[:, None] - lat_p[None, :]).argmin(axis=1)
        d = np.abs(pos - lat_p[j])
        ok = d < 0.3
        score = ok.sum() - (d[ok].mean() if ok.any() else 1.0)
        if best is None or score > best[0]:
            best = (score, a, b, lat_m[j], ok)
    return best[1], best[2], best[3], best[4]


def _fallback_88(x_left: int, x_right: int, y_top: int, y_bot: int, w: int, h: int,
                 why: str) -> Keyboard:
    bw = (x_right - x_left) / 52.0
    midis = np.arange(21, 109)
    x = np.array([x_left + bw * (key_pos(int(m)) - 12.0) for m in midis])
    return Keyboard(y_top, y_bot, bw, midis, x, w, h, method="assumed 88 keys across the width",
                    notes=[why])


# --------------------------------------------------------------------------- main entry
def calibrate(frames: list[np.ndarray], first_note: Optional[int] = None,
              last_note: Optional[int] = None, y_top: Optional[float] = None) -> Keyboard:
    """Detect the keyboard from sample frames (BGR, processing resolution)."""
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames])
    h, w = gray.shape[1:]
    med = np.median(gray, axis=0)
    top, bot = _keyboard_rows(med)
    notes: list[str] = []
    if y_top is not None:
        top = int(y_top)
        bot = max(bot, top + int(0.05 * h))
        notes.append(f"keyboard top set manually to row {top}")
    kh = bot - top

    sub = gray[:, top:bot]
    p20 = np.percentile(sub, 20, axis=0)
    p80 = np.percentile(sub, 80, axis=0)

    # horizontal extent of the keyboard: bright columns on a row crossing only white keys
    rw = p80[int(.72 * kh):max(int(.72 * kh) + 2, int(.84 * kh))].mean(axis=0)
    bright = (rw > 0.4 * np.percentile(rw, 90)).astype(np.uint8).reshape(1, -1)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8)).ravel().astype(bool)
    xs = np.flatnonzero(bright)
    x_left, x_right = (int(xs[0]), int(xs[-1]) + 1) if len(xs) else (0, w)

    # black keys
    rb = p20[int(.12 * kh):max(int(.12 * kh) + 2, int(.38 * kh))].mean(axis=0)
    centres, bw = _black_key_centres(rb, x_left, x_right)
    if len(centres) < 8:
        return _fallback_88(x_left, x_right, top, bot, w, h,
                            f"only {len(centres)} black keys found; assuming a full 88-key keyboard")

    w_est = _estimate_white_width(centres)
    if w_est is None:
        return _fallback_88(x_left, x_right, top, bot, w, h,
                            "black-key spacing was not regular; assuming a full 88-key keyboard")
    a, b, m_rel, ok = _fit_black_pattern(centres, w_est)
    if ok.sum() < 0.7 * len(centres) or ok.sum() < 8:
        return _fallback_88(x_left, x_right, top, bot, w, h,
                            f"black-key pattern did not fit ({int(ok.sum())}/{len(centres)}); "
                            "assuming a full 88-key keyboard")
    notes.append(f"{int(ok.sum())}/{len(centres)} black keys matched the piano pattern, "
                 f"white key width {b:.2f}px")

    # leftmost / rightmost white key in the relative (octave-ambiguous) frame
    kL = int(round((x_left - a) / b))
    kR = int(round((x_right - a) / b)) - 1
    n_white = kR - kL + 1

    shift_sem: Optional[int] = None
    if first_note is not None:
        cand = first_note - white_midi(kL)
        shift_sem = 12 * int(round(cand / 12.0))
        if cand % 12 != 0:
            notes.append(f"--first-note {note_name(first_note)} does not line up with the key at the "
                         "left edge; using the nearest octave")
    else:
        for lo, hi in LAYOUTS:
            k_lo, k_hi = white_index(lo), white_index(hi)
            for s_w in (k_lo - kL, k_lo + 1 - kL):           # +1: first key partly cut off
                if s_w % 7:
                    continue
                if (kL + s_w) in (k_lo, k_lo + 1) and (kR + s_w) in (k_hi - 1, k_hi):
                    shift_sem = 12 * (s_w // 7)
                    break
            if shift_sem is not None:
                notes.append(f"looks like a {hi - lo + 1}-key keyboard ({note_name(lo)}-{note_name(hi)})")
                break
        if shift_sem is None:
            centre = (white_midi(kL) + white_midi(kR)) / 2.0
            shift_sem = 12 * int(round((64 - centre) / 12.0))
            notes.append(f"unusual keyboard size ({n_white} white keys); octave is a guess - "
                         "use --first-note/--last-note if the pitches look shifted")

    lo_m = white_midi(kL) + shift_sem
    hi_m = white_midi(kR) + shift_sem
    if first_note is not None:
        lo_m = first_note
    if last_note is not None:
        hi_m = last_note
    elif first_note is None:
        # include a partly hidden first/last key when the keyboard matches a known layout
        for lo, hi in LAYOUTS:
            if lo <= lo_m <= lo + 2 and hi - 1 <= hi_m <= hi:
                lo_m, hi_m = lo, hi
                break
    midis = np.arange(lo_m, hi_m + 1)
    x = np.array([a + b * key_pos(int(m) - shift_sem) for m in midis])
    # Black keys are not exactly on the white-key grid (they are pushed outward within their
    # group), so use the measured centres, plus the typical offset for keys that were not seen.
    measured = {int(mr) + shift_sem: c for mr, c, o in zip(m_rel, centres, ok) if o}
    off: dict[int, float] = {}
    for pc in BLACK_PCS:
        d = [c - (a + b * key_pos(m - shift_sem)) for m, c in measured.items() if m % 12 == pc]
        off[pc] = float(np.median(d)) if len(d) >= 2 else 0.0
    for i, m in enumerate(midis):
        if int(m) in measured:
            x[i] = measured[int(m)]
        elif is_black(int(m)):
            x[i] += off[int(m) % 12]
    return Keyboard(float(top), float(bot), float(b), midis, x, w, h,
                    method="black-key pattern", notes=notes)


# --------------------------------------------------------------------------- debug image
def draw_overlay(frame_bgr: np.ndarray, kb: Keyboard, strip: tuple[int, int]) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    cv2.line(img, (0, int(kb.y_top)), (w, int(kb.y_top)), (0, 0, 255), 2)
    cv2.rectangle(img, (0, strip[0]), (w - 1, strip[1]), (255, 255, 0), 1)
    x0, x1, valid = kb.windows()
    yt = int(kb.y_top)
    for m, xc, a0, a1, v, blk in zip(kb.midis, kb.x_center, x0, x1, valid, kb.black_mask):
        if not v:
            continue
        col = (60, 60, 255) if blk else (80, 255, 80)
        cv2.line(img, (int(a0), strip[0] - 6), (int(a1), strip[0] - 6), col, 2)
        if int(m) % 12 == 0:
            cv2.putText(img, note_name(int(m)), (int(xc) - 14, yt + int(kb.white_w * 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, kb.white_w / 60), (0, 200, 255), 1, cv2.LINE_AA)
            cv2.line(img, (int(xc), yt), (int(xc), yt + int(kb.white_w * 1.2)), (0, 200, 255), 1)
    return img
