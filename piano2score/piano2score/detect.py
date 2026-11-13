"""Turn the thin "hit line" strip of a falling-notes video into a list of notes.

For every frame we look at one horizontal band just above the keyboard.  Whatever is
brighter/different than the empty background in a key's column means "a note bar is
passing over this key", i.e. the key is sounding.  A bar needs exactly its duration to
sweep past a line at constant speed, so the time a key stays "covered" is the note
length (everything is shifted earlier by a constant, which does not matter).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .keyboard import Keyboard


@dataclass
class RawNote:
    midi: int
    onset: float      # seconds
    offset: float     # seconds
    strength: float = 1.0   # mean coverage (0..1), a rough confidence

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class Detection:
    notes: list[RawNote]
    times: np.ndarray
    coverage: np.ndarray        # (T, K) fraction of each key's window that is covered
    threshold: float            # per-column "different from background" level used
    frame_dt: float


def foreground(strip: np.ndarray, bg_percentile: float = 20.0) -> np.ndarray:
    """Per-column difference from the empty background, (T, W) float32 in 0..255.

    The background is a low percentile of each column over time: bars are brighter than
    the empty screen, and a column is empty most of the time.
    """
    bg = np.percentile(strip, bg_percentile, axis=0).astype(np.float32)   # (W, 3)
    out = np.empty(strip.shape[:2], np.float32)
    step = 512
    for i in range(0, len(strip), step):
        out[i:i + step] = np.abs(strip[i:i + step].astype(np.float32) - bg[None]).max(axis=2)
    return out


def choose_threshold(dcol: np.ndarray, kb: Keyboard) -> float:
    """Pick the "column is covered" level from the data (brightest 1% of key pixels)."""
    x0, x1, valid = kb.windows()
    cols = np.zeros(dcol.shape[1], bool)
    for a, b, v in zip(x0, x1, valid):
        if v:
            cols[a:b] = True
    vals = dcol[:, cols]
    p99 = float(np.percentile(vals, 99)) if vals.size else 0.0
    return float(np.clip(0.25 * p99, 28.0, 80.0))


_SUBSETS: dict[int, np.ndarray] = {}


def _subset_matrix(n: int) -> np.ndarray:
    """(2**n, n) 0/1 matrix listing every subset of n items."""
    if n not in _SUBSETS:
        idx = np.arange(2 ** n)[:, None]
        _SUBSETS[n] = ((idx >> np.arange(n)[None, :]) & 1).astype(np.int32)
    return _SUBSETS[n]


def explain_frames(mask: np.ndarray, kb: Keyboard, min_gain: float = 0.30,
                   key_penalty: float = 0.35, dcol: Optional[np.ndarray] = None,
                   core_frac: float = 0.55, min_level: float = 0.0) -> np.ndarray:
    """Decide which keys are sounding in each frame, (T, K) bool.

    ``mask`` is the (T, W) boolean "this column is covered" image of the strip.  Every
    continuous bright run is explained by the smallest set of keys whose bar footprints
    best reproduce it.  This is what keeps a white key's bar from also lighting up its
    black neighbour (and vice versa), and lets C+D be told apart from C#.

    If ``dcol`` (the brightness difference the mask came from) is given, each run is first
    trimmed to its bright core: the glow around a bar flickers and would otherwise make the
    run wider than the bar, so that it is mistaken for a different key.  Runs whose own peak
    brightness is below ``min_level`` are dropped: they are the dark gap between two bars of
    the same key, lit only by their glow.
    """
    T, W = mask.shape
    w = kb.white_w
    fl, fr = kb.footprints()
    _, _, valid = kb.windows()
    kc = max(3, int(0.12 * w)) | 1                      # close small holes (shaded facets)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, kc), np.uint8))
    d = np.diff(np.pad(closed, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(d == 1)
    _, ends = np.nonzero(d == -1)                       # same rows, same order
    keep = (ends - starts) >= 0.3 * w                   # ignore specks and sparkles
    rows, starts, ends = rows[keep], starts[keep], ends[keep]

    act = np.zeros((T, len(kb.midis)), bool)
    pad = int(np.ceil(w))
    cache: dict[tuple[int, int], tuple[int, ...]] = {}
    for t, a, b in zip(rows.tolist(), starts.tolist(), ends.tolist()):
        if dcol is not None:
            seg = dcol[t, a:b]
            peak = float(np.percentile(seg, 90))
            if peak < min_level:
                continue
            core = np.flatnonzero(seg >= core_frac * peak)
            if len(core) == 0:
                continue
            a, b = a + int(core[0]), a + int(core[-1]) + 1
            if b - a < 0.3 * w:
                continue
        key = (a, b)
        sel = cache.get(key)
        if sel is None:
            sel = _explain_run(a, b, kb, fl, fr, valid, pad, min_gain * w, key_penalty * w)
            cache[key] = sel
        for k in sel:
            act[t, k] = True
    return act


def _explain_run(a: int, b: int, kb: Keyboard, fl: np.ndarray, fr: np.ndarray,
                 valid: np.ndarray, pad: int, min_gain: float, key_penalty: float) -> tuple[int, ...]:
    w = kb.white_w
    i0 = int(np.searchsorted(kb.x_center, a - 0.3 * w))
    i1 = int(np.searchsorted(kb.x_center, b + 0.3 * w))
    cand = [k for k in range(max(0, i0 - 1), min(len(kb.midis), i1 + 1)) if valid[k]]
    # keep the 8 candidates whose footprint overlaps the run the most
    if len(cand) > 8:
        ov = [max(0.0, min(b, fr[k]) - max(a, fl[k])) for k in cand]
        cand = sorted(np.array(cand)[np.argsort(ov)[-8:]].tolist())
    if not cand:
        return ()
    off = a - pad
    L = (b - a) + 2 * pad
    run = np.zeros(L, bool)
    run[a - off:b - off] = True
    F = np.zeros((len(cand), L), np.int32)
    for j, k in enumerate(cand):
        lo, hi = int(round(fl[k] - off)), int(round(fr[k] - off))
        F[j, max(0, lo):max(0, min(L, hi))] = 1
    sub = _subset_matrix(len(cand))
    union = (sub @ F) > 0                                   # (2^n, L)
    mismatch = (union != run[None, :]).sum(axis=1)
    # Every extra key must explain at least ``key_penalty`` of otherwise unexplained bar:
    # halo/sparkle noise widens a run by a few pixels, a genuine second note adds a whole bar.
    cost = mismatch + key_penalty * sub.sum(axis=1)
    best = int(np.argmin(cost))
    if run.sum() - mismatch[best] < min_gain:               # nothing explains this run well enough
        return ()
    return tuple(cand[j] for j in range(len(cand)) if sub[best, j])


def _runs_of(c: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], c, [False]])
    edges = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _split_at_dips(lvl: np.ndarray, ratio: float) -> list[tuple[int, int]]:
    """Split one run at sharp, short drops in the key's brightness.

    Two bars of the same key in quick succession are separated by a thin dark gap that the
    glow around the bars (and the soft edges seen through a thin strip) can fill in, so the
    run looks continuous even though the brightness dips for a frame or two.  A dip is a
    frame well below the rolling median of its neighbours; slow shading changes along a
    long bar follow the median and are not split.
    """
    n = len(lvl)
    if n < 7:
        return [(0, n)]
    padded = np.pad(lvl, 4, mode="edge")
    base = np.array([np.median(padded[i:i + 9]) for i in range(n)])
    dip = lvl < ratio * base
    dip[:2] = False                      # the bar sliding in/out of the strip is not a gap
    dip[-2:] = False
    return _runs_of(~dip)


def detect_notes(times: np.ndarray, strip: np.ndarray, kb: Keyboard,
                 threshold: Optional[float] = None, min_ms: float = 70.0,
                 merge_gap_ms: float = 0.0, dip_ratio: float = 0.72,
                 min_rel_strength: float = 0.5) -> Detection:
    """Detect notes from the strip time series.  See module docstring."""
    dcol = foreground(strip)
    thr = choose_threshold(dcol, kb) if threshold is None else float(threshold)
    x0, x1, valid = kb.windows()
    cols = np.zeros(dcol.shape[1], bool)
    for a, b, v in zip(x0, x1, valid):
        if v:
            cols[a:b] = True
    level_ref = float(np.percentile(dcol[:, cols], 99)) if cols.any() else 255.0
    act = explain_frames(dcol > thr, kb, dcol=dcol, min_level=0.4 * level_ref)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30.0
    min_frames = max(1, int(round(min_ms / 1000.0 / dt)))
    gap_frames = int(round(merge_gap_ms / 1000.0 / dt))

    notes: list[RawNote] = []
    for k, midi in enumerate(kb.midis):
        runs = _runs_of(act[:, k])
        if gap_frames and runs:                          # bridge tiny dropouts
            merged = [runs[0]]
            for s, e in runs[1:]:
                if s - merged[-1][1] <= gap_frames:
                    merged[-1] = (merged[-1][0], e)
                else:
                    merged.append((s, e))
            runs = merged
        for s, e in runs:
            lvl = dcol[s:e, x0[k]:x1[k]].mean(axis=1)
            pieces = _split_at_dips(lvl, dip_ratio) if e - s >= 3 and dip_ratio > 0 else [(0, e - s)]
            for ps, pe in pieces:
                a, b = s + ps, s + pe
                if b - a < min_frames:
                    continue
                strength = float(np.median(lvl[ps:pe]) / max(level_ref, 1.0))
                if strength < 0.15:
                    continue
                t_off = times[b] if b < len(times) else times[-1] + dt
                notes.append(RawNote(int(midi), float(times[a]), float(t_off), strength))
    if notes:
        # Real bars are about equally bright; halo/sparkle ghosts are much dimmer.
        med = float(np.median([n.strength for n in notes]))
        notes = [n for n in notes if n.strength >= min_rel_strength * med]
    notes.sort(key=lambda n: (n.onset, n.midi))
    return Detection(notes, times, act.astype(np.float32), thr, dt)


# --------------------------------------------------------------------------- debug picture
def render_piano_roll(det: Detection, kb: Keyboard, max_width: int = 3000) -> np.ndarray:
    """Key x time picture: grey = raw coverage, coloured bars = detected notes."""
    T, K = det.coverage.shape
    step = max(1, int(np.ceil(T / max_width)))
    cov = det.coverage[: T - T % step].reshape(-1, step, K).max(axis=1)      # (T', K)
    img = (np.clip(cov.T[::-1], 0, 1) * 120).astype(np.uint8)               # low pitch at the bottom
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (img.shape[1], K * 6), interpolation=cv2.INTER_NEAREST)
    midi_to_row = {int(m): i for i, m in enumerate(kb.midis)}
    for n in det.notes:
        r = midi_to_row[n.midi]
        y = (K - 1 - r) * 6
        a = int(n.onset / det.frame_dt / step)
        b = max(a + 1, int(n.offset / det.frame_dt / step))
        cv2.rectangle(img, (a, y), (b, y + 5), (0, 200, 255) if not kb.black_mask[r] else (255, 160, 0), -1)
    return img
