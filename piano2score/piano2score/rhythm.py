"""Tempo estimation and rhythm quantisation.

Detected notes have times in seconds; sheet music needs positions on a beat grid.

1. Notes that start within ~40 ms of each other are one *event* (a chord).
2. The tempo is found by trying many tempos and keeping the one for which the events
   line up best with a regular grid of ``sub`` slots per quarter note (16th notes by
   default), with a mild preference for everyday tempos (this resolves "is it 60 or
   120 bpm" ambiguities).
3. Events are snapped to the grid.  If the performance drifts (human playing,
   ritardando) the grid follows it: each new event is placed using a straight line
   fitted to the most recent events.
4. Note lengths are snapped with the same time->grid mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .detect import RawNote


@dataclass
class QNote:
    midi: int
    start: int        # grid units from the beginning of the piece
    dur: int          # grid units (>= 1)
    hand: Optional[str] = None   # 'R' / 'L' when known, else decided by pitch


@dataclass
class Rhythm:
    bpm: float                    # quarter notes per minute
    sub: int                      # grid units per quarter note
    notes: list[QNote]
    fit: float                    # share of onsets that sit close to the grid (0..1)
    tracked: bool                 # True when a drifting grid was used
    warnings: list[str] = field(default_factory=list)


def cluster_onsets(notes: list[RawNote], tol: float = 0.04) -> tuple[np.ndarray, list[int]]:
    """Group near-simultaneous onsets.  Returns (event times, event index of every note)."""
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset)
    times: list[float] = []
    members: list[list[int]] = []
    start = None
    for i in order:
        t = notes[i].onset
        if start is None or t - start > tol:
            times.append(t)
            members.append([i])
            start = t
        else:
            members[-1].append(i)
    ev_times = np.array([np.mean([notes[i].onset for i in m]) for m in members])
    ev_of = [0] * len(notes)
    for e, m in enumerate(members):
        for i in m:
            ev_of[i] = e
    return ev_times, ev_of


def _resultant(times: np.ndarray, u: np.ndarray) -> np.ndarray:
    """|mean exp(2 pi i t / u)| for many grid spacings u at once."""
    ph = 2j * np.pi * times[None, :] / u[:, None]
    return np.abs(np.exp(ph).mean(axis=1))


def _chunks(ev_times: np.ndarray, window: float) -> list[np.ndarray]:
    t = ev_times - ev_times[0]
    out = []
    for a in np.arange(0.0, t[-1] + window, window):
        c = t[(t >= a) & (t < a + window)]
        if len(c) >= 5:
            out.append(c)
    return out or [t]


def estimate_tempo(ev_times: np.ndarray, sub: int, bpm_range: tuple[float, float] = (40.0, 200.0),
                   prior_center: float = 100.0, window: float = 6.0) -> float:
    """Tempo (quarter notes per minute) whose grid the note starts line up with best.

    Alignment is judged in separate ~6 s windows (each with its own free phase) and the
    results are averaged, so a tempo that drifts over the piece does not wash it out.
    """
    if len(ev_times) < 4:
        return 100.0
    chunks = _chunks(ev_times, window)
    total = float(sum(len(c) for c in chunks))

    def score(bpms: np.ndarray) -> np.ndarray:
        # Beat, half-beat and grid-slot alignment together: a wrong tempo such as 3/4 of the
        # real one can still fit a coarse grid, but it cannot line up the beats as well.
        out = 0.0
        for div in sorted({1, 2, sub}):
            u = 60.0 / (bpms * div)
            out = out + sum(len(c) * _resultant(c, u) for c in chunks) / total
        return out / len({1, 2, sub})

    # Priors that separate a tempo from its doubled/halved alias: everyday tempos, and a
    # typical gap between note starts of about 0.5-1 beat.
    iois = np.diff(np.unique(np.round(ev_times, 3)))
    typical = float(np.median(iois)) if len(iois) else 0.5

    def prior(bpms: np.ndarray) -> np.ndarray:
        tempo = np.exp(-(np.log2(bpms / prior_center) ** 2) / (2 * 0.6 ** 2))
        beats_per_gap = typical * bpms / 60.0
        gap = np.exp(-(np.log2(np.maximum(beats_per_gap, 1e-3) / 0.75) ** 2) / (2 * 0.7 ** 2))
        return (0.5 + 0.5 * tempo) * (0.4 + 0.6 * gap)

    bpms = np.arange(bpm_range[0], bpm_range[1], 0.1)
    best = bpms[int(np.argmax(score(bpms) * prior(bpms)))]
    fine = np.arange(best - 0.6, best + 0.6, 0.01)
    return float(fine[int(np.argmax(score(fine)))])


def _phase(times: np.ndarray, u: float) -> float:
    z = np.exp(2j * np.pi * times / u).mean()
    return float(np.angle(z) * u / (2 * np.pi))


def _fit_global(ev_times: np.ndarray, u: float) -> tuple[float, float, np.ndarray]:
    """Fit t = t0 + g*u (u refined too).  Returns (t0, u, g)."""
    t0 = _phase(ev_times, u)
    for _ in range(8):
        g = np.round((ev_times - t0) / u)
        res = ev_times - (t0 + g * u)
        ok = np.abs(res) < 0.3 * u
        if ok.sum() < 3:
            break
        slope, icpt = np.polyfit(g[ok], ev_times[ok], 1)
        if abs(slope - u) / u > 0.05:        # keep it near the estimate
            break
        u, t0 = float(slope), float(icpt)
    g = np.round((ev_times - t0) / u).astype(int)
    return t0, u, g


def _track_pll(ev_times: np.ndarray, u0: float, t0: float, alpha: float = 0.7,
               beta: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Second-order phase-locked loop over the events.

    ``theta`` is the beat position (in grid units) at the last event and ``v`` the current
    speed in units per second.  Each event is placed on the grid slot nearest to where the
    loop expected it; the leftover error pulls both phase and speed, so the loop follows
    ritardandos and accelerandos instead of lagging behind them like a straight-line fit.
    """
    n = len(ev_times)
    g = np.zeros(n, int)
    err = np.zeros(n)
    theta = float(round((ev_times[0] - t0) / u0))
    g[0] = int(theta)
    v = 1.0 / u0
    for i in range(1, n):
        dt = float(ev_times[i] - ev_times[i - 1])
        pred = theta + v * dt
        gi = max(int(round(pred)), int(g[i - 1]))
        e = gi - pred
        err[i] = e
        theta = pred + alpha * e
        if dt > 0.05:
            v = float(np.clip(v + beta * e / dt, 0.6 / u0, 1.6 / u0))
        g[i] = gi
    return g, err


def _drift_range(ev_times: np.ndarray, t0: float, u: float, window: int = 8) -> float:
    """How far (in beat-grid cycles) the events' phase wanders against a constant-tempo grid.

    Timing jitter averages out inside a window; a tempo that drifts makes the window-mean
    phase travel steadily away from zero.
    """
    if len(ev_times) < window + 2:
        return 0.0
    z = np.exp(2j * np.pi * (ev_times - t0) / u)
    kern = np.ones(window) / window
    zs = np.convolve(z.real, kern, "valid") + 1j * np.convolve(z.imag, kern, "valid")
    ang = np.unwrap(np.angle(zs)) / (2 * np.pi)
    return float(ang.max() - ang.min())


def _linear_map(t0: float, u: float):
    """Time (s) -> grid units for one constant tempo."""
    return lambda t: (np.asarray(t, float) - t0) / u


def _piecewise_map(times: np.ndarray, units: np.ndarray, u: float):
    """Time (s) -> grid units, passing through every event and following its local tempo."""
    def f(t):
        t = np.asarray(t, float)
        out = np.interp(t, times, units)
        out = np.where(t < times[0], units[0] + (t - times[0]) / u, out)
        return np.where(t > times[-1], units[-1] + (t - times[-1]) / u, out)
    return f


def quantize(notes: list[RawNote], sub: int = 4, bpm: Optional[float] = None,
             track: str = "auto", bpm_range: tuple[float, float] = (40.0, 200.0)) -> Rhythm:
    """Snap ``notes`` to a grid of ``sub`` units per quarter note."""
    warnings: list[str] = []
    if not notes:
        return Rhythm(bpm or 100.0, sub, [], 1.0, False, ["no notes were detected"])

    ev_times, ev_of = cluster_onsets(notes)
    given = bpm is not None
    bpm0 = float(bpm) if given else estimate_tempo(ev_times, sub, bpm_range)
    u0 = 60.0 / (bpm0 * sub)

    # model 1: one constant tempo for the whole piece
    if given:
        t0, u = _phase(ev_times, u0), u0
        g = np.round((ev_times - t0) / u).astype(int)
    else:
        t0, u, g = _fit_global(ev_times, u0)
    fit = float(np.mean(np.abs((ev_times - (t0 + g * u)) / u) < 0.2))

    # model 2: a grid that follows the performance (started from the first few seconds)
    use_tracking = False
    drifting = _drift_range(ev_times, t0, u) > 0.4
    if track == "on" or (track == "auto" and (drifting or fit < 0.7) and len(ev_times) >= 12):
        first = ev_times[ev_times <= ev_times[0] + 10.0]
        if given or len(first) < 6:
            t0w, uw = _phase(first, u0), u0
        else:                                    # tempo at the start, not the piece average
            bpm_w = estimate_tempo(first, sub, (bpm0 * 0.7, bpm0 * 1.4), prior_center=bpm0, window=10.0)
            t0w, uw, _ = _fit_global(first, 60.0 / (bpm_w * sub))
        g_t, err_t = _track_pll(ev_times, uw, t0w)
        fit_t = float(np.mean(np.abs(err_t[1:]) < 0.2)) if len(err_t) > 1 else 0.0
        if track == "on" or fit_t > fit + 0.05:
            g, fit, use_tracking, u = g_t, fit_t, True, uw
    to_units = _piecewise_map(ev_times, g.astype(float), u) if use_tracking else _linear_map(t0, u)

    if given:
        bpm_out = float(bpm)
    elif use_tracking and len(ev_times) > 2:
        step = np.diff(g) > 0
        bpm_out = float(60.0 / (np.median(np.diff(ev_times)[step] / np.diff(g)[step]) * sub))
    else:
        bpm_out = 60.0 / (u * sub)
    if fit < 0.6:
        warnings.append(
            f"only {fit * 100:.0f}% of note starts sit on the {sub}-per-beat grid: the tempo may be "
            "wrong or the playing very free. Try --bpm, --subdivision, or --track on.")

    g0 = int(g.min())
    qn: list[QNote] = []
    for i, n in enumerate(notes):
        start = int(g[ev_of[i]]) - g0
        end = int(round(float(to_units(n.offset)))) - g0
        qn.append(QNote(n.midi, start, max(1, end - start), n.hand))
    qn.sort(key=lambda q: (q.start, q.midi))
    return Rhythm(float(bpm_out), sub, qn, fit, use_tracking, warnings)
