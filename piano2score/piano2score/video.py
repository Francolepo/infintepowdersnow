"""Video helpers: probing, sampling frames, and reading the "hit line" strip.

Everything downstream works in a *processing resolution* (at most
``MAX_PROC_WIDTH`` pixels wide).  The decoder gives us frames at the original
resolution, so ``VideoInfo.scale`` converts between the two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

MAX_PROC_WIDTH = 1920


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def scale(self) -> float:
        """Original pixels per processing pixel (>= 1)."""
        return max(1.0, self.width / MAX_PROC_WIDTH)

    @property
    def proc_size(self) -> tuple[int, int]:
        s = self.scale
        return int(round(self.width / s)), int(round(self.height / s))


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0:  # some containers do not report a frame count: count by decoding
        cap = cv2.VideoCapture(path)
        n = 0
        while cap.grab():
            n += 1
        cap.release()
    return VideoInfo(path, float(fps), n, w, h)


def _to_proc(frame: np.ndarray, info: VideoInfo) -> np.ndarray:
    if info.scale == 1.0:
        return frame
    return cv2.resize(frame, info.proc_size, interpolation=cv2.INTER_AREA)


def sample_frames(info: VideoInfo, n: int = 48, t_start: float = 0.0,
                  t_end: Optional[float] = None, margin: float = 0.04) -> list[np.ndarray]:
    """Return ``n`` frames (BGR, processing resolution) spread over the video.

    The first/last ``margin`` of the range is skipped because intros and
    outros often show title cards instead of the piano.
    """
    t_end = info.duration if t_end is None else min(t_end, info.duration)
    f0 = int(t_start * info.fps)
    f1 = max(f0 + 1, int(t_end * info.fps))
    span = f1 - f0
    lo, hi = f0 + int(span * margin), f1 - int(span * margin) - 1
    if hi <= lo:
        lo, hi = f0, f1 - 1
    idxs = np.unique(np.linspace(lo, hi, n).astype(int))
    cap = cv2.VideoCapture(info.path)
    frames: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(_to_proc(fr, info))
    cap.release()
    if not frames:  # seeking failed (rare container/codec combinations): read sequentially
        cap = cv2.VideoCapture(info.path)
        step = max(1, info.frame_count // max(n, 1))
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, fr = cap.retrieve()
                if ok:
                    frames.append(_to_proc(fr, info))
            i += 1
        cap.release()
    if not frames:
        raise RuntimeError("Could not decode any frames from the video.")
    return frames


def read_strip_series(info: VideoInfo, y0: float, y1: float, stride: int = 1,
                      t_start: float = 0.0, t_end: Optional[float] = None,
                      progress: Optional[Callable[[float], None]] = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Read a thin horizontal band from every ``stride``-th frame.

    ``y0``/``y1`` are rows in *processing* coordinates.  Each frame's band is
    averaged over its rows, giving one RGB row per frame: the result is a
    ``(T, W, 3)`` uint8 array plus the ``(T,)`` timestamps.  A thin band is all
    the note detector needs, so memory stays small even for long videos.
    """
    s = info.scale
    r0 = int(np.clip(round(y0 * s), 0, info.height - 1))
    r1 = int(np.clip(round(y1 * s), r0 + 1, info.height))
    pw = info.proc_size[0]
    f_start = int(t_start * info.fps)
    f_end = info.frame_count if t_end is None else min(info.frame_count, int(t_end * info.fps))

    cap = cv2.VideoCapture(info.path)
    if f_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
    rows: list[np.ndarray] = []
    times: list[float] = []
    idx = f_start
    last_report = -1.0
    while idx < f_end:
        if not cap.grab():
            break
        if (idx - f_start) % stride == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                band = fr[r0:r1].mean(axis=0, dtype=np.float32)  # (W_orig, 3)
                if band.shape[0] != pw:
                    band = cv2.resize(band[None], (pw, 1), interpolation=cv2.INTER_AREA)[0]
                rows.append(np.clip(band, 0, 255).astype(np.uint8))
                times.append(idx / info.fps)
        idx += 1
        if progress and (idx - f_start) / max(f_end - f_start, 1) - last_report > 0.02:
            last_report = (idx - f_start) / max(f_end - f_start, 1)
            progress(last_report)
    cap.release()
    if not rows:
        raise RuntimeError("No frames could be read from the selected time range.")
    return np.asarray(times, dtype=np.float64), np.stack(rows)
