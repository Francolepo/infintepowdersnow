"""Fetch a video from YouTube (or any site yt-dlp supports) and make sure OpenCV can read it."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import cv2


def is_url(source: str) -> bool:
    return bool(re.match(r"^(https?://|www\.|youtu\.?be|m\.youtube)", source.strip(), re.I))


def download(url: str, out_dir: Path, max_height: int = 1080, quiet: bool = False) -> tuple[Path, str]:
    """Download the video track only (audio is not needed).  Returns (file path, title)."""
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("yt-dlp is required to download videos: pip install yt-dlp") from e

    out_dir.mkdir(parents=True, exist_ok=True)
    h = int(max_height)
    opts = {
        # H.264 decodes everywhere; fall back to whatever is available
        "format": f"bv[vcodec^=avc1][height<={h}]/bv[height<={h}]/b[height<={h}]/bv/b",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": quiet,
        "no_warnings": quiet,
        "retries": 5,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        reqs = info.get("requested_downloads") or []
        path = Path(reqs[0]["filepath"]) if reqs else Path(ydl.prepare_filename(info))
    return path, info.get("title") or path.stem


def _can_decode(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    return bool(ok)


def ensure_decodable(path: Path, work_dir: Optional[Path] = None) -> Path:
    """Return a path OpenCV can decode, transcoding to H.264 with ffmpeg if necessary
    (YouTube sometimes serves AV1/VP9 that bundled OpenCV builds cannot read)."""
    if _can_decode(path):
        return path
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError(
            f"OpenCV cannot decode {path.name} (probably AV1/VP9) and ffmpeg was not found. "
            "Install ffmpeg, or download an H.264 (mp4) version of the video.")
    out = (work_dir or path.parent) / (path.stem + "_h264.mp4")
    print(f"Transcoding {path.name} to H.264 so it can be read ...")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-an", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", str(out)], check=True)
    return out
