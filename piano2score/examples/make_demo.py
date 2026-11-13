"""Render a synthetic falling-notes video and transcribe it, to see the whole pipeline work.

    python examples/make_demo.py

Writes the sheet music and debug pictures to examples/demo/.  The video imitates tutorials where
gem-like bars fall onto a keyboard and burn up in a glow; the notes are known, so you can
compare the score with the ground truth (Ode to Joy, D major, 92 bpm, slightly human timing).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piano2score import synth            # noqa: E402
from piano2score.cli import main         # noqa: E402

out = Path(__file__).parent / "demo"
with tempfile.TemporaryDirectory() as tmp:
    video = str(Path(tmp) / "demo.mp4")
    notes = synth.demo_piece(bpm=92, jitter=0.012)
    print("Rendering the synthetic video (about half a minute) ...")
    synth.render_video(notes, video, width=960, height=540, fps=30, style="burn", seconds_visible=3.0)
    main([video, "-o", str(out), "--title", "Ode to Joy (synthetic demo)", "--hands", "pitch"])  # cartoon hands: split by pitch
