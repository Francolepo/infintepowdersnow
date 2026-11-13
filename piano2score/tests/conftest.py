"""Shared fixtures.  Rendering a test video takes a few seconds, so it is done once per run."""
import pytest

from piano2score import synth


@pytest.fixture(scope="session")
def mini_video(tmp_path_factory):
    """A short synthetic falling-notes video in the 'burning gems' style, plus its true notes."""
    notes = [n for n in synth.demo_piece(bpm=96, jitter=0.01) if n.t_on < 9.0]
    path = str(tmp_path_factory.mktemp("video") / "mini.mp4")
    synth.render_video(notes, path, width=960, height=540, fps=30, style="burn", seconds_visible=2.8,
                       lead=2.5, tail=1.0)
    return path, notes
