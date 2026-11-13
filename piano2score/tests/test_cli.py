from pathlib import Path

from piano2score.cli import main


def test_cli_end_to_end(mini_video, tmp_path: Path):
    path, truth = mini_video
    out = tmp_path / "out"
    rc = main([path, "-o", str(out), "--title", "Test piece", "--hands", "pitch"])
    assert rc == 0
    assert (out / "Test_piece.musicxml").stat().st_size > 1000
    assert (out / "Test_piece.mid").exists()
    assert (out / "Test_piece.pdf").stat().st_size > 1000
    assert (out / "debug" / "calibration.png").exists()
    assert (out / "debug" / "piano_roll.png").exists()


def test_calibrate_only_stops_early(mini_video, tmp_path: Path):
    path, _ = mini_video
    out = tmp_path / "out2"
    assert main([path, "-o", str(out), "--calibrate-only"]) == 0
    assert (out / "debug" / "calibration.png").exists()
    assert not list(out.glob("*.musicxml"))


def test_missing_file_is_reported(tmp_path: Path):
    assert main([str(tmp_path / "nope.mp4"), "-o", str(tmp_path / "o")]) == 2
