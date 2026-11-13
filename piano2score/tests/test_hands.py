"""Left/right hand assignment.  The logic is tested on simulated hand tracks (so no hand model
or network is needed); the model itself is only smoke-tested when it is available."""
from pathlib import Path

import numpy as np
import pytest

from piano2score import detect, hands, keyboard as K, synth, video
from piano2score.hands import Hand, HandTrack


@pytest.fixture(scope="module")
def kb():
    L = synth._Layout(960, 540)
    img = np.clip(synth._keyboard_image(L), 0, 255).astype(np.uint8)
    return K.calibrate([img])


def _truth_notes():
    truth = synth.hands_piece(bpm=90)
    notes = [detect.RawNote(n.midi, n.t_on, n.t_off) for n in truth]        # onset = press time
    return truth, notes


def _simulated_track(kb, truth, fps=15.0, hidden=None):
    """Hands that move to the notes of their own hand; tips spread over a few keys."""
    xs = dict(zip(kb.midis.tolist(), kb.x_center))
    t_end = max(n.t_off for n in truth)
    times = np.arange(0, t_end, 1.0 / fps)
    pos = {"L": xs[48], "R": xs[66]}
    frames = []
    for t in times:
        fr = []
        for side in ("L", "R"):
            near = [xs[n.midi] for n in truth if n.hand == side and n.t_on - 0.3 <= t < n.t_off]
            if near:
                pos[side] += 0.5 * (np.mean(near) - pos[side])
            if hidden and hidden[0] <= t < hidden[1] and side == hidden[2]:
                continue
            x = pos[side]
            tips = x + np.array([-2, -1, 0, 1, 2]) * kb.white_w * 0.9
            fr.append(Hand(side, tips, np.full(5, 400.0), x, 450.0))
        frames.append(fr)
    return HandTrack(times, frames, 1.0 / fps)


def _accuracy(truth, notes):
    return float(np.mean([a.hand == b.hand for a, b in zip(truth, notes)]))


def test_hands_beat_a_split_by_pitch(kb):
    truth, notes = _truth_notes()
    pitch_acc = float(np.mean([(n.midi >= 60) == (n.hand == "R") for n in truth]))
    stats = hands.assign_hands(notes, kb, _simulated_track(kb, truth), delay=0.0)
    assert pitch_acc < 0.75                       # the test piece really defeats the pitch rule
    assert _accuracy(truth, notes) > 0.95
    assert stats.from_video > 0.8 * stats.total


def test_a_hand_missing_for_a_while_is_covered_by_neighbours(kb):
    truth, notes = _truth_notes()
    track = _simulated_track(kb, truth, hidden=(21.3, 32.0, "L"))        # left hand not visible in bars 9-12
    hands.assign_hands(notes, kb, track, delay=0.0)
    assert _accuracy(truth, notes) > 0.85


def test_no_hands_at_all_falls_back_to_pitch(kb):
    truth, notes = _truth_notes()
    empty = HandTrack(np.array([]), [], 0.1)
    stats = hands.assign_hands(notes, kb, empty, delay=0.0, split_midi=60)
    assert stats.from_pitch == stats.total
    assert all(n.hand == ("R" if n.midi >= 60 else "L") for n in notes)


def _raw(palm_x, label="Right", score=0.9):
    return {"tips_x": np.array([palm_x] * 5), "tips_y": np.array([0.0] * 5), "palm_x": palm_x,
            "palm_y": 0.0, "label": label, "score": score}


def test_sides_come_from_position_not_from_the_models_label():
    # the model calls both hands "Right": the left-hand one is still the leftmost
    frames = [[_raw(900), _raw(1200)]] * 5
    out = hands.resolve_sides(frames, white_w=37.0)
    assert [h.side for h in out[0]] == ["L", "R"]


def test_single_hand_follows_continuity():
    frames = [[_raw(900), _raw(1200)]] * 3 + [[_raw(1190)]] * 2 + [[_raw(910)]] * 2
    out = hands.resolve_sides(frames, white_w=37.0)
    assert [out[i][0].side for i in (3, 4)] == ["R", "R"]
    assert [out[i][0].side for i in (5, 6)] == ["L", "L"]


def test_a_hand_detected_twice_is_merged():
    out = hands.resolve_sides([[_raw(900, score=0.9), _raw(905, score=0.4)]], white_w=37.0)
    assert len(out[0]) == 1


def test_short_dropouts_are_interpolated():
    times = np.arange(10) * 0.1
    h = lambda x: Hand("R", np.full(5, x), np.zeros(5), x, 0.0)          # noqa: E731
    frames = [[h(100.0)], [], [], [h(130.0)]] + [[] for _ in range(6)]
    out = hands.fill_gaps(frames, times)
    assert out[1][0].palm_x == pytest.approx(110.0) and out[2][0].palm_x == pytest.approx(120.0)
    assert out[4][0].label == "held"                                      # tail: held for a moment
    assert out[9] == []                                                   # but not for ever


def test_fall_delay_is_measured_from_two_bands(mini_video):
    path, _ = mini_video
    info = video.probe(path)
    kb = K.calibrate(video.sample_frames(info, 30))
    a, b = kb.strip_rows(3.3), kb.strip_rows(4.8)
    times, strips = video.read_strips(info, [a, b])
    na = detect.detect_notes(times, strips[0], kb).notes
    nb = detect.detect_notes(times, strips[1], kb).notes
    ca = kb.y_top - sum(a) / 2.0
    cb = kb.y_top - sum(b) / 2.0
    t = hands.estimate_fall_delay(na, nb, ca, cb)
    assert t is not None
    L = synth._Layout(960, 540)
    true_speed = (L.top - 0.04 * 540) / 2.8                               # as rendered by the fixture
    assert t.speed == pytest.approx(true_speed, rel=0.08)
    assert t.delay == pytest.approx(ca / true_speed, abs=0.06)


def test_fall_delay_refuses_when_nothing_matches():
    assert hands.estimate_fall_delay([], [], 100.0, 150.0) is None


@pytest.fixture(scope="module")
def detector():
    try:
        d = hands.HandDetector()
    except (RuntimeError, ImportError, OSError) as e:                     # no mediapipe / model / network
        pytest.skip(f"hand model unavailable: {e}")
    yield d
    d.close()


def test_detector_runs_and_finds_no_hands_in_a_blank_frame(detector):
    assert detector.detect(np.zeros((300, 600, 3), np.uint8), 0.0) == []


def test_cli_without_visible_hands_falls_back_to_pitch(mini_video, tmp_path: Path, detector):
    from piano2score.cli import main
    path, _ = mini_video                          # cartoon hands: the model should not see real hands
    out = tmp_path / "o"
    assert main([path, "-o", str(out), "--hands", "auto", "--no-pdf"]) == 0
    assert list(out.glob("*.musicxml"))
