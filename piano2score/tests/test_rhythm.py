import numpy as np
import pytest

from piano2score import rhythm, synth
from piano2score.detect import RawNote
from piano2score.score import decompose


def _accuracy(truth, r):
    exp = sorted((round(n.beat_on * 4), n.midi) for n in truth)
    got = sorted((q.start, q.midi) for q in r.notes)
    return float(np.mean([a == b for a, b in zip(exp, got)]))


def _quantize(truth, **kw):
    raw = [RawNote(n.midi, n.t_on + 3.016, n.t_off + 3.016) for n in truth]
    return rhythm.quantize(raw, sub=4, **kw)


@pytest.mark.parametrize("bpm", [60, 72, 92, 120, 140, 170])
def test_steady_tempo_is_found_and_notes_land_on_the_right_slots(bpm):
    r = _quantize(synth.demo_piece(bpm=bpm))
    assert abs(r.bpm - bpm) < 0.5
    assert _accuracy(synth.demo_piece(bpm=bpm), r) == 1.0


def test_human_timing_jitter():
    accs = []
    for seed in range(1, 6):
        t = synth.demo_piece(bpm=100, jitter=0.02, seed=seed)
        accs.append(_accuracy(t, _quantize(t)))
    assert np.mean(accs) > 0.9


def test_slowing_tempo_is_followed():
    t = synth.demo_piece(bpm=100, drift=0.2)
    r = _quantize(t)
    assert r.tracked
    assert _accuracy(t, r) > 0.9


def test_given_bpm_is_respected():
    t = synth.demo_piece(bpm=100)
    r = _quantize(t, bpm=100)
    assert r.bpm == 100


def test_durations_are_snapped():
    t = synth.demo_piece(bpm=100, articulation=1.0)
    r = _quantize(t)
    exp = sorted((round(n.beat_on * 4), n.midi, round((n.beat_off - n.beat_on) * 4)) for n in t)
    got = sorted((q.start, q.midi, q.dur) for q in r.notes)
    assert exp == got


def test_empty_input():
    r = rhythm.quantize([], sub=4)
    assert r.notes == [] and r.warnings


@pytest.mark.parametrize("pos,length", [(0, 16), (3, 5), (6, 7), (1, 1), (4, 12), (2, 14)])
def test_decompose_sums_and_respects_alignment(pos, length):
    parts = decompose(pos, length, 4)
    assert sum(parts) == length
    p = pos
    for size in parts:
        if size in (16,):
            assert p % 16 == 0
        p += size
