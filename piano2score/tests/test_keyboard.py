import numpy as np
import pytest

from piano2score import keyboard as K
from piano2score import synth


def _frame(width=960, height=540):
    L = synth._Layout(width, height)
    img = np.zeros((height, width, 3), np.float32)
    img[:] = synth._keyboard_image(L)
    return np.clip(img, 0, 255).astype(np.uint8), L


def test_note_names_round_trip():
    assert K.parse_note("A0") == 21 and K.parse_note("C4") == 60 and K.parse_note("C8") == 108
    assert K.parse_note("F#3") == 54 and K.parse_note("Bb2") == 46
    assert K.note_name(60) == "C4" and K.note_name(61) == "C#4"


def test_full_keyboard_is_detected_with_correct_pitches():
    img, L = _frame()
    kb = K.calibrate([img])
    assert (kb.midis[0], kb.midis[-1]) == (21, 108)
    assert kb.method == "black-key pattern"
    assert abs(kb.y_top - L.top) <= 3
    # every key centre within 1.5 px of where it was drawn (white and black)
    err = [abs(x - L.x(int(m))) for m, x in zip(kb.midis, kb.x_center)]
    assert max(err) < 0.1 * kb.white_w, max(err)


@pytest.mark.parametrize("lo,hi", [(21, 108), (28, 103), (36, 96), (36, 84), (41, 77), (48, 72)])
def test_standard_keyboard_sizes_are_recognised(lo, hi):
    """Draw the 88-key board wide, then keep only the keys of a smaller keyboard."""
    img, L = _frame(width=1600)
    first = lo - 1 if K.is_black(lo) else lo
    x0 = int(round(L.x(first) - L.w / 2))
    x1 = int(round(L.x(hi) + L.w / 2))
    kb = K.calibrate([img[:, max(0, x0):x1]])
    assert (int(kb.midis[0]), int(kb.midis[-1])) == (lo, hi)


def test_manual_range_overrides_the_guess():
    img, L = _frame(width=1600)
    x0, x1 = int(L.x(37)), int(L.x(94))             # ragged crop: unusual size, octave is a guess
    kb = K.calibrate([img[:, x0:x1]], first_note=K.parse_note("D2"))
    assert int(kb.midis[0]) == 38


def test_fallback_when_black_keys_cannot_be_found():
    flat = np.zeros((300, 600, 3), np.uint8)
    flat[100:200] = 200                    # a plain bright band: no black keys
    kb = K.calibrate([flat])
    assert kb.method.startswith("assumed 88")


def test_white_windows_avoid_neighbouring_black_bars():
    img, _ = _frame()
    kb = K.calibrate([img])
    x0, x1, valid = kb.windows()
    idx = {int(m): i for i, m in enumerate(kb.midis)}
    c, cs = idx[60], idx[61]               # C4 must stop before the C#4 bar starts
    fl, _ = kb.footprints()
    assert x1[c] - 1 <= fl[cs] + 1
