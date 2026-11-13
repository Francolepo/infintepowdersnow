import numpy as np

from piano2score import detect, keyboard as K, video


def _detect(path, offset=3.3):
    info = video.probe(path)
    kb = K.calibrate(video.sample_frames(info, 30))
    y0, y1 = kb.strip_rows(offset)
    times, strip = video.read_strip_series(info, y0, y1)
    return detect.detect_notes(times, strip, kb), kb


def test_notes_are_recovered_from_a_video(mini_video):
    path, truth = mini_video
    det, kb = _detect(path)
    assert (kb.midis[0], kb.midis[-1]) == (21, 108)

    # the video shows notes a constant time earlier/later than they sound: estimate that shift
    diffs = []
    for n in truth:
        cand = [d.onset - n.t_on for d in det.notes if d.midi == n.midi and abs(d.onset - n.t_on - 2.5) < 1.0]
        if cand:
            diffs.append(min(cand, key=lambda x: abs(x - 2.5)))
    shift = float(np.median(diffs))

    used, onset_err, dur_err = set(), [], []
    for n in truth:
        best = None
        for j, d in enumerate(det.notes):
            e = abs(d.onset - shift - n.t_on)
            if j not in used and d.midi == n.midi and e < 0.12 and (best is None or e < best[0]):
                best = (e, j)
        if best:
            used.add(best[1])
            d = det.notes[best[1]]
            onset_err.append(d.onset - shift - n.t_on)
            dur_err.append(d.duration - (n.t_off - n.t_on))
    recall = len(used) / len(truth)
    precision = len(used) / len(det.notes)
    assert recall > 0.95, recall
    assert precision > 0.95, precision
    assert np.std(onset_err) < 0.04
    assert abs(np.mean(dur_err)) < 0.08


def test_repeated_notes_are_split(mini_video):
    path, truth = mini_video
    det, _ = _detect(path)
    # D4 (62) is struck repeatedly in the piece; each strike must be its own note
    n_true = sum(1 for n in truth if n.midi == 62)
    n_det = sum(1 for d in det.notes if d.midi == 62)
    assert n_det == n_true
