from pathlib import Path

from piano2score import score, synth
from piano2score.detect import RawNote
from piano2score.rhythm import QNote, Rhythm


def _rhythm(notes, bpm=100.0):
    return Rhythm(bpm, 4, notes, 1.0, False)


def test_key_estimation_uses_scale_membership():
    t = synth.demo_piece(bpm=92)
    q = [QNote(n.midi, round(n.beat_on * 4), round((n.beat_off - n.beat_on) * 4)) for n in t]
    tonic, mode, sharps = score.estimate_key(q)
    assert (tonic, mode, sharps) == (2, "major", 2)           # D major, as in the piece


def test_musicxml_round_trip(tmp_path: Path):
    from music21 import converter
    t = synth.demo_piece(bpm=100, articulation=1.0)
    q = [QNote(n.midi, round(n.beat_on * 4), round((n.beat_off - n.beat_on) * 4)) for n in t]
    sc, info = score.build_score(_rhythm(q), title="round trip")
    xml = tmp_path / "s.musicxml"
    score.write_musicxml(sc, xml)

    back = converter.parse(str(xml)).stripTies()
    got = []
    for el in back.recurse().notes:
        for p in (el.pitches if el.isChord else [el.pitch]):
            got.append((int(p.midi), round(float(el.getOffsetInHierarchy(back)) * 4), round(float(el.quarterLength) * 4)))
    assert sorted(got) == sorted((n.midi, n.start, n.dur) for n in q)
    assert info.key == "D major" and info.bars == 10


def test_overlapping_notes_get_their_own_voice(tmp_path: Path):
    # a held whole note with a moving line on top, same hand
    q = [QNote(72, 0, 16), QNote(76, 0, 4), QNote(77, 4, 4), QNote(79, 8, 4), QNote(77, 12, 4)]
    sc, info = score.build_score(_rhythm(q), title="voices")
    assert info.voices_rh == 2
    score.write_musicxml(sc, tmp_path / "v.musicxml")


def test_note_crossing_a_barline_is_tied():
    q = [QNote(64, 12, 8)]                       # starts on beat 4, lasts two beats
    sc, _ = score.build_score(_rhythm(q), title="tie")
    tied = [n for n in sc.recurse().notes if n.tie is not None]
    assert len(tied) == 2


def test_flat_keys_spell_with_flats():
    # B-flat major material
    q = [QNote(m, i * 4, 4) for i, m in enumerate([70, 72, 74, 75, 77, 79, 81, 82])]
    sc, info = score.build_score(_rhythm(q), title="flats")
    names = [n.pitch.name for n in sc.recurse().notes if not n.isChord]
    assert "B-" in names and "A#" not in names
    assert info.key.startswith("Bb")


def test_midi_is_written(tmp_path: Path):
    raw = [RawNote(60, 0.0, 0.5), RawNote(64, 0.5, 1.0)]
    score.write_midi(raw, tmp_path / "x.mid")
    assert (tmp_path / "x.mid").stat().st_size > 20


def test_empty_hand_gets_rests(tmp_path: Path):
    q = [QNote(48, 0, 4), QNote(50, 4, 4)]                    # only left-hand notes
    sc, info = score.build_score(_rhythm(q), title="lh only")
    assert info.notes_rh == 0 and info.notes_lh == 2
    score.write_musicxml(sc, tmp_path / "e.musicxml")
    assert (tmp_path / "e.musicxml").stat().st_size > 500


def test_other_time_signatures():
    for sig, bar_units in (((3, 4), 12), ((6, 8), 12), ((2, 4), 8)):
        q = [QNote(72, i * 4, 4) for i in range(6)]
        sc, info = score.build_score(_rhythm(q), title="ts", time_sig=sig)
        assert info.bars == -(-24 // bar_units), sig


def test_pickup_bar_moves_first_note_to_the_last_beat():
    q = [QNote(72, 0, 4), QNote(74, 4, 4)]
    sc, info = score.build_score(_rhythm(q), title="pickup", pickup_beats=1)
    assert info.bars == 2
    # the first note must sit on beat 4 of bar 1, i.e. 3 beats after the bar starts
    first = next(iter(sc.parts[0].recurse().notes))
    assert abs(float(first.getOffsetInHierarchy(sc.parts[0])) - 3.0) < 1e-6


def test_other_subdivisions_build(tmp_path: Path):
    # 8th-note grid
    q2 = [QNote(72, 0, 2), QNote(74, 2, 1), QNote(76, 3, 3), QNote(77, 6, 2)]
    sc2, _ = score.build_score(Rhythm(80, 2, q2, 1.0, False), title="sub2")
    score.write_musicxml(sc2, tmp_path / "a.musicxml")
    # 8th-note-triplet grid
    q3 = [QNote(72, 0, 1), QNote(74, 1, 1), QNote(76, 2, 1), QNote(77, 3, 3), QNote(79, 6, 6)]
    sc3, _ = score.build_score(Rhythm(80, 3, q3, 1.0, False), title="sub3")
    score.write_musicxml(sc3, tmp_path / "b.musicxml")
    from music21 import converter
    back = converter.parse(str(tmp_path / "b.musicxml"))
    triplets = [n for n in back.recurse().notes if abs(float(n.quarterLength) - 1 / 3) < 1e-6]
    assert len(triplets) == 3
