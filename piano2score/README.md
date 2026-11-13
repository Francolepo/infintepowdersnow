# piano2score

Turn a **falling-notes piano tutorial video** into **sheet music**. Give it a YouTube link or a
video file; get back a PDF score, a MusicXML file you can edit in MuseScore, and a MIDI file.

It is built for videos where coloured bars/gems fall onto a keyboard at the bottom of the screen
(Synthesia-style, including the ones with a real piano and hands filmed from above, where
the notes burn up in a glow when they reach the keys). When real hands are visible, **the
notes are given to the left or right hand by tracking the hands in the video**, not by pitch.

```
python -m piano2score "https://www.youtube.com/watch?v=XXXXXXXXXXX" -o output
python -m piano2score my_tutorial.mp4 -o output
```

```
output/
  <title>.pdf        the score (needs verovio+cairosvg, or MuseScore on PATH)
  <title>.musicxml   open in MuseScore / Finale / Sibelius / Dorico to correct and re-export
  <title>.mid        the performance as played in the video (real timing, not quantised)
  debug/calibration.png   where the program found the keyboard and where it reads the notes
  debug/piano_roll.png    every note it found, key by time (orange = right hand, blue = left)
  debug/hands.png         a frame with the tracked hands and fingertips drawn on it
```

See `examples/demo/` for real output (made from a synthetic video, see below).

## Install

Python 3.9+.

```
pip install -r requirements.txt
```

`yt-dlp` downloads from YouTube (keep it up to date: `pip install -U yt-dlp`; YouTube changes
often). If a download fails, download the video any other way and pass the file instead.
`ffmpeg` is optional (used only to convert video formats OpenCV cannot read).

Hand tracking uses MediaPipe. On the first run it downloads its 8 MB hand model to
`~/.cache/piano2score/` (set `PIANO2SCORE_HAND_MODEL` to use a file you downloaded yourself).
On a bare Linux server MediaPipe may need `sudo apt install libegl1 libgles2`.
`pip install -r requirements.txt` installs `opencv-contrib-python`, which MediaPipe needs; do not
also install `opencv-python` / `opencv-python-headless` next to it (the two conflict).

## How it works

1. **Find the keyboard.** A median over ~48 frames removes the falling notes and most of the hands.
   The black-key pattern (2 keys, gap, 3 keys, ...) is matched against a real piano, which gives every
   key's pitch and x position, even when black keys are not on the ideal grid. 88, 76, 61, 49, 37 and
   25-key boards are recognised; otherwise it falls back to 88 keys across the picture.
2. **Read a thin band of pixels just above the keys, in every frame.** A bar takes exactly its
   duration to sweep past a horizontal line, so the time a key is covered *is* the note length. The band
   sits a few key-widths above the keys, clear of the glow and sparks, and hands never reach it.
3. **Decide which keys are sounding.** Each bright run in the band is explained by the smallest set
   of keys whose bar shapes reproduce it, so a white key's bar is not mistaken for its black neighbour,
   and C+D is not mistaken for C#. Glow around bars is trimmed away, and two bars of the same key
   separated by a thin gap are told apart.
4. **Rhythm.** Notes starting together become chords. The tempo is the one for which note starts line up
   best with a beat grid (16th notes by default). If the playing speeds up or slows down, the grid follows
   it (a phase-locked loop). Lengths are snapped on the same grid, and small overlaps/gaps are tidied.
5. **Which hand plays what.** A hand-landmark model (MediaPipe) tracks both hands and every fingertip.
   The left hand is the left one on screen (with continuity when only one is visible). Notes are read a
   little above the keys, so the speed of the falling bars is measured from two reading bands, which
   tells exactly when each key is pressed. Each note then goes to the hand whose fingertips are closest
   to that key at that moment. Where that is unclear (hands touching, a hand out of view) the note
   follows the neighbouring notes of each hand, and only then falls back to a pitch split.
6. **Notation.** Key estimation (for sharps vs flats), voices for
   overlapping notes, ties across barlines, beaming, accidentals. Written as MusicXML (music21) and
   rendered to PDF (MuseScore if installed, otherwise Verovio).

## Check it before trusting it

```
python -m piano2score video.mp4 --calibrate-only
```
writes `output/debug/calibration.png` in a second. You should see:

* a **red line** on the top edge of the keys (where notes land),
* a **thin cyan band** a few key widths above it (where notes are read),
* small coloured dashes over every key and a label on every C (green = white key, red = black key).

If the labels are on the wrong keys or the lines are in the wrong place, fix it with
`--first-note A0 --last-note C8`, `--keyboard-top ROW`, or `--strip-offset`. Then look at
`debug/piano_roll.png` after a full run: it should look like the piece.

## Options that matter

| Option | Use it when |
|---|---|
| `--start 0:20 --end 3:10` | the video has an intro/outro |
| `--strip-offset 4.5` | the notes burn up in a tall glow, or you see lots of very short junk notes (default 3.3 key-widths above the keys; plain falling bars can use `1`) |
| `--bpm 96` | the estimated tempo is wrong (e.g. half or double) |
| `--time-sig 3/4` | not in 4/4 (default is 4/4; the bar lines depend on it) |
| `--pickup 1` | the piece starts with a one-beat pick-up |
| `--subdivision 2` / `3` | 8th-note grid (slow/free playing) / 8th-note triplets |
| `--hands pitch` | the video shows no hands (or you do not want tracking): split at `--split` instead |
| `--hands video` | insist on hand tracking (error instead of silently falling back) |
| `--split C4` | the pitch split, also the last-resort fallback for notes the hands cannot decide |
| `--track on` | force tempo-following for a rubato performance (`off` to disable) |
| `--first-note`, `--last-note` | the keyboard is not 88 keys and the guess is wrong |
| `--threshold 60` | notes are dim/pale and missed (or bright glow is read as notes) |

## Limits (read this)

* **Tested on synthetic videos and on one screenshot of a real video, not yet on real downloads.**
  On the synthetic videos (burning-gem and plain styles, 24-30 fps, with glow, sparks, hands, noise and
  human-like timing jitter) every note was recovered with onsets within about 10 ms. A real video
  may need `--strip-offset` or `--threshold` tuning: start with `--calibrate-only`.
* **Hand tracking needs the hands in view.** On a screenshot of a real video both hands were found with
  every fingertip on the right finger. On a test video made by pasting those real hands onto a synthetic
  keyboard, notes decided from the video were 98% right and the whole piece 90% right, against 71% for a
  split at middle C. The weak spot is **hands touching or overlapping** (the tracker may find only one) and
  **crossing hands** (the left hand is assumed to be the one on the left). Check `debug/hands.png` and the
  orange/blue `debug/piano_roll.png` (orange = right hand, blue = left). It has not been run on a full real
  video yet.
* The **sustain pedal, dynamics and fingering are not visible** and are not transcribed.
  Expect to tidy the result in MuseScore.
* Free-tempo playing (rubato) is only partly followed; very large tempo changes can derail the grid.
  Compare against `--track off` and `--bpm`.
* Triplets are supported only as a whole-piece 8th-triplet grid (`--subdivision 3`); no mixed
  16th/triplet music.
* Repeated notes with a visual gap shorter than one video frame merge into one longer note.
  Higher frame rates (the default analyses up to 60 fps) help.
* Not supported: 3D-perspective falling notes, notes that fall sideways, scrolling sheet music, keyboards
  that move or zoom during the piece.

## Development

```
pip install pytest
python -m pytest            # about 25 s; renders a small synthetic video
python examples/make_demo.py
```

`piano2score/synth.py` renders synthetic falling-notes videos from a known note list (used by the
tests, to measure accuracy exactly).
