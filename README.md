# audio_spacer

Extend an audio file to a target length by widening its silent gaps,
leaving speech untouched. Extra time is distributed across gaps
proportionally to their lengths and inserted at each gap's midpoint.

## Requirements

- ffmpeg / ffprobe on PATH
- Python 3 with numpy (`python3 -m venv .venv && .venv/bin/pip install numpy`)

## Usage

List detected gaps (dry run):

```sh
python audio_spacer.py input.mp3
```

Extend to a target length:

```sh
python audio_spacer.py input.mp3 output.mp3 -t 6:00
```

Options:

- `-t, --target-length` — target duration, in seconds or `[hh:]mm:ss`
- `-g, --min-gap` — minimum silence length in seconds to count as an
  expandable gap; shorter pauses (within speech) are preserved as-is
  (default 1.0)
- `-d, --threshold-db` — silence threshold in dBFS (default −40)
- `--fill` — `roomtone` (default) loops background noise from the
  quietest window of the gap into the inserted span, avoiding both an
  audible noise-floor drop and looped breath noise; `silence` inserts
  pure digital silence

Fill is inserted at the quietest point of each gap, so breaths at gap
edges are never split or repeated.

## Tests

```sh
.venv/bin/pytest
```
