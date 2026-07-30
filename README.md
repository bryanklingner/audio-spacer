# audio_spacer

Stretch a spoken-word recording (a guided meditation, a talk, a lecture) to a
target length by widening its natural pauses. The speech itself is untouched —
only the silences grow.

Comes as a command-line tool plus a small self-hostable web app ("spacer")
that plays the stretched audio in the browser.

## How it works

1. **Decode** — ffmpeg decodes the input to raw float32 PCM at its native
   sample rate.
2. **Find gaps** — RMS level is computed over 10 ms frames of a mono
   mixdown. Runs of frames below a threshold (default −40 dBFS) lasting at
   least `min_gap` seconds (default 1.0) are "gaps". Shorter pauses — the
   ones inside sentences — are left alone.
3. **Distribute** — the extra time needed to reach the target length is
   split across the gaps proportionally to their lengths, with cumulative
   rounding so the output length is sample-exact.
4. **Insert** — each gap's insertion happens at its *quietest point* (found
   with an O(n) sliding-window energy scan), so breaths and mouth noise at
   gap edges are never cut or repeated. The inserted fill is the gap's own
   room tone: its quietest 250 ms window, palindrome-tiled (forward,
   reversed, forward, …) to the needed length so there are no seams, with
   5 ms crossfades at the boundaries. `--fill silence` inserts digital
   silence instead.
5. **Encode** — ffmpeg encodes the result (192 kbps for lossy formats).

## CLI

Requires ffmpeg/ffprobe on PATH and Python 3 with numpy.

```sh
python3 -m venv .venv && .venv/bin/pip install numpy
.venv/bin/python audio_spacer.py input.mp3              # dry run: list gaps
.venv/bin/python audio_spacer.py input.mp3 out.mp3 -t 45:00
```

Options:

- `-t, --target-length` — target duration, in seconds or `[hh:]mm:ss`
- `-g, --min-gap` — minimum silence length in seconds to count as an
  expandable gap (default 1.0)
- `-d, --threshold-db` — silence threshold in dBFS (default −40)
- `--fill` — `roomtone` (default) or `silence`

## Web app

`server.py` (FastAPI) wraps the tool with a single-page front end
(`static/index.html`): paste a link or upload a file, pick a target length,
and the result plays in the browser. Playback only — no download link is
offered, and results expire after two hours.

Links can be direct audio URLs or pages that embed one: HTML pages are
scanned for an audio URL plus optional `startTime`/`endTime` clip markers
(this is what makes Waking Up share links work — the embedded clip range is
extracted with ffmpeg, not the whole course file). Pasted text is searched
for the first `http(s)` link, so surrounding share-sheet noise is fine.

Endpoints:

- `POST /api/space` — multipart form: `file` or `url`, `target` (seconds),
  optional `min_gap`. Returns a token plus gap metadata (used by the UI to
  draw the output timeline).
- `GET /a/<token>.mp3` — the result, streamed with Range support.
- `GET /api/health` — health check.

Guardrails: 300 MB upload/download cap, 3 h input cap, 6 h target cap,
two concurrent processing jobs, and link fetching refuses non-public
addresses (SSRF).

## Deploying with Docker

```sh
docker build -t audio-spacer .
docker run -d --name audio-spacer -p 8931:8000 --restart unless-stopped audio-spacer
```

Or clone the repo on the host and run `./redeploy.sh`, which pulls the
latest code, rebuilds, and restarts the container in one step.

The app listens on port 8000 in the container. Working files live under
`/data` (override with `SPACER_DATA`); they expire after two hours, so no
volume is needed.

Behind a reverse proxy, allow large uploads and give processing time to
finish, e.g. for nginx:

```nginx
client_max_body_size 300m;
proxy_read_timeout 600s;
```

## Tests

```sh
.venv/bin/pip install numpy pytest fastapi uvicorn requests python-multipart
.venv/bin/pytest
```

The suite builds synthetic audio (tones, seeded noise, silence) and covers
gap detection, the expansion math (sample-exact lengths, proportional
allocation, bit-exact speech preservation), room-tone behavior (including
not looping breath noise), link extraction, and the CLI end to end.
