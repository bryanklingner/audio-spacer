# audio_spacer

> This tool — code, tests, web app, and docs — was created with an LLM
> (Claude). [llm_transcript.md](llm_transcript.md) is a redacted transcript
> of the conversation that built it, showing how it was prompted from first
> idea through debugging, UI iteration, and deployment.

Stretch or squish a spoken-word recording (a guided meditation, a talk, a
lecture) to a target length by widening or shortening its natural pauses. The
speech itself is untouched — only the silences change.

Comes as a command-line tool plus a small self-hostable web app ("spacer")
that plays the respaced audio in the browser.

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
5. **Squish** (target shorter than the input) — instead of inserting,
   time is removed. Each gap keeps at least `min_gap` seconds, and the time
   to remove is split in proportion to each gap's length *beyond* that
   floor, so long silences shrink the most and pauses near the floor
   barely change. Each cut is centered on the gap's quietest point, leaving
   its edges (breaths, trailing reverb) intact, and joined with a 5 ms
   crossfade. The dry run reports the shortest length reachable.
6. **Encode** — ffmpeg encodes the result (192 kbps for lossy formats).

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
  adjustable gap, and the floor no gap is squished below (default 1.0)
- `-d, --threshold-db` — silence threshold in dBFS (default −40)
- `--fill` — `roomtone` (default) or `silence`, used when extending

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

Docker can also build straight from this repository, with no local clone —
which keeps the deployed revision pinned in your compose file rather than
depending on the state of a checkout on the host:

```yaml
services:
  audio-spacer:
    build:
      context: https://github.com/bryanklingner/audio-spacer.git#v1.0.0
    image: audio-spacer:v1.0.0
    ports:
      - "8931:8000"
    restart: unless-stopped
```

Update by changing the pinned ref and running `docker compose up -d --build`.

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
