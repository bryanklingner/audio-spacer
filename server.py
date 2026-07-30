"""Web front end for audio_spacer: upload or link audio, get a spaced version.

Accepts a direct audio URL, a page that embeds one (e.g. a Waking Up share
link, including its clip start/end times), or an uploaded file. Results are
kept on disk under random tokens and expire after TTL_SEC.
"""

import ipaddress
import os
import re
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_spacer import decode, encode, expand, find_gaps, fmt_time, probe

DATA = Path(os.environ.get("SPACER_DATA",
                           os.path.join(tempfile.gettempdir(), "spacer")))
MAX_BYTES = 300 * 2 ** 20
MAX_INPUT_SEC = 3 * 3600
MAX_TARGET_SEC = 6 * 3600
TTL_SEC = 2 * 3600
THRESHOLD_DB = -40.0

AUDIO_URL_RE = re.compile(
    r'https://[^"\'\\\s]+?\.(?:mp3|m4a|aac|ogg|wav)[^"\'\\\s]*')
START_RE = re.compile(r'startTime\\?":\s*([\d.]+)')
END_RE = re.compile(r'endTime\\?":\s*([\d.]+)')
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{12}\.mp3$")
URL_IN_TEXT_RE = re.compile(r'https?://[^\s<>"\']+')
UA = {"User-Agent": "Mozilla/5.0 (audio-spacer)"}

app = FastAPI()
work = threading.Semaphore(2)


def extract_url(text):
    """Pull the first http(s) link out of possibly noisy pasted text."""
    match = URL_IN_TEXT_RE.search(text)
    if not match:
        return None
    return match.group(0).rstrip(".,;:!?)]}’”\"'")


def require_public_host(url):
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(400, "Only http(s) links are supported.")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except OSError:
        raise HTTPException(400, "That link's host could not be found.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise HTTPException(400, "That link points somewhere private.")


def download(url, dst):
    require_public_host(url)
    try:
        with requests.get(url, headers=UA, stream=True,
                          timeout=(10, 120)) as r:
            r.raise_for_status()
            size = 0
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(400, "That file is too large "
                                            "(300 MB max).")
                    f.write(chunk)
    except requests.RequestException:
        raise HTTPException(400, "That link could not be downloaded.")


def fetch_link(url, workdir):
    """Fetch a link into workdir; scrape pages for their embedded audio."""
    require_public_host(url)
    try:
        head = requests.get(url, headers=UA, stream=True, timeout=(10, 60))
        head.raise_for_status()
    except requests.RequestException:
        raise HTTPException(400, "That link could not be fetched.")
    if "text/html" not in head.headers.get("content-type", ""):
        head.close()
        dst = workdir / "input"
        download(url, dst)
        return dst
    with head:
        body = b""
        for chunk in head.iter_content(1 << 16):
            body += chunk
            if len(body) > 4 * 2 ** 20:
                break
    page = body.decode("utf-8", errors="ignore")
    match = AUDIO_URL_RE.search(page)
    if not match:
        raise HTTPException(400, "No audio found at that link.")
    audio_url = match.group(0).split("#")[0]
    require_public_host(audio_url)
    start = START_RE.search(page)
    end = END_RE.search(page)
    dst = workdir / "input.mp3"
    if start and end:
        try:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", start.group(1),
                 "-to", end.group(1), "-i", audio_url, "-c", "copy",
                 str(dst)],
                capture_output=True, check=True, timeout=300)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            raise HTTPException(400, "The audio at that link could not "
                                "be read.")
    else:
        download(audio_url, dst)
    return dst


def cleanup_loop():
    while True:
        cutoff = time.time() - TTL_SEC
        for f in DATA.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
        time.sleep(600)


@app.on_event("startup")
def startup():
    DATA.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=cleanup_loop, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/space")
def space(file: UploadFile | None = File(None), url: str = Form(""),
          target: float = Form(...), min_gap: float = Form(1.0)):
    if target <= 0 or target > MAX_TARGET_SEC:
        raise HTTPException(400, "Target length must be under 6 hours.")
    min_gap = min(max(min_gap, 0.2), 30.0)

    with tempfile.TemporaryDirectory(dir=DATA) as workdir:
        workdir = Path(workdir)
        if file is not None and file.filename:
            src = workdir / "input"
            size = 0
            with open(src, "wb") as f:
                while chunk := file.file.read(1 << 16):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(400, "That file is too large "
                                            "(300 MB max).")
                    f.write(chunk)
        elif url.strip():
            link = extract_url(url)
            if not link:
                raise HTTPException(400, "No link found in the pasted text.")
            src = fetch_link(link, workdir)
        else:
            raise HTTPException(400, "Choose a file or paste a link.")

        with work:
            try:
                sr, channels = probe(str(src))
            except (subprocess.CalledProcessError, KeyError, IndexError):
                raise HTTPException(400, "That doesn't look like an audio "
                                    "file we can read.")
            channels = min(channels, 2)
            audio = decode(str(src), sr, channels)
            duration = len(audio) / sr
            if duration > MAX_INPUT_SEC:
                raise HTTPException(400, "Input is longer than 3 hours.")
            if target < duration:
                raise HTTPException(400, f"Target ({fmt_time(target)}) is "
                                    f"shorter than the audio "
                                    f"({fmt_time(duration)}).")
            gaps = find_gaps(audio, sr, THRESHOLD_DB, min_gap)
            if not gaps and target > duration:
                raise HTTPException(400, "No pauses long enough to widen "
                                    "were found in this audio.")
            extra = int(round(target * sr)) - len(audio)
            result, allocs = expand(audio, sr, gaps, extra, "roomtone")
            token = secrets.token_urlsafe(9)[:12]
            encode(result, str(DATA / f"{token}.mp3"), sr)

    shifted, out_gaps = 0, []
    for (a, b), alloc in zip(gaps, allocs):
        out_gaps.append({"start": (a + shifted) / sr,
                         "len": (b - a + alloc) / sr})
        shifted += alloc
    return {"id": token, "input_seconds": round(duration, 3),
            "output_seconds": round(len(result) / sr, 3),
            "gap_count": len(gaps), "gaps": out_gaps}


@app.get("/a/{name}")
def audio_file(name: str):
    if not TOKEN_RE.match(name) or not (DATA / name).is_file():
        raise HTTPException(404, "This audio has expired.")
    return FileResponse(DATA / name, media_type="audio/mpeg",
                        headers={"Content-Disposition": "inline",
                                 "Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static",
                           html=True))
