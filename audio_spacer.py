#!/usr/bin/env python3
"""Extend an audio file to a target length by widening its silent gaps.

Detects silent regions ("gaps") at least --min-gap seconds long, then
distributes the required extra time across those gaps proportionally to
their lengths. Fill is inserted at the quietest point of each gap and is
either the gap's own room tone, sourced from its quietest window so that
breaths and mouth noise are never looped (default), or digital silence.

Requires ffmpeg/ffprobe on PATH and numpy.
"""

import argparse
import json
import subprocess
import sys

import numpy as np

CROSSFADE_SEC = 0.005  # short fade at insert boundaries to avoid clicks
TONE_WINDOW_SEC = 0.25  # room-tone source window within each gap


def parse_length(s):
    """Parse '330', '330.5', '5:30', or '1:05:30' into seconds."""
    parts = s.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError(f"can't parse length: {s!r}")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(f"can't parse length: {s!r}")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def fmt_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    return int(stream["sample_rate"]), int(stream["channels"])


def decode(path, sr, channels):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path,
         "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio.reshape(-1, channels)


def encode(audio, path, sr):
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "f32le", "-ar", str(sr), "-ch_layout",
           "mono" if audio.shape[1] == 1 else "stereo", "-i", "-"]
    if path.lower().rsplit(".", 1)[-1] in ("mp3", "m4a", "aac", "ogg", "opus"):
        cmd += ["-b:a", "192k"]
    subprocess.run(cmd + [path], input=audio.astype(np.float32).tobytes(),
                   check=True)


def find_gaps(audio, sr, threshold_db, min_gap, frame_ms=10):
    """Return list of (start_sample, end_sample) silent regions >= min_gap."""
    mono = audio.mean(axis=1)
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = len(mono) // frame
    if n_frames == 0:
        return []
    rms = np.sqrt(np.mean(
        mono[:n_frames * frame].reshape(n_frames, frame) ** 2, axis=1))
    silent = rms < 10 ** (threshold_db / 20)

    gaps = []
    edges = np.flatnonzero(np.diff(silent.astype(np.int8)))
    starts = list(edges[~silent[edges]] + 1)
    ends = list(edges[silent[edges]] + 1)
    if silent[0]:
        starts.insert(0, 0)
    if silent[-1]:
        ends.append(n_frames)
    for a, b in zip(starts, ends):
        if (b - a) * frame >= min_gap * sr:
            gaps.append((a * frame, b * frame))
    return gaps


def quietest_window(mono, win):
    """Start index of the lowest-energy window of length win."""
    if len(mono) <= win:
        return 0
    energy = np.convolve(mono.astype(np.float64) ** 2, np.ones(win), "valid")
    return int(np.argmin(energy))


def room_tone(seg, length):
    """Palindrome-tile a segment to the requested length."""
    tiles, forward = [], True
    while sum(len(t) for t in tiles) < length:
        tiles.append(seg if forward else seg[::-1])
        forward = not forward
    return np.concatenate(tiles)[:length]


def expand(audio, sr, gaps, extra_samples, fill):
    total_gap = sum(b - a for a, b in gaps)
    # proportional allocation, cumulative rounding so the total is exact
    allocs, acc = [], 0.0
    for a, b in gaps:
        acc += extra_samples * (b - a) / total_gap
        allocs.append(round(acc) - sum(allocs))

    xf = int(CROSSFADE_SEC * sr)
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
    pieces, prev = [], 0
    for (a, b), alloc in zip(gaps, allocs):
        if alloc <= 0:
            continue
        mono = audio[a:b].mean(axis=1)
        win = max(1, min(len(mono) // 2, int(TONE_WINDOW_SEC * sr)))
        s = quietest_window(mono, win)
        ins = min(max(a + s + win // 2, a + xf), b - xf)
        if fill == "silence":
            chunk = np.zeros((alloc, audio.shape[1]), dtype=np.float32)
        else:
            chunk = room_tone(audio[a + s:a + s + win], alloc).copy()
        if alloc > 2 * xf:
            chunk[:xf] = chunk[:xf] * ramp + audio[ins - xf:ins] * (1 - ramp)
            chunk[-xf:] = chunk[-xf:] * (1 - ramp) + audio[ins:ins + xf] * ramp
        pieces.append(audio[prev:ins])
        pieces.append(chunk)
        prev = ins
    pieces.append(audio[prev:])
    return np.concatenate(pieces), allocs


def main():
    ap = argparse.ArgumentParser(
        description="Extend audio to a target length by widening silent gaps.")
    ap.add_argument("input", help="input audio file")
    ap.add_argument("output", nargs="?",
                    help="output file (omit to just list detected gaps)")
    ap.add_argument("-t", "--target-length", type=parse_length,
                    help="target duration: seconds or [hh:]mm:ss")
    ap.add_argument("-g", "--min-gap", type=float, default=1.0,
                    help="minimum gap length in seconds to expand "
                         "(default: %(default)s)")
    ap.add_argument("-d", "--threshold-db", type=float, default=-40.0,
                    help="silence threshold in dBFS (default: %(default)s)")
    ap.add_argument("--fill", choices=("roomtone", "silence"),
                    default="roomtone",
                    help="what to insert: the gap's own room tone, or pure "
                         "silence (default: %(default)s)")
    args = ap.parse_args()

    sr, channels = probe(args.input)
    audio = decode(args.input, sr, channels)
    duration = len(audio) / sr
    gaps = find_gaps(audio, sr, args.threshold_db, args.min_gap)
    total_gap = sum(b - a for a, b in gaps) / sr

    print(f"{args.input}: {fmt_time(duration)}, {sr} Hz, {channels} ch")
    print(f"{len(gaps)} gaps >= {args.min_gap}s below {args.threshold_db} dBFS"
          f" (total {fmt_time(total_gap)})")
    for a, b in gaps:
        print(f"  {fmt_time(a / sr)} - {fmt_time(b / sr)}  "
              f"({(b - a) / sr:.2f}s)")

    if args.output is None or args.target_length is None:
        if args.output or args.target_length:
            ap.error("need both an output file and --target-length "
                     "(or neither, to just list gaps)")
        return

    extra = int(round(args.target_length * sr)) - len(audio)
    if extra < 0:
        sys.exit(f"error: target ({fmt_time(args.target_length)}) is shorter "
                 f"than the input ({fmt_time(duration)})")
    if not gaps and extra > 0:
        sys.exit("error: no gaps found to expand; try a shorter --min-gap "
                 "or a higher --threshold-db")

    result, allocs = expand(audio, sr, gaps, extra, args.fill)
    for (a, b), alloc in zip(gaps, allocs):
        print(f"  gap at {fmt_time(a / sr)}: +{alloc / sr:.2f}s")
    encode(result, args.output, sr)
    print(f"wrote {args.output}: {fmt_time(len(result) / sr)} "
          f"(+{extra / sr:.2f}s across {len(gaps)} gaps)")


if __name__ == "__main__":
    main()
