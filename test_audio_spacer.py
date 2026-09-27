"""Tests for audio_spacer using synthetic tones, noise, and silence."""

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import audio_spacer
from audio_spacer import (allocate, expand, find_gaps, fmt_time,
                          parse_length, quietest_window, room_tone, squish,
                          squish_capacity)

SR = 8000
FRAME = 80  # samples per 10ms detection frame at SR
XF = int(audio_spacer.CROSSFADE_SEC * SR)
TOOL = Path(audio_spacer.__file__)


def tone(dur, freq=440.0, amp=0.5, ch=1):
    t = np.arange(int(dur * SR)) / SR
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.repeat(x[:, None], ch, axis=1)


def silence(dur, ch=1):
    return np.zeros((int(dur * SR), ch), dtype=np.float32)


def noise(dur, amp=0.001, ch=1, seed=0):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal((int(dur * SR), ch))
            ).astype(np.float32)


def seq(*parts):
    return np.concatenate(parts)


def write_wav(path, audio):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def wav_frames(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes()


def run_cli(*args):
    return subprocess.run([sys.executable, str(TOOL)] + [str(a) for a in args],
                          capture_output=True, text=True)


class TestParseLength:
    @pytest.mark.parametrize("s,expected", [
        ("330", 330.0),
        ("330.5", 330.5),
        ("0", 0.0),
        ("5:30", 330.0),
        ("0:00", 0.0),
        ("1:05:30", 3930.0),
        ("90:00", 5400.0),
        ("1:30.25", 90.25),
    ])
    def test_valid(self, s, expected):
        assert parse_length(s) == expected

    @pytest.mark.parametrize("s", ["", "abc", "1:2:3:4", "5:xx", ":"])
    def test_invalid(self, s):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_length(s)


class TestFmtTime:
    @pytest.mark.parametrize("seconds,expected", [
        (0, "0:00.000"),
        (61.5, "1:01.500"),
        (330, "5:30.000"),
        (3599.999, "59:59.999"),
    ])
    def test_format(self, seconds, expected):
        assert fmt_time(seconds) == expected


class TestFindGaps:
    def test_pure_tone_has_no_gaps(self):
        assert find_gaps(tone(2), SR, -40, 0.5) == []

    def test_all_silence_is_one_gap(self):
        gaps = find_gaps(silence(2), SR, -40, 0.5)
        assert gaps == [(0, 2 * SR)]

    def test_single_gap_position(self):
        audio = seq(tone(1), silence(1), tone(1))
        assert find_gaps(audio, SR, -40, 0.5) == [(SR, 2 * SR)]

    def test_unaligned_boundaries_within_one_frame(self):
        audio = seq(tone(0.997), silence(0.703), tone(1.1))
        [(a, b)] = find_gaps(audio, SR, -40, 0.5)
        assert abs(a - int(0.997 * SR)) <= FRAME
        assert abs(b - int(1.7 * SR)) <= FRAME

    def test_gap_shorter_than_min_gap_ignored(self):
        audio = seq(tone(1), silence(0.3), tone(1))
        assert find_gaps(audio, SR, -40, 0.5) == []

    def test_gap_exactly_min_gap_detected(self):
        audio = seq(tone(1), silence(0.5), tone(1))
        assert len(find_gaps(audio, SR, -40, 0.5)) == 1
        assert find_gaps(audio, SR, -40, 0.51) == []

    def test_leading_and_trailing_gaps(self):
        audio = seq(silence(1), tone(1), silence(1))
        assert find_gaps(audio, SR, -40, 0.5) == \
            [(0, SR), (2 * SR, 3 * SR)]

    def test_multiple_gaps(self):
        audio = seq(tone(0.5), silence(0.6), tone(0.5), silence(0.7),
                    tone(0.5), silence(0.8), tone(0.5))
        assert len(find_gaps(audio, SR, -40, 0.5)) == 3

    def test_quiet_noise_counts_as_gap(self):
        audio = seq(tone(1), noise(1, amp=0.001), tone(1))
        assert len(find_gaps(audio, SR, -40, 0.5)) == 1

    def test_threshold_boundary(self):
        # sine amp 0.005 -> RMS ~0.0035: below -40 dBFS, above -60 dBFS
        audio = seq(tone(1), tone(1, amp=0.005), tone(1))
        assert len(find_gaps(audio, SR, -40, 0.5)) == 1
        assert find_gaps(audio, SR, -60, 0.5) == []

    def test_stereo(self):
        audio = seq(tone(1, ch=2), silence(1, ch=2), tone(1, ch=2))
        assert find_gaps(audio, SR, -40, 0.5) == [(SR, 2 * SR)]

    def test_min_gap_zero_catches_short_silences(self):
        audio = seq(tone(1), silence(0.02), tone(1))
        assert len(find_gaps(audio, SR, -40, 0.01)) == 1
        assert find_gaps(audio, SR, -40, 0.5) == []

    def test_empty_audio(self):
        assert find_gaps(silence(0), SR, -40, 0.5) == []

    def test_audio_shorter_than_one_frame(self):
        assert find_gaps(tone(0.005), SR, -40, 0.5) == []
        assert find_gaps(silence(0.005), SR, -40, 0.5) == []


class TestQuietestWindow:
    def test_finds_quiet_half(self):
        mono = seq(tone(0.5, amp=0.5), noise(0.5)).ravel()
        s = quietest_window(mono, 800)
        assert s >= int(0.5 * SR) - 1

    def test_window_covering_everything(self):
        assert quietest_window(np.ones(100, dtype=np.float32), 100) == 0
        assert quietest_window(np.ones(100, dtype=np.float32), 500) == 0

    def test_avoids_loud_burst_in_middle(self):
        mono = seq(noise(0.3), tone(0.2, amp=0.05), noise(0.3, seed=1)).ravel()
        s = quietest_window(mono, 1600)
        burst = (int(0.3 * SR), int(0.5 * SR))
        assert s + 1600 <= burst[0] or s >= burst[1]


class TestRoomTone:
    def test_exact_length(self):
        seg = noise(0.25)
        for length in (1, 10, 1999, 2000, 2001, 9999):
            assert room_tone(seg, length).shape == (length, 1)

    def test_values_come_from_segment(self):
        seg = noise(0.25)
        out = room_tone(seg, 9999)
        assert np.isin(out.ravel(), seg.ravel()).all()

    def test_palindrome_continuity_at_tile_boundaries(self):
        seg = noise(0.25)
        n = len(seg)
        out = room_tone(seg, 3 * n)
        # forward|reverse boundary repeats seg's last sample;
        # reverse|forward boundary repeats seg's first sample
        assert out[n - 1] == out[n]
        assert out[2 * n - 1] == out[2 * n]

    def test_stereo(self):
        assert room_tone(noise(0.25, ch=2), 5000).shape == (5000, 2)


class TestExpand:
    def gaps_for(self, audio, min_gap=0.5):
        return find_gaps(audio, SR, -40, min_gap)

    def test_output_length_exact(self):
        audio = seq(tone(1), silence(1), tone(1))
        gaps = self.gaps_for(audio)
        for extra in (1, 7, 41, 4000, 16001):
            result, _ = expand(audio, SR, gaps, extra, "roomtone")
            assert len(result) == len(audio) + extra

    def test_allocations_sum_to_extra(self):
        audio = seq(tone(1), silence(0.5), tone(1), silence(1.5), tone(1))
        gaps = self.gaps_for(audio)
        for extra in (1, 2, 3, 999, 12345):
            _, allocs = expand(audio, SR, gaps, extra, "silence")
            assert sum(allocs) == extra
            assert all(a >= 0 for a in allocs)

    def test_proportional_allocation(self):
        audio = seq(tone(1), silence(0.5), tone(1), silence(1.5), tone(1))
        gaps = self.gaps_for(audio)
        assert [b - a for a, b in gaps] == [4000, 12000]
        _, allocs = expand(audio, SR, gaps, 8000, "silence")
        assert allocs == [2000, 6000]

    def test_single_gap_gets_everything(self):
        audio = seq(tone(1), silence(1), tone(1))
        _, allocs = expand(audio, SR, self.gaps_for(audio), 5000, "silence")
        assert allocs == [5000]

    def test_rounding_with_extra_smaller_than_gap_count(self):
        audio = seq(tone(0.5), silence(1), tone(0.5), silence(1),
                    tone(0.5), silence(1), tone(0.5))
        gaps = self.gaps_for(audio)
        assert len(gaps) == 3
        _, allocs = expand(audio, SR, gaps, 1, "silence")
        assert sum(allocs) == 1
        assert all(a >= 0 for a in allocs)

    def test_zero_extra_returns_identical_audio(self):
        audio = seq(tone(1), silence(1), tone(1))
        result, allocs = expand(audio, SR, self.gaps_for(audio), 0, "silence")
        assert np.array_equal(result, audio)
        assert allocs == [0]

    def test_no_gaps_zero_extra(self):
        audio = tone(2)
        result, allocs = expand(audio, SR, [], 0, "silence")
        assert np.array_equal(result, audio)
        assert allocs == []

    def test_speech_preserved_exactly(self):
        audio = seq(tone(1, 300), noise(1), tone(1, 700), noise(1, seed=1),
                    tone(1, 500))
        gaps = self.gaps_for(audio)
        assert len(gaps) == 2
        for fill in ("roomtone", "silence"):
            result, allocs = expand(audio, SR, gaps, 4321, fill)
            (a0, b0), (a1, b1) = gaps
            off0, off1 = allocs[0], allocs[0] + allocs[1]
            assert np.array_equal(result[:a0], audio[:a0])
            assert np.array_equal(result[b0 + off0:a1 + off0], audio[b0:a1])
            assert np.array_equal(result[b1 + off1:], audio[b1:])

    def test_silence_fill_inserts_exact_zeros(self):
        audio = seq(tone(1), noise(1), tone(1))
        [(a, b)] = self.gaps_for(audio)
        extra = 4000
        result, _ = expand(audio, SR, [(a, b)], extra, "silence")
        gap_region = result[a:b + extra]
        zeros = np.sum(np.all(gap_region == 0, axis=1))
        assert extra - 2 * XF <= zeros <= extra

    def test_roomtone_fill_stays_below_threshold(self):
        audio = seq(tone(1), noise(1), tone(1))
        gaps = self.gaps_for(audio)
        extra = 8000
        result, _ = expand(audio, SR, gaps, extra, "roomtone")
        new_gaps = find_gaps(result, SR, -40, 0.5)
        assert len(new_gaps) == len(gaps)
        grown = (sum(b - a for a, b in new_gaps)
                 - sum(b - a for a, b in gaps))
        assert abs(grown - extra) <= 2 * FRAME

    def test_roomtone_does_not_loop_breath_noise(self):
        # a "breath": audible burst inside the gap, still below -40 dBFS
        breath_amp = 0.012
        gap = seq(noise(0.4), tone(0.2, amp=breath_amp, freq=200),
                  noise(0.4, seed=1))
        audio = seq(tone(1), gap, tone(1))
        [(a, b)] = self.gaps_for(audio)
        assert b - a == len(gap)  # breath did not split the gap
        extra = 8000
        result, _ = expand(audio, SR, [(a, b)], extra, "roomtone")
        loud = 0.008
        orig_breath = np.sum(np.abs(audio[a:b]) > loud)
        new_breath = np.sum(np.abs(result[a:b + extra]) > loud)
        assert new_breath <= orig_breath + XF

    def test_extra_smaller_than_crossfade(self):
        audio = seq(tone(1), silence(1), tone(1))
        for extra in (1, 2 * XF - 1, 2 * XF):
            result, _ = expand(audio, SR, self.gaps_for(audio), extra,
                               "roomtone")
            assert len(result) == len(audio) + extra

    def test_stereo_preserved(self):
        audio = seq(tone(1, ch=2), silence(1, ch=2), tone(1, ch=2))
        result, _ = expand(audio, SR, self.gaps_for(audio), 4000, "roomtone")
        assert result.shape == (len(audio) + 4000, 2)
        assert np.array_equal(result[:, 0], result[:, 1])

    def test_expanded_all_silence_file(self):
        audio = silence(2)
        result, allocs = expand(audio, SR, [(0, len(audio))], 8000, "silence")
        assert len(result) == 3 * SR
        assert allocs == [8000]


class TestExtractUrl:
    @pytest.mark.parametrize("text,expected", [
        ("https://x.org/a.mp3", "https://x.org/a.mp3"),
        ("listen to this: https://x.org/a.mp3 so good",
         "https://x.org/a.mp3"),
        ("Check out\nhttps://x.org/clip?id=1&t=2.\nAmazing.",
         "https://x.org/clip?id=1&t=2"),
        ("(https://x.org/a).", "https://x.org/a"),
        ("“https://x.org/a”", "https://x.org/a"),
        ("http://x.org/first https://y.org/second", "http://x.org/first"),
        ("no link here", None),
        ("", None),
    ])
    def test_extract(self, text, expected):
        import server
        assert server.extract_url(text) == expected


class TestAllocate:
    def test_proportional(self):
        assert allocate(8000, [4000, 12000]) == [2000, 6000]

    def test_exact_sum_with_rounding(self):
        for total in (1, 2, 7, 1001):
            assert sum(allocate(total, [3, 3, 3])) == total

    def test_zero_weights_get_nothing(self):
        assert allocate(100, [0, 5, 0, 5]) == [0, 50, 0, 50]

    def test_all_zero_weights(self):
        assert allocate(0, [0, 0]) == [0, 0]


class TestSquish:
    def gaps_for(self, audio, min_gap=0.5):
        return find_gaps(audio, SR, -40, min_gap)

    def test_output_length_exact(self):
        audio = seq(tone(1), noise(3), tone(1), noise(2, seed=1), tone(1))
        gaps = self.gaps_for(audio)
        for remove in (1, 7, 41, 4000, squish_capacity(SR, gaps, 0.5)):
            result, allocs = squish(audio, SR, gaps, remove, 0.5)
            assert len(result) == len(audio) - remove
            assert sum(allocs) == -remove

    def test_removal_proportional_to_excess_over_floor(self):
        audio = seq(tone(1), silence(1.5), tone(1), silence(3.5), tone(1))
        gaps = self.gaps_for(audio)
        # excess over a 0.5s floor: 1s and 3s
        _, allocs = squish(audio, SR, gaps, 8000, 0.5)
        assert allocs == [-2000, -6000]

    def test_gap_at_floor_untouched(self):
        audio = seq(tone(1), silence(0.5), tone(1), silence(2), tone(1))
        gaps = self.gaps_for(audio)
        _, allocs = squish(audio, SR, gaps, 4000, 0.5)
        assert allocs == [0, -4000]

    def test_capacity(self):
        audio = seq(tone(1), silence(1.5), tone(1), silence(0.5), tone(1))
        gaps = self.gaps_for(audio)
        assert squish_capacity(SR, gaps, 0.5) == SR
        assert squish_capacity(SR, gaps, 2.0) == 0

    def test_too_much_removal_raises(self):
        audio = seq(tone(1), silence(1.5), tone(1))
        gaps = self.gaps_for(audio)
        with pytest.raises(ValueError):
            squish(audio, SR, gaps, SR + 1, 0.5)

    def test_speech_preserved_exactly(self):
        audio = seq(tone(1, 300), noise(2), tone(1, 700), noise(3, seed=1),
                    tone(1, 500))
        gaps = self.gaps_for(audio)
        result, allocs = squish(audio, SR, gaps, 12345, 0.5)
        (a0, b0), (a1, b1) = gaps
        off0, off1 = allocs[0], allocs[0] + allocs[1]
        assert np.array_equal(result[:a0], audio[:a0])
        assert np.array_equal(result[b0 + off0:a1 + off0], audio[b0:a1])
        assert np.array_equal(result[b1 + off1:], audio[b1:])

    def test_keeps_gap_edges(self):
        # breath-like bursts at each edge of a long gap survive the cut
        audio = seq(tone(1), noise(0.3, amp=0.005), noise(4, seed=1),
                    noise(0.3, amp=0.005, seed=2), tone(1))
        gaps = self.gaps_for(audio)
        (a, b), = gaps
        result, (alloc,) = squish(audio, SR, gaps, 3 * SR, 0.8)
        edge = int(0.35 * SR)
        assert np.array_equal(result[:a + edge], audio[:a + edge])
        assert np.array_equal(result[b + alloc - edge:], audio[b - edge:])

    def test_squished_gap_stays_quiet(self):
        audio = seq(tone(1), noise(4), tone(1))
        gaps = self.gaps_for(audio)
        result, (alloc,) = squish(audio, SR, gaps, 2 * SR, 0.5)
        (a, b), = gaps
        rms = np.sqrt(np.mean(result[a:b + alloc] ** 2))
        assert rms < 10 ** (-40 / 20)

    def test_stereo_preserved(self):
        audio = seq(tone(1, ch=2), silence(2, ch=2), tone(1, ch=2))
        result, _ = squish(audio, SR, self.gaps_for(audio), SR, 0.5)
        assert result.shape == (3 * SR, 2)

    def test_zero_removal_returns_identical_audio(self):
        audio = seq(tone(1), silence(2), tone(1))
        result, allocs = squish(audio, SR, self.gaps_for(audio), 0, 0.5)
        assert np.array_equal(result, audio)
        assert allocs == [0]

    def test_tiny_floor_still_leaves_room_for_crossfade(self):
        audio = seq(tone(1), silence(1), tone(1))
        gaps = self.gaps_for(audio, min_gap=0)
        cap = squish_capacity(SR, gaps, 0)
        result, _ = squish(audio, SR, gaps, cap, 0)
        assert len(result) == len(audio) - cap
        assert cap == SR - 2 * XF


class TestCLI:
    @pytest.fixture
    def speech_wav(self, tmp_path):
        path = tmp_path / "in.wav"
        write_wav(path, seq(tone(1), silence(1), tone(1)))
        return path

    def test_dry_run_lists_gaps(self, speech_wav):
        r = run_cli(speech_wav)
        assert r.returncode == 0
        assert "1 gaps" in r.stdout

    def test_extend_to_exact_length(self, speech_wav, tmp_path):
        out = tmp_path / "out.wav"
        r = run_cli(speech_wav, out, "-t", "5")
        assert r.returncode == 0, r.stderr
        assert wav_frames(out) == 5 * SR

    def test_extend_with_mmss_target(self, speech_wav, tmp_path):
        out = tmp_path / "out.wav"
        r = run_cli(speech_wav, out, "-t", "0:04")
        assert r.returncode == 0, r.stderr
        assert wav_frames(out) == 4 * SR

    def test_target_equal_to_duration(self, speech_wav, tmp_path):
        out = tmp_path / "out.wav"
        r = run_cli(speech_wav, out, "-t", "3")
        assert r.returncode == 0, r.stderr
        assert wav_frames(out) == 3 * SR

    def test_target_below_shortest_possible_fails(self, speech_wav,
                                                  tmp_path):
        r = run_cli(speech_wav, tmp_path / "out.wav", "-t", "1")
        assert r.returncode != 0
        assert "shortest possible" in r.stderr

    def test_dry_run_reports_shortest_possible(self, speech_wav):
        r = run_cli(speech_wav, "-g", "0.5")
        assert "shortest possible: 0:02.500" in r.stdout

    def test_squish_to_exact_length(self, tmp_path):
        path = tmp_path / "long.wav"
        write_wav(path, seq(tone(1), silence(4), tone(1), silence(3),
                            tone(1)))
        out = tmp_path / "out.wav"
        r = run_cli(path, out, "-t", "5")
        assert r.returncode == 0, r.stderr
        assert wav_frames(out) == 5 * SR

    def test_no_gaps_fails(self, tmp_path):
        path = tmp_path / "tone.wav"
        write_wav(path, tone(2))
        r = run_cli(path, tmp_path / "out.wav", "-t", "5")
        assert r.returncode != 0
        assert "no gaps" in r.stderr

    def test_output_without_target_fails(self, speech_wav, tmp_path):
        r = run_cli(speech_wav, tmp_path / "out.wav")
        assert r.returncode == 2

    def test_target_without_output_fails(self, speech_wav):
        r = run_cli(speech_wav, "-t", "5")
        assert r.returncode == 2

    def test_min_gap_excludes_short_gaps(self, speech_wav, tmp_path):
        r = run_cli(speech_wav, tmp_path / "out.wav", "-t", "5", "-g", "1.5")
        assert r.returncode != 0
        assert "no gaps" in r.stderr

    def test_threshold_flag(self, tmp_path):
        # gap is quiet noise: a gap at -40 dBFS, not at -80 dBFS
        path = tmp_path / "noisy.wav"
        write_wav(path, seq(tone(1), noise(1, amp=0.003), tone(1)))
        assert "1 gaps" in run_cli(path).stdout
        assert "0 gaps" in run_cli(path, "-d", "-80").stdout

    def test_mp3_output(self, speech_wav, tmp_path):
        out = tmp_path / "out.mp3"
        r = run_cli(speech_wav, out, "-t", "5")
        assert r.returncode == 0, r.stderr
        probed = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(out)],
            capture_output=True, text=True, check=True)
        assert abs(float(probed.stdout) - 5.0) < 0.15

    def test_stereo_wav(self, tmp_path):
        path = tmp_path / "stereo.wav"
        write_wav(path, seq(tone(1, ch=2), silence(1, ch=2), tone(1, ch=2)))
        out = tmp_path / "out.wav"
        r = run_cli(path, out, "-t", "4")
        assert r.returncode == 0, r.stderr
        assert wav_frames(out) == 4 * SR

    def test_empty_wav_dry_run(self, tmp_path):
        path = tmp_path / "empty.wav"
        write_wav(path, silence(0))
        r = run_cli(path)
        assert r.returncode == 0, r.stderr
        assert "0 gaps" in r.stdout

    def test_empty_wav_extend_fails(self, tmp_path):
        path = tmp_path / "empty.wav"
        write_wav(path, silence(0))
        r = run_cli(path, tmp_path / "out.wav", "-t", "1")
        assert r.returncode != 0
        assert "no gaps" in r.stderr

    def test_sub_frame_wav_dry_run(self, tmp_path):
        path = tmp_path / "tiny.wav"
        write_wav(path, tone(0.005))
        r = run_cli(path)
        assert r.returncode == 0, r.stderr
        assert "0 gaps" in r.stdout

    def test_all_silence_wav_extends(self, tmp_path):
        path = tmp_path / "quiet.wav"
        write_wav(path, silence(2))
        out = tmp_path / "out.wav"
        r = run_cli(path, out, "-t", "4")
        assert r.returncode == 0, r.stderr
        assert wav_frames(out) == 4 * SR

    def test_missing_input_fails(self, tmp_path):
        r = run_cli(tmp_path / "nope.wav")
        assert r.returncode != 0
