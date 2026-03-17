"""Phonk remix engine for short WAV/MP3 inputs.

Processing pipeline
-------------------
1) Validate input (WAV/MP3, <= 15 seconds)
2) Slow down to 0.75x and drop pitch by 2-4 semitones
3) Detect BPM + beat positions with librosa
4) Apply aggressive low-end EQ + saturation/distortion (ffmpeg)
5) Add dark, wide reverb and stereo widening
6) Overlay punchy cowbell loop synced to detected beat grid
7) Loudness control + limiter to avoid clipping
8) Export to WAV
"""

from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

MAX_DURATION_SECONDS = 15.0
DEFAULT_OUTPUT_PATH = "output_phonk.wav"
SUPPORTED_EXTENSIONS = {".wav", ".mp3"}


class AudioValidationError(ValueError):
    """Raised when input audio fails validation checks."""


@dataclass(frozen=True)
class BeatAnalysis:
    bpm: float
    first_beat_ms: int


def _validate_input_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise AudioValidationError(f"Input file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported extension '{path.suffix}'. Use WAV or MP3."
        )


def _load_audio_mono(path: Path, sr: int = 44100) -> tuple[np.ndarray, int]:
    samples, sample_rate = librosa.load(path, sr=sr, mono=True)
    duration = librosa.get_duration(y=samples, sr=sample_rate)
    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Input is {duration:.2f}s long; max allowed is {MAX_DURATION_SECONDS:.0f}s."
        )
    return samples, sample_rate


def _slow_and_pitch_shift(samples: np.ndarray, sr: int, speed: float = 0.75) -> np.ndarray:
    slowed = librosa.effects.time_stretch(samples, rate=speed)
    semitones = random.randint(-4, -2)
    return librosa.effects.pitch_shift(slowed, sr=sr, n_steps=semitones)


def _analyze_beats(samples: np.ndarray, sr: int) -> BeatAnalysis:
    tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr)
    bpm = float(np.squeeze(tempo)) if np.size(tempo) else 120.0
    if not bpm or np.isnan(bpm):
        bpm = 120.0
    bpm = float(np.clip(bpm, 60.0, 180.0))

    if beat_frames is None or len(beat_frames) == 0:
        first_beat_ms = 0
    else:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        first_beat_ms = max(0, int(beat_times[0] * 1000))

    return BeatAnalysis(bpm=bpm, first_beat_ms=first_beat_ms)


def _run_ffmpeg_filter(input_path: Path, output_path: Path, filter_chain: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-af",
        filter_chain,
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _apply_tone_and_texture(input_path: Path, output_path: Path, aggressive: bool) -> None:
    """Strong sub/low boost (20-150Hz) + gritty saturation + clip-safe limiting."""
    if aggressive:
        low_boost = "firequalizer=gain_entry='entry(20,10);entry(60,12);entry(110,9);entry(150,7);entry(300,1);entry(1000,0)'"
        drive = "acompressor=threshold=-24dB:ratio=3.2:attack=8:release=110,acrusher=bits=7:mode=log:aa=1,asoftclip=type=tanh:threshold=0.88"
    else:
        low_boost = "firequalizer=gain_entry='entry(20,7);entry(60,8);entry(110,6);entry(150,4);entry(300,1);entry(1000,0)'"
        drive = "acompressor=threshold=-20dB:ratio=2.4:attack=12:release=140,acrusher=bits=9:mode=log:aa=1,asoftclip=type=tanh:threshold=0.92"

    # alimiter keeps output clean before further spatial processing.
    chain = f"{low_boost},{drive},alimiter=limit=0.93"
    _run_ffmpeg_filter(input_path, output_path, chain)


def _add_dark_wide_reverb(audio: AudioSegment, aggressive: bool) -> AudioSegment:
    """Dark atmospheric reverb with stereo-friendly tails for headphones."""
    tail_1 = audio.low_pass_filter(3200) - (7 if aggressive else 9)
    tail_2 = audio.low_pass_filter(2600) - (10 if aggressive else 12)
    tail_3 = audio.low_pass_filter(1900) - (13 if aggressive else 15)

    wet = audio.overlay(tail_1.pan(-0.35), position=95)
    wet = wet.overlay(tail_2.pan(0.35), position=220)
    wet = wet.overlay(tail_3.pan(-0.15), position=360)
    return wet


def _apply_stereo_widen(audio: AudioSegment, aggressive: bool) -> AudioSegment:
    """Simple Haas-style widening for better headphone image."""
    mid = audio
    side_l = audio.pan(-0.75) - (10 if aggressive else 12)
    side_r = audio.pan(0.75) - (10 if aggressive else 12)

    widened = mid.overlay(side_l, position=0).overlay(side_r, position=14 if aggressive else 11)
    return widened


def _build_cowbell_hit(duration_ms: int = 115, aggressive: bool = False) -> AudioSegment:
    """Synth punchy metallic cowbell layer (louder and sharper for phonk)."""
    tone_a = Sine(850).to_audio_segment(duration=duration_ms).apply_gain(-7 if aggressive else -9)
    tone_b = Sine(1220).to_audio_segment(duration=duration_ms).apply_gain(-8 if aggressive else -10)
    click = WhiteNoise().to_audio_segment(duration=28).high_pass_filter(2200).apply_gain(-15 if aggressive else -17)
    body = WhiteNoise().to_audio_segment(duration=duration_ms).band_pass_filter(1450, 2).apply_gain(-20)

    hit = tone_a.overlay(tone_b).overlay(body).overlay(click)
    hit = hit.high_pass_filter(650).fade_in(2).fade_out(int(duration_ms * 0.82))
    return hit


def _overlay_cowbell(audio: AudioSegment, beat: BeatAnalysis, aggressive: bool) -> AudioSegment:
    """Overlay cowbell with precise BPM/beat alignment (eighth notes)."""
    beat_ms = int(round(60_000.0 / max(1.0, beat.bpm)))
    step_ms = max(80, beat_ms // 2)
    hit_duration = min(140, step_ms)

    cowbell_hit = _build_cowbell_hit(duration_ms=hit_duration, aggressive=aggressive)
    cowbell_hit = cowbell_hit.apply_gain(3 if aggressive else 1)

    loop = AudioSegment.silent(duration=len(audio), frame_rate=audio.frame_rate)
    start = min(max(0, beat.first_beat_ms), max(0, len(audio) - 1))

    for pos in range(start, len(audio), step_ms):
        loop = loop.overlay(cowbell_hit, position=pos)

    return audio.overlay(loop)


def _finalize_level(audio: AudioSegment, aggressive: bool) -> AudioSegment:
    """Normalize with conservative true-peak headroom to prevent clipping."""
    target_peak = -1.2 if aggressive else -1.0
    gain = target_peak - audio.max_dBFS
    out = audio.apply_gain(gain)

    # Small safety margin after all overlays.
    if out.max_dBFS > -0.3:
        out = out.apply_gain(-0.8)
    return out


def make_phonk_remix(
    input_file: str | Path,
    output_file: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    aggressive: bool = False,
) -> Path:
    """Create a professional phonk remix from a short WAV/MP3 input.

    Args:
        input_file: Source WAV/MP3 file (<= 15s).
        output_file: Target output WAV path.
        aggressive: Enables stronger bass/drive/stereo/cowbell tuning.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    _validate_input_path(input_path)
    samples, sr = _load_audio_mono(input_path)

    transformed = _slow_and_pitch_shift(samples, sr=sr, speed=0.75)
    beat = _analyze_beats(transformed, sr)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        stretched_path = temp_dir_path / "stretched.wav"
        toned_path = temp_dir_path / "toned.wav"

        sf.write(stretched_path, transformed, sr)
        _apply_tone_and_texture(stretched_path, toned_path, aggressive=aggressive)

        base = AudioSegment.from_file(toned_path)
        reverbed = _add_dark_wide_reverb(base, aggressive=aggressive)
        widened = _apply_stereo_widen(reverbed, aggressive=aggressive)
        with_cowbell = _overlay_cowbell(widened, beat=beat, aggressive=aggressive)
        final_audio = _finalize_level(with_cowbell, aggressive=aggressive)

        final_audio.export(output_path, format="wav")

    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a phonk-style remix.")
    parser.add_argument("input_file", help="Path to input .wav or .mp3 (max 15s)")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Enable stronger bass, grit, stereo width, and cowbell intensity.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    out = make_phonk_remix(args.input_file, args.output, aggressive=args.aggressive)
    mode = "aggressive" if args.aggressive else "standard"
    print(f"Phonk remix exported ({mode} mode): {out}")


if __name__ == "__main__":
    main()
