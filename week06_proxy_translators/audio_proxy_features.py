"""Create tabular proxy features from WAV audio files.

This routine uses only Python's standard WAV reader plus NumPy. It is intended
for quick Week 6 proxy modeling, not final audio representation learning.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import pandas as pd


def pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def audio_features(path: Path, max_seconds_for_fft: float) -> dict[str, float | int | str]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    samples = pcm_to_float(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    duration_seconds = frame_count / sample_rate if sample_rate else 0.0
    abs_samples = np.abs(samples)
    zero_crossings = np.mean(np.abs(np.diff(np.signbit(samples)))) if len(samples) > 1 else 0.0

    fft_count = min(len(samples), int(sample_rate * max_seconds_for_fft)) if sample_rate else len(samples)
    fft_samples = samples[:fft_count]
    if len(fft_samples) >= 2:
        spectrum = np.abs(np.fft.rfft(fft_samples))
        freqs = np.fft.rfftfreq(len(fft_samples), d=1.0 / sample_rate)
        spectral_centroid = float((freqs * spectrum).sum() / max(spectrum.sum(), 1e-12))
        low_energy = float(spectrum[freqs < 500].sum() / max(spectrum.sum(), 1e-12))
        high_energy = float(spectrum[freqs >= 2000].sum() / max(spectrum.sum(), 1e-12))
    else:
        spectral_centroid = 0.0
        low_energy = 0.0
        high_energy = 0.0

    return {
        "filename": path.name,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": float(duration_seconds),
        "frame_count": frame_count,
        "mean_abs_amplitude": float(abs_samples.mean()) if len(abs_samples) else 0.0,
        "rms_amplitude": float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0,
        "peak_abs_amplitude": float(abs_samples.max()) if len(abs_samples) else 0.0,
        "silence_fraction": float((abs_samples < 0.01).mean()) if len(abs_samples) else 0.0,
        "zero_crossing_rate": float(zero_crossings),
        "spectral_centroid_hz": spectral_centroid,
        "low_frequency_energy_fraction": low_energy,
        "high_frequency_energy_fraction": high_energy,
    }


def attach_labels(features: pd.DataFrame, labels_path: Path | None, filename_column: str, label_column: str) -> pd.DataFrame:
    if labels_path is None:
        return features
    labels = pd.read_csv(labels_path)
    expected = {filename_column, label_column}
    missing = expected.difference(labels.columns)
    if missing:
        raise ValueError(f"labels file is missing columns: {sorted(missing)}")
    labels = labels[[filename_column, label_column]].rename(columns={filename_column: "filename"})
    return features.merge(labels, on="filename", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tabular proxy features from WAV audio files.")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-seconds-for-fft", type=float, default=10.0)
    args = parser.parse_args()

    pattern = "**/*.wav" if args.recursive else "*.wav"
    paths = sorted(args.audio_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No WAV files found under {args.audio_dir}")

    features = pd.DataFrame([audio_features(path, args.max_seconds_for_fft) for path in paths])
    features = attach_labels(features, args.labels, args.filename_column, args.label_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
