"""Create tabular proxy features from an image folder.

This is a Week 6 starter routine, not a CNN replacement. It extracts simple
metadata, color, brightness, and edge-summary features so you can test whether
tabular models have any signal for your portfolio image problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def image_features(path: Path) -> dict[str, float | int | str]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0

    height, width, _ = arr.shape
    gray = arr.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1)).mean() if width > 1 else 0.0
    dy = np.abs(np.diff(gray, axis=0)).mean() if height > 1 else 0.0
    channel_means = arr.mean(axis=(0, 1))
    channel_stds = arr.std(axis=(0, 1))

    return {
        "filename": path.name,
        "relative_path": str(path),
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else 0.0,
        "pixel_count": width * height,
        "red_mean": float(channel_means[0]),
        "green_mean": float(channel_means[1]),
        "blue_mean": float(channel_means[2]),
        "red_std": float(channel_stds[0]),
        "green_std": float(channel_stds[1]),
        "blue_std": float(channel_stds[2]),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "dark_pixel_fraction": float((gray < 0.2).mean()),
        "bright_pixel_fraction": float((gray > 0.8).mean()),
        "edge_proxy_mean_abs_diff": float((dx + dy) / 2.0),
    }


def collect_images(image_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in image_dir.glob(pattern) if path.suffix.lower() in IMAGE_SUFFIXES)


def attach_labels(
    features: pd.DataFrame,
    labels_path: Path | None,
    filename_column: str,
    label_column: str,
) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="Build tabular proxy features from images.")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--filename-column", default="filename")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    paths = collect_images(args.image_dir, args.recursive)
    if not paths:
        raise SystemExit(f"No image files found under {args.image_dir}")

    rows = [image_features(path) for path in paths]
    features = attach_labels(pd.DataFrame(rows), args.labels, args.filename_column, args.label_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
