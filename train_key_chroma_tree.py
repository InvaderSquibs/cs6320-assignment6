#!/usr/bin/env python3
"""Train a tree baseline on cached chroma tabular features (Assignment 6)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
ASSIGNMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ASSIGNMENT_DIR / "outputs" / "tree_runs"

sys.path.insert(0, str(BEATPORT_DIR))

from chroma_features import CHROMA_INPUT_DIM, filter_chunks_with_chroma, load_chroma_matrix  # noqa: E402
from config_loader import load_config  # noqa: E402
from paths import key_chroma_dir, key_chunks_path  # noqa: E402


def evaluate_split(
    model: Pipeline,
    *,
    X: np.ndarray,
    y_enc: np.ndarray,
    sample_weight: np.ndarray | None,
    split_name: str,
) -> dict[str, float | int | str]:
    proba = model.predict_proba(X)
    preds = model.predict(X)
    return {
        "split": split_name,
        "n_chunks": len(y_enc),
        "loss": float(log_loss(y_enc, proba, sample_weight=sample_weight)),
        "accuracy": float(accuracy_score(y_enc, preds, sample_weight=sample_weight)),
        "macro_f1": float(
            f1_score(y_enc, preds, average="macro", zero_division=0, sample_weight=sample_weight)
        ),
    }


def train_tree_baseline(args: argparse.Namespace) -> dict:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = BEATPORT_DIR / out_dir

    chunks = pd.read_csv(key_chunks_path(out_dir, args.clip_sec))
    chroma_dir = key_chroma_dir(out_dir, args.clip_sec)
    chunks = filter_chunks_with_chroma(chunks, chroma_dir)

    encoder = LabelEncoder()
    train_df = chunks[chunks["split"] == "train"]
    encoder.fit(train_df["key_24"])

    splits = {
        name: chunks[chunks["split"] == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }

    X = {name: load_chroma_matrix(df, chroma_dir) for name, df in splits.items()}
    y = {name: encoder.transform(df["key_24"]) for name, df in splits.items()}
    weights = {
        name: df["sample_weight"].to_numpy(dtype=np.float64) for name, df in splits.items()
    }

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=args.max_iter,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    random_state=args.seed,
                    early_stopping=False,
                ),
            ),
        ]
    )

    print(
        f"Training {args.model_name} on {len(splits['train'])} train chunks "
        f"({CHROMA_INPUT_DIM}-d chroma, seed={args.seed})..."
    )
    model.fit(X["train"], y["train"], clf__sample_weight=weights["train"])

    metrics = [
        evaluate_split(
            model,
            X=X[name],
            y_enc=y[name],
            sample_weight=weights[name],
            split_name=name,
        )
        for name in ("train", "val", "test")
    ]

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "run_name": args.run_name,
        "model_name": args.model_name,
        "clip_sec": args.clip_sec,
        "seed": args.seed,
        "input": {
            "feature_type": "chroma_cqt_mean_std",
            "input_dim": CHROMA_INPUT_DIM,
            "description": "12 pitch-class means + 12 stds from librosa chroma_cqt",
            "standardized": True,
        },
        "model_settings": {
            "classifier": "HistGradientBoostingClassifier",
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "sample_weight": "chunk sample_weight column (1 / n_chunks per song)",
        },
        "metrics": metrics,
        "test_accuracy": metrics[-1]["accuracy"],
        "test_macro_f1": metrics[-1]["macro_f1"],
        "test_loss": metrics[-1]["loss"],
        "val_accuracy": metrics[1]["accuracy"],
        "val_macro_f1": metrics[1]["macro_f1"],
        "val_loss": metrics[1]["loss"],
    }

    meta_path = run_dir / f"{args.run_name}_metadata.json"
    metrics_path = run_dir / f"{args.run_name}_metrics.csv"
    meta_path.write_text(json.dumps(metadata, indent=2))
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    print(f"Wrote {meta_path}")
    print(f"Wrote {metrics_path}")
    print(
        f"Test: acc={metadata['test_accuracy']:.3f}  "
        f"macro_f1={metadata['test_macro_f1']:.3f}  loss={metadata['test_loss']:.3f}"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train HistGradientBoosting baseline on cached chroma vectors."
    )
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--run-name", default="exp6_chroma_tree")
    parser.add_argument("--model-name", default="HistGradientBoosting (chroma tabular)")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    train_tree_baseline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
