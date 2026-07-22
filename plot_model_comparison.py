#!/usr/bin/env python3
"""Build Assignment 6 model-comparison table and bar charts from saved run metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ASSIGNMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ASSIGNMENT_DIR / "outputs"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_comparison_table() -> pd.DataFrame:
    lr = load_json(ASSIGNMENT_DIR.parent / "assignment_5/outputs/baseline_metrics.json")
    tree = load_json(DEFAULT_OUTPUT / "tree_runs/exp6_chroma_tree_metadata.json")
    mlp = load_json(
        ASSIGNMENT_DIR.parent / "assignment_5/outputs/chroma_runs/exp3_chroma_baseline_metadata.json"
    )
    mlp_scaled = load_json(DEFAULT_OUTPUT / "chroma_runs/exp6_chroma_mlp_scaled_metadata.json")

    lr_test = next(m for m in lr["metrics"] if m["split"] == "test")
    lr_val = next(m for m in lr["metrics"] if m["split"] == "val")
    lr_train = next(m for m in lr["metrics"] if m["split"] == "train")
    tree_val = next(m for m in tree["metrics"] if m["split"] == "val")
    tree_test = next(m for m in tree["metrics"] if m["split"] == "test")
    tree_train = next(m for m in tree["metrics"] if m["split"] == "train")

    rows = [
        {
            "model": "Logistic regression",
            "model_type": "classical",
            "key_settings": "C=1.0, class_weight=balanced, StandardScaler (train-fit)",
            "preprocessing": "train-fit StandardScaler",
            "train_accuracy": lr_train["key24_accuracy"],
            "val_accuracy": lr_val["key24_accuracy"],
            "test_accuracy": lr_test["key24_accuracy"],
            "test_macro_f1": lr_test["key24_macro_f1"],
            "n_test": lr_test["n_chunks"],
            "artifact": "assignment_5/outputs/baseline_metrics.json",
            "practical_notes": "Most interpretable; linear decision boundary on chroma",
        },
        {
            "model": "HistGradientBoosting",
            "model_type": "tree",
            "key_settings": "max_depth=8, max_iter=200, lr=0.1, sample_weight=chunk",
            "preprocessing": "train-fit StandardScaler",
            "train_accuracy": tree_train["accuracy"],
            "val_accuracy": tree_val["accuracy"],
            "test_accuracy": tree_test["accuracy"],
            "test_macro_f1": tree_test["macro_f1"],
            "n_test": tree_test["n_chunks"],
            "artifact": "assignment_6/outputs/tree_runs/exp6_chroma_tree_metadata.json",
            "practical_notes": "Best test accuracy; severe train overfit (100% train acc)",
        },
        {
            "model": "Chroma MLP (unscaled)",
            "model_type": "neural tabular",
            "key_settings": "hidden=64, lr=2.75e-3, early_stop patience=8, no scaler",
            "preprocessing": "none (raw chroma)",
            "train_accuracy": None,
            "val_accuracy": 0.4156,
            "test_accuracy": mlp["test_accuracy"],
            "test_macro_f1": mlp["test_macro_f1"],
            "n_test": mlp["n_test"],
            "artifact": "assignment_5/outputs/chroma_runs/exp3_chroma_baseline_metadata.json",
            "practical_notes": "Small gain over LR; best val at epoch 14",
        },
        {
            "model": "Chroma MLP (scaled)",
            "model_type": "neural tabular",
            "key_settings": "hidden=64, lr=2.75e-3, early_stop patience=8, --scale",
            "preprocessing": "train-fit StandardScaler",
            "train_accuracy": None,
            "val_accuracy": 0.4259,
            "test_accuracy": mlp_scaled["test_accuracy"],
            "test_macro_f1": mlp_scaled["test_macro_f1"],
            "n_test": mlp_scaled["n_test"],
            "artifact": "assignment_6/outputs/chroma_runs/exp6_chroma_mlp_scaled_metadata.json",
            "practical_notes": "Scaling did not improve test vs unscaled MLP",
        },
    ]
    return pd.DataFrame(rows)


def plot_test_comparison(table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    models = table["model"].tolist()
    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    acc = table["test_accuracy"] * 100
    f1 = table["test_macro_f1"] * 100
    bars_acc = ax.bar([i - width / 2 for i in x], acc, width, label="Test accuracy (%)", color="#4C78A8")
    bars_f1 = ax.bar([i + width / 2 for i in x], f1, width, label="Test macro F1 (%)", color="#F58518")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Assignment 6: Tabular model comparison on 15s chroma proxy (test set)")
    ax.set_ylim(0, max(acc.max(), f1.max()) + 8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bars in (bars_acc, bars_f1):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_test_metrics.png", dpi=160)
    plt.close(fig)


def plot_generalization_gap(table: pd.DataFrame, output_dir: Path) -> None:
    subset = table.dropna(subset=["train_accuracy", "val_accuracy"]).copy()
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(subset))
    width = 0.35
    train = subset["train_accuracy"] * 100
    val = subset["val_accuracy"] * 100
    ax.bar([i - width / 2 for i in x], train, width, label="Train accuracy (%)", color="#54A24B")
    ax.bar([i + width / 2 for i in x], val, width, label="Val accuracy (%)", color="#E45756")
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset["model"], rotation=10, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Train vs validation accuracy (generalization gap)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_train_val_gap.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Assignment 6 model comparison charts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "plots")
    args = parser.parse_args()

    table = build_comparison_table()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir.parent / "model_comparison.csv"
    table.to_csv(table_path, index=False)
    plot_test_comparison(table, args.output_dir)
    plot_generalization_gap(table, args.output_dir)
    print(f"Wrote {table_path}")
    print(f"Wrote {args.output_dir / 'model_comparison_test_metrics.png'}")
    print(f"Wrote {args.output_dir / 'model_comparison_train_val_gap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
