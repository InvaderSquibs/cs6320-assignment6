#!/usr/bin/env python3
"""Extract, train, and compare rich chroma (84-d) vs baseline chroma (24-d)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
BEATPORT_DIR = REPO_ROOT / "beatport"
ASSIGNMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ASSIGNMENT_DIR / "outputs" / "chroma_rich"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BEATPORT_DIR))

from chroma_features import (  # noqa: E402
    CHROMA_INPUT_DIM,
    CHROMA_RICH_INPUT_DIM,
    CHROMA_RICH_TIME_BINS,
    extract_clip_dataset,
    filter_chunks_with_chroma,
    load_chroma_matrix,
)
from config_loader import load_config  # noqa: E402
from models.key_baseline import Key24Classifier  # noqa: E402
from paths import key_chroma_rich_dir, key_chunks_path, key_waveforms_dir  # noqa: E402
from waveforms import filter_chunks_with_waveforms  # noqa: E402


def resolve_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg["output_dir"])
    return BEATPORT_DIR / out_dir if not out_dir.is_absolute() else out_dir


def ensure_rich_cache(out_dir: Path, clip_sec: int, cfg: dict) -> Path:
    rich_dir = key_chroma_rich_dir(out_dir, clip_sec)
    chunks = pd.read_csv(key_chunks_path(out_dir, clip_sec))
    waveform_dir = key_waveforms_dir(out_dir, clip_sec)
    with_waveforms = filter_chunks_with_waveforms(chunks, waveform_dir)
    with_rich = filter_chunks_with_chroma(with_waveforms, rich_dir)
    if len(with_rich) < len(with_waveforms):
        print(
            f"Extracting rich chroma for clip_{clip_sec}s "
            f"({len(with_waveforms) - len(with_rich)} missing / {len(with_waveforms)} total)..."
        )
        extract_clip_dataset(out_dir, clip_sec, cfg, rich=True)
    else:
        print(f"Rich chroma cache already present ({len(with_rich)} chunks) at {rich_dir}")
    return rich_dir


def load_split_tables(chunks: pd.DataFrame, chroma_dir: Path) -> dict[str, pd.DataFrame]:
    chunks = filter_chunks_with_chroma(chunks, chroma_dir)
    return {
        name: chunks[chunks["split"] == name].reset_index(drop=True)
        for name in ("train", "val", "test")
    }


def train_logistic(
    splits: dict[str, pd.DataFrame],
    chroma_dir: Path,
) -> dict:
    X_train = load_chroma_matrix(splits["train"], chroma_dir)
    y_train = splits["train"]["key_24"].tolist()
    w_train = splits["train"]["sample_weight"].to_numpy(dtype=np.float64)
    model = Key24Classifier(C=1.0)
    model.fit(X_train, y_train, sample_weight=w_train)

    rows = []
    for name, df in splits.items():
        X = load_chroma_matrix(df, chroma_dir)
        y = df["key_24"].tolist()
        w = df["sample_weight"].to_numpy(dtype=np.float64)
        pred = model.predict(X)
        y_enc = model.label_encoder_.transform(y)
        pred_enc = model.label_encoder_.transform(pred)
        rows.append(
            {
                "split": name,
                "n_chunks": len(df),
                "accuracy": float(accuracy_score(y_enc, pred_enc, sample_weight=w)),
                "macro_f1": float(
                    f1_score(y_enc, pred_enc, average="macro", zero_division=0, sample_weight=w)
                ),
            }
        )
    return {"model": "Logistic regression", "metrics": rows}


def train_tree(
    splits: dict[str, pd.DataFrame],
    chroma_dir: Path,
    *,
    seed: int,
) -> dict:
    encoder = LabelEncoder()
    encoder.fit(splits["train"]["key_24"])

    X = {n: load_chroma_matrix(splits[n], chroma_dir) for n in splits}
    y = {n: encoder.transform(splits[n]["key_24"]) for n in splits}
    w = {n: splits[n]["sample_weight"].to_numpy(dtype=np.float64) for n in splits}

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=seed,
                    early_stopping=False,
                ),
            ),
        ]
    )
    model.fit(X["train"], y["train"], clf__sample_weight=w["train"])

    rows = []
    for name in splits:
        proba = model.predict_proba(X[name])
        pred = model.predict(X[name])
        rows.append(
            {
                "split": name,
                "n_chunks": len(splits[name]),
                "accuracy": float(accuracy_score(y[name], pred, sample_weight=w[name])),
                "macro_f1": float(
                    f1_score(y[name], pred, average="macro", zero_division=0, sample_weight=w[name])
                ),
                "loss": float(log_loss(y[name], proba, sample_weight=w[name])),
            }
        )
    return {"model": "HistGradientBoosting", "metrics": rows}


def train_mlp(
    splits: dict[str, pd.DataFrame],
    chroma_dir: Path,
    *,
    input_dim: int,
    run_name: str,
    output_dir: Path,
    seed: int,
    skip_if_exists: bool,
) -> dict:
    meta_path = output_dir / f"{run_name}_metadata.json"
    if skip_if_exists and meta_path.exists():
        print(f"Reusing existing MLP run {run_name}")
        return json.loads(meta_path.read_text())

    cmd = [
        sys.executable,
        str(ASSIGNMENT_DIR.parent / "assignment_5" / "train_key_chroma.py"),
        "--run-name",
        run_name,
        "--clip-sec",
        "15",
        "--learning-rate",
        "0.00275",
        "--epochs",
        "30",
        "--scale",
        "--quiet",
        "--feature-variant",
        "rich" if input_dim == CHROMA_RICH_INPUT_DIM else "baseline",
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
    ]
    if input_dim == CHROMA_RICH_INPUT_DIM:
        cmd.extend(["--hidden-dim", "128"])
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=BEATPORT_DIR)
    return json.loads(meta_path.read_text())


def summarize_run(model: str, feature_set: str, input_dim: int, payload: dict) -> dict:
    if "metrics" in payload:
        test = next(m for m in payload["metrics"] if m["split"] == "test")
        val = next(m for m in payload["metrics"] if m["split"] == "val")
        train = next(m for m in payload["metrics"] if m["split"] == "train")
        return {
            "feature_set": feature_set,
            "input_dim": input_dim,
            "model": model,
            "train_accuracy": train["accuracy"],
            "val_accuracy": val["accuracy"],
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
        }
    return {
        "feature_set": feature_set,
        "input_dim": input_dim,
        "model": model,
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": payload["test_accuracy"],
        "test_macro_f1": payload["test_macro_f1"],
    }


def load_baseline_reference() -> list[dict]:
    """Pull best baseline (24-d) numbers from prior Assignment 6 runs."""
    comp = ASSIGNMENT_DIR / "outputs" / "model_comparison.csv"
    if not comp.exists():
        return []
    df = pd.read_csv(comp)
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "feature_set": "baseline_24d",
                "input_dim": 24,
                "model": row["model"],
                "train_accuracy": row["train_accuracy"] if pd.notna(row["train_accuracy"]) else None,
                "val_accuracy": row["val_accuracy"] if pd.notna(row["val_accuracy"]) else None,
                "test_accuracy": row["test_accuracy"],
                "test_macro_f1": row["test_macro_f1"],
            }
        )
    return rows


def plot_comparison(table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rich = table[table["feature_set"] == "rich_84d"].copy()
    if rich.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rich))
    width = 0.35
    acc = rich["test_accuracy"] * 100
    f1 = rich["test_macro_f1"] * 100
    ax.bar(x - width / 2, acc, width, label="Test accuracy (%)", color="#4C78A8")
    ax.bar(x + width / 2, f1, width, label="Test macro F1 (%)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(rich["model"], rotation=15, ha="right")
    ax.set_title("Rich chroma (84-d) tabular models — test set")
    ax.set_ylabel("Percent")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "chroma_rich_test_metrics.png", dpi=160)
    plt.close(fig)

    baseline = table[table["feature_set"] == "baseline_24d"]
    if baseline.empty:
        return

    best_baseline = baseline.loc[baseline["test_accuracy"].idxmax()]
    best_rich = rich.loc[rich["test_accuracy"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["Baseline 24-d\n(best test acc)", "Rich 84-d\n(best test acc)"]
    vals = [best_baseline["test_accuracy"] * 100, best_rich["test_accuracy"] * 100]
    colors = ["#BAB0AC", "#59A14F"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(
        f"Best tabular model per feature set\n"
        f"{best_baseline['model']} vs {best_rich['model']}"
    )
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.annotate(f"{val:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "chroma_baseline_vs_rich_best.png", dpi=160)
    plt.close(fig)


def write_summary_md(table: pd.DataFrame, path: Path) -> None:
    rich = table[table["feature_set"] == "rich_84d"].sort_values("test_accuracy", ascending=False)
    baseline = table[table["feature_set"] == "baseline_24d"].sort_values("test_accuracy", ascending=False)
    best_r = rich.iloc[0]
    best_b = baseline.iloc[0]
    delta = (best_r["test_accuracy"] - best_b["test_accuracy"]) * 100

    lines = [
        "# Rich chroma benchmark (84-d)",
        "",
        "**Feature recipe:** 4 temporal bins × 12 pitch means + 12 global stds + 24 Krumhansl–Schmuckler profile correlations.",
        "",
        "## Rich chroma results (test set)",
        "",
        "| Model | Val acc | Test acc | Test macro F1 |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in rich.iterrows():
        val = f"{row['val_accuracy'] * 100:.1f}%" if row["val_accuracy"] is not None else "—"
        lines.append(
            f"| {row['model']} | {val} | {row['test_accuracy'] * 100:.1f}% | {row['test_macro_f1']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## vs baseline 24-d (prior best)",
            "",
            f"- Baseline best: **{best_b['model']}** @ {best_b['test_accuracy'] * 100:.1f}% test acc",
            f"- Rich best: **{best_r['model']}** @ {best_r['test_accuracy'] * 100:.1f}% test acc",
            f"- Delta: **{delta:+.1f} pp** on test accuracy",
            "",
            "## Interpretation",
            "",
        ]
    )
    if delta > 1.0:
        lines.append(
            "Rich chroma improves the tabular proxy enough to keep chroma-based modeling in the portfolio plan before trying pretrained embeddings."
        )
    elif delta > 0:
        lines.append(
            "Rich chroma gives a modest bump; tabular chroma may be near its ceiling on 15s clips."
        )
    else:
        lines.append(
            "Rich chroma did not beat the baseline 24-d recipe on test accuracy; the simpler summary may generalize better."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark rich vs baseline chroma tabular features.")
    parser.add_argument("--clip-sec", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--reuse-mlp", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = resolve_out_dir(cfg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_extract:
        rich_dir = ensure_rich_cache(out_dir, args.clip_sec, cfg)
    else:
        rich_dir = key_chroma_rich_dir(out_dir, args.clip_sec)

    chunks = pd.read_csv(key_chunks_path(out_dir, args.clip_sec))
    splits = load_split_tables(chunks, rich_dir)

    results: list[dict] = []
    lr = train_logistic(splits, rich_dir)
    results.append(summarize_run("Logistic regression", "rich_84d", CHROMA_RICH_INPUT_DIM, lr))

    tree = train_tree(splits, rich_dir, seed=args.seed)
    results.append(summarize_run("HistGradientBoosting", "rich_84d", CHROMA_RICH_INPUT_DIM, tree))

    if not args.skip_mlp:
        mlp = train_mlp(
            splits,
            rich_dir,
            input_dim=CHROMA_RICH_INPUT_DIM,
            run_name="exp_rich_chroma_mlp",
            output_dir=OUTPUT_DIR / "mlp_runs",
            seed=args.seed,
            skip_if_exists=args.reuse_mlp,
        )
        val_acc = None
        hist_path = OUTPUT_DIR / "mlp_runs" / "exp_rich_chroma_mlp_history.csv"
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
            best_idx = hist["validation_loss"].idxmin()
            val_acc = float(hist.loc[best_idx, "validation_accuracy"])
        results.append(
            {
                "feature_set": "rich_84d",
                "input_dim": CHROMA_RICH_INPUT_DIM,
                "model": "Chroma MLP (scaled)",
                "train_accuracy": None,
                "val_accuracy": val_acc,
                "test_accuracy": mlp["test_accuracy"],
                "test_macro_f1": mlp["test_macro_f1"],
            }
        )

    results.extend(load_baseline_reference())
    table = pd.DataFrame(results)
    table_path = OUTPUT_DIR / "chroma_rich_comparison.csv"
    table.to_csv(table_path, index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_recipe": {
            "input_dim": CHROMA_RICH_INPUT_DIM,
            "components": [
                f"{CHROMA_RICH_TIME_BINS} time bins x 12 pitch means",
                "12 global pitch stds",
                "24 K-S major/minor profile correlations",
            ],
        },
        "results": results,
    }
    meta_path = OUTPUT_DIR / "chroma_rich_benchmark.json"
    meta_path.write_text(json.dumps(payload, indent=2))

    plot_comparison(table, OUTPUT_DIR / "plots")
    write_summary_md(table, OUTPUT_DIR / "chroma_rich_summary.md")

    print(f"Wrote {table_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {OUTPUT_DIR / 'chroma_rich_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
