"""Create tabular proxy features from event, sequence, or time-series rows.

The output has one row per entity. Use it to test whether summary statistics,
lags, and trends provide enough signal for a tabular model before sequence
models are covered later in the course.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def numeric_columns(data: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [col for col in data.select_dtypes(include=[np.number]).columns if col not in exclude]


def slope_from_order(values: pd.Series) -> float:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return 0.0
    x = np.arange(len(clean), dtype=float)
    return float(np.polyfit(x, clean, deg=1)[0])


def summarize_group(group: pd.DataFrame, value_columns: list[str], time_column: str | None) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"row_count": len(group)}

    if time_column:
        times = pd.to_datetime(group[time_column], errors="coerce")
        valid_times = times.dropna()
        row["valid_time_count"] = int(valid_times.shape[0])
        if len(valid_times) >= 2:
            span_seconds = (valid_times.max() - valid_times.min()).total_seconds()
            row["time_span_seconds"] = float(span_seconds)
            row["events_per_day"] = float(len(group) / max(span_seconds / 86400.0, 1e-9))
        else:
            row["time_span_seconds"] = 0.0
            row["events_per_day"] = 0.0

    for col in value_columns:
        values = group[col]
        clean = values.dropna()
        diffs = clean.diff().dropna()
        row[f"{col}_mean"] = float(clean.mean()) if len(clean) else 0.0
        row[f"{col}_std"] = float(clean.std(ddof=0)) if len(clean) else 0.0
        row[f"{col}_min"] = float(clean.min()) if len(clean) else 0.0
        row[f"{col}_max"] = float(clean.max()) if len(clean) else 0.0
        row[f"{col}_first"] = float(clean.iloc[0]) if len(clean) else 0.0
        row[f"{col}_last"] = float(clean.iloc[-1]) if len(clean) else 0.0
        row[f"{col}_last_minus_first"] = row[f"{col}_last"] - row[f"{col}_first"]
        row[f"{col}_slope_by_order"] = slope_from_order(clean)
        row[f"{col}_mean_abs_change"] = float(diffs.abs().mean()) if len(diffs) else 0.0
        row[f"{col}_missing_fraction"] = float(values.isna().mean())

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tabular proxy features from sequence rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entity-column", required=True)
    parser.add_argument("--time-column", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--value-columns", default="", help="Comma-separated numeric columns. Default: all numeric inputs.")
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    if args.entity_column not in data.columns:
        raise ValueError(f"Missing entity column: {args.entity_column}")

    exclude = {args.entity_column}
    if args.label_column:
        exclude.add(args.label_column)
    value_columns = [col.strip() for col in args.value_columns.split(",") if col.strip()]
    if not value_columns:
        value_columns = numeric_columns(data, exclude)
    missing_values = set(value_columns).difference(data.columns)
    if missing_values:
        raise ValueError(f"Missing value columns: {sorted(missing_values)}")

    rows = []
    for entity, group in data.groupby(args.entity_column, sort=False):
        if args.time_column and args.time_column in group.columns:
            group = group.sort_values(args.time_column)
        row = {args.entity_column: entity}
        row.update(summarize_group(group, value_columns, args.time_column))
        if args.label_column:
            labels = group[args.label_column].dropna()
            row[args.label_column] = labels.iloc[-1] if len(labels) else np.nan
        rows.append(row)

    features = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
