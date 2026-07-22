"""Create node-level tabular proxy features from an edge list.

This routine avoids graph neural networks and networkx. It summarizes each node
with degree-style features so you can test whether graph structure alone gives a
useful tabular baseline for Assignment 6.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_node_features(edges: pd.DataFrame, source_col: str, target_col: str, weight_col: str | None) -> pd.DataFrame:
    work = edges[[source_col, target_col] + ([weight_col] if weight_col else [])].copy()
    work[source_col] = work[source_col].astype(str)
    work[target_col] = work[target_col].astype(str)
    work["_weight"] = work[weight_col] if weight_col else 1.0
    work["_weight"] = pd.to_numeric(work["_weight"], errors="coerce").fillna(1.0)

    out_degree = work.groupby(source_col).size().rename("out_degree")
    in_degree = work.groupby(target_col).size().rename("in_degree")
    out_weight = work.groupby(source_col)["_weight"].sum().rename("out_weight_sum")
    in_weight = work.groupby(target_col)["_weight"].sum().rename("in_weight_sum")

    nodes = pd.Index(sorted(set(work[source_col]).union(set(work[target_col]))), name="node")
    features = pd.DataFrame(index=nodes)
    features = features.join(out_degree, how="left").join(in_degree, how="left")
    features = features.join(out_weight, how="left").join(in_weight, how="left").fillna(0)
    features["total_degree"] = features["out_degree"] + features["in_degree"]
    features["total_weight_sum"] = features["out_weight_sum"] + features["in_weight_sum"]
    features["degree_balance_out_minus_in"] = features["out_degree"] - features["in_degree"]

    undirected = pd.concat(
        [
            work[[source_col, target_col]].rename(columns={source_col: "node", target_col: "neighbor"}),
            work[[target_col, source_col]].rename(columns={target_col: "node", source_col: "neighbor"}),
        ],
        ignore_index=True,
    )
    neighbor_counts = undirected.groupby("node")["neighbor"].nunique().rename("unique_neighbor_count")
    self_loops = work[work[source_col] == work[target_col]].groupby(source_col).size().rename("self_loop_count")
    features = features.join(neighbor_counts, how="left").join(self_loops, how="left").fillna(0)

    neighbor_degree = features["total_degree"].rename("neighbor_total_degree")
    neighbor_summary = undirected.merge(neighbor_degree, left_on="neighbor", right_index=True, how="left")
    neighbor_mean = neighbor_summary.groupby("node")["neighbor_total_degree"].mean().rename("mean_neighbor_degree")
    features = features.join(neighbor_mean, how="left").fillna(0)
    return features.reset_index()


def attach_labels(features: pd.DataFrame, labels_path: Path | None, node_column: str, label_column: str) -> pd.DataFrame:
    if labels_path is None:
        return features
    labels = pd.read_csv(labels_path)
    expected = {node_column, label_column}
    missing = expected.difference(labels.columns)
    if missing:
        raise ValueError(f"labels file is missing columns: {sorted(missing)}")
    labels = labels[[node_column, label_column]].rename(columns={node_column: "node"})
    labels["node"] = labels["node"].astype(str)
    features["node"] = features["node"].astype(str)
    return features.merge(labels, on="node", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build node-level tabular proxy features from an edge list.")
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-column", default="source")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--weight-column", default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--node-column", default="node")
    parser.add_argument("--label-column", default="label")
    args = parser.parse_args()

    edges = pd.read_csv(args.edges)
    required = {args.source_column, args.target_column}
    if args.weight_column:
        required.add(args.weight_column)
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"edges file is missing columns: {sorted(missing)}")

    features = build_node_features(edges, args.source_column, args.target_column, args.weight_column)
    features = attach_labels(features, args.labels, args.node_column, args.label_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features)} rows and {len(features.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
