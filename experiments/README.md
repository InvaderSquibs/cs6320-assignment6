# Assignment 6 Experiments — Hypothesis Before You Run

Each experiment records **changed setting → hypothesis → expected behavior → observed → diagnosis** (same shape as Assignments 4–5). Artifacts live under `outputs/`.

| # | Report | Role |
| --- | --- | --- |
| 1 | [exp1_tree_baseline.md](exp1_tree_baseline.md) | Classical tree baseline on 24-d chroma proxy |
| 2 | [exp2_mlp_scaled.md](exp2_mlp_scaled.md) | Neural tabular MLP + train-fit scaling diagnostic |
| 3 | [exp3_model_comparison.md](exp3_model_comparison.md) | Fair comparison: LR vs tree vs MLP → model-choice call |
| 4 | [exp4_rich_chroma.md](exp4_rich_chroma.md) | Appendix: richer 84-d chroma proxy (staged improvement) |

**Reproducibility:** seed `42`, device recorded in each `*_metadata.json` (local MPS / CPU runs).

**CHPC:** not required — local metadata, history CSVs, logs, and plots are the run evidence (same teacher guidance as Assignment 5).

**Reused Assignment 5 evidence** (copied under `outputs/reused_a5/` for a self-contained repo):

- Logistic regression chroma baseline
- Unscaled chroma MLP (`exp3_chroma_baseline`)
