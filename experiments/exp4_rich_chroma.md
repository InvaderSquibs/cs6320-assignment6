# Experiment Report: rich chroma 84-d (Exp 4 — appendix)

> **Reproducibility tag:** `seed=42` · clip `15s` · feature variant `rich`  
> Local run evidence: `outputs/chroma_rich/`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `chroma_rich` benchmark |
| seed | **42** |
| change | Feature recipe only: 24-d mean/std → 84-d (4 time bins × 12 means + 12 stds + 24 KS profile correlations) |
| hypothesis | Adding coarse temporal bins and key-template correlations will lift tabular models above the ~40% tree ceiling on 24-d features. |
| expected_signal | Modest lift (a few points) for tree and/or MLP; still below deployment-quality key detection. |
| observed_signal | modest_lift_tree_still_best — rich tree **42.6%** test (+2.3 pp vs 40.3%); MLP/LR still ~35% |
| fixed_conditions | same splits, seed, model families, evaluation metrics |
| limitation | still_hand_built_proxy (not pretrained embeddings) |

## Hypothesis (pre-run)

**Hypothesis:** An 84-d rich chroma vector preserves more within-chunk harmonic structure than global mean/std alone, so the same tree/MLP/LR comparison will improve, with tree remaining competitive.

**Expected behavior:** Best rich model ≥ best 24-d model; gain large enough to keep chroma in the staged plan, but not large enough to claim a final production model.

## Setup

| Component | Dim | Purpose |
| --- | --- | --- |
| 4 time bins × 12 pitch means | 48 | Coarse harmonic timing |
| 12 global pitch stds | 12 | Stability per pitch class |
| 24 Krumhansl–Schmuckler correlations | 24 | Key-template match scores |
| **Total** | **84** | |

Models: HistGradientBoosting, scaled MLP (`hidden=128`), logistic regression — same evaluation protocol.

## Commands

```bash
cd beatport && source .venv/bin/activate
python chroma_features.py --rich --clip-sec 15
python ../assignment_6/run_chroma_rich_benchmark.py
```

## Observed behavior

| Feature set | Best model | Test acc | Test macro F1 |
| --- | --- | --- | --- |
| Baseline 24-d | HistGradientBoosting | 40.3% | 0.31 |
| **Rich 84-d** | **HistGradientBoosting** | **42.6%** | **0.36** |

Rich MLP / LR remain ~35% test — MLP still trails the tree.

**Vs expected:** Confirmed modest lift; tabular ceiling still visible on 15s clips.

## Diagnosis

- Richer hand-built chroma helps (~2 pp) but remains near a tabular ceiling.
- Next fidelity step: pretrained audio embeddings / modality-specific models, not more MLP width on chroma summaries.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/chroma_rich/chroma_rich_comparison.csv` | Full rich comparison table |
| `outputs/chroma_rich/chroma_rich_summary.md` | Short summary |
| `outputs/chroma_rich/chroma_rich_benchmark.json` | Machine-readable results |
| `outputs/chroma_rich/mlp_runs/exp_rich_chroma_mlp_*` | Rich MLP history/metadata/log |
| `outputs/chroma_rich/plots/*.png` | Comparison charts |
