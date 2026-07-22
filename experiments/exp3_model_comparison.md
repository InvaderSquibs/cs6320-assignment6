# Experiment Report: model comparison (Exp 3)

> **Reproducibility tag:** `seed=42` · clip `15s` · shared song-level splits  
> Core Assignment 6 deliverable: fair model-choice argument on the chroma tabular proxy.  
> Local run evidence: `outputs/model_comparison.csv`, `outputs/plots/`, plus Exp 1–2 / reused A5 artifacts

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `model_comparison` (aggregate) |
| seed | **42** |
| change | Same target/split/metrics across LR, HistGradientBoosting, unscaled MLP, scaled MLP |
| hypothesis | If the 24-d chroma proxy is adequate, a neural MLP should clearly beat classical baselines on held-out test; otherwise recommend the simpler model. |
| expected_signal | At least one classical baseline ≥ ~35% test; neural win only if gain is large enough to justify opacity/cost. |
| observed_signal | tree_wins_accuracy_mlp_not_justified — tree **40.3%** test > unscaled MLP **36.7%** > scaled MLP / LR ~**34.6–34.7%** |
| fixed_conditions | `key_24`, song-level 70/15/15, chunk accuracy + macro F1, prediction-time chroma only |
| limitation | focused_comparison_not_architecture_search |

## Hypothesis (pre-run)

**Lesson grounding:** Week 6 — compare models fairly; recommend responsibly; neural complexity needs evidence, not hype.

**Hypothesis:** Under a shared protocol, the chroma MLP will either (a) beat LR and tree by a clear margin worth the maintenance cost, or (b) fail to justify itself, in which case LR/tree remain the portfolio tabular baselines.

**Expected behavior:** Compact results table with val for selection / test for final reporting; a one-sentence recommendation that names which model is justified and what would change the call.

## Setup (shared protocol)

| Piece | Choice |
| --- | --- |
| Target | `key_24` (24-class) |
| Features | 24-d chroma CQT mean/std (portfolio-derived proxy) |
| Split | Song-level 70/15/15 (locked from A4/A5) |
| Metrics | Chunk accuracy, macro F1 |
| Selection | Val evidence for MLP checkpoint; test reported once for final table |
| Embeddings | Not used — all inputs continuous numeric |

## Commands

```bash
cd beatport && source .venv/bin/activate
python ../assignment_6/plot_model_comparison.py
```

## Observed behavior (test set)

| Model | Val acc | Test acc | Test macro F1 | Practical note |
| --- | --- | --- | --- | --- |
| Logistic regression (A5) | 40.5% | 34.6% | 0.31 | Most interpretable |
| HistGradientBoosting | 40.3% | **40.3%** | 0.31 | Best test acc; 100% train |
| Chroma MLP unscaled (A5) | 41.6% | 36.7% | **0.33** | Best neural F1; small gain over LR |
| Chroma MLP scaled | 42.6% | 34.7% | 0.31 | Scaling did not improve test |

**Vs majority sanity (~10%):** all models clearly beat chance — proxy carries harmonic signal.

**Vs expected:** Branch (b). Neural tabular is **not** justified over the tree baseline on accuracy, or over LR given cost/opacity.

## Diagnosis / model-choice call

- **Tabular proxy:** justified as a bounded diagnostic baseline.
- **Neural tabular MLP:** not justified on this 24-d proxy.
- **Keep:** LR or tree on chroma as sanity-check / monitoring baselines.
- **Next portfolio step:** richer audio representation (clip length, spectrogram, pretrained embeddings) — not deeper MLP tuning.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/model_comparison.csv` | Consolidated metrics |
| `outputs/plots/model_comparison_test_metrics.png` | Test comparison chart |
| `outputs/plots/model_comparison_train_val_gap.png` | Generalization gap chart |
| `outputs/reused_a5/baseline_metrics/` | LR baseline |
| Exp 1–2 artifact dirs | Tree + scaled MLP |
