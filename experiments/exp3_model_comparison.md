# Experiment Report: model comparison (Exp 3)

> **Reproducibility tag:** `seed=42` · clip `15s` · shared song-level splits  
> Local run evidence: `outputs/model_comparison.csv`, `outputs/plots/`, plus Exp 1–2 / prior baseline artifacts

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `model_comparison` (aggregate) |
| seed | **42** |
| change | Same target/split/metrics across LR, HistGradientBoosting, unscaled MLP, scaled MLP |
| hypothesis | If the 24-d chroma proxy is adequate, an MLP will clearly beat classical baselines on held-out test; otherwise the simpler model is the better choice. |
| expected_signal | At least one classical baseline ≥ ~35% test; MLP preferred only if the gain clearly offsets opacity and cost. |
| observed_signal | tree_wins_accuracy — tree **40.3%** test > unscaled MLP **36.7%** > scaled MLP / LR ~**34.6–34.7%** |
| fixed_conditions | `key_24`, song-level 70/15/15, chunk accuracy + macro F1, prediction-time chroma only |
| limitation | focused_comparison_not_architecture_search |

## Hypothesis (pre-run)

**Hypothesis:** Under a shared protocol, the chroma MLP will either (a) beat LR and tree by a clear margin worth the maintenance cost, or (b) fail that bar, in which case LR/tree remain the tabular baselines.

**Expected behavior:** Compact results table with val for selection / test for final reporting; a recommendation based on held-out metrics and practical constraints.

## Setup (shared protocol)

| Piece | Choice |
| --- | --- |
| Target | `key_24` (24-class) |
| Features | 24-d chroma CQT mean/std |
| Split | Song-level 70/15/15 |
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
| Logistic regression | 40.5% | 34.6% | 0.31 | Most interpretable |
| HistGradientBoosting | 40.3% | **40.3%** | 0.31 | Best test acc; 100% train |
| Chroma MLP unscaled | 41.6% | 36.7% | **0.33** | Best neural F1; small gain over LR |
| Chroma MLP scaled | 42.6% | 34.7% | 0.31 | Scaling did not improve test |

**Vs majority (~10%):** all models clearly beat chance — the proxy carries harmonic signal.

**Vs expected:** Branch (b). Tree leads on accuracy; MLP does not outperform tree and only modestly exceeds LR.

## Diagnosis / model-choice call

- **Tabular proxy:** useful diagnostic baseline (above chance, cheap, pitch-class readable).
- **Neural tabular MLP:** not selected for this proxy given tree’s better test accuracy and higher cost/opacity.
- **Keep:** LR or tree on chroma as sanity-check / monitoring baselines.
- **Next:** richer audio representation (clip length, spectrogram, pretrained embeddings) — not deeper MLP tuning.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/model_comparison.csv` | Consolidated metrics |
| `outputs/plots/model_comparison_test_metrics.png` | Test comparison chart |
| `outputs/plots/model_comparison_train_val_gap.png` | Generalization gap chart |
| `outputs/reused_a5/baseline_metrics/` | LR baseline |
| Exp 1–2 artifact dirs | Tree + scaled MLP |
