# Experiment Report: exp6_chroma_tree (Exp 1)

> **Reproducibility tag:** `seed=42` · clip `15s` · train-fit `StandardScaler`  
> Classical tree baseline for Assignment 6 fair comparison.  
> Local run evidence: `outputs/tree_runs/exp6_chroma_tree_*`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `exp6_chroma_tree` |
| seed | **42** |
| change | Model family: add `HistGradientBoostingClassifier` on the same 24-d chroma proxy used by A5 LR / MLP |
| hypothesis | A nonlinear tree ensemble on pitch-class summaries should beat or match logistic regression (~34.6% test) without needing neural complexity. |
| expected_signal | Test accuracy in the mid-to-high 30%s or low 40%s; possible train overfit with val/test still aligned. |
| observed_signal | tree_best_test_but_train_memorized — test acc **40.3%**, train acc **100%**, val ≈ test (~40.3%) |
| fixed_conditions | same song-level 70/15/15 splits, seed 42, 15s chroma, sample_weight = 1/n_chunks |
| limitation | single_tree_config (not a hyperparameter tournament) |

## Hypothesis (pre-run)

**Lesson grounding:** Week 6 fair comparison recipe — a simple classical baseline must give the neural tabular model something meaningful to beat.

**Hypothesis:** HistGradientBoosting on train-scaled chroma will capture nonlinear pitch-class interactions that linear LR misses, lifting test accuracy above the A5 LR band (~34.6%) while remaining cheap and interpretable enough as a monitoring baseline.

**Expected behavior:** Val/test accuracy ≥ LR; train–val gap may be large (trees can memorize) but val and test should still agree if the split is honest.

## Setup

| Setting | Value |
| --- | --- |
| Input | 24-d chroma CQT mean/std |
| Preprocessing | `StandardScaler` fit on **train only** |
| Model | `HistGradientBoostingClassifier` (`max_depth=8`, `max_iter=200`, `lr=0.1`) |
| Sample weight | chunk `1 / n_chunks` per song |
| Device | CPU (sklearn) |

## Commands

```bash
cd beatport && source .venv/bin/activate
python ../assignment_6/train_key_chroma_tree.py --run-name exp6_chroma_tree
```

## Observed behavior

| Split | Accuracy | Macro F1 |
| --- | --- | --- |
| train | **100%** | 1.00 |
| val | 40.3% | 0.33 |
| test | **40.3%** | 0.31 |

**Vs expected:** Confirmed on test lift vs LR (+~5.7 pp). Unexpected severity of train memorization (100% train) — still usable as a baseline because val/test align, but monitoring must watch train→prod gap.

## Diagnosis

- Tree is the **strongest 24-d tabular accuracy** in the main comparison.
- Severe train overfit is a practical constraint (interpretability/monitoring), not a leakage indictment of the split.
- Next: neural MLP under the same protocol to test whether added complexity is justified.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/tree_runs/exp6_chroma_tree_metadata.json` | Config + split metrics + command |
| `outputs/tree_runs/exp6_chroma_tree_metrics.csv` | Split metric table |
