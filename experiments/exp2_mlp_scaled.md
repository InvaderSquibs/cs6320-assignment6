# Experiment Report: exp6_chroma_mlp_scaled (Exp 2)

> **Reproducibility tag:** `seed=42` · device `mps` · clip `15s` · early-stop patience `8`  
> Local run evidence: `outputs/chroma_runs/exp6_chroma_mlp_scaled_*`  
> Comparator (unscaled): `outputs/reused_a5/chroma_runs/exp3_chroma_baseline_*`

## One-change experiment log

| Field | Value |
| --- | --- |
| run_id | `exp6_chroma_mlp_scaled` |
| seed | **42** |
| change | Preprocessing only vs unscaled MLP: enable train-fit `StandardScaler` (`--scale`); keep architecture / LR / patience |
| hypothesis | Train-fit normalization will stabilize training and may improve generalization vs raw chroma inputs. |
| expected_signal | Val-selected checkpoint; test accuracy at least matching unscaled MLP (~36.7%) if scaling helps; no worse than LR (~34.6%). |
| observed_signal | scaling_did_not_help_test — best val at **epoch 3** (val acc 42.6%), test acc **34.7%** (below unscaled 36.7%) |
| fixed_conditions | same splits, seed, hidden=64, lr=2.75e-3, patience=8, 15s chroma |
| limitation | one_focused_control (normalization), not a regularization sweep |

## Hypothesis (pre-run)

**Hypothesis:** Applying train-fit StandardScaler to the same chroma MLP will improve or at least preserve test performance relative to the unscaled run, because feature scales across pitch-class means/stds differ.

**Expected behavior:** Early-stop selects a clear best-val epoch; test accuracy ≥ unscaled MLP if preprocessing was the missing piece; learning curves show faster early progress than unscaled.

## Setup

| Setting | Value |
| --- | --- |
| Input | 24-d chroma CQT mean/std |
| Preprocessing | `StandardScaler` fit on **train only** (`--scale`) |
| Model | Chroma MLP (`hidden_dim=64`) |
| Optimizer | Adam, `lr=0.00275`, `weight_decay=0`, `dropout=0` |
| Epochs | 30 requested → ran 11 (patience 8) |
| Selection rule | lowest validation loss → **epoch 3** |
| Device | MPS |

## Commands

```bash
cd beatport && source .venv/bin/activate

PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp6_chroma_mlp_scaled \
  --learning-rate 0.00275 --epochs 30 --scale --quiet \
  --output-dir ../assignment_6/outputs/chroma_runs

python ../assignment_5/plot_learning_curves.py \
  ../assignment_5/outputs/chroma_runs/exp3_chroma_baseline_history.csv \
  ../assignment_6/outputs/chroma_runs/exp6_chroma_mlp_scaled_history.csv \
  --output-dir ../assignment_6/outputs/chroma_runs/plots
```

## Observed behavior

| Run | Best epoch (val loss) | Val acc | Test acc | Test macro F1 |
| --- | --- | --- | --- | --- |
| Unscaled MLP | 14 | 41.6% | **36.7%** | **0.33** |
| Scaled MLP | **3** | 42.6% | 34.7% | 0.31 |

**Vs expected:** Partially contradicted. Scaling produced a strong early val peak but worse test than unscaled. Early stopping and train-fit normalization ran as designed; they did not raise held-out accuracy enough to prefer the MLP over LR/tree.

## Diagnosis

- MLP trains under this protocol, but held-out gains vs LR are small and tree remains stronger on accuracy.
- Control tested: train-fit scaling + validation early stopping.
- Next: consolidate LR + tree + both MLPs into one comparison table for the model-choice call.

## Artifacts

| File | Purpose |
| --- | --- |
| `outputs/chroma_runs/exp6_chroma_mlp_scaled_history.csv` | Epoch metrics |
| `outputs/chroma_runs/exp6_chroma_mlp_scaled_metadata.json` | Config + test scores + command |
| `outputs/chroma_runs/exp6_chroma_mlp_scaled.log` | Training log |
| `outputs/chroma_runs/plots/*.png` | Unscaled vs scaled learning curves |
| `outputs/reused_a5/chroma_runs/exp3_chroma_baseline_*` | Unscaled comparator |
