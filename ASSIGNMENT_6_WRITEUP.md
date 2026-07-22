# Beatport EDM Key — Tabular Proxy Model Comparison

**Project:** Beatport EDM Key Dataset — 24-class major/minor key from audio chunks  
**Representation:** Portfolio-derived chroma tabular proxy (audio → structured features)  
**Clip length:** 15 seconds  
**Random seed:** 42  
**Repo:** [https://github.com/InvaderSquibs/cs6320-assignment6](https://github.com/InvaderSquibs/cs6320-assignment6)

Run artifacts: `outputs/` (including prior LR and unscaled MLP copies under `outputs/reused_a5/`). Experiment cards: `experiments/`. Each `*_metadata.json` records the command, seed, and metrics.

**Compute:** local Mac — MLP on MPS, sklearn on CPU. Evidence: history CSVs, metadata JSON, training logs, and plots.

---

## Part A: Tabular Representation and Model Comparison

### Problem, stakeholder, and representation

| Item | Detail |
| --- | --- |
| **Dataset** | [Beatport EDM Key Dataset](https://doi.org/10.5281/zenodo.1101082) (Faraldo, 2017), Tier A filters (`confidence ≥ 2`, single standard major/minor) |
| **Task** | Multiclass classification — predict `key_24` (24 major/minor keys) from a single audio chunk |
| **Stakeholder** | DJs/producers seeking a *practice* key hint for harmonic mixing — not an authoritative analyzer |
| **Primary data type** | Audio (non-tabular) |
| **Tabular proxy** | One row per 15s chunk; 24 numeric columns from chroma CQT summary |

**Proxy transformation** (`beatport/chroma_features.py`):

1. Load cached 15s stereo waveform per chunk.
2. Collapse to mono; compute `librosa.feature.chroma_cqt`.
3. Aggregate over time → 12 pitch-class **means** + 12 **stds** → 24-d float vector.
4. Cache as `{track_id}_{chunk_idx}.npz` under `datasets/clip_15s/features/key_chroma/`.

**Why chroma (vs generic amplitude/spectral proxies):** duration and coarse spectral shape do not encode pitch-class structure. For key detection we used chroma CQT summaries.

**What the proxy preserves:** relative energy across the 12 pitch classes (harmonic content summary).  
**What it loses:** temporal order within the chunk, timbre, polyphony detail, modulation within a chunk, and any non-harmonic cues.

### Embeddings

**Not used.** All model inputs are continuous numeric pitch-class statistics. There are no high-cardinality categorical fields (e.g. artist ID, track ID) in the feature matrix. The 24-class key label is the prediction target, not an input embedding candidate.

### Preprocessing and leakage controls

| Rule | Implementation |
| --- | --- |
| Split before fitting transforms | Song-level 70/15/15 split locked in `beatport/config.yaml`; same split reused across all models |
| Train-fit normalization | `StandardScaler` fit on **train chroma only** for LR, tree, and scaled MLP |
| Prediction-time availability | Chroma computed from audio chunk only — no future labels or post-hoc metadata |
| Sample weighting | `1 / n_chunks` per song |
| Chunk filter | 8,969 chunks with cached chroma (7 chunks dropped for missing cache) |

**Split counts** (prior split audit; chroma-filtered counts match within ±1 chunk):

| Split | Songs | Chunks (15s, chroma) |
| --- | --- | --- |
| train | ~785 | 6,285 |
| val | ~168 | 1,350 |
| test | ~169 | 1,334–1,335 |

### Models compared

Same target (`key_24`), same evaluation procedure, same metrics (chunk accuracy, macro F1). Validation used for neural early stopping / checkpoint selection; **test reported once** for final comparison.

| Model | Type | Key settings | Preprocessing |
| --- | --- | --- | --- |
| Logistic regression | Classical | `C=1.0`, `class_weight=balanced` | Train-fit `StandardScaler` |
| HistGradientBoosting | Tree | `max_depth=8`, `max_iter=200`, `lr=0.1` | Train-fit `StandardScaler` |
| Chroma MLP (unscaled) | Neural tabular | `hidden=64`, `lr=2.75e-3`, patience=8 | Raw chroma |
| Chroma MLP (scaled) | Neural tabular | same + `--scale` | Train-fit `StandardScaler` |

Neural training controls we applied:

- Validation-aware early stopping on val loss (both MLP runs).
- Train-fit normalization via scaled MLP (`--scale`).
- Checkpoint selection: unscaled best val loss at **epoch 14**; scaled at **epoch 3**.
- One diagnostic variant (scaled vs unscaled). Details: `experiments/exp2_mlp_scaled.md`.

### Results table (test set)

Consolidated in `outputs/model_comparison.csv`. Visualization: `outputs/plots/model_comparison_test_metrics.png`.

| Model | Val acc | Test acc | Test macro F1 | Practical notes |
| --- | --- | --- | --- | --- |
| Logistic regression | 40.5% | **34.6%** | 0.31 | Most interpretable; stable train→val gap (~7.6 pp) |
| HistGradientBoosting | 40.3% | **40.3%** | 0.31 | **Best test accuracy**; 100% train acc → severe overfit |
| Chroma MLP (unscaled) | 41.6% | **36.7%** | **0.33** | Best macro F1 among neural runs; small gain over LR |
| Chroma MLP (scaled) | 42.6% | **34.7%** | 0.31 | Scaling did not improve test vs unscaled MLP |

*Note:* LR evaluation used 1,335 test chunks; tree/MLP runs used 1,334 (7 missing chroma caches). Difference is negligible for comparison.

**Train vs val generalization gap** (`outputs/plots/model_comparison_train_val_gap.png`):

- LR: train 48.1% → val 40.5% (plausible gap).
- Tree: train **100%** → val 40.3% (memorization on train; val/test still align).
- MLPs: val peaks ~42% but test lands ~35–37% (mild overfitting; scaling did not fix it).

**MLP learning curves** (unscaled vs scaled): `outputs/chroma_runs/plots/loss_learning_curve.png`, `validation_metric_learning_curve.png`.

### Model-choice recommendation

**Tabular chroma proxy.** We hypothesized that pitch-class summaries would carry usable key signal above majority (~10%). We observed all models in the mid-30s to low-40s test accuracy. We treat chroma tabular features as a useful diagnostic baseline: cheap, pitch-class interpretable, and clearly above chance.

**Neural MLP vs simpler models.** We hypothesized that an MLP on the same proxy would outperform LR and tree enough to offset opacity and training cost. We observed the opposite on accuracy: tree **40.3%** > unscaled MLP **36.7%** > scaled MLP / LR ~**34.6–34.7%**. Unscaled MLP had the best macro F1 (0.33) but only ~2 pp above LR on accuracy. We recommend keeping LR or tree on chroma as monitoring baselines and not adopting the neural tabular MLP for this proxy.

**Next representation step.** Keep LR/tree chroma checks in the harness. Pursue richer audio representations (longer clips, spectrogram CNN, pretrained embeddings) rather than deeper MLP tuning on 24-d summaries.

### Practical constraints

| Constraint | LR | Tree | MLP |
| --- | --- | --- | --- |
| Interpretability | High (linear weights on pitch classes) | Medium (splits hard to explain per prediction) | Low |
| Training cost | Seconds (CPU) | ~30s (CPU) | ~7–15 min (MPS) |
| Inference cost | Trivial | Trivial | Trivial at this size |
| Maintainability | Simple sklearn pipeline | sklearn; watch overfit/monitoring | PyTorch + scaler state + checkpoint |
| Monitoring | Stable train/val gap | **Alert if train→prod gap widens** (100% train) | Track val plateau / calibration |

### Responsible-use concern

**Automation bias:** A DJ may treat any model output as ground truth. At ~35–40% chunk accuracy with poorly calibrated confidence (prior error analysis), automated key labels could mislead harmonic mixing decisions. **Mitigation:** Present outputs as suggestions requiring human review; do not auto-tag tracks for commercial use without listener QA. **Scope limit:** EDM-only training data — no evidence of cross-genre fairness or representativeness.

---

## Part B: Portfolio Checkpoint and Model-Choice Note

| Item | Status |
| --- | --- |
| **Data readiness** | Tier A filtered (1,122 songs); 15s waveforms + chroma cached; song-level splits audited |
| **Baseline / model status** | Tabular proxy complete. Tree ~40% test acc > MLP ~37% > LR ~35%. Raw waveform CNN previously failed (~2–4%). |
| **Next planned experiment** | Extract chroma for **20s and 30s** clips and rerun the same tabular comparison — tests whether short clips weaken harmonic signal before moving to spectrogram / transfer learning |
| **Expected staged improvement** | Longer windows may lift all tabular models modestly; final system likely needs modality-specific deep audio, not tabular MLP alone |
| **Model-choice implication** | Tabular methods are **indirectly relevant**: they show harmonic signal in structured features but are insufficient for deployment. Evidence supports **simple tabular baselines for monitoring**, not neural tabular as the final model |
| **Prior audit updates** | **Still untested:** clip-length comparison (20s/30s), EDM-only generalization, rare-key fairness. **Confirmed:** chroma tabular beats majority; tree can overfit train while val/test stay ~40% |
| **Tabular / embeddings relevance** | **Indirect** — chroma proxy for diagnostics. **Embeddings not relevant** to current feature matrix. **Simpler baselines directly relevant** — tree/LR should stay in the evaluation harness |

---

## Experiment reports

| Exp | Report | Role |
| --- | --- | --- |
| 1 | `experiments/exp1_tree_baseline.md` | Tree baseline on 24-d chroma |
| 2 | `experiments/exp2_mlp_scaled.md` | Scaled MLP vs unscaled MLP |
| 3 | `experiments/exp3_model_comparison.md` | Fair comparison → model-choice call |
| 4 | `experiments/exp4_rich_chroma.md` | Appendix: 84-d rich chroma |

## Artifact index

| Category | Path |
| --- | --- |
| Tree baseline script | `train_key_chroma_tree.py` |
| Comparison plots script | `plot_model_comparison.py` |
| Consolidated metrics | `outputs/model_comparison.csv` |
| Model comparison charts | `outputs/plots/model_comparison_test_metrics.png`, `model_comparison_train_val_gap.png` |
| Tree run | `outputs/tree_runs/exp6_chroma_tree_*` |
| Scaled MLP run | `outputs/chroma_runs/exp6_chroma_mlp_scaled_*` (history, metadata, log) |
| MLP learning curves | `outputs/chroma_runs/plots/` |
| Prior LR baseline | `outputs/reused_a5/baseline_metrics/` |
| Prior unscaled MLP | `outputs/reused_a5/chroma_runs/exp3_chroma_baseline_*` |
| Proxy source | `../beatport/chroma_features.py` |
| Rich chroma benchmark | `outputs/chroma_rich/chroma_rich_*`, `run_chroma_rich_benchmark.py` |

## Commands (reference)

```bash
cd beatport && source .venv/bin/activate

# Tree baseline
python ../assignment_6/train_key_chroma_tree.py --run-name exp6_chroma_tree

# Scaled MLP
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp6_chroma_mlp_scaled \
  --learning-rate 0.00275 --epochs 30 --scale --quiet \
  --output-dir ../assignment_6/outputs/chroma_runs

# MLP learning curves (unscaled vs scaled)
python ../assignment_5/plot_learning_curves.py \
  ../assignment_5/outputs/chroma_runs/exp3_chroma_baseline_history.csv \
  ../assignment_6/outputs/chroma_runs/exp6_chroma_mlp_scaled_history.csv \
  --output-dir ../assignment_6/outputs/chroma_runs/plots

# Model comparison table + bar charts
python ../assignment_6/plot_model_comparison.py
```

---

## Appendix: Rich chroma benchmark (84-d)

After the 24-d comparison, we upgraded the tabular proxy to test a richer chroma-only representation before pretrained embeddings.

**Recipe** (`beatport/chroma_features.py --rich`):

| Component | Dim | Purpose |
| --- | --- | --- |
| 4 time bins × 12 pitch means | 48 | Coarse harmonic timing within chunk |
| 12 global pitch stds | 12 | Stability per pitch class |
| 24 Krumhansl–Schmuckler profile correlations | 24 | Key-template match scores |

**Best chroma-only model:** `HistGradientBoosting` on rich features — **42.6% test acc** (+2.3 pp vs 40.3% tree on 24-d).

| Feature set | Best model | Test acc | Test macro F1 |
| --- | --- | --- | --- |
| Baseline 24-d | HistGradientBoosting | 40.3% | 0.31 |
| **Rich 84-d** | **HistGradientBoosting** | **42.6%** | **0.36** |

Full table: `outputs/chroma_rich/chroma_rich_comparison.csv`  
Plots: `outputs/chroma_rich/plots/chroma_baseline_vs_rich_best.png`

**Observation:** richer hand-built chroma helped modestly (~2 pp) and still looks near a tabular ceiling on 15s clips. Next fidelity step: pretrained audio embeddings.

```bash
python chroma_features.py --rich --clip-sec 15
python ../assignment_6/run_chroma_rich_benchmark.py
```
