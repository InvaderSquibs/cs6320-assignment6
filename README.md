# CS6320 Assignment 6

Tabular modeling and model-choice justification on the Beatport EDM Key portfolio task (24-class major/minor key from 15s audio chunks), using a portfolio-derived **chroma tabular proxy**.

**Repository:** [https://github.com/InvaderSquibs/cs6320-assignment6](https://github.com/InvaderSquibs/cs6320-assignment6)

**Author:** Zach Walton

## What’s in this repo

| Path | Purpose |
| --- | --- |
| `ASSIGNMENT_6_WRITEUP.md` | Part A + Part B writeup |
| `experiments/` | Hypothesis-case report cards (changed setting → expected → observed → diagnosis) |
| `train_key_chroma_tree.py` | HistGradientBoosting baseline on cached chroma |
| `plot_model_comparison.py` | Fair comparison table + charts |
| `run_chroma_rich_benchmark.py` | Appendix: 84-d rich chroma comparison |
| `week06_proxy_translators/` | Instructor Week 6 proxy starters (reference; portfolio uses chroma) |
| `outputs/` | Lightweight local run evidence (history, metadata, logs, plots) — **no model weights** |
| `outputs/reused_a5/` | Copied A5 LR + unscaled MLP evidence for a self-contained comparison |

## Experiment summary

| Exp | Hypothesis focus | Local result (seed 42) |
| --- | --- | --- |
| 1 `exp6_chroma_tree` | Tree beats LR on 24-d chroma | **40.3%** test; 100% train (memorize) |
| 2 `exp6_chroma_mlp_scaled` | Train-fit scaling helps MLP | Val peak early; test **34.7%** (worse than unscaled 36.7%) |
| 3 model comparison | Neural justified over baselines? | **No** — tree wins accuracy; keep simple baselines |
| 4 rich chroma (appendix) | 84-d features lift ceiling | Tree **42.6%** (+2.3 pp); still tabular ceiling |

Full report cards: [`experiments/README.md`](experiments/README.md).

## Setup (local Mac)

Training depends on the sibling `beatport/` package and cached chroma features (not shipped here — too large / audio-derived).

```bash
cd ../beatport
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Tree baseline
python ../assignment_6/train_key_chroma_tree.py --run-name exp6_chroma_tree

# Scaled MLP (requires assignment_5/train_key_chroma.py in the course monorepo)
PYTHONUNBUFFERED=1 python ../assignment_5/train_key_chroma.py \
  --run-name exp6_chroma_mlp_scaled \
  --learning-rate 0.00275 --epochs 30 --scale --quiet \
  --output-dir ../assignment_6/outputs/chroma_runs

python ../assignment_6/plot_model_comparison.py
```

Reproduce commands for each run are in the matching `outputs/*/*_metadata.json` and experiment markdown.

## Run evidence

Primary runs were executed **locally** (`device=mps` for MLP; CPU for sklearn; `seed=42`). History CSVs, metadata JSON, logs, and plots are under `outputs/`.

**CHPC is not required** for this submission — local training evidence at CHPC-comparable detail (command, seed, device, metrics, logs/curves) is sufficient per course guidance used in Assignment 5.
