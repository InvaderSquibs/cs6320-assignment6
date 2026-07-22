# Submission package — tabular proxy model comparison

**Author:** Zach Walton  
**GitHub (code + run evidence):** [https://github.com/InvaderSquibs/cs6320-assignment6](https://github.com/InvaderSquibs/cs6320-assignment6)

Re-run `scripts/build_submission_zip.sh` after updates.

## Layout

```
assignment_6_submission_v1.zip
├── README_SUBMISSION.txt
├── writeup/
│   └── ASSIGNMENT_6_WRITEUP.md          # Part A + Part B
└── part_a/
    ├── code/                            # scripts + experiment reports (no caches / weights)
    └── run_evidence/                    # lightweight local proof of computation
        ├── experiments/                 # hypothesis-case report cards
        ├── tree_runs/                   # HistGradientBoosting metadata + metrics
        ├── chroma_runs/                 # scaled MLP history, metadata, log, plots
        ├── reused_a5/                   # prior LR + unscaled MLP evidence
        ├── chroma_rich/                 # appendix rich-chroma comparison
        ├── model_comparison.csv
        └── plots/                       # comparison charts
```

## What proves the experiments ran

| Role | Run | Evidence |
| --- | --- | --- |
| Classical tree | `exp6_chroma_tree` | `*_metadata.json`, `*_metrics.csv` |
| Neural tabular (scaled) | `exp6_chroma_mlp_scaled` | history, metadata, log, learning-curve plots |
| Prior LR / unscaled MLP | reused baselines | `outputs/reused_a5/` |
| Fair comparison | aggregate | `model_comparison.csv` + plots |
| Rich chroma (appendix) | 84-d benchmark | `chroma_rich/*` |

Each `*_metadata.json` records device (`mps` or CPU), seed (`42`), command, and metrics.

Hypothesis / expected / observed / diagnosis: `experiments/exp*.md`.

## Excluded from the zip (intentional)

- Model weights (`*.pt`)
- Audio / chroma `.npz` feature stores
- Assignment prompt (`requirements.md`)
- Lecture PDF

## Build

```bash
./scripts/build_submission_zip.sh
```

Output: `dist/assignment_6_submission_v1.zip`
