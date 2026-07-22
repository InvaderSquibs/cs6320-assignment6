# Rich chroma benchmark (84-d)

**Feature recipe:** 4 temporal bins × 12 pitch means + 12 global stds + 24 Krumhansl–Schmuckler profile correlations.

## Rich chroma results (test set)

| Model | Val acc | Test acc | Test macro F1 |
| --- | --- | --- | --- |
| HistGradientBoosting | 42.8% | 42.6% | 0.362 |
| Chroma MLP (scaled) | 37.9% | 35.2% | 0.321 |
| Logistic regression | 40.0% | 35.1% | 0.314 |

## vs baseline 24-d (prior best)

- Baseline best: **HistGradientBoosting** @ 40.3% test acc
- Rich best: **HistGradientBoosting** @ 42.6% test acc
- Delta: **+2.3 pp** on test accuracy

## Interpretation

Rich chroma improves the tabular proxy enough to keep chroma-based modeling in the portfolio plan before trying pretrained embeddings.
