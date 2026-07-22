# Week 6 Proxy Translators

These starter scripts help you create tabular proxy features from portfolio data that may not be natively tabular. Use them for Assignment 6 when you need a bounded way to test whether tabular baselines or neural tabular models can produce useful evidence for your portfolio problem.

The proxy table is not your final representation. It is a Week 6 diagnostic tool. A simple proxy can show that structured features are useful, inadequate, or worth combining later with CNNs, transfer learning, sequence models, transformers, or other methods covered after Week 6.

## Files

- `image_proxy_features.py`: image folder to one row per image with size, color, brightness, and edge-summary features.
- `text_proxy_features.py`: text CSV to one row per document with length, lexical, keyword, and optional TF-IDF/SVD features.
- `sequence_proxy_features.py`: event or time-series CSV to one row per entity with counts, time-span, aggregate, trend, and change features.
- `audio_proxy_features.py`: WAV folder to one row per audio file with duration, amplitude, silence, zero-crossing, and simple spectral features.
- `graph_proxy_features.py`: edge-list CSV to one row per node with degree, weighted-degree, neighbor-degree, and graph-membership features.
- `pyproject.toml`: minimal dependencies for running the scripts with `uv`.

## Environment

From this folder, run scripts with `uv` if available:

```bash
uv run python3 image_proxy_features.py --image-dir path/to/images --output outputs/image_proxy.csv
```

If you do not use `uv`, use a Python environment that can import:

- `numpy`
- `pandas`
- `pillow`
- `scikit-learn`

The audio and graph scripts avoid specialized dependencies such as `librosa` or `networkx`.

## Suggested Workflow

1. Pick the script that matches your primary data type.
2. Generate a proxy CSV.
3. Join labels or targets if they are not already included.
4. Split the proxy table using the same evaluation policy from your Assignment 4 charter where possible.
5. Train one simple baseline and one neural tabular model.
6. Explain what the proxy preserves, what it loses, and whether tabular modeling is justified for your portfolio problem.

## Example Commands

Images:

```bash
uv run python3 image_proxy_features.py \
  --image-dir portfolio_images \
  --labels labels.csv \
  --filename-column filename \
  --label-column label \
  --output outputs/image_proxy.csv
```

Text:

```bash
uv run python3 text_proxy_features.py \
  --input documents.csv \
  --text-column text \
  --id-column document_id \
  --label-column label \
  --tfidf-components 20 \
  --output outputs/text_proxy.csv
```

Sequences or time series:

```bash
uv run python3 sequence_proxy_features.py \
  --input events.csv \
  --entity-column user_id \
  --time-column event_time \
  --label-column outcome \
  --output outputs/sequence_proxy.csv
```

Audio:

```bash
uv run python3 audio_proxy_features.py \
  --audio-dir wav_files \
  --labels labels.csv \
  --filename-column filename \
  --label-column label \
  --output outputs/audio_proxy.csv
```

Graphs:

```bash
uv run python3 graph_proxy_features.py \
  --edges edges.csv \
  --source-column source \
  --target-column target \
  --weight-column weight \
  --labels node_labels.csv \
  --node-column node \
  --label-column label \
  --output outputs/graph_proxy.csv
```

## Submission Reminder

In your Assignment 6 writeup, state which proxy script you used, what columns you kept, and what the proxy cannot represent. A decent tabular result does not prove the proxy is final; a weak tabular result does not prove the full portfolio project is impossible.
