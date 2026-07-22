#!/usr/bin/env bash
# Build lightweight submission zip: writeup + code + local run evidence.
# Excludes model weights and feature caches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="${ROOT}/dist/staging"
ZIP="${ROOT}/dist/assignment_6_submission_v1.zip"
EVID="${STAGING}/part_a/run_evidence"

rm -rf "${STAGING}"
mkdir -p \
  "${STAGING}/writeup" \
  "${STAGING}/part_a/code" \
  "${EVID}/experiments" \
  "${EVID}/tree_runs" \
  "${EVID}/chroma_runs/plots" \
  "${EVID}/reused_a5/baseline_metrics" \
  "${EVID}/reused_a5/chroma_runs" \
  "${EVID}/chroma_rich/mlp_runs" \
  "${EVID}/chroma_rich/plots" \
  "${EVID}/plots" \
  "${ROOT}/dist"

# --- Writeup ---
cp "${ROOT}/ASSIGNMENT_6_WRITEUP.md" "${STAGING}/writeup/ASSIGNMENT_6_WRITEUP.md"
cp "${ROOT}/SUBMISSION.md" "${STAGING}/README_SUBMISSION.txt"

# --- Runnable code (no outputs, no weights) ---
rsync -a \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.git/' \
  --exclude 'dist/' \
  --exclude 'outputs/' \
  --exclude 'ASSIGNMENT_6_WRITEUP.md' \
  --exclude 'SUBMISSION.md' \
  --exclude 'requirements.md' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'part060-000-deep-learning-tabular-data.pdf' \
  "${ROOT}/" "${STAGING}/part_a/code/"

# --- Hypothesis-case experiment reports ---
cp "${ROOT}/experiments/"*.md "${EVID}/experiments/"

# --- Tree runs ---
rsync -a \
  --include '*/' \
  --include '*.json' \
  --include '*.csv' \
  --exclude '*' \
  "${ROOT}/outputs/tree_runs/" "${EVID}/tree_runs/"

# --- Scaled MLP runs + plots ---
for suffix in history.csv metadata.json; do
  src="${ROOT}/outputs/chroma_runs/exp6_chroma_mlp_scaled_${suffix}"
  [[ -f "${src}" ]] && cp "${src}" "${EVID}/chroma_runs/"
done
[[ -f "${ROOT}/outputs/chroma_runs/exp6_chroma_mlp_scaled.log" ]] && \
  cp "${ROOT}/outputs/chroma_runs/exp6_chroma_mlp_scaled.log" "${EVID}/chroma_runs/"
if [[ -d "${ROOT}/outputs/chroma_runs/plots" ]]; then
  rsync -a --include '*.png' --exclude '*' \
    "${ROOT}/outputs/chroma_runs/plots/" "${EVID}/chroma_runs/plots/"
fi

# --- Reused prior baselines ---
rsync -a \
  --include '*/' \
  --include '*.md' \
  --include '*.csv' \
  --include '*.json' \
  --include '*.log' \
  --exclude '*' \
  "${ROOT}/outputs/reused_a5/" "${EVID}/reused_a5/"

# --- Comparison table + plots ---
[[ -f "${ROOT}/outputs/model_comparison.csv" ]] && \
  cp "${ROOT}/outputs/model_comparison.csv" "${EVID}/"
if [[ -d "${ROOT}/outputs/plots" ]]; then
  rsync -a --include '*.png' --exclude '*' \
    "${ROOT}/outputs/plots/" "${EVID}/plots/"
fi

# --- Rich chroma appendix (no .pt) ---
rsync -a \
  --include '*/' \
  --include '*.md' \
  --include '*.csv' \
  --include '*.json' \
  --include '*.log' \
  --include '*.png' \
  --exclude '*.pt' \
  --exclude '*' \
  "${ROOT}/outputs/chroma_rich/" "${EVID}/chroma_rich/"

rm -f "${ZIP}"
(
  cd "${STAGING}"
  zip -r "${ZIP}" writeup part_a README_SUBMISSION.txt
)

echo "Wrote ${ZIP}"
unzip -l "${ZIP}" | tail -40
du -h "${ZIP}"

if unzip -l "${ZIP}" | grep -E '\.pt$' >/dev/null; then
  echo "ERROR: zip contains .pt weights" >&2
  exit 1
fi
if unzip -l "${ZIP}" | grep -E 'requirements\.md$' >/dev/null; then
  echo "ERROR: zip contains requirements.md" >&2
  exit 1
fi
echo "OK: no .pt weights; no requirements.md"
