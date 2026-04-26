#!/usr/bin/env bash
# Pull the dataset and checkpoints from HuggingFace + the GitHub Release into
# ./data/ and ./checkpoints/, leaving the layout that every notebook in the
# repo expects. Idempotent — re-running only re-downloads missing files.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
HF_DATASET="<USER>/quant-thesis-nasdaq-2020-2022"
HF_MODELS="<USER>/quant-thesis-checkpoints"
GH_RELEASE_TAG="v1.0.0"
GH_REPO="<USER>/<REPO>"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "Installing huggingface-hub CLI..."
  pip install --quiet "huggingface-hub>=0.24"
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI 'gh' is required for the Release artifact step." >&2
  echo "Install: https://cli.github.com/  (or download the tarball manually)" >&2
  exit 1
fi

echo "==> Downloading dataset from HuggingFace ($HF_DATASET) into data/"
huggingface-cli download "$HF_DATASET" \
  --repo-type dataset \
  --local-dir "$REPO_ROOT/data" \
  --local-dir-use-symlinks False

echo "==> Downloading large model checkpoints from HuggingFace ($HF_MODELS)"
huggingface-cli download "$HF_MODELS" \
  --local-dir "$REPO_ROOT/checkpoints" \
  --local-dir-use-symlinks False

echo "==> Downloading per-ticker LSTM checkpoints from GitHub Release $GH_RELEASE_TAG"
mkdir -p "$REPO_ROOT/checkpoints"
cd "$REPO_ROOT/checkpoints"
gh release download "$GH_RELEASE_TAG" --repo "$GH_REPO" --pattern 'checkpoints-*.tar.gz'
for f in checkpoints-*.tar.gz; do
  echo "  extracting $f"
  tar -xzf "$f"
  rm -f "$f"
done

echo "==> Verifying checkpoints (NaN / Inf / weights_only sanity)"
cd "$REPO_ROOT"
python tools/verify_checkpoints.py

echo
echo "Done."
echo "  data/        : $(du -sh data 2>/dev/null | cut -f1)"
echo "  checkpoints/ : $(du -sh checkpoints 2>/dev/null | cut -f1)"
echo
echo "Next: jupyter lab notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb"
