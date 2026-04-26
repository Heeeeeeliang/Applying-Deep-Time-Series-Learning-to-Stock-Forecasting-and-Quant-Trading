# Checkpoints

This directory is empty in the source repo — checkpoints are published as
tarballs attached to GitHub Release v1.0.0 and downloaded manually.
`MANIFEST.csv` in this directory enumerates every checkpoint with its
size, training context, SHA-1 prefix, layer, and (for redistributed
files) `relative_path` — the path inside the extracted tarball, i.e.
where to find the file on disk under `checkpoints/`.

## Hosting plan

| Target | Files | Total size | Why |
|---|---:|---:|---|
| **GitHub Release v1.0.0** (asset `checkpoints-<layer>.tar.gz`, four tarballs) | 75 | ~45 MB | Per-ticker LSTM / CNN / LightGBM weights — small, version-locked to the source release. |

The four tarballs are `checkpoints-ceiling_baseline.tar.gz`,
`checkpoints-layer1_volatility.tar.gz`,
`checkpoints-layer2_turning_point.tar.gz`, and
`checkpoints-unknown.tar.gz`. Each extracts under `checkpoints/<layer>/`.

## Layer mapping (for the manifest's `layer` column)

| Layer code | Description |
|---|---|
| `layer1_volatility` | LightGBM and LSTM volatility forecasters consumed by Layer 2 |
| `layer2_turning_point` | MultiScaleCNN top/bottom heads |
| `layer3_trade_filter` | LightGBM meta-label gates (top, bottom, combined) |
| `ceiling_baseline` | The 9-method ceiling-study models (vanilla / attention / transformer LSTM). Published for replication only — they are baselines, not part of the deployed system. |
| `end_to_end_pipeline` | Multi-stock phase-1 unified model used for the integrated end-to-end notebook. |
| `exploratory` | Early 1-min LSTM runs retained for transparency; not used in the final ablation. |

## Manual download

From the repository root:

```bash
gh release download v1.0.0 --pattern "checkpoints-*.tar.gz"
for tar in checkpoints-*.tar.gz; do tar -xzf "$tar"; done
```

This populates `checkpoints/<layer>/...` for every redistributed weight.
Tarballs land at the repo root after `gh release download` and can be
deleted after extraction.

## Integrity check

After downloading, walk the manifest and load each redistributed
checkpoint to surface a corrupted download or a NaN-poisoned checkpoint
(the failure mode we hit during the original training where a CNN
checkpoint was saved post-divergence with NaN weights):

```python
import torch, joblib, pandas as pd
from pathlib import Path

m = pd.read_csv("checkpoints/MANIFEST.csv")
# Only the GitHub-Release-shipped rows have files on disk after extraction.
m = m[m["publish_target"] == "github_release"]

assert len(m) == m["relative_path"].nunique(), "MANIFEST relative_path collision"

errors = 0
for _, row in m.iterrows():
    path = Path("checkpoints") / row["relative_path"]
    if not path.exists():
        print(f"MISSING {path}")
        errors += 1
        continue
    try:
        if path.suffix == ".joblib":
            joblib.load(path)
        else:
            torch.load(path, weights_only=True)
    except Exception as e:
        print(f"FAIL    {path}: {e}")
        errors += 1
        continue
print(f"\n{len(m) - errors}/{len(m)} checkpoints loaded successfully.")
```

The PyTorch loads use `weights_only=True` so a malformed pickle cannot
execute arbitrary code at load time.

## Not redistributed

Six checkpoints listed in `MANIFEST.csv` with `publish_target =
not_redistributed` are deliberately excluded from Release v1.0.0:

- **TimesFM fine-tunes** — `timesfm_ft_AAPL_1hour_v2.pt`,
  `timesfm_ft_AAPL_1hour_v3_directional.pt`, `timesfm_ft_AAPL_1hour.pt`,
  `timesfm_ft_QQQ_1hour.pt`. Evaluated in the ceiling study (Ch. 4.1) but
  the weights are not redistributed; the fine-tuning recipe is
  reproducible from the official
  [google-research/timesfm](https://github.com/google-research/timesfm)
  repo combined with the data and split files in this repo.
- **`multistock_phase1/best_model.pt` (1.7 GB)** and
  **`timesfm_finetuned/best_model.pt` (1.9 GB)** — intermediate
  exploratory checkpoints; not required to reproduce the headline
  backtest (Run 11) or the ceiling-study tables.
