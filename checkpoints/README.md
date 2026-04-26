# Checkpoints

This directory is empty in the source repo — checkpoints are published
separately (see hosting plan below) and downloaded by `quickstart.sh`.

## Hosting plan

Source: `MANIFEST.csv` in this directory enumerates every checkpoint with
its size, training context, SHA-1 prefix, layer, and publication target.

| Target | Files | Total size | Why |
|---|---:|---:|---|
| **HuggingFace** (`<USER>/quant-thesis-checkpoints`) | 6 | ~7.1 GB | TimesFM fine-tunes (4 × 925 MB) + multistock_phase1 (1.8 GB) + timesfm best_model (2.1 GB). Single-file size > GitHub's per-file limit, and total is too large for a Release artifact. |
| **GitHub Release** (tag `v1.0.0`, asset `checkpoints-<layer>.tar.gz`) | 75 | ~51 MB | Per-ticker LSTM / CNN / LightGBM weights. Small, version-locked to source releases. |

## Layer mapping (for the manifest's `layer` column)

| Layer code | Description |
|---|---|
| `layer1_volatility` | LightGBM and LSTM volatility forecasters consumed by Layer 2 |
| `layer2_turning_point` | MultiScaleCNN top/bottom heads |
| `layer3_trade_filter` | LightGBM meta-label gates (top, bottom, combined) |
| `ceiling_baseline` | The 9-method ceiling-study models (vanilla / attention / transformer LSTM, TimesFM zero-shot and fine-tuned). Published for replication only — they are baselines, not part of the deployed system. |
| `end_to_end_pipeline` | Multi-stock phase-1 unified model used for the integrated end-to-end notebook. |
| `exploratory` | Early 1-min LSTM runs retained for transparency; not used in the final ablation. |

## Verifying after download

After `quickstart.sh` finishes:

```bash
python tools/verify_checkpoints.py
```

The script reads `MANIFEST.csv`, loads each file with
`torch.load(path, weights_only=True)` (or `joblib.load` for `.joblib`),
walks every tensor, and exits non-zero if any contains NaN or Inf. This
catches the failure mode we hit during the original training where a CNN
checkpoint was saved post-divergence with NaN weights. Re-run the check
locally before publishing any new checkpoint to the manifest.
