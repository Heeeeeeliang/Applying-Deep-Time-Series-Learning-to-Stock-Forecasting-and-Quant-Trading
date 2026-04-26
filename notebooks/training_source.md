# Training Source Notebook

This is the original notebook used to train the shipped model weights. It is preserved as-is for reproducibility. Paths and configuration reflect the original training environment (Google Colab + Google Drive).

Weights produced by this notebook:

- `multiscale_cnn_bottom.pt`  → Layer 2 CNN (downtrend detector)
- `multiscale_cnn_top.pt`     → Layer 2 CNN (uptrend detector)
- `lgb_bottom_v1/`            → Layer 2 meta-label classifier
- `lgb_top_v1/`               → Layer 2 meta-label classifier

Layer 1 volatility LightGBM weights (`lightgbm_v3`, `lightgbm_v3_flat`) were produced by a separate training script not included in this release.
