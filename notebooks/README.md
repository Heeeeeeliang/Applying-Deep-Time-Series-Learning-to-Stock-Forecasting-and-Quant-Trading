# Notebooks

Each subfolder corresponds to one chapter / figure group of the thesis. Every
notebook begins with a CONFIG dict pointing to `../../data/` relative paths so
that the full pipeline runs top-to-bottom on a fresh clone.

| Folder | Thesis section | What it produces |
|---|---|---|
| `01_direct_prediction_ceiling/` | Ch. 4.1 — Direct Prediction Ceiling Study | `results/figures/fig4_1{a,b,c,d}_*.{pdf,png}`, `results/ch2_standardised_results.json` — the 9-method ~50% DA ceiling evidence |
| `02_volatility_layer/` | Ch. 4.2 — Volatility Layer | `results/vol_prediction_v3_results.json`, LightGBM vol predictions, vol_lstm per-ticker checkpoints |
| `03_turning_point_layer/` | Ch. 4.3 — Turning-Point Layer | `results/meta_label_v2_results.json`, `cnn_{top,bottom}_predictions.csv`, `multiscale_cnn_{top,bottom}.pt` |
| `04_backtest_and_ablation/` | Ch. 4.4 — Backtest & Ablation | `results/ablation_table.csv`, `results/backtest_metrics.json`, `results/figures/fig_ablation*.{pdf,png}`, `fig_equity_*.{pdf,png}` |

## Runtime expectations

| Notebook tier | CPU-only OK? | GPU suggested | Approx runtime |
|---|---|---|---|
| Ceiling study (01) | yes for baselines; no for TimesFM-ft | T4 or A100 for TimesFM-ft | 30 min (baselines) / 3–5 h (all incl. TimesFM-ft) |
| Volatility (02) | yes | optional | 10–20 min |
| Turning point (03) | no | T4 minimum | 45–90 min |
| Backtest & ablation (04) | yes | n/a | 15 min per run × 11 runs |

## CONFIG convention

Every notebook's first code cell is:

```python
CONFIG = {
    "data_dir": "../../data",          # processed + features + splits
    "results_dir": "../../results",    # outputs go here
    "checkpoints_dir": "../../checkpoints",
    "seed": 42,
    "device": "cuda",                  # or "cpu"
    # Colab fallback (commented by default):
    # "data_dir": "/content/drive/MyDrive/thesis_data",
}
```

The commented Colab fallback is kept as a one-line reference — uncomment if
running on Colab with the dataset mounted from Drive.
