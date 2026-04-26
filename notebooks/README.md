# Notebooks

Each subfolder corresponds to one chapter / figure group of the thesis. Every
notebook begins with a CONFIG dict pointing to `../../data/` relative paths so
that the full pipeline runs top-to-bottom on a fresh clone.

| Folder / file | Thesis section | What it produces |
|---|---|---|
| `01_direct_prediction_ceiling/` | Ch. 4.1 — Direct Prediction Ceiling Study | The 9-method ~50% DA ceiling evidence (figures + standardised results table) |
| `training_source.ipynb` | Ch. 4.3 — Turning-Point Layer | Original Colab notebook that trained the shipped Layer-2 weights (`multiscale_cnn_{top,bottom}.pt`, `lgb_{top,bottom}_v1/`). See `training_source.md` for the per-weight mapping. |

## Runtime expectations

| Notebook tier | CPU-only OK? | GPU suggested | Approx runtime |
|---|---|---|---|
| Ceiling study (01) | yes for baselines; no for TimesFM-ft | T4 or A100 for TimesFM-ft | 30 min (baselines) / 3–5 h (all incl. TimesFM-ft) |
| Training source (Layer-2) | no | T4 minimum | 45–90 min |

## CONFIG convention

Every notebook's first code cell is:

```python
CONFIG = {
    "data_dir": "../../data",          # processed + features + splits
    "checkpoints_dir": "../../checkpoints",
    "seed": 42,
    "device": "cuda",                  # or "cpu"
    # Colab fallback (commented by default):
    # "data_dir": "/content/drive/MyDrive/thesis_data",
}
```

The commented Colab fallback is kept as a one-line reference — uncomment if
running on Colab with the dataset mounted from Drive.
