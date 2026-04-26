# Notebook roles in `01_direct_prediction_ceiling/`

| Notebook | Role | When to run |
|---|---|---|
| `03_baseline_models.ipynb` | **Headline table producer.** Trains all 9 methods (Naive / Hist Mean / LinReg / LightGBM / LSTM / Attn-LSTM / Transformer-LSTM / TimesFM ZS / TimesFM FT) and renders the 9-row ceiling table from the repo README. | **Run this to reproduce the headline table.** |
| `01_nine_method_ceiling.ipynb` | Sequence-model grid sweep and per-ticker breakdown. | Optional — supports `03` with finer-grained results. |
| `02_missing_metrics.ipynb` | Cleanup pass that fills in metrics `03` didn't compute on its first run. | Optional — supplementary only. Has a known schema bug (`load_splits` reads index-only splits CSVs then accesses OHLCV columns); not gating the ceiling reproduction. |

## To reproduce the README's headline table

1. Follow the data extraction recipe in [`data/README.md`](../../data/README.md).
2. Activate the env (conda or pip+venv per top-level [`README.md`](../../README.md)).
3. Run

   ```bash
   jupyter nbconvert --to notebook --execute \
     notebooks/01_direct_prediction_ceiling/03_baseline_models.ipynb \
     --output executed_03.ipynb \
     --ExecutePreprocessor.timeout=3600
   ```

4. The resulting `results/*_predictions.csv` files plus the table cell's
   stdout reproduce **7 of 9 rows** of the README's headline table on the
   default pip+venv environment. The 2 TimesFM rows
   (`TimesFM zero-shot`, `TimesFM fine-tuned`) require the optional
   Python 3.11 environment described in the markdown cell directly
   above the TimesFM cell (cell 13) inside the notebook.
