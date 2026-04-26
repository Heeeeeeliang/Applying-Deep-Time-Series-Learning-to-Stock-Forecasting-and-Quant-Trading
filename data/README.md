# Dataset card

## Source
- **Origin:** Databento historical US equities, NASDAQ ITCH (`xnas-itch`) feed
- **Product:** OHLCV bars at 1-minute, 15-minute, 1-hour, and daily resolutions
- **Date range:** 2020-01-07 → 2022-09-30 (inclusive; ~692 trading days)

## Data Source and License

**Source:** Databento (https://databento.com), XNAS.ITCH dataset
(Nasdaq TotalView-ITCH), accessed under a standard historical data
subscription.

**What is redistributed here:** Bar-aggregated OHLCV data and engineered
features derived from Nasdaq historical data (2020-01-07 through
2022-09-30). No raw ITCH tick messages are included.

**License basis:** Per Databento's public policy, historical (T+1)
market data does not require additional licensing for redistribution
or use. Nasdaq historical data is among the venues for which
redistribution is permitted after the 24-hour real-time window.
See: https://databento.com/blog/introduction-market-data-licensing

**Derived features:** Engineered features (technical indicators,
rolling statistics, etc.) computed from the OHLCV bars are
redistributed under the MIT License of this repository.

**Attribution requested:** If you use this dataset in academic or
commercial work, please cite both this repository and Databento as
the data source.

**Disclaimer:** This dataset is provided for research and educational
purposes. No warranty of accuracy or completeness. Users are
responsible for verifying suitability for their own use cases and
for compliance with any terms of service that may apply to their
Databento subscription if they re-download data.

## Tickers (8)
AAPL, MSFT, NVDA, QQQ, SPY, TSLA, GOOG, GOOGL

> **Note on GOOG vs GOOGL.** Both tickers were initially ingested. GOOG was
> removed from the final Layer-2/3 training set because its post-split series
> overlaps with GOOGL and caused duplicated turning-point events that inflated
> win-rate by a small but systematic amount. GOOG remains in `processed/` and
> `features/` for completeness; the Layer-2/3 pipelines exclude it.

## Schema

### `processed/<TICKER>_<FREQ>.csv`
Cleaned OHLCV bars. Columns: `timestamp, open, high, low, close, volume`.
`timestamp` is UTC ISO-8601. NYSE session hours only (09:30–16:00 ET); pre-/
post-market bars dropped.

### `features/<TICKER>_<FREQ>_features.csv`
46 columns total: `ts_event`, the 5 OHLCV bars, and 40 engineered features.
Feature families (counts derived from the actual column list):
- **Returns (3):** `returns`, `log_returns`, `price_change`
- **Range (3):** `price_range`, `price_range_pct`, `high_low_ratio`
- **Trend / moving averages (10):** `ma_{5,10,20,60}` and the corresponding
  `ma_ratio_{5,10,20,60}`, plus `ema_12`, `ema_26`
- **MACD (2):** `macd`, `macd_signal`
- **Volatility (2):** `volatility_20`, `volatility_60` (rolling std of returns)
- **Momentum (7):** `rsi`, `momentum_{5,10,20}`, `roc_{5,10,20}`
- **Bollinger (5):** `bb_middle`, `bb_std`, `bb_upper`, `bb_lower`, `bb_width`
- **Volume (3):** `volume_ma_20`, `relative_volume`, `volume_change`
- **Position-in-bar (2):** `close_to_high`, `close_to_low`
- **Calendar (3):** `hour`, `day_of_week`, `month`

Full list with formulas: `data/feature_dictionary.md`.

### `splits/<TICKER>_<FREQ>/{train,val,test}.csv`
Index-only CSVs (just `timestamp` column) defining the split membership.
Chronological split, **70% / 10% / 20%** by trading days, computed per ticker.
Use these to filter rows from the corresponding `processed/` and `features/`
files at load time.

- `splits/split_summary.json` — the exact date boundaries and row counts.
- `splits/USAGE_EXAMPLE.py` — minimal load-and-split example.

### `tp_data/` (Layer-2 artifacts, optional input)
- `<TICKER>_turning_points.csv` — zigzag-labelled tops/bottoms with the zigzag
  threshold used (see `tp_data/zigzag_config.csv`)
- `<TICKER>_{top,bottom}_{train,val,test}.npz` — pre-sliced CNN input windows

### `vol_data/` (Layer-1 artifacts, optional input)
- `<TICKER>_vol_{train,val,test}.csv` — realised-vol targets with feature joins
- `vol_data/numpy/<TICKER>_vol_{train,val,test}.npz` — numpy versions

## Hosting

Large files are published as a single tarball
(`apexquant_data_v1.tar.gz`, ~1.4 GB) attached to GitHub Release v1.0.0.

What lives where:
- **In this git repo:** `splits/` (small CSVs), `README.md`, `feature_dictionary.md`, `splits/split_summary.json`, `splits/USAGE_EXAMPLE.py`
- **In Release v1.0.0:** everything else (`processed/`, `features/`, `tp_data/`, `vol_data/`, raw Databento bar CSVs)

Download and verify, from the repository root:

```bash
gh release download v1.0.0 --pattern "apexquant_data_v1.tar.gz"
echo "fb94e857c0918f4308956eea0f0de646050c07f90ee21a6d6baec5f5905daaa4  apexquant_data_v1.tar.gz" | sha256sum -c -
tar -xzf apexquant_data_v1.tar.gz --strip-components=1 -C data/
```

`--strip-components=1` drops the tarball's top-level
`apexquant_data_v1/` directory; `-C data/` lands the contents under
the existing `data/` directory in the cloned repo. After extraction,
`data/processed/`, `data/features/`, `data/tp_data/`, and
`data/vol_data/` are populated alongside the existing `data/splits/`.

## Provenance notes

1. `GOOG_1min_features.csv` was regenerated after a column-ordering fix on
   2026-02-14; earlier versions have 51 columns, final version has 53.
2. Turning-point labels use a volatility-scaled zigzag threshold of
   `1.5 × σ_20`. See `tp_data/zigzag_config.csv` for per-ticker values.
3. Volatility targets are realised vol over the next 20 bars (`sqrt(sum(r²))`),
   clipped at the 99.5th percentile within each ticker's training window to
   control the tail.
