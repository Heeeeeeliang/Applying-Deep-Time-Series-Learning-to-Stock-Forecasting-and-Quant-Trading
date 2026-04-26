# Feature dictionary

The 40 engineered features in `data/features/<TICKER>_<FREQ>_features.csv`,
in column order. All features are computed bar-by-bar with no forward-looking
information; the audit script that verifies this is
`notebooks/01_direct_prediction_ceiling/05_feature_lookahead_audit.ipynb`.

`r_t = log(close_t / close_{t-1})`. `c_t, h_t, l_t, o_t, v_t` are
close/high/low/open/volume of the current bar.

## Returns (3)

| Column | Formula |
|---|---|
| `returns` | `(c_t / c_{t-1}) - 1` |
| `log_returns` | `log(c_t / c_{t-1})` |
| `price_change` | `c_t - c_{t-1}` |

## Range (3)

| Column | Formula |
|---|---|
| `price_range` | `h_t - l_t` |
| `price_range_pct` | `(h_t - l_t) / c_{t-1}` |
| `high_low_ratio` | `h_t / l_t` |

## Moving averages and MA ratios (10)

| Column | Formula |
|---|---|
| `ma_{5,10,20,60}` | rolling mean of `close` over N bars |
| `ma_ratio_{5,10,20,60}` | `c_t / ma_N` |
| `ema_12`, `ema_26` | exponential moving average, span = N |

## MACD (2)

| Column | Formula |
|---|---|
| `macd` | `ema_12 - ema_26` |
| `macd_signal` | EMA of `macd` with span = 9 |

## Volatility (2)

| Column | Formula |
|---|---|
| `volatility_20` | rolling std of `returns` over 20 bars |
| `volatility_60` | rolling std of `returns` over 60 bars |

## Momentum (7)

| Column | Formula |
|---|---|
| `rsi` | Wilder's RSI over 14 bars |
| `momentum_{5,10,20}` | `c_t - c_{t-N}` |
| `roc_{5,10,20}` | `(c_t / c_{t-N}) - 1` (rate of change) |

## Bollinger Bands (5, period 20, k = 2)

| Column | Formula |
|---|---|
| `bb_middle` | rolling mean of `close` over 20 bars |
| `bb_std` | rolling std of `close` over 20 bars |
| `bb_upper` | `bb_middle + 2 * bb_std` |
| `bb_lower` | `bb_middle - 2 * bb_std` |
| `bb_width` | `(bb_upper - bb_lower) / bb_middle` |

## Volume (3)

| Column | Formula |
|---|---|
| `volume_ma_20` | rolling mean of `volume` over 20 bars |
| `relative_volume` | `v_t / volume_ma_20` |
| `volume_change` | `(v_t / v_{t-1}) - 1` |

## Position-in-bar (2)

| Column | Formula |
|---|---|
| `close_to_high` | `(h_t - c_t) / (h_t - l_t)` (0 = at high, 1 = at low) |
| `close_to_low` | `(c_t - l_t) / (h_t - l_t)` (0 = at low, 1 = at high) |

## Calendar (3)

| Column | Formula |
|---|---|
| `hour` | hour-of-day in NY local time, integer 0–23 |
| `day_of_week` | Monday = 0, Sunday = 6 |
| `month` | 1–12 |

## Notes on standardisation

The CSVs ship the raw values. Models that need standardised inputs (the LSTM
family, the MultiScaleCNN) compute mean/std on the **training split only**
and apply the same transform to val/test — see `data/splits/USAGE_EXAMPLE.py`
for the canonical loader. LightGBM consumes the raw features unchanged.
