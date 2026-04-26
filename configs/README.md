# Configs

`ablation/` contains one YAML per ablation run, extracted verbatim from the
embedded config block of each run's `metrics.json`. The filenames preserve
the cumulative run order from the thesis (run 01 → run 11).

**Run 11** (`run11_ai_full_trail.yaml`) is the final / canonical
configuration that produces the headline numbers in the README. It is the
only run marked `is_final: true`.

| Run | Config | What changed from previous run |
|---:|---|---|
| 1 | `run01_ema_rsi_baseline.yaml` | technical baseline only |
| 2 | `run02_ema_rsi_with_sizing.yaml` | + position sizing |
| 3 | `run03_ai_layer2_only.yaml` | switch to AI Layer 2 (TP detection), no Vol Gate |
| 4 | `run04_ai_no_vol_gate.yaml` | + Layer-2 direction signal, still no Vol Gate |
| 5 | `run05_ai_vol_gate_only.yaml` | **+ Vol Gate** (this is the +63 pp jump) |
| 6 | `run06_ai_baseline.yaml` | + Trend bypass |
| 7 | `run07_ai_signal_reversal.yaml` | + Signal-reversal exit |
| 8 | `run08_ai_vol_collapse.yaml` | + Vol-collapse exit |
| 9 | `run09_ai_tranche_exit.yaml` | tranche exit alone (sensitivity check) |
| 10 | `run10_ai_full_tranche.yaml` | full system with tranche exit |
| **11** | **`run11_ai_full_trail.yaml`** | **full system with trail-stop exit (final)** |

To reproduce a single run end-to-end, point the ApexQuant backtester at the
relevant YAML and the data + checkpoints layout described in
`../README.md` and `../data/README.md`.
