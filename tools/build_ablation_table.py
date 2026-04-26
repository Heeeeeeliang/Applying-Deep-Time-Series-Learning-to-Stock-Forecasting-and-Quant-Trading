#!/usr/bin/env python3
"""
Build results/ablation_table.csv from the 11 ablation run dirs.

Reads metrics.json from each run dir under RUNS_ROOT, writes a single CSV
with one row per run in cumulative order. Run 11 (ai_full_trail) is the
final/canonical configuration referenced in the README and thesis.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

CONFIG = {
    # Local source of the 2026-03-16 ablation run set (outside this repo —
    # produced by the ApexQuant platform, not shipped with this research
    # release). Override with APEXQUANT_RUNS_DIR or edit the default.
    "runs_root": Path(os.environ.get("APEXQUANT_RUNS_DIR", "runs")),
    "prefix":    "20260316_140416_",
    "out_csv":   Path(__file__).resolve().parents[1] / "results" / "ablation_table.csv",
}

# (run#, suffix, label) — cumulative order matches fig_ablation.py
RUNS = [
    ( 1, "ema_rsi_baseline",    "EMA + RSI baseline"),
    ( 2, "ema_rsi_with_sizing", "+ Position sizing"),
    ( 3, "ai_layer2_only",      "TP detection only (no Vol Gate)"),
    ( 4, "ai_no_vol_gate",      "TP + Direction (no Vol Gate)"),
    ( 5, "ai_vol_gate_only",    "+ Vol Gate"),
    ( 6, "ai_baseline",         "+ Trend bypass"),
    ( 7, "ai_signal_reversal",  "+ Signal reversal"),
    ( 8, "ai_vol_collapse",     "+ Vol collapse"),
    ( 9, "ai_tranche_exit",     "Tranche exit (standalone)"),
    (10, "ai_full_tranche",     "+ Full + Tranche"),
    (11, "ai_full_trail",       "+ Full + Trail-Stop (final)"),
]

COLUMNS = [
    "run", "label", "config_dir",
    "total_return", "annualized_return",
    "sharpe_ratio", "sortino_ratio", "calmar_ratio",
    "max_drawdown", "max_drawdown_duration_days",
    "total_trades", "win_rate", "profit_factor",
    "avg_trade_pnl", "avg_win", "avg_loss",
    "avg_bars_held", "long_trades", "short_trades",
    "is_final",
]


def main() -> int:
    rows = []
    for run, suffix, label in RUNS:
        cfg_dir = f"{CONFIG['prefix']}{suffix}"
        path = CONFIG["runs_root"] / cfg_dir / "metrics.json"
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        m = json.loads(path.read_text())["metrics"]
        rows.append({
            "run": run,
            "label": label,
            "config_dir": cfg_dir,
            **{k: m[k] for k in COLUMNS if k in m},
            "is_final": int(run == 11),
        })

    CONFIG["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    with CONFIG["out_csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {CONFIG['out_csv']}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
