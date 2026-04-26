#!/usr/bin/env python3
"""Generate ablation equity curve overlay figure with market benchmarks."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)

# ── Run definitions ───────────────────────────────────────────────────────────
# RUNS_DIR + DATA_DIR point to outputs of the separate ApexQuant platform
# repo, which are not shipped in this research release. Override via the
# env vars below, or drop run CSVs into the default relative paths.
RUNS_DIR = Path(os.environ.get("APEXQUANT_RUNS_DIR", "runs"))
DATA_DIR = Path(os.environ.get("APEXQUANT_DATA_DIR", "data/features"))

RUNS = [
    # (short_label, legend_label,              directory,                              color,     lw,  ls,    alpha, zorder)
    ("EMA+RSI",     "Technical (EMA+RSI)",     "20260313_150154_ema_rsi_baseline",     "#AAAAAA", 1.0, "--",  0.6,   2),
    ("+ Sizing",    "+ Position Sizing",       "20260313_190125_ema_rsi_with_sizing",  "#999999", 1.0, "--",  0.6,   2),
    ("TP only",     "TP Detection (no VG)",    "20260313_184505_ai_layer2_only",       "#888888", 1.0, "--",  0.6,   2),
    ("TP+Dir",      "TP + Direction (no VG)",  "20260313_150154_ai_no_vol_gate",       "#777777", 1.0, "--",  0.6,   2),
    ("Vol Gate",    "Vol Gate + TP + Dir",     "20260313_150154_ai_vol_gate_only",     "#E74C3C", 1.4, "-",   0.85,  4),
    ("Trend",       "+ Trend Bypass",          "20260313_150154_ai_baseline",          "#E67E22", 1.4, "-",   0.85,  4),
    ("Sig Rev",     "+ Sig Rev + Vol Collapse","20260313_150154_ai_vol_collapse",      "#F1C40F", 1.4, "-",   0.85,  4),
    ("Tranche",     "+ All Exits (Tranche)",   "20260313_150154_ai_full_tranche",      "#2980B9", 2.0, "-",   1.0,   6),
    ("Trail Stop",  "+ All Exits (Trail Stop)","20260313_151451_ai_full_trail",        "#1A1A1A", 2.5, "-",   1.0,   7),
]

# ── Load helpers ──────────────────────────────────────────────────────────────
def load_equity(dirname):
    path = RUNS_DIR / dirname / "equity.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    daily = df["portfolio_value"].resample("1D").last().dropna()
    return daily / daily.iloc[0] * 100

def load_benchmark(ticker):
    path = DATA_DIR / f"{ticker}_1hour.csv"
    df = pd.read_csv(path, parse_dates=["ts_event"])
    df = df.set_index("ts_event").sort_index()
    df.index = df.index.tz_localize(None)
    daily = df["close"].resample("1D").last().dropna()
    return daily / daily.iloc[0] * 100

# ── Load all data ─────────────────────────────────────────────────────────────
curves = []
for short, legend, dirname, *_ in RUNS:
    c = load_equity(dirname)
    curves.append(c)
    print(f"  {short:12s}  final={c.iloc[-1]:.1f}")

spy = load_benchmark("SPY")
qqq = load_benchmark("QQQ")
print(f"  {'SPY':12s}  final={spy.iloc[-1]:.1f}  ({spy.iloc[-1]-100:+.1f}%)")
print(f"  {'QQQ':12s}  final={qqq.iloc[-1]:.1f}  ({qqq.iloc[-1]-100:+.1f}%)")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcdefaults()
plt.rc("font", family="DejaVu Sans", size=9)
plt.rc("axes", labelsize=11, titlesize=13)
plt.rc("xtick", labelsize=9)
plt.rc("ytick", labelsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.5)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# ── 100 baseline ──
ax.axhline(100, color="#CCCCCC", linewidth=0.7, linestyle=":", zorder=0)

# ── Market benchmarks (subtle, behind strategies) ──
ax.plot(spy.index, spy.values,
        color="#27AE60", lw=1.1, ls="-", alpha=0.55, zorder=1)
ax.plot(qqq.index, qqq.values,
        color="#8E44AD", lw=1.1, ls="-", alpha=0.55, zorder=1)

# ── Strategy curves ──
for i, (curve, run_def) in enumerate(zip(curves, RUNS)):
    short, legend, dirname, col, lw, ls, alpha, zo = run_def
    ax.plot(curve.index, curve.values,
            color=col, linewidth=lw, linestyle=ls, alpha=alpha,
            zorder=zo)

# ── Right-edge annotations ────────────────────────────────────────────────────
# Collect all end-points: (final_value, end_date, label_text, color, bold)
annots = []
for curve, run_def in zip(curves, RUNS):
    short, legend, dirname, col, lw, *_ = run_def
    annots.append((curve.iloc[-1], curve.index[-1], short, col, lw >= 2.0))

# Benchmarks
spy_ret = spy.iloc[-1] - 100
qqq_ret = qqq.iloc[-1] - 100
annots.append((spy.iloc[-1], spy.index[-1], f"SPY ({spy_ret:+.0f}%)", "#27AE60", False))
annots.append((qqq.iloc[-1], qqq.index[-1], f"QQQ ({qqq_ret:+.0f}%)", "#8E44AD", False))

# Sort by value, then space labels to avoid overlap
annots.sort(key=lambda x: x[0])
min_gap = 4.0
placed = []
for val, dt, txt, col, bold in annots:
    y = val
    for py in placed:
        if abs(y - py) < min_gap:
            y = py + min_gap
    placed.append(y)

    dy = y - val
    ax.annotate(
        f"{txt}  {val:.0f}",
        xy=(dt, val),
        xytext=(14, dy),
        textcoords="offset points",
        fontsize=7, fontweight="bold" if bold else "normal",
        color=col, va="center",
        arrowprops=dict(arrowstyle="-", color=col, lw=0.3, alpha=0.4)
            if abs(dy) > 4 else None,
    )

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.set_xlabel("Date")
ax.set_ylabel("Normalised portfolio value (start = 100)")
ax.set_title(
    "Ablation Study: Equity Curves with Market Benchmarks (2020\u20132022)",
    fontsize=13, fontweight="bold", pad=14,
)

# ── Compact legend (3 groups) ─────────────────────────────────────────────────
legend_elements = [
    # Group 1: Losing strategies (single representative entry)
    Line2D([0], [0], color="#999999", lw=1.0, ls="--", alpha=0.7,
           label="Runs 1\u20134: No Vol Gate (losing)"),
    # Group 2: Intermediate
    Line2D([0], [0], color="#E74C3C", lw=1.4, label="Run 5: + Vol Gate"),
    Line2D([0], [0], color="#E67E22", lw=1.4, label="Run 6: + Trend Bypass"),
    Line2D([0], [0], color="#F1C40F", lw=1.4, label="Run 7: + Sig Rev / Vol Collapse"),
    # Group 3: Full system
    Line2D([0], [0], color="#2980B9", lw=2.0, label="Run 8: + Tranche Exit"),
    Line2D([0], [0], color="#1A1A1A", lw=2.5, label="Run 9: + Trail Stop (final)"),
    # Benchmarks
    Line2D([0], [0], color="#27AE60", lw=1.1, alpha=0.55, label="SPY Buy-and-Hold"),
    Line2D([0], [0], color="#8E44AD", lw=1.1, alpha=0.55, label="QQQ Buy-and-Hold"),
]
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.01, 0.98),
          fontsize=7.5, framealpha=0.92, edgecolor="#CCCCCC",
          handlelength=2.2, labelspacing=0.4)

# ── Despine ──
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.autofmt_xdate(rotation=30)
fig.tight_layout()

# ── Save ──
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_ablation_curves.{ext}", dpi=300,
                bbox_inches="tight", facecolor="white")
plt.close(fig)

p = OUT / "fig_ablation_curves.png"
print(f"\nSaved: {p}  ({p.stat().st_size / 1024:.0f} KB)")
p2 = OUT / "fig_ablation_curves.pdf"
print(f"Saved: {p2}  ({p2.stat().st_size / 1024:.0f} KB)")
