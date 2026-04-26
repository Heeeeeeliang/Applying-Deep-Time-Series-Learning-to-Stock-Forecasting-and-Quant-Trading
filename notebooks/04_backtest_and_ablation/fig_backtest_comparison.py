#!/usr/bin/env python3
"""
Generate 2 publication-quality backtest comparison figures for thesis.
  Figure 1: Equity curve comparison (AI vs EMA+RSI baseline)
  Figure 2: Ablation study bar chart (incremental component contribution)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("thesis_figures")
OUT.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD EQUITY CURVES
# ══════════════════════════════════════════════════════════════════════════════
# RUNS_DIR + BM_DIR (below) point to outputs of the separate ApexQuant
# platform repo, which are not shipped in this research release. Override
# via APEXQUANT_RUNS_DIR / APEXQUANT_DATA_DIR env vars.
RUNS_DIR = Path(os.environ.get("APEXQUANT_RUNS_DIR", "runs"))
AI_CSV   = RUNS_DIR / "20260313_151451_ai_full_trail"  / "equity.csv"
BASE_CSV = RUNS_DIR / "20260313_141728_ema_rsi_baseline" / "equity.csv"

def load_equity(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df["portfolio_value"]

ai_raw   = load_equity(AI_CSV)
base_raw = load_equity(BASE_CSV)

# Resample to daily (last value per calendar day) for clean plotting
ai_daily   = ai_raw.resample("1D").last().dropna()
base_daily = base_raw.resample("1D").last().dropna()

# Normalise to 100
ai_norm   = ai_daily / ai_daily.iloc[0] * 100
base_norm = base_daily / base_daily.iloc[0] * 100

# ── Market benchmarks ─────────────────────────────────────────────────────────
BM_DIR = Path(os.environ.get("APEXQUANT_DATA_DIR", "data/features"))

def load_benchmark(ticker):
    df = pd.read_csv(BM_DIR / f"{ticker}_1hour.csv", parse_dates=["ts_event"])
    df = df.set_index("ts_event").sort_index()
    df.index = df.index.tz_localize(None)
    daily = df["close"].resample("1D").last().dropna()
    return daily / daily.iloc[0] * 100

spy_norm = load_benchmark("SPY")
qqq_norm = load_benchmark("QQQ")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcdefaults()
plt.rc("font", family="DejaVu Sans", size=9)
plt.rc("axes", labelsize=11, titlesize=12)
plt.rc("xtick", labelsize=9)
plt.rc("ytick", labelsize=9)

BLACK   = "#1A1A1A"
DGRAY   = "#555555"
LGRAY   = "#B0B0B0"
VLGRAY  = "#E0E0E0"


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    p = OUT / f"{name}.png"
    print(f"  saved  {p}  ({p.stat().st_size / 1024:.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Equity Curve Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig1_equity():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    ax.set_facecolor("white")

    # Light horizontal gridlines only
    ax.yaxis.grid(True, color=VLGRAY, linewidth=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # ── 2022 bear-market shading ──
    shade_start = pd.Timestamp("2022-01-01")
    shade_end   = min(ai_norm.index.max(), base_norm.index.max(),
                      pd.Timestamp("2022-12-31"))
    ax.axvspan(shade_start, shade_end, color="#CCCCCC", alpha=0.08, zorder=0)

    # ── Market benchmarks ──
    ax.plot(spy_norm.index, spy_norm.values,
            color="#27AE60", linewidth=1.1, alpha=0.55, label="SPY Buy-and-Hold", zorder=1)
    ax.plot(qqq_norm.index, qqq_norm.values,
            color="#8E44AD", linewidth=1.1, alpha=0.55, label="QQQ Buy-and-Hold", zorder=1)

    # ── Strategy lines ──
    ax.plot(base_norm.index, base_norm.values,
            color=DGRAY, linewidth=1.2, label="EMA+RSI Baseline", zorder=2)
    ax.plot(ai_norm.index, ai_norm.values,
            color=BLACK, linewidth=1.8, label="AI Strategy (Trail Stop)", zorder=3)

    # ── 100 reference line ──
    ax.axhline(100, color=LGRAY, linewidth=0.8, linestyle="--", zorder=1)

    # ── Annotate final values ──
    # AI final
    ai_end_val = ai_norm.iloc[-1]
    ai_end_dt  = ai_norm.index[-1]
    ax.annotate(f"{ai_end_val:.1f}  (+{ai_end_val - 100:.1f}%)",
                xy=(ai_end_dt, ai_end_val),
                xytext=(15, 5), textcoords="offset points",
                fontsize=9, fontweight="bold", color=BLACK,
                arrowprops=dict(arrowstyle="-", color=LGRAY, lw=0.6))

    # Baseline final
    base_end_val = base_norm.iloc[-1]
    base_end_dt  = base_norm.index[-1]
    ax.annotate(f"{base_end_val:.1f}  ({base_end_val - 100:.1f}%)",
                xy=(base_end_dt, base_end_val),
                xytext=(15, -10), textcoords="offset points",
                fontsize=9, fontweight="bold", color=DGRAY,
                arrowprops=dict(arrowstyle="-", color=LGRAY, lw=0.6))

    # ── Benchmark annotations ──
    for bm, bm_col, bm_name, y_off in [
        (spy_norm, "#27AE60", "SPY", -12),
        (qqq_norm, "#8E44AD", "QQQ", 8),
    ]:
        bm_end = bm.iloc[-1]
        bm_dt  = bm.index[-1]
        ax.annotate(f"{bm_name} {bm_end:.0f}  ({bm_end - 100:+.1f}%)",
                    xy=(bm_dt, bm_end),
                    xytext=(15, y_off), textcoords="offset points",
                    fontsize=8, color=bm_col,
                    arrowprops=dict(arrowstyle="-", color=bm_col, lw=0.5, alpha=0.5))

    # ── Stats box (top-left) ──
    stats_text = (
        "AI Strategy:        Return +67.11%  |  Sharpe 2.14  |  MaxDD  \u22129.90%\n"
        "EMA+RSI Baseline:   Return \u221246.75%  |  Sharpe \u22121.54  |  MaxDD \u221251.25%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white",
                 edgecolor=LGRAY, alpha=0.95)
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment="top", fontfamily="monospace",
            bbox=props, zorder=10)

    # ── Axes formatting ──
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised portfolio value (start = 100)")
    ax.set_title("Portfolio Equity Curve: AI Strategy vs. Technical Baseline (2020\u20132022)",
                 fontsize=13, fontweight="bold", pad=14)

    # X-axis date range
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.87),
              fontsize=9, framealpha=0.9, edgecolor=LGRAY)
    despine(ax)

    # Bear market label (use axes transform for x to avoid ordinal issues)
    ax.text(pd.Timestamp("2022-06-01"), ax.get_ylim()[0] + 3,
            "2022 bear market", fontsize=8, color="#999999",
            ha="center", va="bottom", style="italic")

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    save(fig, "fig_equity_curve")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ablation Study Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
ABLATION = [
    ("Technical\n(EMA+RSI)",           -46.75, -1.54),
    ("+ Position\nSizing",             -41.64, -0.92),
    ("TP Detection\n(no Vol Gate)",    -39.04, -1.98),
    ("TP + Direction\n(no Vol Gate)",  -38.84, -1.97),
    ("Vol Gate +\nTP + Direction",     +24.57, +1.29),
    ("+ Trend\nBypass",                +26.11, +1.35),
    ("+ Signal Rev\n+ Vol Collapse",   +26.11, +1.38),
    ("+ All Exits\n(Tranche)",         +68.40, +2.13),
    ("+ All Exits\n(Trail Stop)",      +67.11, +2.14),
]

def fig2_ablation():
    labels   = [a[0] for a in ABLATION]
    returns  = [a[1] for a in ABLATION]
    sharpes  = [a[2] for a in ABLATION]
    n = len(labels)
    x = np.arange(n)

    # Colour logic: negative=light gray, positive=dark gray, final=black
    def bar_colors(vals, final_black=True):
        cols = []
        for i, v in enumerate(vals):
            if i == n - 1 and final_black:
                cols.append(BLACK)
            elif v >= 0:
                cols.append(DGRAY)
            else:
                cols.append(LGRAY)
        return cols

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), facecolor="white",
                                    sharey=False)
    fig.suptitle("Ablation Study: Incremental Contribution of Each System Component",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color=VLGRAY, linewidth=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    # ── Left: Total Return (%) ──
    cols_ret = bar_colors(returns)
    bars1 = ax1.bar(x, returns, color=cols_ret, edgecolor="white",
                    linewidth=0.6, width=0.72)
    ax1.axhline(0, color="#C04040", linewidth=1.0, linestyle="--", zorder=3)
    ax1.set_ylabel("Total return (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5, ha="center")

    # Value labels
    for i, (bar, v) in enumerate(zip(bars1, returns)):
        va = "bottom" if v >= 0 else "top"
        offset = 1.5 if v >= 0 else -1.5
        txt = f"{v:+.1f}%"
        if i == n - 1:
            txt += "  \u2605"  # star for final config
        ax1.text(i, v + offset, txt, ha="center", va=va, fontsize=7.5,
                 fontweight="bold" if i == n - 1 else "normal", color=BLACK)

    # Annotation arrow: Vol Gate jump (Run 4 → Run 5, index 3 → 4)
    ax1.annotate(
        "Vol Gate:\n+63.4 pp",
        xy=(4, returns[4]),
        xytext=(2.2, returns[4] + 20),
        fontsize=8.5, fontweight="bold", color="#C04040",
        arrowprops=dict(arrowstyle="-|>", color="#C04040", lw=1.5),
        ha="center",
    )

    despine(ax1)

    # ── Right: Sharpe Ratio ──
    cols_shp = bar_colors(sharpes)
    bars2 = ax2.bar(x, sharpes, color=cols_shp, edgecolor="white",
                    linewidth=0.6, width=0.72)
    ax2.axhline(0, color="#C04040", linewidth=1.0, linestyle="--", zorder=3)
    ax2.set_ylabel("Sharpe ratio")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5, ha="center")

    for i, (bar, v) in enumerate(zip(bars2, sharpes)):
        va = "bottom" if v >= 0 else "top"
        offset = 0.08 if v >= 0 else -0.08
        txt = f"{v:+.2f}"
        if i == n - 1:
            txt += "  \u2605"
        ax2.text(i, v + offset, txt, ha="center", va=va, fontsize=7.5,
                 fontweight="bold" if i == n - 1 else "normal", color=BLACK)

    # Annotation arrow: Vol Gate jump in Sharpe (Run 4 → Run 5)
    ax2.annotate(
        "Vol Gate:\n+3.26",
        xy=(4, sharpes[4]),
        xytext=(2.2, sharpes[4] + 0.7),
        fontsize=8.5, fontweight="bold", color="#C04040",
        arrowprops=dict(arrowstyle="-|>", color="#C04040", lw=1.5),
        ha="center",
    )

    despine(ax2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating thesis backtest figures ...\n")
    fig1_equity()
    fig2_ablation()
    print("\nDone.")
