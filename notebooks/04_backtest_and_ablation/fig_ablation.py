#!/usr/bin/env python3
"""
Generate 3 thesis figures from the 11-run ablation study (2026-03-16 run set).
  Figure 1: fig_ablation_curves.png   — All 11 equity curves + SPY/QQQ
  Figure 2: fig_equity_comparison.png — AI Final vs EMA+RSI vs SPY
  Figure 3: fig_ablation_bars.png     — Return + Sharpe bar charts
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# RUNS_DIR + BM_DIR come from the separate ApexQuant platform repo's backtest
# outputs, which are not shipped in this research release. Either point the
# env vars below at your local copy or drop the run CSVs into the default
# relative paths before executing this script.
OUT      = Path("thesis_figures")
OUT.mkdir(exist_ok=True)
RUNS_DIR = Path(os.environ.get("APEXQUANT_RUNS_DIR", "runs"))
BM_DIR   = Path(os.environ.get("APEXQUANT_DATA_DIR", "data/features"))
PREFIX   = "20260316_140416_"

# ── 11-run ablation definitions (in cumulative order) ─────────────────────────
#  (run#, short_label, dir_suffix, color, lw, ls, alpha)
RUNS = [
    ( 1, "Technical\n(EMA+RSI)",         "ema_rsi_baseline",     "#AAAAAA", 1.0, "--", 0.60),
    ( 2, "+ Position\nSizing",           "ema_rsi_with_sizing",  "#999999", 1.0, "--", 0.60),
    ( 3, "TP Detection\n(no Vol Gate)",  "ai_layer2_only",       "#888888", 1.0, "--", 0.60),
    ( 4, "TP + Direction\n(no Vol Gate)","ai_no_vol_gate",       "#777777", 1.0, "--", 0.60),
    ( 5, "Vol Gate +\nTP + Direction",   "ai_vol_gate_only",     "#E74C3C", 1.8, "-",  0.90),
    ( 6, "+ Trend\nBypass",              "ai_baseline",          "#E67E22", 1.5, "-",  0.85),
    ( 7, "+ Signal\nReversal",           "ai_signal_reversal",   "#F39C12", 1.5, "-",  0.85),
    ( 8, "+ Vol\nCollapse",              "ai_vol_collapse",      "#27AE60", 1.5, "-",  0.85),
    ( 9, "Tranche Exit\n(standalone)",   "ai_tranche_exit",      "#16A085", 1.5, "-",  0.85),
    (10, "+ All Exits\n(Tranche)",       "ai_full_tranche",      "#2980B9", 2.0, "-",  1.00),
    (11, "+ All Exits\n(Trail Stop)",    "ai_full_trail",        "#1A1A1A", 2.5, "-",  1.00),
]

# ── Load helpers ──────────────────────────────────────────────────────────────
def load_equity(suffix):
    path = RUNS_DIR / f"{PREFIX}{suffix}" / "equity.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    daily = df["portfolio_value"].resample("1D").last().dropna()
    return daily / daily.iloc[0] * 100

def load_metrics(suffix):
    path = RUNS_DIR / f"{PREFIX}{suffix}" / "metrics.json"
    with open(path) as f:
        return json.load(f)["metrics"]

def load_benchmark(ticker):
    df = pd.read_csv(BM_DIR / f"{ticker}_1hour.csv", parse_dates=["ts_event"])
    df = df.set_index("ts_event").sort_index()
    df.index = df.index.tz_localize(None)
    daily = df["close"].resample("1D").last().dropna()
    # Trim to backtest period
    daily = daily[daily.index >= "2020-01-02"]
    daily = daily[daily.index <= "2022-09-30"]
    return daily / daily.iloc[0] * 100

# ── Load all data ─────────────────────────────────────────────────────────────
print("Loading equity curves...")
curves  = {}
metrics = {}
for num, label, suffix, *_ in RUNS:
    curves[num]  = load_equity(suffix)
    metrics[num] = load_metrics(suffix)
    ret = metrics[num]["total_return"]
    sr  = metrics[num]["sharpe_ratio"]
    print(f"  Run {num:2d}: {ret:+7.2%}  Sharpe={sr:.2f}  final={curves[num].iloc[-1]:.0f}")

spy = load_benchmark("SPY")
qqq = load_benchmark("QQQ")
print(f"  SPY:   {spy.iloc[-1]-100:+.1f}%   QQQ: {qqq.iloc[-1]-100:+.1f}%")

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcdefaults()
plt.rc("font", family="DejaVu Sans", size=9)
plt.rc("axes", labelsize=11, titlesize=13)
plt.rc("xtick", labelsize=9)
plt.rc("ytick", labelsize=9)

BLACK   = "#1A1A1A"
DGRAY   = "#555555"
LGRAY   = "#CCCCCC"

def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    p = OUT / f"{name}.png"
    print(f"  saved {p}  ({p.stat().st_size/1024:.0f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Ablation Equity Curves (all 11 + benchmarks)
# ══════════════════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="white")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.5, alpha=0.25)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # 100 baseline
    ax.axhline(100, color=LGRAY, lw=0.7, ls=":", zorder=0)

    # 2022 bear market shading
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-09-30"),
               color="#F5F5F5", zorder=0)

    # Benchmarks (behind everything)
    ax.plot(spy.index, spy.values, color="#888888", lw=1.2, alpha=0.7, zorder=1)
    ax.plot(qqq.index, qqq.values, color="#AAAAAA", lw=1.2, alpha=0.7, zorder=1)

    # Strategy curves
    for num, label, suffix, col, lw, ls, alpha in RUNS:
        c = curves[num]
        ax.plot(c.index, c.values, color=col, lw=lw, ls=ls, alpha=alpha,
                zorder=2 + num)

    # ── Right-edge annotations ──
    annots = []
    for num, label, suffix, col, lw, *_ in RUNS:
        c = curves[num]
        short = label.split("\n")[0].replace("+ ", "").strip()
        if num == 11: short = "Trail Stop"
        elif num == 10: short = "Tranche"
        elif num == 9: short = "Tranche (solo)"
        elif num == 8: short = "Vol Collapse"
        elif num == 7: short = "Sig Rev"
        elif num == 6: short = "Trend Bypass"
        elif num == 5: short = "Vol Gate"
        elif num == 4: short = "TP+Dir"
        elif num == 3: short = "TP only"
        elif num == 2: short = "+ Sizing"
        elif num == 1: short = "EMA+RSI"
        annots.append((c.iloc[-1], c.index[-1], short, col, lw >= 2.0))

    # Benchmarks
    annots.append((spy.iloc[-1], spy.index[-1],
                   f"SPY ({spy.iloc[-1]-100:+.0f}%)", "#888888", False))
    annots.append((qqq.iloc[-1], qqq.index[-1],
                   f"QQQ ({qqq.iloc[-1]-100:+.0f}%)", "#AAAAAA", False))

    # Sort and space labels
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
        ax.annotate(f"{txt}  {val:.0f}", xy=(dt, val),
                    xytext=(14, dy), textcoords="offset points",
                    fontsize=6.5, fontweight="bold" if bold else "normal",
                    color=col, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.3, alpha=0.4)
                        if abs(dy) > 4 else None)

    # Axes
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-10-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised portfolio value (start = 100)")
    ax.set_title(
        "Ablation Study: Portfolio Equity Curves by Configuration (2020\u20132022 Q3)",
        fontsize=13, fontweight="bold", pad=14)

    # Subtitle
    ax.text(0.5, 1.01, "Each configuration adds one component to the previous",
            transform=ax.transAxes, ha="center", fontsize=9, color=DGRAY,
            style="italic")

    # Compact legend
    legend_elements = [
        Line2D([0], [0], color="#999999", lw=1.0, ls="--", alpha=0.7,
               label="Runs 1\u20134: No Vol Gate (losing)"),
        Line2D([0], [0], color="#E74C3C", lw=1.8, label="Run 5: + Vol Gate"),
        Line2D([0], [0], color="#E67E22", lw=1.5, label="Run 6: + Trend Bypass"),
        Line2D([0], [0], color="#F39C12", lw=1.5, label="Run 7: + Signal Reversal"),
        Line2D([0], [0], color="#27AE60", lw=1.5, label="Run 8: + Vol Collapse"),
        Line2D([0], [0], color="#16A085", lw=1.5, label="Run 9: Tranche (standalone)"),
        Line2D([0], [0], color="#2980B9", lw=2.0, label="Run 10: Full + Tranche"),
        Line2D([0], [0], color="#1A1A1A", lw=2.5, label="Run 11: Full + Trail Stop"),
        Line2D([0], [0], color="#888888", lw=1.2, alpha=0.7, label="SPY Buy-and-Hold"),
        Line2D([0], [0], color="#AAAAAA", lw=1.2, alpha=0.7, label="QQQ Buy-and-Hold"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(0.01, 0.96), fontsize=7, framealpha=0.92,
              edgecolor=LGRAY, handlelength=2.2, labelspacing=0.35)

    # Footnote
    ax.text(0.99, -0.12,
            "Models trained on first 70% of data. Fixed weights applied across full "
            "backtest period.\n15-min data available through 2022-09-30.",
            transform=ax.transAxes, fontsize=6.5, color="#999999",
            ha="right", va="top")

    despine(ax)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    save(fig, "fig_ablation_curves")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Equity Comparison (AI Final vs EMA+RSI vs SPY)
# ══════════════════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#EEEEEE", lw=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # 2022 bear market shading
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-09-30"),
               color="#F5F5F5", zorder=0)

    # 100 baseline
    ax.axhline(100, color=LGRAY, lw=0.8, ls="--", zorder=1)

    # SPY
    ax.plot(spy.index, spy.values,
            color="#888888", lw=1.2, alpha=0.7, label="SPY Buy-and-Hold", zorder=2)

    # EMA+RSI
    c1 = curves[1]
    ax.plot(c1.index, c1.values,
            color=DGRAY, lw=1.2, label="EMA+RSI Baseline (Run 1)", zorder=3)

    # AI Final
    c11 = curves[11]
    ax.plot(c11.index, c11.values,
            color=BLACK, lw=2.0, label="AI Strategy - Trail Stop (Run 11)", zorder=4)

    # End-value annotations
    for c, name, col, bold, yoff in [
        (c11, "AI Final", BLACK, True, 6),
        (spy, "SPY", "#888888", False, -10),
        (c1,  "EMA+RSI", DGRAY, False, -8),
    ]:
        v = c.iloc[-1]
        ax.annotate(f"{name}  {v:.0f} ({v-100:+.1f}%)",
                    xy=(c.index[-1], v), xytext=(14, yoff),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold" if bold else "normal", color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.5))

    # Stats box
    m11 = metrics[11]
    m1  = metrics[1]
    spy_ret = spy.iloc[-1] - 100
    stats = (
        f"AI Final:    Return {m11['total_return']:+.2%}  |  "
        f"Sharpe {m11['sharpe_ratio']:.2f}  |  "
        f"MaxDD {m11['max_drawdown']:.2%}\n"
        f"EMA+RSI:     Return {m1['total_return']:+.2%}  |  "
        f"Sharpe {m1['sharpe_ratio']:.2f}  |  "
        f"MaxDD {m1['max_drawdown']:.2%}\n"
        f"SPY B&H:     Return {spy_ret:+.1f}%"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="white",
                 edgecolor=LGRAY, alpha=0.95)
    ax.text(0.02, 0.97, stats, transform=ax.transAxes,
            fontsize=8, va="top", fontfamily="monospace", bbox=props, zorder=10)

    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-10-15"))
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalised portfolio value (start = 100)")
    ax.set_title(
        "Portfolio Equity Curve: AI Strategy vs. Technical Baseline vs. Market",
        fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.83),
              fontsize=9, framealpha=0.9, edgecolor=LGRAY)
    despine(ax)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    save(fig, "fig_equity_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Ablation Bar Chart (Return + Sharpe)
# ══════════════════════════════════════════════════════════════════════════════
def fig3():
    labels  = [r[1].replace("\n", " ") for r in RUNS]
    returns = [metrics[r[0]]["total_return"] * 100 for r in RUNS]
    sharpes = [metrics[r[0]]["sharpe_ratio"] for r in RUNS]
    n = len(RUNS)
    x = np.arange(n)

    def bar_colors(vals):
        cols = []
        for i, v in enumerate(vals):
            if i == n - 1:       cols.append(BLACK)
            elif v >= 0:         cols.append(DGRAY)
            else:                cols.append("#BBBBBB")
        return cols

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6), facecolor="white")
    fig.suptitle("Ablation Study: Incremental Contribution of Each System Component",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        ax.yaxis.grid(True, color="#EEEEEE", lw=0.5)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)

    # ── Left: Total Return ──
    bars1 = ax1.bar(x, returns, color=bar_colors(returns),
                    edgecolor="white", lw=0.6, width=0.72)
    ax1.axhline(0, color="#C04040", lw=1.0, ls="--", zorder=3)
    ax1.set_ylabel("Total return (%)")
    ax1.set_xticks(x)
    # Shorter labels for x-axis
    short_xlabels = ["EMA+RSI", "+Sizing", "TP only", "TP+Dir",
                     "Vol Gate", "+Trend", "+SigRev", "+VolCol",
                     "Tranche\n(solo)", "Full+\nTranche", "Full+\nTrail"]
    ax1.set_xticklabels(short_xlabels, fontsize=7, ha="center")

    for i, v in enumerate(returns):
        va = "bottom" if v >= 0 else "top"
        off = 1.5 if v >= 0 else -1.5
        txt = f"{v:+.1f}%"
        if i == n - 1: txt += "  *"
        ax1.text(i, v + off, txt, ha="center", va=va, fontsize=6.5,
                 fontweight="bold" if i == n-1 else "normal", color=BLACK)

    # Vol Gate annotation
    ax1.annotate("Vol Gate:\n+63.5 pp",
                 xy=(4, returns[4]), xytext=(2.2, returns[4] + 22),
                 fontsize=8.5, fontweight="bold", color="#C04040",
                 arrowprops=dict(arrowstyle="-|>", color="#C04040", lw=1.5),
                 ha="center")
    despine(ax1)

    # ── Right: Sharpe Ratio ──
    bars2 = ax2.bar(x, sharpes, color=bar_colors(sharpes),
                    edgecolor="white", lw=0.6, width=0.72)
    ax2.axhline(0, color="#C04040", lw=1.0, ls="--", zorder=3)
    ax2.set_ylabel("Sharpe ratio")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_xlabels, fontsize=7, ha="center")

    for i, v in enumerate(sharpes):
        va = "bottom" if v >= 0 else "top"
        off = 0.08 if v >= 0 else -0.08
        txt = f"{v:+.2f}"
        if i == n - 1: txt += "  *"
        ax2.text(i, v + off, txt, ha="center", va=va, fontsize=6.5,
                 fontweight="bold" if i == n-1 else "normal", color=BLACK)

    # Vol Gate annotation
    ax2.annotate("Vol Gate:\n+3.27",
                 xy=(4, sharpes[4]), xytext=(2.2, sharpes[4] + 0.8),
                 fontsize=8.5, fontweight="bold", color="#C04040",
                 arrowprops=dict(arrowstyle="-|>", color="#C04040", lw=1.5),
                 ha="center")
    despine(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig_ablation_bars")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating 3 thesis figures (11-run ablation)...\n")
    fig1()
    fig2()
    fig3()
    print("\nDone. 6 files saved (3 PDF + 3 PNG).")
