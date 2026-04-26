#!/usr/bin/env python3
"""
Generate 6 publication-quality figures for direct prediction model comparison.
Run in Google Colab or any Python environment with matplotlib, seaborn, numpy, pandas.
Saves each figure as PDF + PNG (300 dpi).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# DATA — single source of truth for all 6 figures
# ══════════════════════════════════════════════════════════════════════════════
DATA = pd.DataFrame([
    ("LSTM Regression",          0.4986, 16181,  -0.347,  0.6356,  2.9687,  0.9730, ""),
    ("Attention-LSTM",           0.5059, 25045,   1.856,  0.0317,  3.4315,  0.9600, "*"),
    ("Transformer+LSTM",         0.5035, 26037,   1.141,  0.1270,  4.0820,  0.9494, ""),
    ("LightGBM v1",              0.4887, 16391,  -2.897,  0.9981,  1.6823,  0.9904, ""),
    ("LightGBM v2 (Alpha158)",   0.4833, 15971,  -4.229,  1.0000,  1.5034,  0.9930, ""),
    ("TimesFM 2.5",              0.4876, 16398,  -3.173,  0.9992,  1.6763,  0.9919, ""),
    ("VMD-LSTM (Global)",        0.7209,  2233,  20.876,  0.0000,  0.4591,  0.9982, "***"),
    ("VMD-LSTM (Rolling)",       0.4627,  2354,  -3.622,  0.9999, 13.2761, -0.5447, ""),
    ("Wavelet-LSTM",             0.5140, 16188,   3.568,  0.0002,  4.1218,  0.9320, "***"),
], columns=["Model", "DA", "N", "z", "p", "RMSE", "R2", "Sig"])

DATA["DA_pct"]    = DATA["DA"] * 100
DATA["CI_half"]   = 1.96 * np.sqrt(DATA["DA"] * (1 - DATA["DA"]) / DATA["N"]) * 100
DATA["deviation"] = DATA["DA_pct"] - 50.0

# ── Colour palette ────────────────────────────────────────────────────────────
C_STEEL    = "#4878A8"
C_SIG      = "#2B5D8C"
C_RED      = "#D04040"
C_ORANGE   = "#E8963E"
C_GRAY     = "#AAAAAA"
C_GREEN    = "#5B9F5B"
C_REDSOFT  = "#C75050"

# Category colours for Figure 6
CAT_NN     = "#4878A8"   # neural-network family (blue)
CAT_TREE   = "#5A9E6F"   # tree / ensemble (green)
CAT_FOUND  = "#7B6CB0"   # foundation model  (purple)
CAT_LEAK   = "#D04040"   # VMD-Global leakage (red)
CAT_ROLL   = "#E8963E"   # VMD-Rolling (orange)

def _model_color(name):
    if "Global" in name:  return C_RED
    if "Rolling" in name: return C_ORANGE
    return C_STEEL

def _model_color_sig(row):
    if "Global"  in row.Model: return C_RED
    if "Rolling" in row.Model: return C_ORANGE
    if row.Sig:                return C_SIG
    return C_STEEL

def _cat_color(name):
    if "Global"  in name: return CAT_LEAK
    if "Rolling" in name: return CAT_ROLL
    if "LightGBM" in name: return CAT_TREE
    if "TimesFM"  in name: return CAT_FOUND
    return CAT_NN

# ── Global style ──────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("font", family="DejaVu Sans", size=9)
plt.rc("axes", labelsize=11, titlesize=12)
plt.rc("xtick", labelsize=9)
plt.rc("ytick", labelsize=9)

def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {stem}.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Horizontal bar chart  (DA % comparison)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_bar():
    df = DATA.sort_values("DA_pct", ascending=True).reset_index(drop=True)
    is_global = df.Model.str.contains("Global")

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))
    colors  = [_model_color_sig(r) for _, r in df.iterrows()]
    hatches = ["///" if "Global" in m else "" for m in df.Model]

    bars = ax.barh(y, df.DA_pct, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.7)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)

    # 50 % baseline
    ax.axvline(50, color="#555555", ls="--", lw=1.0, zorder=0)
    ax.text(50.15, len(df) - 0.3, "Random baseline (50 %)",
            fontsize=8, color="#555555", va="top")

    # Y-tick labels with significance stars
    labels = []
    for _, r in df.iterrows():
        lbl = r.Model
        if r.Sig:
            lbl += f"  {r.Sig}"
        labels.append(lbl)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Value annotations at bar tips
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r.DA_pct + 0.3, i, f"{r.DA_pct:.1f}%", va="center",
                fontsize=8.5,
                fontweight="bold" if r.Sig else "normal",
                color=C_RED if "Global" in r.Model else "#333333")

    # Leakage callout
    g_idx = df.index[is_global][0]
    ax.annotate("DATA LEAKAGE", xy=(df.loc[g_idx, "DA_pct"], g_idx),
                xytext=(df.loc[g_idx, "DA_pct"] - 14, g_idx + 1.1),
                fontsize=9, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))

    ax.set_xlabel("Directional accuracy (%)")
    ax.set_xlim(44, 76)
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig4_1a_bar_da_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Forest plot  (DA with 95 % CI)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_forest():
    df = DATA.sort_values("DA_pct", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))

    ax.axvline(50, color="#555555", ls="--", lw=1.0, zorder=0)

    for i, (_, r) in enumerate(df.iterrows()):
        lo = r.DA_pct - r.CI_half
        hi = r.DA_pct + r.CI_half
        crosses_50 = lo <= 50 <= hi

        if "Global" in r.Model:
            col, marker, ms = C_RED, "D", 10
        elif "Rolling" in r.Model:
            col, marker, ms = C_ORANGE, "s", 8
        else:
            col = C_GRAY if crosses_50 else C_SIG
            marker, ms = "o", 5 + 1.2 * np.log(r.N / 1000)

        # whiskers
        ax.plot([lo, hi], [i, i], color=col, lw=2.0, solid_capstyle="round")
        ax.plot([lo, lo], [i - 0.15, i + 0.15], color=col, lw=1.5)
        ax.plot([hi, hi], [i - 0.15, i + 0.15], color=col, lw=1.5)
        # point estimate
        ax.plot(r.DA_pct, i, marker=marker, color=col, markersize=ms,
                markeredgecolor="white" if "Global" in r.Model else col,
                markeredgewidth=1.2 if "Global" in r.Model else 0, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.Model}  (N={r.N:,})" for _, r in df.iterrows()])
    ax.set_xlabel("Directional accuracy (%)")

    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SIG,
               markersize=8, label="Significant (CI excludes 50%)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GRAY,
               markersize=8, label="Not significant"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_RED,
               markersize=8, markeredgecolor="white", markeredgewidth=1,
               label="VMD-Global (leakage)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_ORANGE,
               markersize=8, label="VMD-Rolling"),
        Line2D([0], [0], color="#555555", ls="--", lw=1,
               label=r"$H_0$: DA = 50%"),
    ]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.9)
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig4_1b_forest_plot")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Lollipop chart  (deviation from 50 % baseline)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_lollipop():
    df = DATA.sort_values("deviation", ascending=False).reset_index(drop=True)
    vmd_global_dev = df.loc[df.Model.str.contains("Global"), "deviation"].values[0]
    CAP = 8.0  # visual cap for VMD-Global

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))

    # reference band and line
    ax.axhline(0, color="#555555", lw=1.0, zorder=0)
    ax.axhspan(-2, 2, color="#F0F0F0", zorder=0)
    ax.text(len(df) - 0.5, 1.7, "+/-2 pp noise band",
            fontsize=7, color="#999999", ha="right", va="top")

    for i, (_, r) in enumerate(df.iterrows()):
        dev = r.deviation
        is_global  = "Global"  in r.Model
        is_rolling = "Rolling" in r.Model
        vis_dev = min(dev, CAP) if dev > CAP else dev

        if   is_global:  col = C_RED
        elif is_rolling: col = C_ORANGE
        elif dev > 0:    col = C_GREEN
        else:            col = C_REDSOFT

        # stem
        ax.plot([i, i], [0, vis_dev], color=col, lw=2.0, solid_capstyle="round")
        # marker sized by |z|
        ms = 5 + min(abs(r.z), 20) * 0.5
        marker = "D" if is_global else ("s" if is_rolling else "o")
        ax.plot(i, vis_dev, marker=marker, color=col, markersize=ms,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)

        # axis-break annotation for VMD-Global
        if is_global:
            ax.annotate(f"+{vmd_global_dev:.1f} pp\n(DA = 72.1%)",
                        xy=(i, CAP), xytext=(i + 0.8, CAP + 0.5),
                        fontsize=8, fontweight="bold", color=C_RED,
                        arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2),
                        ha="left")
            # break marks
            ax.plot([i - 0.15, i + 0.15], [CAP - 0.2, CAP + 0.2],
                    color=C_RED, lw=1.5)
            ax.plot([i - 0.15, i + 0.15], [CAP - 0.5, CAP - 0.1],
                    color=C_RED, lw=1.5)

    xlabels = []
    for _, r in df.iterrows():
        lbl = r.Model.replace(" (Global)", "\n(Global)")
        lbl = lbl.replace(" (Rolling)", "\n(Rolling)")
        lbl = lbl.replace(" (Alpha158)", "\n(Alpha158)")
        if r.Sig:
            lbl += f"  {r.Sig}"
        xlabels.append(lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8, ha="center")
    ax.set_ylabel("Deviation from 50% baseline (pp)")
    ax.set_ylim(-5.5, CAP + 2.5)
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig4_1c_lollipop_deviation")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Heatmap  (multi-metric comparison)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_heatmap():
    df = DATA.copy()
    # Ordering: keep logical grouping
    order = [
        "LSTM Regression", "Attention-LSTM", "Transformer+LSTM",
        "LightGBM v1", "LightGBM v2 (Alpha158)", "TimesFM 2.5",
        "VMD-LSTM (Global)", "VMD-LSTM (Rolling)", "Wavelet-LSTM",
    ]
    df = df.set_index("Model").loc[order]

    fig, axes = plt.subplots(1, 3, figsize=(12, 6),
                             gridspec_kw={"width_ratios": [1, 1, 1], "wspace": 0.05})

    metrics = [
        ("DA_pct",  "DA (%)",   "RdYlGn",   False),
        ("RMSE",    "RMSE",     "RdYlGn_r", False),
        ("R2",      "R-squared","RdYlGn",   False),
    ]

    for ax, (col, title, cmap, _) in zip(axes, metrics):
        vals = df[[col]].copy()

        # Normalise for colour mapping only
        vmin, vmax = vals[col].min(), vals[col].max()
        # widen range slightly for better contrast
        span = vmax - vmin if vmax != vmin else 1
        norm_vals = (vals[col] - vmin) / span

        # Build annotation strings
        if col == "DA_pct":
            annot = vals[col].map(lambda v: f"{v:.1f}")
        elif col == "RMSE":
            annot = vals[col].map(lambda v: f"{v:.2f}")
        else:
            annot = vals[col].map(lambda v: f"{v:.4f}")
        annot_arr = annot.values.reshape(-1, 1)

        sns.heatmap(
            norm_vals.values.reshape(-1, 1),
            ax=ax, annot=annot_arr, fmt="",
            cmap=cmap, cbar=False,
            linewidths=1.5, linecolor="white",
            xticklabels=[title],
            yticklabels=df.index if ax is axes[0] else False,
            annot_kws={"fontsize": 10, "fontweight": "bold"},
            vmin=0, vmax=1,
        )
        ax.set_xticklabels([title], fontsize=11, fontweight="bold")
        ax.tick_params(left=False, bottom=False)

        # Highlight VMD-Global row with red border
        global_row = list(df.index).index("VMD-LSTM (Global)")
        rect = plt.Rectangle((0, global_row), 1, 1, fill=False,
                              edgecolor=C_RED, lw=2.5, clip_on=False)
        ax.add_patch(rect)

    axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=9)
    fig.suptitle("Multi-metric comparison across direct prediction models",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fig4_1d_heatmap_metrics")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Radar / spider chart  (multi-dimensional model profile)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_radar():
    # Select 5 representative models
    sel_names = [
        "LSTM Regression",
        "Attention-LSTM",
        "LightGBM v2 (Alpha158)",
        "VMD-LSTM (Global)",
        "Wavelet-LSTM",
    ]
    df = DATA.set_index("Model").loc[sel_names].copy()

    # Axes: DA%, 1-normalised_RMSE, R², 1-p, normalised_log(N)
    # All mapped to [0, 1] where higher = better
    full = DATA  # use full dataset for normalisation bounds

    # DA normalised
    da_min, da_max = full.DA_pct.min(), full.DA_pct.max()
    df["norm_DA"] = (df.DA_pct - da_min) / (da_max - da_min)

    # RMSE: lower = better → invert
    rmse_min, rmse_max = full.RMSE.min(), full.RMSE.max()
    df["norm_RMSE_inv"] = 1 - (df.RMSE - rmse_min) / (rmse_max - rmse_min)

    # R²: higher = better (clamp negatives to 0 for visual)
    r2_min, r2_max = max(full.R2.min(), -1), full.R2.max()
    df["norm_R2"] = (df.R2.clip(lower=-1) - r2_min) / (r2_max - r2_min)

    # 1 - p  (higher = more significant)
    df["norm_sig"] = 1 - df.p

    # log(N) normalised
    logn_min, logn_max = np.log(full.N.min()), np.log(full.N.max())
    df["norm_logN"] = (np.log(df.N) - logn_min) / (logn_max - logn_min)

    categories = ["DA", "1-RMSE\n(norm)", "R-squared", "1-p\n(signif.)", "log N\n(norm)"]
    value_cols = ["norm_DA", "norm_RMSE_inv", "norm_R2", "norm_sig", "norm_logN"]
    N_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, N_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors_radar = {
        "LSTM Regression":        C_STEEL,
        "Attention-LSTM":         C_SIG,
        "LightGBM v2 (Alpha158)": CAT_TREE,
        "VMD-LSTM (Global)":      C_RED,
        "Wavelet-LSTM":           "#8B6DB0",
    }
    line_styles = {
        "VMD-LSTM (Global)": "--",  # dashed → indicates leakage
    }

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)

    for name in sel_names:
        vals = df.loc[name, value_cols].values.tolist()
        vals += vals[:1]
        ls = line_styles.get(name, "-")
        col = colors_radar[name]
        ax.plot(angles, vals, linewidth=1.8, linestyle=ls, label=name, color=col)
        ax.fill(angles, vals, alpha=0.12, color=col)

    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7,
                       color="#888888")

    # Legend outside
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05),
                    fontsize=9, framealpha=0.9)
    # Annotate leakage in legend
    for text in leg.get_texts():
        if "Global" in text.get_text():
            text.set_color(C_RED)
            text.set_fontweight("bold")

    fig.tight_layout()
    _save(fig, "fig4_1e_radar_profile")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Grouped bar chart  (DA % by model category + significance)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_grouped_bar():
    # Define display order and category labels
    order = [
        "LSTM Regression", "Attention-LSTM", "Transformer+LSTM",
        "Wavelet-LSTM",
        "LightGBM v1", "LightGBM v2 (Alpha158)",
        "TimesFM 2.5",
        "VMD-LSTM (Global)", "VMD-LSTM (Rolling)",
    ]
    df = DATA.set_index("Model").loc[order].reset_index()

    cat_labels = {
        "LSTM Regression":        "Neural network",
        "Attention-LSTM":         "Neural network",
        "Transformer+LSTM":       "Neural network",
        "Wavelet-LSTM":           "Neural network",
        "LightGBM v1":            "Tree / ensemble",
        "LightGBM v2 (Alpha158)": "Tree / ensemble",
        "TimesFM 2.5":            "Foundation model",
        "VMD-LSTM (Global)":      "VMD-LSTM (leakage)",
        "VMD-LSTM (Rolling)":     "VMD-LSTM (corrected)",
    }
    df["Category"] = df.Model.map(cat_labels)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    colors = [_cat_color(m) for m in df.Model]

    bars = ax.bar(x, df.DA_pct, color=colors, edgecolor="white",
                  linewidth=0.6, width=0.7)

    # Hatching for VMD-Global
    for bar, m in zip(bars, df.Model):
        if "Global" in m:
            bar.set_hatch("///")

    # 50 % baseline
    ax.axhline(50, color="#555555", ls="--", lw=1.0, zorder=0)
    ax.text(len(df) - 0.3, 50.3, "50%", fontsize=8, color="#555555", ha="right")

    # Value + significance annotations
    for i, (_, r) in enumerate(df.iterrows()):
        va = "bottom" if r.DA_pct >= 50 else "top"
        offset = 0.4 if r.DA_pct >= 50 else -0.4
        txt = f"{r.DA_pct:.1f}%"
        if r.Sig:
            txt += f" {r.Sig}"
        ax.text(i, r.DA_pct + offset, txt, ha="center", va=va, fontsize=8,
                fontweight="bold" if r.Sig else "normal",
                color=C_RED if "Global" in r.Model else "#333333")

    # Leakage annotation
    g_idx = list(df.Model).index("VMD-LSTM (Global)")
    ax.annotate("DATA LEAKAGE",
                xy=(g_idx, df.iloc[g_idx].DA_pct),
                xytext=(g_idx - 1.5, df.iloc[g_idx].DA_pct + 3),
                fontsize=9, fontweight="bold", color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))

    # X labels
    xlabels = [m.replace(" (Alpha158)", "\n(Alpha158)")
                .replace(" (Global)", "\n(Global)")
                .replace(" (Rolling)", "\n(Rolling)")
               for m in df.Model]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_ylim(44, 78)

    # Category legend
    leg_elements = [
        mpatches.Patch(facecolor=CAT_NN,    label="Neural network"),
        mpatches.Patch(facecolor=CAT_TREE,  label="Tree / ensemble"),
        mpatches.Patch(facecolor=CAT_FOUND, label="Foundation model"),
        mpatches.Patch(facecolor=CAT_LEAK,  label="VMD-LSTM (leakage)", hatch="///",
                       edgecolor="white"),
        mpatches.Patch(facecolor=CAT_ROLL,  label="VMD-LSTM (corrected)"),
    ]
    ax.legend(handles=leg_elements, loc="upper left", fontsize=8, framealpha=0.9)
    _despine(ax)
    fig.tight_layout()
    _save(fig, "fig4_1f_grouped_bar_category")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating 6 direct-prediction comparison figures ...\n")
    fig1_bar()
    fig2_forest()
    fig3_lollipop()
    fig4_heatmap()
    fig5_radar()
    fig6_grouped_bar()
    print(f"\nDone. 12 files saved (6 PDF + 6 PNG).")
