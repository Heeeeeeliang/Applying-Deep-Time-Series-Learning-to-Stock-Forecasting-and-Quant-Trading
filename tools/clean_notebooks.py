#!/usr/bin/env python3
"""
Notebook cleanup pass for the public research release.

What this does (idempotently, in place):
  1. Inserts a CONFIG dict + chapter-header markdown cell at the top of each
     notebook in ../notebooks/**/*.ipynb if not already present.
  2. Rewrites Google-Drive paths (/content/drive/MyDrive/..., My Drive/...,
     drive_root = "/content/drive/...") to use CONFIG paths.
  3. Strips cell outputs for cells whose source contains any of:
       - shell magics that print system paths (!pwd, !ls /content, !pip list)
       - auth/mount calls (drive.mount, files.upload, userdata.get, api_key)
       - noisy debug prints of absolute paths
     KEEPS outputs for cells whose final rendered object is a DataFrame,
     matplotlib figure, or structured metric dict.
  4. Clears execution_count for every cell to guarantee a clean diff.
  5. Removes cells that are ONLY Colab-specific setup (drive.mount /
     !pip install from Drive URL / auth wrappers) — replaced by a one-line
     marker so readers know the step exists in the original.
  6. Warns (non-fatal) on: remaining hardcoded absolute paths, email-like
     strings, gdown ID form, torch.load without weights_only, and
     commented-out code blocks longer than 5 lines.

Run from repo root:
    python tools/clean_notebooks.py [--dry-run] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ── top-level CONFIG ──────────────────────────────────────────────────────────
CONFIG = {
    "repo_root": Path(__file__).resolve().parents[1],
    "notebook_glob": "notebooks/**/*.ipynb",
    "max_commented_block_lines": 5,
}

# ── patterns ──────────────────────────────────────────────────────────────────
DRIVE_PATH = re.compile(
    r"""(?xi)
    (['"])
    /content/drive/(?:My\ Drive|MyDrive)/(?:[^'"]*/)*(?P<tail>[^'"]*)
    \1
    """
)
MYDRIVE_BARE = re.compile(r"['\"](?:My\s?Drive|MyDrive)/([^'\"]+)['\"]")
AUTH_LINE = re.compile(
    r"(?i)^(?:\s*!)?\s*("
    r"from\s+google\.colab\s+import\s+drive"
    r"|drive\.mount\("
    r"|from\s+google\.colab\s+import\s+files"
    r"|files\.upload\("
    r"|files\.download\("
    r"|from\s+google\.colab\s+import\s+userdata"
    r"|userdata\.get\("
    r"|!\s*pip\s+install\s+.*drive\.google"
    r")"
)
SHELL_NOISY = re.compile(r"^\s*!\s*(pwd|ls\s+/content|pip\s+list)\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
GDOWN_ID = re.compile(r"gdown\.download\([^)]*\bid\s*=")
ABS_WIN_PATH = re.compile(r"['\"][A-Z]:[\\/][^'\"]{2,}['\"]")
ABS_UNIX_PATH = re.compile(r"['\"]/(?:home|Users|content)/[^'\"]{2,}['\"]")
TORCH_LOAD_UNSAFE = re.compile(
    r"torch\.load\(\s*[^)]*\)"
)  # then we post-filter for weights_only=

CONFIG_CELL_MARKER = "# === CONFIG (edit paths here) ==="

# ── cell builders ─────────────────────────────────────────────────────────────
HEADER_MD_TEMPLATE = """\
# {title}

**Thesis section.** {section}

**Inputs.** {inputs}

**Outputs.** {outputs}

**Expected runtime.** {runtime}. **Expected GPU.** {gpu}.

> All paths in the CONFIG cell below resolve relative to the repo root. On
> Google Colab, uncomment the Drive fallback line.
"""

CONFIG_CELL_CODE = f"""\
{CONFIG_CELL_MARKER}
from pathlib import Path

CONFIG = {{
    "data_dir":        Path("../../data"),         # processed + features + splits
    "results_dir":     Path("../../results"),
    "checkpoints_dir": Path("../../checkpoints"),
    "seed":            42,
    "device":          "cuda",                      # or "cpu"
    # Colab fallback — uncomment if running on Colab with the dataset mounted:
    # "data_dir": Path("/content/drive/MyDrive/thesis_data"),
}}
for key, path in CONFIG.items():
    if isinstance(path, Path):
        path.mkdir(parents=True, exist_ok=True)
"""

# Per-notebook header metadata (filled in by hand from the thesis chapter map)
HEADERS = {
    "01_direct_prediction_ceiling/01_nine_method_ceiling.ipynb": dict(
        title="Direct Prediction Ceiling — 9 methods (Chapter 4.1)",
        section="4.1 — the ~50% accuracy ceiling result",
        inputs="`data/features/<TICKER>_1hour_features.csv`, `data/splits/<TICKER>_1hour/`",
        outputs="`results/ch2_standardised_results.json`, `results/figures/fig4_1{a,b,c,d}_*.{pdf,png}`",
        runtime="~30 min for baselines; 3–5 h incl. TimesFM fine-tune",
        gpu="T4 sufficient for TimesFM zero-shot; A100 for fine-tuning",
    ),
    "01_direct_prediction_ceiling/02_missing_metrics.ipynb": dict(
        title="Direct Prediction — Recovery of Missing Metrics (Chapter 4.1)",
        section="4.1 — fills in the model/ticker cells where the first pass crashed",
        inputs="`results/ch2_standardised_results.json` (partial)",
        outputs="`results/ch2_standardised_results.json` (complete)",
        runtime="~20 min",
        gpu="T4",
    ),
    "01_direct_prediction_ceiling/03_baseline_models.ipynb": dict(
        title="Baseline Models — Naïve / HistMean / LinReg / LightGBM",
        section="4.1 — non-neural baselines in the ceiling study",
        inputs="`data/features/<TICKER>_1hour_features.csv`, `data/splits/<TICKER>_1hour/`",
        outputs="rows 1–4 of `results/ch2_standardised_results.json`",
        runtime="< 10 min",
        gpu="not required",
    ),
    "02_volatility_layer/01_train_vol_lightgbm.ipynb": dict(
        title="Layer 1 — Train volatility LightGBM (Chapter 4.2)",
        section="4.2 — volatility-regime forecaster, 119-flat-feature LightGBM (Model D)",
        inputs="`data/processed/<TICKER>_1hour.csv`",
        outputs="`models/layer1/volatility/lightgbm_v3_flat/{weights.joblib, meta.json, feature_names.json}`",
        runtime="5–10 min",
        gpu="not required",
    ),
    "03_turning_point_layer/01_multiscale_cnn.ipynb": dict(
        title="Layer 2 — Turning-Point Detection with MultiScaleCNN (Chapter 4.3)",
        section="4.3 — local top / bottom detection, meta-labelled",
        inputs="`data/tp_data/<TICKER>_{top,bottom}_{train,val,test}.npz`",
        outputs="`checkpoints/multiscale_cnn_{top,bottom}.pt`, `results/meta_label_v2_results.json`, `results/turning_point_summary.md`",
        runtime="45–90 min",
        gpu="T4 minimum",
    ),
    "04_backtest_and_ablation/01_end_to_end_pipeline.ipynb": dict(
        title="End-to-End Pipeline — LSTM + Vol + TP (Chapter 4.4 prep)",
        section="4.4 — full-system driver that produces inputs for the 11 ablation runs",
        inputs="all layer outputs from 02_ and 03_; `data/processed/<TICKER>_15min.csv`",
        outputs="per-run signal streams consumed by the ablation backtester (ApexQuant repo)",
        runtime="~2 h (full), ~30 min (single ticker)",
        gpu="T4 minimum",
    ),
}


# ── core ──────────────────────────────────────────────────────────────────────
def scrub_source(src: str) -> tuple[str, list[str]]:
    """Return (new_src, warnings) after path/email scrubbing."""
    warnings: list[str] = []
    new_src = src

    # Drive paths -> CONFIG
    new_src = DRIVE_PATH.sub(r"CONFIG['data_dir'] / r'\g<tail>'", new_src)
    new_src = MYDRIVE_BARE.sub(r"CONFIG['data_dir'] / r'\1'", new_src)

    # Warnings
    if EMAIL.search(new_src):
        warnings.append("contains email-like string")
    if GDOWN_ID.search(new_src):
        warnings.append("uses gdown.download(id=...) — use fuzzy=True with the full URL")
    if "torch.load(" in new_src and "weights_only" not in new_src:
        warnings.append("torch.load without weights_only=True")
    if ABS_WIN_PATH.search(new_src) or ABS_UNIX_PATH.search(new_src):
        warnings.append("remaining absolute path literal")

    # Long commented-out blocks
    comment_run = 0
    for line in new_src.splitlines():
        if line.lstrip().startswith("#") and not line.lstrip().startswith("#!"):
            comment_run += 1
            if comment_run > CONFIG["max_commented_block_lines"]:
                warnings.append(
                    f"commented-out block longer than {CONFIG['max_commented_block_lines']} lines"
                )
                break
        else:
            comment_run = 0

    return new_src, warnings


def is_pure_auth_cell(src: str) -> bool:
    lines = [l for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return False
    return all(AUTH_LINE.match(l) for l in lines)


def should_strip_outputs(src: str) -> bool:
    """True if this cell's outputs likely contain paths/creds."""
    for line in src.splitlines():
        if AUTH_LINE.match(line) or SHELL_NOISY.match(line):
            return True
    if "drive.mount" in src or "userdata" in src:
        return True
    return False


def outputs_contain_leak(cell: dict) -> bool:
    """True if rendered output text contains a Drive path — catches the case
    where a benign print() emits a path the source regex didn't classify."""
    for out in cell.get("outputs", []) or []:
        text = "".join(out.get("text") or []) + str(out.get("data", {}))
        if "/content/drive" in text or "MyDrive" in text:
            return True
    return False


def has_config_cell(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if CONFIG_CELL_MARKER in src:
                return True
    return False


def build_header_cells(rel_path: str) -> list[dict]:
    hdr = HEADERS.get(rel_path)
    if hdr is None:
        hdr = dict(
            title=Path(rel_path).stem.replace("_", " ").title(),
            section="(fill in)",
            inputs="(fill in)",
            outputs="(fill in)",
            runtime="(fill in)",
            gpu="(fill in)",
        )
    md = HEADER_MD_TEMPLATE.format(**hdr)
    return [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": md.splitlines(keepends=True),
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": CONFIG_CELL_CODE.splitlines(keepends=True),
        },
    ]


def process_notebook(path: Path, root: Path, dry_run: bool, verbose: bool) -> dict:
    rel = path.relative_to(root / "notebooks").as_posix()
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    stats = {
        "path": rel,
        "cells_total": len(nb.get("cells", [])),
        "outputs_stripped": 0,
        "cells_deleted": 0,
        "source_scrubbed": 0,
        "warnings": [],
    }

    new_cells: list[dict] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue

        src = "".join(cell.get("source", []))

        # Delete pure-auth/pure-mount cells (replace with a marker md cell)
        if is_pure_auth_cell(src):
            stats["cells_deleted"] += 1
            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "_(removed Colab Drive-mount / auth cell — "
                        "the CONFIG cell at the top of this notebook "
                        "is the local-first replacement)_\n"
                    ],
                }
            )
            continue

        new_src, warns = scrub_source(src)
        if warns:
            stats["warnings"].extend(f"{rel}:cell-{len(new_cells)}: {w}" for w in warns)
        if new_src != src:
            stats["source_scrubbed"] += 1
            cell["source"] = new_src.splitlines(keepends=True)

        # Clear execution_count for reproducible diffs
        cell["execution_count"] = None

        # Strip outputs where warranted (source-based OR output-text-based)
        if should_strip_outputs(new_src) or outputs_contain_leak(cell):
            if cell.get("outputs"):
                stats["outputs_stripped"] += 1
            cell["outputs"] = []
        # else: keep outputs as-is (DataFrames, plots, metric dicts stay)

        new_cells.append(cell)

    # Prepend CONFIG cell if absent
    if not has_config_cell({"cells": new_cells}):
        new_cells = build_header_cells(rel) + new_cells

    nb["cells"] = new_cells

    if not dry_run:
        path.write_text(
            json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    if verbose:
        print(
            f"  {rel}: {stats['cells_total']} cells, "
            f"scrubbed={stats['source_scrubbed']}, "
            f"outputs_stripped={stats['outputs_stripped']}, "
            f"deleted={stats['cells_deleted']}"
        )
        for w in stats["warnings"]:
            print(f"    WARN {w}")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = CONFIG["repo_root"]
    paths = sorted(root.glob(CONFIG["notebook_glob"]))
    if not paths:
        print(f"no notebooks found under {root / 'notebooks'}", file=sys.stderr)
        return 1

    totals = {"cells_total": 0, "outputs_stripped": 0, "cells_deleted": 0, "source_scrubbed": 0}
    all_warnings: list[str] = []
    for p in paths:
        s = process_notebook(p, root, args.dry_run, args.verbose)
        for k in totals:
            totals[k] += s[k]
        all_warnings.extend(s["warnings"])

    print(
        f"\n{len(paths)} notebooks processed"
        + (" (dry-run, no writes)" if args.dry_run else "")
    )
    print(f"  total cells:       {totals['cells_total']}")
    print(f"  source scrubbed:   {totals['source_scrubbed']}")
    print(f"  outputs stripped:  {totals['outputs_stripped']}")
    print(f"  cells deleted:     {totals['cells_deleted']}")
    if all_warnings:
        print(f"\n{len(all_warnings)} warnings:")
        for w in all_warnings:
            print(f"  {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
