#!/usr/bin/env python3
"""
Walk the source archives and produce a single CSV manifest of every model
checkpoint in the project, with size, training context, and proposed
publication target (GitHub Release vs. HuggingFace).

Threshold: anything ≥ HF_THRESHOLD_BYTES goes to HuggingFace, the rest
ships as a GitHub Release artifact bundle.

Output: ../checkpoints/MANIFEST.csv
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Author-local source archives containing the training checkpoints — not
# shipped with this research release. Point APEXQUANT_SOURCE_ARCHIVES at the
# parent directory that holds the 8 Drive-exported shard folders
# (毕设-20260311T013659Z-3-001 … -008), or edit the default below.
# Names match the `source_archive` column of checkpoints/MANIFEST.csv.
_SHARD_PARENT = Path(os.environ.get("APEXQUANT_SOURCE_ARCHIVES", "source_archives"))
_SHARD_NAMES = [f"毕设-20260311T013659Z-3-{i:03d}" for i in range(1, 9)]

CONFIG = {
    "search_roots": [_SHARD_PARENT / name / "毕设" for name in _SHARD_NAMES],
    "extensions":          {".pt", ".joblib", ".pkl"},
    "hf_threshold_bytes":  100 * 1024 * 1024,    # 100 MB
    "out_csv":             Path(__file__).resolve().parents[1] / "checkpoints" / "MANIFEST.csv",
    # Track files of identical size+name as duplicates; keep first occurrence.
    "dedupe_by_size_and_name": True,
    "hf_repo":             "<USER>/quant-thesis-checkpoints",
    "release_tag":         "v1.0.0",
}

LAYER_HINTS = [
    ("multiscale_cnn",      "layer2_turning_point"),
    ("cnn_top",             "layer2_turning_point"),
    ("cnn_bottom",          "layer2_turning_point"),
    ("vol_lstm",            "layer1_volatility"),
    ("vol_multiscale_lstm", "layer1_volatility"),
    ("lightgbm",            "layer1_volatility"),
    ("lgb_top",             "layer3_trade_filter"),
    ("lgb_bottom",          "layer3_trade_filter"),
    ("lgb_v2",              "layer3_trade_filter"),
    ("attention_lstm",      "ceiling_baseline"),
    ("transformer_lstm",    "ceiling_baseline"),
    ("vanilla_lstm",        "ceiling_baseline"),
    ("attn_lstm_direction", "ceiling_baseline"),
    ("timesfm",             "ceiling_baseline"),
    ("multistock_phase1",   "end_to_end_pipeline"),
    ("lstm_1min",           "exploratory"),
    ("lstm_regression_1min","exploratory"),
]

COLUMNS = [
    "filename", "layer", "ticker_or_scope", "size_bytes", "size_human",
    "publish_target", "hf_path", "release_asset",
    "source_archive", "source_relpath", "sha1_short",
    "training_date", "notes",
]


def classify_layer(name: str) -> str:
    n = name.lower()
    for key, layer in LAYER_HINTS:
        if key in n:
            return layer
    return "unknown"


def extract_ticker(name: str) -> str:
    for t in ("AAPL", "MSFT", "NVDA", "QQQ", "SPY", "TSLA", "GOOGL", "GOOG"):
        if f"_{t}." in name or f"_{t}_" in name or name.endswith(f"_{t}.pt"):
            return t
    return "all"


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def sha1_short(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        # Hash only the first 8 MB — enough to identify duplicates
        # without scanning multi-GB files end to end.
        data = f.read(8 * chunk)
        h.update(data)
    return h.hexdigest()[:12]


def main() -> int:
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for root in CONFIG["search_roots"]:
        if not root.exists():
            print(f"missing search root: {root}", file=sys.stderr)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in CONFIG["extensions"]:
                key = (p.name, p.stat().st_size)
                if CONFIG["dedupe_by_size_and_name"] and key in seen:
                    continue
                seen.add(key)
                size = p.stat().st_size
                layer = classify_layer(p.name)
                ticker = extract_ticker(p.name)
                target = "huggingface" if size >= CONFIG["hf_threshold_bytes"] else "github_release"
                hf_path = (
                    f"{layer}/{p.name}" if target == "huggingface" else ""
                )
                release_asset = (
                    f"checkpoints-{layer}.tar.gz" if target == "github_release" else ""
                )
                rows.append({
                    "filename":         p.name,
                    "layer":            layer,
                    "ticker_or_scope":  ticker,
                    "size_bytes":       size,
                    "size_human":       human_size(size),
                    "publish_target":   target,
                    "hf_path":          hf_path,
                    "release_asset":    release_asset,
                    "source_archive":   root.parent.name,
                    "source_relpath":   p.relative_to(root).as_posix(),
                    "sha1_short":       sha1_short(p),
                    "training_date":    datetime.fromtimestamp(
                        p.stat().st_mtime, tz=timezone.utc
                    ).date().isoformat(),
                    "notes":            "",
                })

    # Write manifest
    CONFIG["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    with CONFIG["out_csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # Summary
    by_target: dict[str, list[dict]] = {"huggingface": [], "github_release": []}
    for r in rows:
        by_target[r["publish_target"]].append(r)

    print(f"wrote {CONFIG['out_csv']}")
    print(f"  {len(rows)} unique checkpoints (deduped by name + size)")
    for t, items in by_target.items():
        total = sum(r["size_bytes"] for r in items)
        print(f"  {t}: {len(items)} files, {human_size(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
