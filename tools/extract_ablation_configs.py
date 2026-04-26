#!/usr/bin/env python3
"""
Extract one YAML per ablation run from each metrics.json (which embeds the
config) under the source RUNS_ROOT, and write them to ../configs/ablation/.

Run 11 (ai_full_trail) is named 'run11_ai_full_trail.yaml' and is the
canonical configuration referenced from the README and CITATION.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG = {
    # Local source of the 2026-03-16 ablation run set (outside this repo —
    # produced by the ApexQuant platform, not shipped with this research
    # release). Override with APEXQUANT_RUNS_DIR or edit the default.
    "runs_root":  Path(os.environ.get("APEXQUANT_RUNS_DIR", "runs")),
    "prefix":     "20260316_140416_",
    "out_dir":    Path(__file__).resolve().parents[1] / "configs" / "ablation",
}

RUNS = [
    ( 1, "ema_rsi_baseline"),
    ( 2, "ema_rsi_with_sizing"),
    ( 3, "ai_layer2_only"),
    ( 4, "ai_no_vol_gate"),
    ( 5, "ai_vol_gate_only"),
    ( 6, "ai_baseline"),
    ( 7, "ai_signal_reversal"),
    ( 8, "ai_vol_collapse"),
    ( 9, "ai_tranche_exit"),
    (10, "ai_full_tranche"),
    (11, "ai_full_trail"),
]


def to_yaml(d: dict, indent: int = 0) -> str:
    """Tiny YAML emitter — avoids a PyYAML dependency for ~50 lines of config."""
    out_lines = []
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            out_lines.append(f"{pad}{k}:")
            out_lines.append(to_yaml(v, indent + 1))
        elif isinstance(v, list):
            out_lines.append(f"{pad}{k}:")
            for item in v:
                if isinstance(item, dict):
                    out_lines.append(f"{pad}- ")
                    out_lines.append(to_yaml(item, indent + 1))
                else:
                    out_lines.append(f"{pad}- {item!r}")
        elif isinstance(v, bool):
            out_lines.append(f"{pad}{k}: {str(v).lower()}")
        elif v is None:
            out_lines.append(f"{pad}{k}: null")
        elif isinstance(v, (int, float)):
            out_lines.append(f"{pad}{k}: {v}")
        else:
            out_lines.append(f"{pad}{k}: {v}")
    return "\n".join(out_lines)


def main() -> int:
    CONFIG["out_dir"].mkdir(parents=True, exist_ok=True)
    for run_num, suffix in RUNS:
        src = CONFIG["runs_root"] / f"{CONFIG['prefix']}{suffix}" / "metrics.json"
        if not src.exists():
            print(f"missing: {src}", file=sys.stderr)
            return 1
        m = json.loads(src.read_text())
        config = {
            "run_number": run_num,
            "is_final":   run_num == 11,
            "config":     m.get("config"),
            "display_name": m.get("display_name"),
            "description":  m.get("description"),
            "fee_structure": m.get("fee_structure", {}),
            "ablation":      m.get("ablation", {}),
            "overrides":     m.get("overrides", {}),
        }
        out = CONFIG["out_dir"] / f"run{run_num:02d}_{suffix}.yaml"
        marker = "  ← FINAL" if run_num == 11 else ""
        out.write_text(
            f"# Ablation run {run_num}{marker}\n"
            f"# {config['display_name']}\n\n" + to_yaml(config) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {out.name}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
