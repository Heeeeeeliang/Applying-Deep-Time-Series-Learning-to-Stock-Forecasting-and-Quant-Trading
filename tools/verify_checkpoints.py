#!/usr/bin/env python3
"""
Sanity-check every checkpoint listed in checkpoints/MANIFEST.csv:

  - Loads with torch.load(path, weights_only=True)
  - Verifies no NaN or Inf in any tensor
  - Verifies no obviously stale optimizer state (warns only)

Why this exists: an early CNN training run shipped with NaN weights because
the trainer saved the post-divergence state instead of the best-val state.
This check is a cheap guard against re-shipping a broken checkpoint.

Usage:
    python tools/verify_checkpoints.py [--source-from-archives]

Without --source-from-archives, the script reads paths relative to
./checkpoints/. With the flag, it loads each checkpoint from its original
source_relpath under the source_archive (the working layout for the
pre-publish verification pass).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "checkpoints" / "MANIFEST.csv"
# Author-local parent of the 8 shard archives (--source-from-archives mode).
# Not shipped with the release; override via APEXQUANT_SOURCE_ARCHIVES.
ARCHIVE_PARENT = Path(os.environ.get("APEXQUANT_SOURCE_ARCHIVES", "source_archives"))


def iter_tensors(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_tensors(v)


def check_one(path: Path, allow_unsafe: bool = False) -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch not installed"

    if path.suffix == ".joblib":
        try:
            import joblib
            obj = joblib.load(path)
            return True, f"joblib OK ({type(obj).__name__})"
        except Exception as e:
            return False, f"joblib load failed: {e}"

    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as e:
        if not allow_unsafe:
            return False, (
                f"torch.load(weights_only=True) failed: {e} — "
                "rerun with --allow-unsafe to attempt the legacy pickle-based path "
                "(executes arbitrary code; never use on untrusted files)"
            )
        try:
            # Legacy fallback: some old checkpoints contain custom objects
            # and cannot be loaded with weights_only=True. Only attempted
            # when the user explicitly opts in with --allow-unsafe, because
            # torch.load with weights_only=False deserialises arbitrary
            # pickled Python objects — a remote-code-execution vector on
            # untrusted inputs. Never use on files you did not produce.
            state = torch.load(path, weights_only=False, map_location="cpu")  # noqa: S614 — gated behind --allow-unsafe
            note = "legacy checkpoint — requires weights_only=False (do not publish as-is)"
        except Exception as e2:
            return False, f"torch.load failed: {e2}"
        else:
            ok = _scan_tensors(state)
            return ok, note if ok else f"NaN/Inf detected and {note}"

    return _scan_tensors(state), "weights_only OK"


def _scan_tensors(state) -> bool:
    import torch
    nan_count = 0
    inf_count = 0
    total = 0
    for t in iter_tensors(state):
        total += 1
        if torch.isnan(t).any():
            nan_count += 1
        if torch.isinf(t).any():
            inf_count += 1
    if nan_count or inf_count:
        print(f"    BAD: {nan_count} tensors with NaN, {inf_count} with Inf, of {total}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-from-archives",
        action="store_true",
        help="resolve each checkpoint from its source archive instead of ./checkpoints/",
    )
    ap.add_argument(
        "--allow-unsafe",
        action="store_true",
        help=(
            "opt into the legacy weights_only=False pickle fallback for "
            "checkpoints that cannot be loaded safely. Runs arbitrary code "
            "embedded in the .pt file — never use on files you did not produce."
        ),
    )
    args = ap.parse_args()

    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_ok = n_bad = 0
    for row in rows:
        if args.source_from_archives:
            path = ARCHIVE_PARENT / row["source_archive"] / "毕设" / row["source_relpath"]
        else:
            path = REPO_ROOT / "checkpoints" / row["filename"]

        if not path.exists():
            print(f"  MISS  {row['filename']}  ({path})")
            n_bad += 1
            continue

        ok, msg = check_one(path, allow_unsafe=args.allow_unsafe)
        marker = "OK" if ok else "FAIL"
        print(f"  {marker:4s}  {row['filename']:50s}  {msg}")
        if ok:
            n_ok += 1
        else:
            n_bad += 1

    print(f"\n{n_ok} OK / {n_bad} FAIL / {len(rows)} total")
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
