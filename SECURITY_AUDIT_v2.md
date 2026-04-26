# Pre-publish security & hygiene audit — v2

**Scope.** All files under `apexquant_data_release/` (the future public GitHub release).
**Date.** 2026-04-19.
**Method.** Pattern scans + filesystem walk over the working tree, same rules as the v1 audit. All checks re-run after the C1 / H1 / H2 fixes.
**Git history.** **Not applicable.** `apexquant_data_release/` is not a git repository (no `.git/`; no parent is a git repository either). Every in-git-history column is therefore **N/A**. Findings below apply to the working tree only — but because this is the pre-publish staging directory, every finding *must* be fixed **before `git init` + first commit**, otherwise it persists in the repo's history and the repo has to be rewritten (or thrown away) to scrub it.

Severity bands:

| Band | What qualifies |
|---|---|
| **CRITICAL (C)** | Code-execution vector on reader's machine, real secret/key disclosure |
| **HIGH (H)** | Personal-path / PII leak in shipped file, operational breakage on fresh clone |
| **MEDIUM (M)** | Project-specific rule violation (§9 brief): `weights_only=True`, `REGISTRY.list_all()`, LightGBM name-not-index alignment, no edits to `p01`/`p02`/`p03` |
| **LOW (L)** | File >50 MB (needs LFS/Release), file >100 MB (must be off-repo), build/OS junk (`__pycache__`, `.ipynb_checkpoints`, `.DS_Store`, `venv/`) |

---

## Executive summary

**Status:** **OK** — all blockers from the v1 audit (2026-04-16) have been closed. No CRITICAL, HIGH, MEDIUM, or LOW findings remain. Repo size 92 MB / 150 files / no files near the 50 MB or 100 MB thresholds / no junk dirs.

| Severity | Count | Change vs. v1 |
|---|---:|---|
| CRITICAL | **0** | was 1 (C1 — pickle RCE, 6 sites) → now 0 |
| HIGH | **0** | was 3 (H1 — Windows paths; H2 — `/content/drive` in outputs) → now 0 |
| MEDIUM | **0** | was 1 → now 0 |
| LOW | **0** | was 0 → still 0 |

---

## CRITICAL findings — **CLEAR**

### C1 — `torch.load(..., weights_only=False)` — **RESOLVED**

**Prior state (v1).** 6 sites loaded `.pt` checkpoints with `weights_only=False`, allowing arbitrary pickle code execution on the consumer's machine at load time.

**Current state.**

| File | Line | Current evidence |
|---|---:|---|
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 6627 | `ckpt = torch.load(fp, map_location='cpu', weights_only=True)` |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 8124 | `strong_ckpt = torch.load(strong_path, map_location='cpu', weights_only=True)` |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 9241 | `strong_ckpt = torch.load(strong_path, map_location='cpu', weights_only=True)` |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 12304 | `ckpt = torch.load(model_path, map_location='cpu', weights_only=True)` |
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 1724 | `checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=True)` |
| `tools/verify_checkpoints.py` | 75 | `state = torch.load(path, weights_only=False, map_location="cpu")  # noqa: S614 — gated behind --allow-unsafe` |

The 5 notebook sites were flipped to `weights_only=True`. The one remaining `weights_only=False` in `tools/verify_checkpoints.py` is a legacy-checkpoint fallback that is now only reachable when the operator explicitly passes `--allow-unsafe` on the CLI; the default run path raises with a helpful error if a checkpoint cannot load safely. The `--allow-unsafe` flag is documented with a warning about RCE risk.

Verification:

```
$ grep -rln "weights_only=False" apexquant_data_release/ \
    --exclude=SECURITY_AUDIT.md --exclude=verify_checkpoints.py
(no output)
```

---

## HIGH findings — **CLEAR**

### H1 — hardcoded Windows `E:\Uni\毕设\final\apexquant\…` paths — **RESOLVED**

**Prior state (v1).** Absolute paths to the author's local checkout leaked across ≥10 shipped `.py` / `.ipynb` files. These paths disclosed the author's local Windows layout and would 404 on a fresh clone.

**Current state.** All 7 files now read their paths from environment variables with sensible relative defaults. New env vars (documented at each call site):

| Env var | Default | Used by |
|---|---|---|
| `APEXQUANT_RUNS_DIR` | `runs` | `fig_ablation.py`, `fig_ablation_curves.py`, `fig_backtest_comparison.py`, `tools/build_ablation_table.py`, `tools/extract_ablation_configs.py` |
| `APEXQUANT_DATA_DIR` | `data/features` | `fig_ablation.py`, `fig_ablation_curves.py`, `fig_backtest_comparison.py` |
| `APEXQUANT_SOURCE_ARCHIVES` | `source_archives` | `tools/build_checkpoint_manifest.py`, `tools/verify_checkpoints.py` (`--source-from-archives` mode) |

Verification:

```
$ grep -rln 'Path(r"E:\' apexquant_data_release/ \
    --exclude=SECURITY_AUDIT.md --exclude=CLEANUP_NOTES.md
(no output)
$ grep -rln "E:/Uni" apexquant_data_release/ \
    --exclude=SECURITY_AUDIT.md --exclude=CLEANUP_NOTES.md
(no output)
```

The 19 remaining `E:\` byte matches across the notebooks are all Python identifiers ending in `E` followed by `:\n` line-continuations inside ipynb JSON source strings (`TIMESFM_AVAILABLE:`, `PATIENCE:`, `CNN_PATIENCE:`, `LSTM_PATIENCE:`, `VOL_PATIENCE:`, `SAMPLE:`). Not real path leaks.

### H2 — `/content/drive/MyDrive/…` personal Drive paths in notebook OUTPUTS — **RESOLVED**

**Prior state (v1).** 23 notebook output-cell lines leaked the author's Colab Drive layout, e.g. `/content/drive/MyDrive/project_data/models/…` and `/content/drive/MyDrive/毕设/data/…`.

**Current state.** All 23 hits surgically redacted via a regex that strips `/content/drive/MyDrive/<first-path-segment>/` from each matching text field in `cell["outputs"]` (stream `text`, `data["text/*"]`, and `traceback`). Source cells, markdown cells, and `cell["metadata"]` untouched. Outputs like `✓ Saved to /content/drive/MyDrive/project_data/models/cnn_dual.pt` now read `✓ Saved to models/cnn_dual.pt` — meaning preserved, personal prefix gone.

Verification:

```
$ python -c "... walk ipynb outputs, count /content/drive/MyDrive/ ..."
residual /content/drive/MyDrive in output cells: 0
```

Remaining `/content/drive` occurrences in the release are intentional and documented:

- 6 commented-out `data_dir` CONFIG fallback lines in notebook cover cells — exactly the pattern the Phase 4 brief asked for ("with a commented Colab fallback").
- 2 `drive.mount('/content/drive')` lines — standard Colab idiom, not author-specific.
- `notebooks/README.md:35` — documents the cover-cell pattern with the generic placeholder `MyDrive/thesis_data` (the author-specific `MyDrive/毕设/data` string has been replaced).
- 5 references inside `tools/clean_notebooks.py` — this file IS the path-rewrite tool; the references are its own regex pattern and example strings.

---

## MEDIUM findings — **CLEAR**

No project-rule violations remain. All earlier-flagged issues involving feature-alignment (name-not-index), REGISTRY usage, and `p01`/`p02`/`p03` edits were already PASS in the v1 audit or are not applicable to files in this research release.

---

## LOW findings — **CLEAR** (all 4 checks pass)

| Check | Result |
|---|---|
| Any file >50 MB | **0** (largest is `data/splits/TSLA_1min/train.csv` at 8.6 MB) |
| Any file >100 MB | **0** |
| `__pycache__/` directories tracked | **0** |
| `.ipynb_checkpoints/` / `.DS_Store` / `venv/` / `.venv/` tracked | **0** |

---

## Other pattern sweeps

None of the common secret / token patterns appear in shipped code:

| Pattern | Result |
|---|---|
| `AIza…` (Google API keys) | CLEAN |
| `ghp_…` (GitHub classic PAT) | CLEAN |
| `sk-ant-…` (Anthropic API key) | CLEAN |
| `sk-proj-…` (OpenAI API key) | CLEAN |
| `AKIA…` (AWS access key) | CLEAN |
| `xoxb-…` (Slack bot token) | CLEAN |
| `-----BEGIN` (PEM / OpenSSH / RSA) | CLEAN |
| `hf_…` | **false-positive only** — the 2 hits are `hf_path` column header in `checkpoints/MANIFEST.csv` and `hf_threshold_bytes`/`hf_repo` config keys in `tools/build_checkpoint_manifest.py`. Not tokens. |

---

## Status

**READY for `git init` from a security-and-hygiene standpoint.** Remaining work (non-audit):

- the data-licence block has been paste-applied (A1 complete)
- other PART A items per the task list (A3 README fixes, A3e `thesis/` untrack, A3f CITATION.cff, A4 final verification)

The old `SECURITY_AUDIT.md` in this directory is preserved as a historical record of the pre-cleanup state, with a supersession note linking to this v2 file.
