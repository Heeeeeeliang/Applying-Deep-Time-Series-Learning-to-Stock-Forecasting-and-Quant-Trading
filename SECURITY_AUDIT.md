# Pre-publish security & hygiene audit

> Superseded by SECURITY_AUDIT_v2.md (2026-04-19). Preserved as a historical record of the pre-cleanup state.

**Scope.** All files under `release_repo/` (the future public GitHub release).
**Date.** 2026-04-16.
**Method.** Pattern scans + filesystem walk over the working tree.
**Git history.** **Not applicable.** `release_repo/` is not a git repository
(no `.git/` directory; no parent is a git repository either). Every
in-git-history column is therefore **N/A**. All findings below apply to the
working tree only — but because this is the pre-publish staging directory,
every finding *must* be fixed **before `git init` + first commit**, otherwise
it persists in the repo's history and the repo has to be rewritten (or thrown
away) to scrub it.

Severity bands used below:

| Band | What qualifies |
|---|---|
| **CRITICAL (C)** | Code-execution vector on reader's machine, real secret/key disclosure |
| **HIGH (H)** | Personal-path / PII leak in shipped file, operational breakage on fresh clone |
| **MEDIUM (M)** | Project-specific rule violation (brief §9): `weights_only=True`, `REGISTRY.list_all()`, LightGBM name-not-index alignment, no edits to `p01`/`p02`/`p03` |
| **LOW (L)** | File >50 MB (needs LFS/Release), file >100 MB (must be off-repo), build/OS junk (`__pycache__`, `.ipynb_checkpoints`, `.DS_Store`, `venv/`) |

---

## Executive summary

**Status:** **BLOCK.** 5 distinct findings (1 CRITICAL, 3 HIGH, 1 MEDIUM),
none LOW. The CRITICAL is a pickle RCE vector against any consumer of the
shipped checkpoints; the HIGH findings disclose the author's local Windows
layout and Google Drive structure. Repo is 8.1 MB; no files near any size
threshold; no junk/venv/cache.

| Severity | Count | Sites |
|---|---:|---|
| CRITICAL | 1 | 6 |
| HIGH | 3 | 7 + 21 + 2 = **30** lines across **10** files |
| MEDIUM | 1 finding (+ 3 project rules PASS) | 5 |
| LOW | 0 findings (3 checks PASS) | — |

**Net new vs. prior `SECURITY_AUDIT.md`:** Finding **H1** (hardcoded Windows
`E:\Uni\毕设\...` paths in shipped `.py` source) was absent from the previous
audit's findings table (only mentioned in passing in `notebooks/CLEANUP_NOTES.md`).
It is the largest surface by line-count and the most disclosive.

---

## CRITICAL findings

### C1 — `torch.load(..., weights_only=False)` — 6 sites (pickle RCE against readers)

| File | Line | Redacted evidence | In git history |
|---|---:|---|---|
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 6627 | `ckpt = torch.load(fp, map_location='cpu', weights_only=False)` | N/A |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 8124 | `strong_ckpt = torch.load(strong_path, map_location='cpu', weights_only=False)` | N/A |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 9241 | `strong_ckpt = torch.load(strong_path, map_location='cpu', weights_only=False)` | N/A |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 12304 | `ckpt = torch.load(model_path, map_location='cpu', weights_only=False)` | N/A |
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 1724 | `checkpoint = torch.load(cnn_model_path, map_location=device, weights_only=False)` | N/A |
| `tools/verify_checkpoints.py` | 66 | `state = torch.load(path, weights_only=False, map_location="cpu")` | N/A |

**Why it matters.** `weights_only=False` opts *out* of the PyTorch safe-load
path and deserialises arbitrary Python objects. A malicious `.pt` at the
named path executes attacker-controlled code on the consumer's machine at
load time. Project brief §9 explicitly requires `weights_only=True` in every
shipped example. Six sites explicitly pass `False`.

`tools/verify_checkpoints.py:66` is a *fallback branch* inside the
verification script that warns and flags the file; it is arguably
intentional for the verification pass, but it is still shipped code that
executes unsafe load. It must be gated behind an explicit CLI flag
(e.g. `--allow-unsafe`) or removed, not left on by default.

**Suggested fix.** Replace `weights_only=False` with `weights_only=True`
everywhere. If a checkpoint genuinely needs arbitrary Python objects
(optimizer state, custom classes), split it on the producer side into
`state_dict.pt` + `metadata.json` and ship the pair; do not ship a
checkpoint that requires unsafe load. For `verify_checkpoints.py:66`, put
the fallback behind `if args.allow_unsafe:` and default the flag to
`False`.

---

## HIGH findings

### H1 — Hardcoded Windows personal paths `E:\Uni\毕设\...` in shipped `.py` source — 21 lines across 7 files

Not in notebook outputs — in the **committed source** of seven Python files.
These paths disclose: author's drive letter layout, the Chinese project
folder name **毕设** ("final-year project"), a sibling local project named
`apexquant`, and dated source-archive folder names. They also make every
one of these scripts fail on any machine that is not the author's.

| File | Line | Redacted evidence | In git history |
|---|---:|---|---|
| `tools/build_checkpoint_manifest.py` | 22 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-001\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 23 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-002\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 24 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-003\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 25 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-004\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 26 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-005\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 27 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-006\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 28 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-007\毕设")` | N/A |
| `tools/build_checkpoint_manifest.py` | 29 | `Path(r"E:\Uni\毕设\final\对话\毕设-2026…-008\毕设")` | N/A |
| `tools/verify_checkpoints.py` | 30 | `ARCHIVE_PARENT = Path(r"E:\Uni\毕设\final\对话")` | N/A |
| `tools/extract_ablation_configs.py` | 16 | `"runs_root": Path(r"E:\Uni\毕设\final\apexquant\results\runs")` | N/A |
| `tools/build_ablation_table.py` | 19 | `"runs_root": Path(r"E:\Uni\毕设\final\apexquant\results\runs")` | N/A |
| `notebooks/04_backtest_and_ablation/fig_backtest_comparison.py` | 22 | `AI_CSV = Path(r"E:\Uni\毕设\final\apexquant\results\runs" …)` | N/A |
| `notebooks/04_backtest_and_ablation/fig_backtest_comparison.py` | 24 | `BASE_CSV = Path(r"E:\Uni\毕设\final\apexquant\results\runs" …)` | N/A |
| `notebooks/04_backtest_and_ablation/fig_backtest_comparison.py` | 44 | `BM_DIR = Path(r"E:\Uni\毕设\final\apexquant\data\features")` | N/A |
| `notebooks/04_backtest_and_ablation/fig_ablation.py` | 21 | `RUNS_DIR = Path(r"E:\Uni\毕设\final\apexquant\results\runs")` | N/A |
| `notebooks/04_backtest_and_ablation/fig_ablation.py` | 22 | `BM_DIR = Path(r"E:\Uni\毕设\final\apexquant\data\features")` | N/A |
| `notebooks/04_backtest_and_ablation/fig_ablation_curves.py` | 16 | `RUNS_DIR = Path(r"E:\Uni\毕设\final\apexquant\results\runs")` | N/A |
| `notebooks/04_backtest_and_ablation/fig_ablation_curves.py` | 17 | `DATA_DIR = Path(r"E:\Uni\毕设\final\apexquant\data\features")` | N/A |

**Why it matters.** `notebooks/CLEANUP_NOTES.md:32` acknowledges the class
of problem ("Windows `E:\Uni\...` paths that the regex did not catch") but
only for two notebook cells — none of the seven `.py` files above were
listed. These files are worse than a notebook output because they are the
canonical shipped code: they will be imported and stared at by any reader,
and they will crash immediately unless the reader happens to share the
author's filesystem. The repeated Chinese string **毕设** in the manifest
script also discloses cultural/regional context the author may not want
public.

**Suggested fix.** Replace each path with a configurable constant:

```python
# before
RUNS_DIR = Path(r"E:\Uni\毕设\final\apexquant\results\runs")

# after (choose one):
RUNS_DIR = Path(os.environ.get("APEX_RUNS_DIR", "./external/runs"))
# or, accept via argparse:
RUNS_DIR = Path(args.runs_dir)
```

For `tools/build_checkpoint_manifest.py`, move the eight `search_roots`
into a side file (`tools/archive_roots.local.txt`, gitignored) or accept
via `--roots` on the CLI. Both options let the reader run the script
against *their* local copy of the source archive while keeping the
author's layout out of the repo.

Before committing, re-run: `grep -rn "E:\\\\Uni\\|apexquant\\|毕设" release_repo/`
should return zero source-code hits (documentation mentions are fine).

### H2 — Google Drive / MyDrive paths embedded in notebook cell outputs — 3 notebooks, ~21 lines

`tools/clean_notebooks.py` strips outputs from cells whose **source**
matches an auth pattern, but it does not strip outputs whose **rendered
text** contains a Drive path. Several cell outputs therefore still quote
`/content/drive/MyDrive/...`.

| File | Lines | Redacted evidence (one representative per file) | In git history |
|---|---|---|---|
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 761, 779, 782, 1669, 1670, 1696, 1697, 1702, 7520 | `"  ✓ Vol LSTM saved to /content/drive/MyDrive/project_data/models/vol_lstm_v3.pt\n"` (line 761) | N/A |
| `notebooks/03_turning_point_layer/01_multiscale_cnn.ipynb` | 7518, 7527, 7536, 8306, 8461, 8603, 9477 | `"  ✓ Saved to /content/drive/MyDrive/project_data/models/cnn_dual_5min_end_point.pt\n"` (line 7518) | N/A |
| `notebooks/01_direct_prediction_ceiling/03_baseline_models.ipynb` | 11730–11736 | `"/content/drive/MyDrive/毕设/data/results/plots/fig1_da_bar.png\n"` (line 11730) | N/A |

**Why it matters.** These are shipped cell outputs. Any reader who clones
the repo sees the author's personal Drive layout (`MyDrive/project_data/…`
and `MyDrive/毕设/data/…`). The `毕设` folder again discloses regional
context.

**Suggested fix.** Two options (pick one):

1. **Preferred:** Re-execute these three notebooks on the cleaned local
   layout so their outputs reflect `./checkpoints/...` paths, not Drive.
2. **Minimal patch:** Extend `tools/clean_notebooks.py::should_strip_outputs`
   to scrub outputs whose rendered text contains `/content/drive` or
   `MyDrive`:

   ```python
   for out in cell.get("outputs", []) or []:
       text = "".join(out.get("text") or []) + str(out.get("data", {}))
       if "/content/drive" in text or "MyDrive" in text:
           return True
   ```

   Then re-run `python tools/clean_notebooks.py --verbose`.

### H3 — Live Colab `drive.mount('/content/drive')` calls in 2 notebooks — 4 lines across 2 files

The cleanup script removes cells that are **only** Colab auth. Two
notebooks have auth lines adjacent to other content, so the classifier
left them in place as executable code.

| File | Lines | Redacted evidence | In git history |
|---|---|---|---|
| `notebooks/01_direct_prediction_ceiling/01_nine_method_ceiling.ipynb` | 157 | `"from google.colab import drive\n"` | N/A |
| `notebooks/01_direct_prediction_ceiling/01_nine_method_ceiling.ipynb` | 158 | `"drive.mount('/content/drive')\n"` | N/A |
| `notebooks/01_direct_prediction_ceiling/02_missing_metrics.ipynb` | 107 | `"from google.colab import drive\n"` | N/A |
| `notebooks/01_direct_prediction_ceiling/02_missing_metrics.ipynb` | 108 | `"drive.mount('/content/drive')\n"` | N/A |

**Why it matters.** Two-part harm: (1) on any non-Colab environment the
import fails and the whole notebook will not run top-to-bottom on a fresh
clone — a Phase-4 contract violation; (2) it signals that this notebook
was authored against a personal Google account's Drive, and invites a
reader to expect Drive paths downstream.

**Suggested fix.** Wrap in a Colab guard so it is a no-op locally, or
replace with a commented fallback that matches the CONFIG pattern used
elsewhere:

```python
# Local run — use CONFIG['data_dir']. Colab fallback:
# from google.colab import drive; drive.mount('/content/drive')
```

Alternatively, move the auth pair into its own cell so the existing
`is_pure_auth_cell` classifier removes it on the next cleanup pass.

---

## MEDIUM findings

### M1 — `torch.load(...)` without an explicit `weights_only=True` arg — 5 sites (project rule)

Brief §9 mandates `weights_only=True` *explicitly*. PyTorch 2.4+ defaults
to `True`, so these are not exploitable today, but they violate the rule
and will silently regress if anyone pins an older PyTorch.

| File | Line | Redacted evidence | In git history |
|---|---:|---|---|
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 971 | `"    state = torch.load(path, map_location='cpu')\n"` | N/A |
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 2943 | `"    state = torch.load(vol_model_path, map_location=device)\n"` | N/A |
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 2986 | `"    state = torch.load(cnn_model_path, map_location=device)\n"` | N/A |
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 6691 | `"    state = torch.load(cnn_model_path, map_location=device)\n"` | N/A |
| `notebooks/04_backtest_and_ablation/01_end_to_end_pipeline.ipynb` | 7392 | `"state = torch.load(cfg.CNN_MODEL_PATH, map_location='cpu')\n"` | N/A |

**Suggested fix.** Add `weights_only=True` to every call. Line 7392 is
in a later cell that `CLEANUP_NOTES.md` did not enumerate — the cleanup
script regex (`"torch.load(" in new_src and "weights_only" not in new_src`
at `tools/clean_notebooks.py:176`) should have caught it; verify the
per-cell scan is not early-returning on the first match.

---

## PASS list (checked, clean)

| Check | Band | Evidence |
|---|---|---|
| API keys / Bearer / PEM / AWS keys / OpenAI keys / GitHub PATs / GCP keys | C | Pattern `api_key\|API_KEY\|secret\|SECRET\|Bearer \|private_key\|-----BEGIN\|sk-[a-zA-Z0-9]\|ghp_\|gho_\|github_pat_\|AKIA[0-9A-Z]\|AIza`. Only hits are: a docstring in `tools/clean_notebooks.py:12` enumerating what to scan for, a pip-resolver warning string in `03_baseline_models.ipynb:11885`, two word occurrences in `README.md:13` / `.gitignore:17`. No actual credential material. |
| Email addresses | H | Pattern `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`. Zero hits. |
| Real-name / handle leaks (`liheliang`, `李鹤亮`, `李赫亮`, OS UID `14811`, `LIHELIANG`) | H | Zero hits across all committed files. |
| `gdown.download(id=...)` form | M | Zero executable hits. Only reference is the warning string in `tools/clean_notebooks.py:175` and a diagram label `"Google Drive · gdown sync"` in `thesis/figures/fig_architecture_diagrams.py:387` (a text annotation, not a call). |
| Hardcoded model names where `REGISTRY.list_all()` should be used | M | Zero matches for `MODEL_NAME = "…"` / `MODELS = […]` enumeration patterns in shipped code. |
| LightGBM inference aligning features by **index** not **name** | M | Zero matches for `lgb.predict(X[:, N:M])` / `booster.predict(features[N])` / `.predict(…iloc[…, N:M])`. Training and inference paths use `feat_cols` lists and `feature_name=` args. `data/feature_dictionary.md` documents canonical column order. |
| Modifications to original Colab `p01` / `p02` / `p03` files | M | N/A — no `p01` / `p02` / `p03` filenames or references exist in `release_repo/`. The rule applies to the source-of-truth Colab files, which are outside this repo. |
| Files > 50 MB | L | Zero hits. Largest file: `results/figures/fig_ablation_curves.png` = 865 KB. Total repo = 8.1 MB. |
| Files > 100 MB | L | Zero hits. |
| `__pycache__` / `*.pyc` / `.ipynb_checkpoints` / `.DS_Store` / `Thumbs.db` | L | Zero hits. |
| Virtual-env directories (`venv/`, `.venv/`, `env/`) | L | Zero hits. |
| `.gitignore` covers cache, venvs, large artifacts, source archives, OS junk, credentials | — | Yes. See `.gitignore:1-41`. |

---

## Informational (not security, flagged for Phase-5 cleanup)

### INFO-1 — Unreplaced placeholder strings in 6 user-visible files

Not a security issue, but these must be replaced before the first public
push or the README will ship with literal `<USER>/<REPO>` strings and
CITATION with `<FAMILY-NAME>` / `<GIVEN-NAMES>`. 21 occurrences across
7 files.

| File | Lines with placeholders |
|---|---|
| `CITATION.cff` | 6, 7, 11, 31, 32, 35 |
| `README.md` | 7, 78, 79, 146, 148, 150, 276 |
| `quickstart.sh` | 8, 9, 11 |
| `checkpoints/README.md` | 13 |
| `data/README.md` | 65 |
| `tools/build_checkpoint_manifest.py` | 36 |

Enumeration: `grep -rn '<[A-Z][A-Z_-]*>' release_repo/ | grep -vE '<TICKER>|<FREQ>|<OPTIONAL'`
— the `<TICKER>` / `<FREQ>` placeholders in data-schema docs are intended
documentation tokens and should be left alone.

---

## Severity counts & recommendation

| Severity | Findings | Affected sites |
|---|---:|---:|
| CRITICAL | 1 | 6 |
| HIGH | 3 | ≈30 |
| MEDIUM | 1 (+3 project-rule checks PASS) | 5 |
| LOW | 0 (3 checks PASS) | 0 |
| INFO | 1 | 21 |

**Recommendation: BLOCK.**

Do not push to a public remote in this state. The CRITICAL + HIGH findings
together expose the author's Windows filesystem, Google Drive layout, and
a pickle-deserialisation RCE vector against every downstream consumer. All
five must be fixed; then re-run this audit (or wire the scans into
`tools/security_audit.sh`) and confirm zero CRITICAL / HIGH before
`git init`.

Because the repo is not yet under version control, fixes applied now leave
no history behind. Delay any `git init` until at least CRITICAL and HIGH
are resolved — otherwise the disclosures persist as historical commits
and require a history rewrite to remove.

## Out of scope for this audit

- **Databento redistribution licence.** The data card claims permission;
  not verified with Databento legal here.
- **Thesis PDF metadata.** `thesis/thesis.pdf` is not yet in the tree;
  when added, scrub with `exiftool` (Word / LaTeX templates routinely
  leak author name and OS username in PDF metadata).
- **Checkpoint NaN/Inf sanity.** `tools/verify_checkpoints.py` exists but
  has not been run in this pass — CRITICAL-C1 affects it anyway.
- **Figma SVG embed.** `README.md` references an SVG that ships as PDF/PNG;
  fix the link or export the SVG.
