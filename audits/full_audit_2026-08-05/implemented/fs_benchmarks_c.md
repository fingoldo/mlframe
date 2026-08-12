# Audit report: cluster `fs_benchmarks_c`

## Scope

- The `*.py` files directly inside `src/mlframe/feature_selection/_benchmarks/` (not its subdirectories),
  sorted alphabetically, **last third by count** (files 134-200 of 200, i.e. `bench_polynomial_bases.py`
  through `profile_wellbore_mrmr_only_100k.py`) — 67 files.
- All files under `src/mlframe/feature_selection/_benchmarks/kernel_tuning_cache/` — 6 files
  (`__init__.py`, `auto_tune.py`, `_auto_tune_sweeps_a.py`, `_auto_tune_sweeps_b.py`, `cli.py`, `dispatch.py`).
- All files under `src/mlframe/feature_selection/_benchmarks/wide_data_scaling/` — 4 `.py` files
  (`_progress_shared.py`, `h1_bench.py`, `h1_gpu_large.py`, `h2_bench.py`); `RESULTS.md` and
  `raw_progress_2026-06-18.txt` are documentation/data artifacts, not code, and were not reviewed as code.
- `results/` and `_results/` under `_benchmarks/` contain only JSON/data artifacts — skipped per instructions.
- `filters/**` (MRMR engine) and `shap_proxied_fs/**` are out of scope (separate, already-closed audit); this
  report only reviews *benchmark/profiling harness* code that lives in `_benchmarks/`, even where that code
  imports from the out-of-scope packages.

**Files reviewed: 77**
**LOC reviewed: 10,975** (7,618 in the last-third slice + 2,973 in `kernel_tuning_cache/` + 384 in
`wide_data_scaling/`)

## Summary by dimension

1. **Correctness bugs**: no crashes/wrong-formula bugs found in the benchmark logic itself. One
   deterministic-crash robustness bug found in `wide_data_scaling/_progress_shared.py` (FS_BENCHMARKS_C-1).
2. **ML correctness (leakage/reproducibility/etc.)**: none found. Every A/B and honest-eval bench in this
   slice correctly splits train/holdout, seeds its RNGs, and states its metric is OOS/honest where relevant
   (e.g. `bench_rfecv_importance_agg.py`, `bench_shap_interaction_proxy.py`, `bench_warp_linear_tiebreak.py`).
3. **Computational efficiency**: no unnecessary `.copy()`/O(n^2) issues found in-scope; the reviewed files
   are themselves benchmark harnesses measuring efficiency of production code (out of scope), not vice versa.
4. **Edge cases / robustness**: the hardcoded-path crash (FS_BENCHMARKS_C-1) and the dev-machine-specific
   `sys.path` inserts (FS_BENCHMARKS_C-3) are the only robustness gaps found.
5. **Test coverage gaps**: not flagged as a dimension issue — these are dev bench/profile scripts, not
   library code with a test-coverage contract; each performs its own inline bit-identity/oracle assertions
   (e.g. `bench_rankgauss_replay_iter110.py`, `profile_cluster_aggregate.py::_bitident_check`), which is the
   correct pattern for this kind of file.
6. **Code quality / architecture**: dead/superseded legacy sweep functions left exported after a documented
   migration (FS_BENCHMARKS_C-2); one incompletely-applied refactor (FS_BENCHMARKS_C-9); a fragile
   identity-keyed cache (FS_BENCHMARKS_C-10).
7. **OSS/hygiene**: repeated `Wave N (date)` process-metadata comments violating the project's own
   comment-style rule (FS_BENCHMARKS_C-4); four separate stale-docstring/doc-drift issues
   (FS_BENCHMARKS_C-5, -6, -7, -8).

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|---|---|---|---|---|---|
| FS_BENCHMARKS_C-1 | P1 | `wide_data_scaling/_progress_shared.py:8,13` | `ck()` opens a hardcoded `D:/Temp/synergy_scale_bench/progress.txt` for append with no directory creation; crashes with `FileNotFoundError` on any host/session where that exact folder doesn't already exist, mid-sweep, in both `h1_bench.py` and `h2_bench.py`. | Add `os.makedirs(os.path.dirname(PROG), exist_ok=True)` once at module import (or lazily inside `ck()`), and make `PROG` overridable via an env var instead of a hardcoded `D:` path. | Grep-based scanner: any module-level string literal used as the first arg to `open(..., "a")`/`open(..., "w")` where the containing function has no `os.makedirs`/`Path.mkdir` call reachable before it. |
| FS_BENCHMARKS_C-2 | P2 | `kernel_tuning_cache/_auto_tune_sweeps_a.py:937` (`ensure_batch_pair_mi_tuning`), `_auto_tune_sweeps_b.py:145,366,473` (`ensure_cat_fe_perm_kernel_tuning`, `ensure_unary_elementwise_tuning`, `ensure_rff_matmul_tuning`) | Four legacy `ensure_*_tuning`/`_run_sweep_*` pairs remain fully defined and re-exported in `auto_tune.__all__` after their kernels were migrated to the new `pyutilz.performance.kernel_tuning` registry (`cli.py`'s own `_refresh_via_new_registry` docstring explains the legacy sweep "wrote regions without a backend_choice / code_version, which silently shadowed the new dispatcher"). Nothing stops a future caller/test from importing and calling the legacy `ensure_batch_pair_mi_tuning` etc. directly, re-introducing exactly that documented collision. | Delete the four superseded sweep bodies (or turn them into thin `raise NotImplementedError("migrated to pyutilz registry; use cli.py's refresh-<kernel>")` stubs) now that the CLI no longer calls them, instead of leaving working-but-dangerous code reachable. | Meta-test: for every kernel name registered via the new `kernel_tuner(kernel_name=...)` registry, grep the codebase for any *other* function that calls `KernelTuningCache.update("<same kernel_name>", ...)` directly — flag any kernel name written by two different code paths. |
| FS_BENCHMARKS_C-3 | P2 | `wide_data_scaling/h1_bench.py:10-11`, `h1_gpu_large.py:9-10`, `h2_bench.py:10` | All three scripts do `sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/src")` — a dev-machine-specific absolute path inserted at position 0. If that path exists on a given machine but holds a stale/different mlframe checkout, it silently takes priority over the properly pip-installed/editable package for the whole process, matching the "stale package shadowing" bug class already documented elsewhere in this project's memory notes. | Drop the hardcoded insert; rely on the package being installed (editable or otherwise) like every other script in `_benchmarks/`, or derive the path via `Path(__file__).resolve().parents[N]`. | Grep scanner: `sys\.path\.(insert|append)\(.*r?["\']([A-Z]:[\\/]|/home/|/Users/)` — any absolute, drive-lettered or hardcoded-home path literal passed to `sys.path` mutation. |
| FS_BENCHMARKS_C-4 | P3 | `kernel_tuning_cache/cli.py:47,90`; `auto_tune.py:139`; `_auto_tune_sweeps_a.py:401,529,892` | Multiple `# Wave N (date): ...` / `# WAVE N: ...` comments embed process/audit-phase metadata (wave numbers + dates) directly in source, which this project's own CLAUDE.md explicitly forbids ("No process/audit metadata in code comments: no phase/wave markers ... date stamps ... that belongs in git history"). | Reword each comment to state the current WHY only (e.g. "drop the redundant isfile precheck; try-open directly" with no wave/date prefix); the history stays in git log. | Regex scanner over all first-party comments: `#.*\bWAVE\s*\d+\b` (case-insensitive) — already effectively the rule this project states it wants enforced; wire it into the code_audit comment-hygiene checker. |
| FS_BENCHMARKS_C-5 | P3 | `bench_pr4_methods.py:67,84-86` | `_run_knockoffs`'s docstring says the FDR threshold uses "the proper Barber-Candes FDR-controlled threshold (default q=0.2)", but its actual parameter is `fdr_q: float = 0.5`, and the underlying `_bc_threshold(W, q: float = 0.1)` has yet a third, independent default (0.1) that is never actually used since every call site passes `q=fdr_q` explicitly. Three different numbers for what reads as one policy value. | Pick one canonical default (the one actually exercised, 0.5), fix the docstring to match it, and either drop `_bc_threshold`'s unused 0.1 default or make it mirror the real default so a bare call isn't silently different from every real call site. | A docstring/signature cross-checker: extract `default=...` / `q=0.2`-style numeric claims from docstrings via regex and diff them against the actual `inspect.signature` default of the same-named parameter. |
| FS_BENCHMARKS_C-6 | P3 | `collision_census.py:14-16` vs `69-71` | Module docstring's example usage shows `[--n-features 100 --max-order 2]` as CLI flags, but the actual `argparse.ArgumentParser` in `main()` only defines `--out` — those flags do not exist and passing them would error out. | Either implement `--n-features`/`--max-order` (overriding the hardcoded `grid` list) or remove them from the docstring's usage example. | Doc-vs-CLI checker: for scripts using `argparse`, extract every `--flag` token appearing inside the top-of-file docstring's fenced/indented "Run::" block and diff against `parser.add_argument` calls in the same file; flag any docstring flag with no matching `add_argument`. |
| FS_BENCHMARKS_C-7 | P3 | `profile_wellbore_mrmr_only_100k.py:11-13` | The docstring's `nsys profile`/`ncu` example commands both invoke `profile_wellbore_mrmr_only.py`, but the actual module is `profile_wellbore_mrmr_only_100k.py` — copy/paste-run instructions reference a file that doesn't exist. | Update the two example commands to the real filename. | Doc-vs-filename checker: for any docstring code block containing a `.py` token, verify it matches `os.path.basename(__file__)` when the surrounding sentence clearly refers to "this script". |
| FS_BENCHMARKS_C-8 | P3 | `kernel_tuning_cache/cli.py:10-27` vs `338-345` | The module docstring's subcommand list stops at `refresh-discretize-2d-array` / `refresh-all`; it omits `refresh-batch-mi-noise-gate` and `refresh-fe-gpu-pairs-mi`, both of which are fully implemented (`_cmd_refresh_batch_mi_noise_gate`, `_cmd_refresh_fe_gpu_pairs_mi`), registered as subparsers, folded into `refresh-all`, and present in the final dispatch dict. | Add the two missing subcommands to the docstring's list. | Doc-vs-CLI checker (see FS_BENCHMARKS_C-6's idea): diff the docstring's enumerated subcommand names against `sub.add_parser(...)` calls in the same file; flag any subparser with no docstring entry. |
| FS_BENCHMARKS_C-9 | P3 | `bench_usability_corr_f32_f64.py:53,70` (use) vs `70` (definition) | Module-level `_f64_cache: dict = {}` is defined at the very bottom of the file (after `main()`, which already references it via `globals()` lookup at call time) and is keyed by `id(y)` — correctness here depends entirely on `pairs` (built once and held for the whole `main()` call) keeping every `y` array alive so its `id()` can't be reused by an unrelated object. It happens to be correct today, but is a fragile, easy-to-break pattern if the pair list is ever streamed/released early. | Move `_f64_cache` to the top of the file (or make it a local `dict` passed explicitly between the two dtype passes instead of a module global), and key by loop index instead of `id(y)`. | Grep scanner: `id\(` used as a dict/set key anywhere in the codebase — flag every hit for manual review, since identity-as-cache-key is only ever safe under an explicit, provable object-lifetime guarantee. |
| FS_BENCHMARKS_C-10 | P3 | `wide_data_scaling/h1_gpu_large.py:11-16` vs `_progress_shared.py:1-16` | `h1_gpu_large.py` re-implements the `PROG`/`ck()` progress-log helper inline instead of importing the shared `wide_data_scaling._progress_shared.ck` that `h1_bench.py`/`h2_bench.py` already use — defeating the stated purpose of that shared module ("independently duplicated across those scripts, consolidated here so a fix can't silently drift out of sync across copies"). The drift has already happened: the inline copy formats timestamps as `%H:%M:%S` while the shared helper uses `%Y-%m-%d %H:%M:%S`. | Delete the inline `PROG`/`ck` in `h1_gpu_large.py` and import from `_progress_shared` like its two siblings. | AST-similarity scanner across sibling files in the same directory: hash small (3-15 line) function bodies and flag near-duplicate definitions of the same function name across files in one package — exactly what would have caught this immediately after the "consolidation" claim. |

## Counts

- P0: 0
- P1: 1
- P2: 2
- P3: 7

## Narrative

**FS_BENCHMARKS_C-1** (`_progress_shared.py`). `ck()` is the sole progress-logging mechanism for the two
multi-minute sweep scripts (`h1_bench.py` iterates over 6 widths x 3 seeds x up to 3 backends; `h2_bench.py`
iterates over 3 leak-levels x 5 seeds x up to 5 criteria) and is called after *every* inner iteration.
`PROG = r"D:/Temp/synergy_scale_bench/progress.txt"` is a bare absolute path with no `os.makedirs` anywhere
in the file or its two call sites. On any machine (or any session on the same machine after `D:/Temp` gets
cleaned) where `D:/Temp/synergy_scale_bench/` does not already exist, the very first `ck()` call raises
`FileNotFoundError`, aborting the whole sweep before a single result prints. I verified via `Grep` that no
other file in the repo creates this directory. Rated P1 rather than P2 because — unlike a rare edge case —
the failure mode is the *default* state for any environment other than the one where this exact folder was
manually created once.

**FS_BENCHMARKS_C-2** (`kernel_tuning_cache/_auto_tune_sweeps_a.py` / `_b.py`). `cli.py`'s
`_refresh_via_new_registry` docstring states plainly that the legacy sweep for `batch_pair_mi` /
`cat_fe_perm_kernel` / `unary_elementwise` / `rff_matmul` "wrote regions without a backend_choice /
code_version, which silently shadowed the new dispatcher" — i.e. this is a *known, previously-live* bug
class for these four kernels. I confirmed via `Grep` that the CLI itself was fixed (it now routes all four
through `_refresh_via_new_registry`, bypassing the legacy `ensure_*_tuning` entirely), but the legacy
`_run_sweep_batch_pair_mi`/`ensure_batch_pair_mi_tuning` and its three siblings are still fully implemented,
still exported through `auto_tune.__all__`, and I found no call site anywhere in `src/` or `tests/` besides
their own module and the (now-bypassing) `__all__` re-export. This is exactly the kind of landmine that gets
re-triggered by a well-intentioned future change (a new test calling the "obvious" `ensure_batch_pair_mi_tuning`
name, or a revert of the CLI fix) — the fix for the *symptom* (the CLI) shipped, but the *cause* (two writers
for one cache key) was never removed.

**FS_BENCHMARKS_C-3** (`wide_data_scaling/h1_bench.py`, `h1_gpu_large.py`, `h2_bench.py`). All three scripts
insert a hardcoded `D:/Upd/Programming/PythonCodeRepository/mlframe/src` at `sys.path[0]` before importing
`mlframe`. `sys.path.insert` on a non-existent path is a silent no-op, so on most machines this merely adds
dead weight — but on any machine where that path *does* exist (e.g. a colleague who happens to use the same
drive letter and directory convention, or a stale second checkout on the original author's own machine after
this worktree moved), it takes priority over the correctly pip-installed/editable `mlframe`, silently running
whatever code sits at that path instead. This is the same class of bug already called out in this project's
own memory notes ("stale installed pyutilz crashed every mlframe import") applied to a different package.

**FS_BENCHMARKS_C-4** through **FS_BENCHMARKS_C-10** are documentation-drift and hygiene findings, each
verified by direct comparison of the docstring/comment text against the actual code in the same file (cited
line numbers). FS_BENCHMARKS_C-4 (`Wave N` markers) is a repeated pattern I found via `Grep` across 3 of the
6 `kernel_tuning_cache` files — I cited representative lines rather than every occurrence to avoid padding
the table, but the meta-test (a regex over all first-party comments) would catch every instance in one pass.
FS_BENCHMARKS_C-10 is the only finding in this batch with directly-observable behavioral drift (not just
staleness): the inline `ck()` copy in `h1_gpu_large.py` uses a different timestamp format than the "shared"
helper it was supposed to replace, proving the consolidation the shared module's own docstring claims did
not fully happen.

## Dimensions with zero findings

- **Data leakage / OOF boundary violations**: none. Every honest-eval bench in scope (`bench_rfecv_*.py`,
  `bench_shap_*.py`, `bench_warp_linear_tiebreak.py`, `bench_supervised_fs_default.py`,
  `bench_stability_orchestration.py`) correctly fits on train-only data and scores on a held-out split the
  selector never saw, and the code/comments are explicit about which number is val/OOF/test-honest.
- **Reproducibility / unseeded RNG**: every synthetic-data generator in scope seeds its
  `np.random.default_rng(...)` explicitly; no bare `np.random.rand()`/global-state RNG use found.
- **Mutable default arguments**: none found (all function defaults across the 77 files are `None`, `str`,
  `int`, `float`, or immutable tuples).
- **GPU/CPU dispatch correctness**: the `kernel_tuning_cache` dispatch logic (`dispatch.py`) correctly
  distinguishes `None` (no verdict) from empty/stale results, gates online relearning behind an explicit
  env var, and documents non-obvious footguns in its own comments (e.g. the `dims={...}` vs `**dims` trap
  in `lookup_pairwise_corr_backend`) rather than actually containing that bug.
- **Class-imbalance / calibration correctness**: not directly exercised by this cluster's benchmarks in a
  way that revealed a bug; where imbalance is deliberately part of a bed (e.g.
  `bench_rfecv_importance_agg.py`'s `imbalanced` scenario), `class_weight="balanced"` is used correctly.

## Report path

`C:/Users/Admin/Machine learning/mlframe/audits/full_audit_2026-08-05/fs_benchmarks_c.md`
