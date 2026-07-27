# orth_basis_b — `_orthogonal_univariate_fe/` package internals (device-born scoring, dedup, extra-basis detectors)

This cluster is the second of two covering `_orthogonal_univariate_fe/` (the first, `orth_basis_a.md`, covers
the sibling `hermite_fe/` pair-optimiser cluster and misidentifies its own file list as this package — see its
own note). This package is the **default-on production entry point** for univariate orthogonal-basis FE inside
`MRMR.fit` (`fe_univariate_basis_enable=True` by default; `hybrid_orth_mi_fe_with_recipes` is on the auto-wired
path) — already flagged by this same wave's `x_efficiency_architecture.md` (X_EFFICIENCY_ARCHITECTURE-3/4) as a
13-file, ~5.5k-LOC subpackage that fell through every other cluster's scope because its own module docstring
(`__init__.py:37-40`) falsely claims "NOT wired into MRMR.fit by default". This report covers the 13 assigned
files: the package facade (`__init__.py`), the four device-born "SF1/SF1b/SF1c :311-H2D-collapse" resident MI
scorers (`_uplift_univariate_resident.py`, `_extra_basis_resident.py`, `_gpu_resident_cross_basis.py`,
`_orth_gpu_resident.py`), the fit-scoped scoring/dedup memos (`_orth_scoring_memo.py`, `_orth_dedup.py`), the
bench-rejected opt-in imbalance-aware MI (`_imbalance_mi.py`), the batch-MI backend dispatcher
(`_orth_mi_backends.py`), the extra-basis (spline/Fourier/chirp/wavelet) generator pair
(`_orth_extra_basis_fe.py` + `_orth_extra_basis_fe_generate.py`), the pair cross-basis generator
(`_orth_pair_cross_fe.py`), and the GPU-resident Fourier-frequency detector twin
(`_fourier_detect_gpu_resident.py`). Angle 9 (SQL/HTTP/UI) confirmed N/A — no such surface anywhere in this
cluster. Angle 7 (security): no `eval`/`exec`/`subprocess`/`pickle.load`/`yaml.load`; every GPU/device path is
try/except-guarded with a documented host fallback.

Engineering quality is unusually high throughout: nearly every device-born resident scorer carries an explicit
parity/selection-equivalence contract, a documented bench-rejection trail (REJECTED != DELETED convention
honored), and detailed "why" comments. The residual findings below are narrow and mostly P2/P3 — this cluster's
biggest issue (X_EFFICIENCY_ARCHITECTURE-3/4, the semi-supervised aux-pool GPU-resident gap and the false
"not wired" docstring) was already raised by the cross-cutting report and is not re-litigated here beyond a
one-line status note.

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| ORTH_BASIS_B-1 | (cross-ref) | docs | `__init__.py:37-40` | "NOT wired into MRMR.fit by default" is false (`fe_univariate_basis_enable` defaults True). | Already raised as X_EFFICIENCY_ARCHITECTURE-4 (P1); not re-scored here. |
| ORTH_BASIS_B-2 | (cross-ref) | cpu_gpu_parity | `_orth_gpu_resident.py` vs `__init__.py:334-342` | GPU-resident univariate builder never consults the semi-supervised unlabeled pool the host builder fits basis-preprocess params on. | Already raised as X_EFFICIENCY_ARCHITECTURE-3 (P1); not re-scored here. |
| ORTH_BASIS_B-3 | P2 | thread-safety | `_fourier_detect_gpu_resident.py:50-81` (`_SPLIT_MASK_CACHE`, `_SUBSAMPLE_IDX_CACHE`) | Two module-level plain `dict`s memoize the seeded held-out split masks / row-subsample indices, read via `.get()` then written via `[key] =` with no lock and no eviction — the exact "hand-rolled unlocked module-level cache" pattern the cross-cutting report's X_EFFICIENCY_ARCHITECTURE-5 catalogued 12 instances of and empirically reproduced a crash for (`dictionary changed size during iteration`, `KeyError`) at one sibling site. This file was not in that finding's list of 12. Currently low-reachability (the whole module is gated behind `MLFRAME_FE_GPU_STRICT_RESIDENT`, default OFF, and joblib worker processes each get their own dict so cross-process races don't apply) — but any future in-process thread-parallel FE-family fan-out over this resident path would hit the same read-then-write race the reproduced instance did. Values are read-only after insertion (never mutated in place), so a torn read cannot corrupt a *value*, only race on the insert itself (worst case: harmless duplicate recompute, or in CPython's dict implementation a concurrent resize during iteration elsewhere in the same dict — the reproduced crash class). | NEW; not named by `x_efficiency_architecture.md`'s 12-site list, not raised by 2026-07-20 (this GPU twin post-dates it) or `orth_basis_a.md`. |
| ORTH_BASIS_B-4 | P2 | architecture / fragility | `_orth_extra_basis_fe.py:896-900` and `_orth_extra_basis_fe_generate.py:36-47` | The two files import from each other at MODULE SCOPE (not lazily/in-body like every other cross-file dependency in this package): `_orth_extra_basis_fe.py`'s bottom re-exports `generate_extra_basis_features` / `hybrid_orth_extra_basis_fe_with_recipes` / `_build_recipe_from_meta` FROM `_orth_extra_basis_fe_generate.py`, which itself imports 9 names FROM `_orth_extra_basis_fe.py` at its own top. This resolves today only because every current caller triggers the import via `_orth_extra_basis_fe.py` first (`__init__.py` reaches it indirectly through `_orth_gpu_resident.py:16`'s `from ._orth_extra_basis_fe import _is_int_as_cat_axis`, itself imported before `__init__.py`'s own direct import at line 837) — by the time `_orth_extra_basis_fe.py` executes its bottom-of-file import, all 9 names `_orth_extra_basis_fe_generate.py` needs are already bound in the partially-initialized module. Any future caller that imports `_orth_extra_basis_fe_generate` FIRST (e.g. a new sibling doing `from ._orth_extra_basis_fe_generate import generate_extra_basis_features` directly, or a reordering of `_orth_gpu_resident.py`'s imports) would hit `ImportError: cannot import name 'generate_extra_basis_features' from partially initialized module` — the exact class of bug CLAUDE.md's monolith-split rule warns about ("AST-audit the sibling for unresolved names before commit... a moved function/class... can NameError at first call if it references a parent-module name with no matching import"), except manifesting as an import-order-dependent ImportError rather than a NameError. Grepped all current callers (`_fe_auto_escalation.py`, two `_benchmarks/` scripts, `_orth_gpu_resident.py`, `__init__.py`) — none import `_orth_extra_basis_fe_generate` directly today, so the bug is currently latent, not live. | NEW. |
| ORTH_BASIS_B-5 | P3 | docs | `_uplift_univariate_resident.py:7`, `_extra_basis_resident.py:7`, `_gpu_resident_cross_basis.py:7-10`, `_orth_mi_backends.py` (comment sites throughout) | All three device-born resident-scorer modules' docstrings cite a fixed line number `_orth_mi_backends.py:311` as "the" host H2D upload site their device twin collapses. In the CURRENT file that exact line (311) sits inside an unrelated `bench-attempt-rejected` prose comment block; the actual host `cp.asarray(X)` upload the docstrings describe is now at line 347 (`Xd = cp.asarray(np.ascontiguousarray(np.asarray(X, dtype=np.float64)))`). The file has evidently grown since these citations were written (the same drift pattern the cross-cutting report found in docstring "wired" claims) — a maintainer trying to verify the claim by opening `_orth_mi_backends.py:311` today lands on the wrong code. Harmless (the mechanism description is still correct, only the pinned line number is stale) but repeated in 3+ places, so worth fixing in one pass. | NEW. |
| ORTH_BASIS_B-6 | P2 | efficiency (documented, no action needed) | `_orthogonal_univariate_fe/__init__.py:600,608,673,680,824` (`hybrid_orth_mi_fe` / `hybrid_orth_mi_fe_with_recipes` no-candidate paths) | `return X.copy(), scores` / `return X.copy(), pd.DataFrame(...)` on every no-engineered-column early exit — 4 sites in this package alone. Confirmed part of the 39-file, module-wide pattern the cross-cutting report's X_EFFICIENCY_ARCHITECTURE-2 already catalogued and proposed fixing generically (the caller's `_appended`-then-conditional-merge convention never reads the returned frame on the no-op path). Listed here only so the disposition table for THIS cluster's files is complete; not a new finding, no separate fix needed beyond that proposal's module-wide sweep. | Covered by X_EFFICIENCY_ARCHITECTURE-2 (P1, cross-cutting); this is 4 of its 39 sites. |
| ORTH_BASIS_B-7 | P3 | test-coverage | `_orth_scoring_memo.py` (whole file, 159 LOC) | Zero test hits for `orth_scoring_memo_scope`, `cached_raw_mi_baseline`, or `cached_dense_finite_corr_matrix` anywhere under `tests/` (grep-confirmed). The module's own docstring claims parity with `dedup_collinear_memo_scope()` (`_orth_dedup.py`), which itself DOES have dedicated tests (`test_dedup_source_cols_perf.py`, `test_orth_dedup_dense_block_backend_dispatch.py`) — this memo's cache-key correctness (the `_col_hash`/`_content_hash` keying, the nesting-safe scope reuse, the "outside an active scope this is unconditionally a fresh call" fallback) has no regression pin at all. Low severity because the memo is a pure optional performance layer with an explicit fallback to the unmemoized call on any miss/absence, so a cache-key collision would only silently return a STALE value for a differently-keyed column under an active scope — worth a parity test (memoized vs unmemoized result, byte-identical) given the correctness-adjacent nature of any content-hash cache. | NEW; not flagged by `orth_basis_a.md` (out of its scope) or the 2026-07-20 audit (module postdates it, per its 2026-06-21 mirrors-comment). |
| ORTH_BASIS_B-8 | P3 | test-coverage | `_fourier_detect_gpu_resident.py` (whole file, 334 LOC) | Zero test hits for `detect_fourier_freqs_for_col_gpu` (grep-confirmed) — no GPU-vs-CPU selection-equivalence parity test exists for this specific detector twin, unlike most other resident twins in this cluster which each cite a named parity test in their docstrings (`test_device_born_cross_basis_parity.py`, `test_extra_basis_device_born_parity.py`, `test_resident_311_residual_parity.py`). The module's own docstring explicitly frames itself as "RESIDENCY COMPLETENESS... not a wall win" and default-OFF (`MLFRAME_FE_GPU_STRICT_RESIDENT`), which is presumably why it fell outside the other twins' parity-test convention — but that convention exists precisely to pin selection-equivalence claims like this file's own ("the returned frequency list matches the CPU detector within the coarse-grid tolerance"), and right now that claim is unverified by any automated test. | NEW; this specific gap not named in the 2026-07-20 `gpu_residency.md` roadmap's proposal #3 (which lists 4 other untested `device_born_*` flags but not this one, since it is a detector twin rather than a `device_born_*`-flagged family). |

## Test coverage summary (angle 6)

Coverage is strong for the higher-traffic files: `_orth_dedup.py` has 3 dedicated test files (perf regression +
2 GPU backend-dispatch tests); the device-born resident scorers collectively have 3 named parity test files
(`test_device_born_cross_basis_parity.py`, `test_extra_basis_device_born_parity.py`,
`test_resident_311_residual_parity.py`); `_imbalance_mi.py` has its own dedicated regression file
(`test_imbalance_mi.py`, cited by its own docstring); the pair-cross generator has biz_value coverage
(`test_cross_basis_pair_orthpoly.py`) and a read-only-column regression test shared with the triplet/quad
family. The two files flagged above (`_orth_scoring_memo.py`, `_fourier_detect_gpu_resident.py`) are the only
genuine zero-coverage gaps found in this cluster's 13 files; both are low-severity (pure optional perf layer /
residency-completeness twin with an explicit CPU fallback on any fault), not correctness-critical production
defaults.

## Proposals

1. **Add a `LockedBoundedCache`-style guard (or at minimum a `threading.Lock`) around `_SPLIT_MASK_CACHE` /
   `_SUBSAMPLE_IDX_CACHE`** (closes ORTH_BASIS_B-3): cheapest as an adoption of the cross-cutting report's
   proposed shared primitive (X_EFFICIENCY_ARCHITECTURE-5's proposal #5) once it exists; until then a bare
   `threading.Lock()` around the get-or-insert in `_seeded_split_masks`/`_seeded_subsample_idx` costs nothing
   on the common single-thread path and removes the latent race.
2. **Break the `_orth_extra_basis_fe.py` <-> `_orth_extra_basis_fe_generate.py` circular top-level import**
   (closes ORTH_BASIS_B-4): move the bottom-of-file re-export in `_orth_extra_basis_fe.py` into a lazy
   in-function import (matching the "lazy-imported in-body to avoid a cycle" convention every other cross-file
   dependency in this package already follows, per both files' own docstrings), or invert the split so
   `_orth_extra_basis_fe_generate.py` is a pure leaf module with no back-import. Add a one-line smoke test that
   imports `_orth_extra_basis_fe_generate` directly (not via the package `__init__`) to lock in the fix and
   catch any future reintroduction.
3. **Fix the stale `_orth_mi_backends.py:311` line citation** (closes ORTH_BASIS_B-5) across the 3+ docstrings
   that reference it — either drop the specific line number and describe the site by function name
   (`_mi_classif_batch`'s host-input `cp.asarray` branch), which won't go stale on future edits, or add a
   grep-based CI check (per the cross-cutting report's proposal #4, extended to catch stale line-number
   citations generally, not just "not wired" claims).
4. **Add a `orth_scoring_memo_scope` parity test** (closes ORTH_BASIS_B-7): assert `cached_raw_mi_baseline` /
   `cached_dense_finite_corr_matrix` return byte-identical results with and without an active memo scope on a
   small fixture, plus a nesting-scope test (inner scope reuses outer cache).
5. **Add a `detect_fourier_freqs_for_col_gpu` parity test** (closes ORTH_BASIS_B-8): mirror the pattern already
   used by the 3 other resident-twin parity tests in this cluster — force the resident flag on, compare the
   returned frequency list against the CPU `_detect_fourier_freqs_for_col` on the same fixture within the
   documented coarse-grid tolerance.
