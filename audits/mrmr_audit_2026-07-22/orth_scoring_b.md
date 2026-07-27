# orth_scoring_b -- audit report (2026-07-22)

## Scope

This cluster covers the "second half" of the orthogonal-poly scorer zoo: the parametric
pre-selectors (Lasso, Elastic Net), the meta-scorer fingerprint/rule-cascade dispatcher, the
per-column bootstrap-LCB auto-scorer and its rank-fusion-ensemble sibling, the conditional
basis-routing FE, the multi-way cross-basis families (triplet / quadruplet, plus their split-out
recipe-builder siblings), the per-cluster shared-basis FE, the highly-correlated-pair diff-basis
FE, and the Param-Oracle-backed unified scorer selector (`_oracle_scorer_select.py`). All of
these are opt-in `MRMR.fit()` FE layers (never on by default) that append engineered
`orth_*`-family columns and back them with `EngineeredRecipe`s for leakage-free
`MRMR.transform()` replay. None of the 13 files touch a database, network socket, or browser-facing
surface -- confirmed by direct read of every file (no SQL/HTTP/UI angle applies to this cluster).

This is a re-audit of a cluster the 2026-07-20 16-agent audit already covered
(`c7b_orth_scoring_fe.md`, 10 findings). Commits `f067e0d44` (Fix B-17: preprocess_params
freeze) and `2cc59a6b1` (Fix B-18: y-densification) landed on top of that audit and fixed most
of the affected files in this list (`_orthogonal_elasticnet_fe.py`, `_orthogonal_lasso_fe.py`,
`_orthogonal_routing_fe.py`, `_orthogonal_triplet_fe.py` (+ its new `_orthogonal_triplet_fe_recipes.py`
sibling), `_orthogonal_quadruplet_fe.py` (+ its new `_orthogonal_quadruplet_fe_recipes.py`
sibling), `_orthogonal_cluster_basis_fe.py`, `_orthogonal_diff_basis_fe.py`) -- all verified
fixed by direct code read below. The new finding this pass surfaces is that a **monolith split**
(`b48a48de6`, 2026-06-06, carving `_orth_auto_scorer_fe.py` out of `_orthogonal_scorer_auto_fe.py`
under the 900-LOC guideline) happened *before* the B-17/B-18 fix commits, and the fix-propagation
pass on 2026-07-20 grepped/patched the *parent* file's remaining (ensemble) function but missed
the *carved-out sibling's* function entirely -- both bug classes are confirmed still live in
`_orth_auto_scorer_fe.py`, reproduced empirically below. A genuinely new concurrency bug was also
found and reproduced in `_oracle_scorer_select.py`'s module-level rows cache.

## Findings

| ID | Severity | Category | File:Line | Summary | Prior-audit status |
|----|----------|----------|-----------|---------|---------------------|
| ORTH_SCORING_B-1 | P1 | bug | `_orth_auto_scorer_fe.py:64-65` | `_score_plug_in` coerces non-integer `y` via bare `y_arr.astype(np.int64)` (truncates, does not densify) -- the exact B-18 bug class, reintroduced because this function was carved into a new sibling file (`b48a48de6`, 2026-06-06) *before* the 2026-07-20 B-18 fix pass, which patched `_orthogonal_scorer_auto_fe.py` (its parent) but never followed the split into this sibling. Empirically reproduced: `y=[0.1,0.2,...]` perfectly separated by `x` scores MI=0.693 via the correct densifying path but MI=0.0 through this function (both classes truncate to `0`). Corrupts Layer-68's per-column scorer choice (`select_best_scorer_per_column`) AND, via `_oracle_scorer_select.OracleScorerSelector.benchmark_all_scorers` (which imports this same function transitively), persists a permanently-wrong `quality=0.0` for `plug_in` into the on-disk Param-Oracle store for any dataset with a low-cardinality float-encoded target. | NEW (post-split blind spot of fixed prior finding B-18) |
| ORTH_SCORING_B-2 | P1 | bug | `_orth_auto_scorer_fe.py:518-583` | `hybrid_orth_mi_auto_scorer_fe_with_recipes` never freezes the fit-time basis-preprocess params into the emitted `orth_univariate` recipes (`build_orth_univariate_recipe(name=..., src_name=..., basis=..., degree=...)` at line 579 has no `preprocess_params=` kwarg and the function doesn't even import `_evaluate_basis_column`) -- the exact B-17 bug class. Confirmed still open by direct code read AND by `git show f067e0d44 --stat`, which touched `_orthogonal_scorer_auto_fe.py` (8 lines, fixing only the ensemble function that stayed in the parent) but never touched `_orth_auto_scorer_fe.py` -- the same split-before-fix blind spot as B-1. `MRMR.transform()` on a row-sliced or distribution-shifted test frame will silently refit the z-score/min-max axis instead of replaying it, per the B-17 root-cause analysis. No test in the suite exercises row-sliced replay parity for this function (`test_biz_value_mrmr_auto_scorer_selection.py`'s pickle-roundtrip test only asserts recipe `basis`/`degree` keys are preserved, never a value-level replay-vs-fit check). | NEW (post-split blind spot of fixed prior finding B-17) |
| ORTH_SCORING_B-3 | P1 | concurrency / thread-safety | `_oracle_scorer_select.py:87-111` | `_cached_read_rows`'s module-level `_ROWS_CACHE` dict is read/evicted/written with three separate, unsynchronized operations (`len(...)` check, `_ROWS_CACHE.pop(next(iter(_ROWS_CACHE)))`, `_ROWS_CACHE[key] = rows`) with no lock. Empirically reproduced under concurrent access (32 threads x 500 calls each against distinct fake stores, `sys.setswitchinterval` tightened to force interleaving): 225 crashes, a mix of `RuntimeError: dictionary changed size during iteration` (two threads both call `next(iter(_ROWS_CACHE))` while a third mutates it between the two) and `KeyError` (`.pop(key)` on a key another thread already evicted). Under the *default* Python switch interval a 3200-call stress test produced 0 crashes, so the failure needs real thread contention (e.g. a joblib-parallel grid search fitting several `MRMR` models concurrently against enough distinct dataset fingerprints to fill/evict the 8-slot cache, or a store whose `read_rows()` I/O is slow enough to widen the window) -- reported as P1 (confirmed-reachable, contention-gated) rather than P0 (always-crashes). No thread-safety test exists (`test_oracle_scorer_rows_cache.py` only covers single-threaded mtime invalidation). | NEW |
| ORTH_SCORING_B-4 | P2 | dead_code | `_orthogonal_scorer_auto_fe.py:175-197, 240-254` | The nested `_call()` helper's `"plug_in"` and `"copula"` branches are unreachable: `_BATCHABLE = {"plug_in", "copula"}` means any scorer name in that set is always routed through `_batch_scores` instead, so `_call("plug_in", ...)` / `_call("copula", ...)` are dead code paths (both call sites guard with `if s in _batch_scorers: ... else: _call(...)`, and those two names are always in `_batch_scorers` whenever requested). Not a correctness bug today, but a maintenance foot-gun -- a future refactor that removes a name from `_BATCHABLE` would silently start exercising previously-untested code. | NEW |
| ORTH_SCORING_B-5 | P2 | inefficiency / dead_code | `_orthogonal_cluster_basis_fe.py:234-235` | `detect_clusters_by_correlation`'s `max_cluster_size` truncation branch computes `dense_names[dense_names.index(name)]` for each surviving member -- this is a no-op (`dense_names.index(name)` finds the position of `name` in the list, then re-indexes to get `name` back), so it's an O(p) linear-scan re-lookup that produces exactly the value already in hand (`name`). Harmless (only triggers on clusters exceeding `max_cluster_size=20`, and `p` is bounded by the numeric-column count of the DCD/correlation-cluster candidate pool) but should just be `members = sorted(name for (_, name) in mean_corr[: int(max_cluster_size)])`. | NEW |
| ORTH_SCORING_B-6 | P2 | code_quality | multiple files, e.g. `_orthogonal_elasticnet_fe.py:345`, `_orthogonal_lasso_fe.py` (same pattern), `_orthogonal_triplet_fe.py:649,679`, `_orthogonal_quadruplet_fe.py:609,641`, `_orthogonal_diff_basis_fe.py:584`, `_orthogonal_cluster_basis_fe.py:298,326,463,592,673` | Pervasive process/audit-metadata-in-comments across nearly every file in this cluster: `"mrmr_audit_2026-07-20 B-17: ..."`, `"REPLAY-FIDELITY FIX (2026-06-13): ..."` (repeated verbatim in 4+ files), `"2026-06-03 (audit cluster-aggregate-6/7): ..."` (repeated 5x in one file), `"W6 abs-MAD floor instrumentation"`. CLAUDE.md explicitly bans this class of comment ("no phase/wave markers, finding IDs, date stamps ... that belongs in git history / the PR description") and calls it a "repeated complaint." Cosmetic only -- flagged because it's systemic across this whole cluster, not a one-off, and the fix commits that introduced most of these are recent enough that the convention violation is still fresh (not legacy debt to grandfather in). | NEW (style-convention violation, not previously called out in c7b) |
| ORTH_SCORING_B-7 | P2 | bug (exception swallowing) | `_oracle_scorer_select.py:122-126` | `_quality_objective`'s `except Exception: q = float("nan")` has no logging and is broader than the module's own numeric-error conventions elsewhere in the cluster; a malformed bake-off output (e.g. a future refactor changing the closure's return shape) silently persists `quality=NaN` into the on-disk Param-Oracle store with zero diagnostic trace. | Still open (prior report c7b, unfixed) |
| ORTH_SCORING_B-8 | P2 | bug (cache correctness) | `_oracle_scorer_select.py:100-103` | `_cached_read_rows`'s staleness key is `(store_path, os.path.getmtime(path))`; on a coarse-mtime filesystem, two writes within the same resolution window collide on cache key and the second write's rows are invisible until a later write changes mtime. Low real-world impact on Windows NTFS (100ns resolution). | Still open (prior report c7b, unfixed) |
| ORTH_SCORING_B-9 | P2 | bug (exception swallowing) | `_orthogonal_meta_scorer_fe.py:208-210, 276-284` | `fingerprint_signal`'s inner per-column Pearson/Spearman/symmetric-Pearson probes still use bare `except Exception: r = float("nan")` (3 sites), which is broader than the module's own declared `_NUMERIC_ERRORS` convention used by every *outer* try/except in the same function, and contradicts the module docstring's explicit invariant that a genuine programming error must propagate. | Still open (prior report c7b, unfixed) |
| ORTH_SCORING_B-10 (informational, no action) | -- | verification | `_orthogonal_elasticnet_fe.py`, `_orthogonal_lasso_fe.py`, `_orthogonal_routing_fe.py`, `_orthogonal_triplet_fe.py`/`_orthogonal_triplet_fe_recipes.py`, `_orthogonal_quadruplet_fe.py`/`_orthogonal_quadruplet_fe_recipes.py`, `_orthogonal_cluster_basis_fe.py`, `_orthogonal_diff_basis_fe.py` | Confirmed FIXED (not re-listed as findings): B-17 (missing `preprocess_params` freeze) and B-18 (truncating y-coercion) are both resolved in all 7 of these files via direct code read -- each now calls `_evaluate_basis_column(..., return_params=True)` (or threads through an already-computed `basis_params`) before building its recipe, and each uses a proper `_coerce_y_int64` densify-via-`np.unique(return_inverse=True)` helper. | RESOLVED, see commits `f067e0d44` / `2cc59a6b1` |

## CPU/GPU parity

`_orthogonal_triplet_fe.py` / `_orthogonal_quadruplet_fe.py` are the only two files in this
cluster with an optional GPU-resident scoring path (`raw_and_product_mi_resident`, gated by
`_crossbasis_device_born_on()`). Both callers fall back to the exact host path (`None` return)
on any cupy absence/failure/non-strict-mode, matching the documented contract; no new parity gap
found in the caller-side code owned by this cluster (the resident kernel itself lives in
`_orthogonal_univariate_fe/_gpu_resident_cross_basis.py`, outside this file list, and was already
covered by the 2026-07-20 audit's `c4_gpu_infra` / `gpu_residency` reports). Every other file in
this cluster (elasticnet, lasso, meta_scorer, scorer_auto, auto_scorer, routing, cluster_basis,
diff_basis, oracle_scorer_select) is host-only with no GPU path at all -- consistent with the
prior audit's "GPU-resident scoring exists only for cross-basis families" design finding, still
true today.

## Module size

All 13 files are within the ~800-900 LOC guideline: `_orthogonal_elasticnet_fe.py` 359,
`_orthogonal_lasso_fe.py` 449, `_orthogonal_meta_scorer_fe.py` 649, `_orthogonal_scorer_auto_fe.py`
681, `_orth_auto_scorer_fe.py` 583, `_orthogonal_routing_fe.py` 584, `_orthogonal_triplet_fe.py`
698, `_orthogonal_triplet_fe_recipes.py` 104, `_orthogonal_quadruplet_fe.py` 659,
`_orthogonal_quadruplet_fe_recipes.py` 116, `_orthogonal_cluster_basis_fe.py` **809**,
`_orthogonal_diff_basis_fe.py` 655, `_oracle_scorer_select.py` 370. Only
`_orthogonal_cluster_basis_fe.py` is close to the ceiling and worth splitting (e.g. carve
`detect_clusters_by_correlation` + `compute_cluster_aggregate` into a
`_orthogonal_cluster_basis_fe_helpers.py` sibling) before the next feature lands on it, mirroring
the `_orth_auto_scorer_fe.py` precedent -- see Proposals.

## mypy

`mypy --cache-dir=.mlframe_mypy_cache_shared --ignore-missing-imports` on all 13 files: **0
errors**. All parameters use `Optional[T] = None` correctly, return annotations match actual
returns, and no bare `object`/implicit-Optional patterns were found.

## Security

Confirmed no DB/network/UI surface in any of the 13 files (grep for
`eval\(|exec\(|subprocess|os\.system|pickle\.load|yaml\.load` across the whole cluster: zero
hits). `OracleScorerSelector.__getstate__`/`__setstate__` round-trip only `store_path` +
scalar config (no array data), consistent with the Param-Oracle's stat-only design. The one
path-shaped input in this cluster, `OracleScorerSelector(store_path=...)`, is passed straight to
`ParamOracle(store_path, ...)` with no validation in these files; `mlframe.utils._param_oracle`
(outside this file list) is where `default_store_dir` resolution/sanitisation would need to live
-- flagged for the owning cluster's attention, not re-audited here since the resolution code
itself isn't in this file set.

## Test coverage

Reasonable coverage overall: `test_orthogonal_scorer_bugfixes.py` pins 6 bug classes across
elasticnet/lasso/hsic/ksg/adaptive_degree/meta_scorer/copula/dcor/cmim/tc;
`test_biz_value_mrmr_auto_scorer_selection.py` has 6 scorer-selection contract classes (dCor/
plug-in/copula per-signal-family wins, AUC lift, pickle/clone); `test_param_oracle_scorer_unify.py`
and `test_oracle_scorer_rows_cache.py` cover the oracle unification and its read-cache. Gaps
confirmed:
* No test exercises `_orth_auto_scorer_fe.py`'s `_score_plug_in` with a genuinely fractional
  low-cardinality `y` (every fixture in `test_biz_value_mrmr_auto_scorer_selection.py` builds
  `y` via `.astype(int)` on a boolean condition) -- this is precisely why ORTH_SCORING_B-1 was
  never caught.
* No replay-parity (fit-vs-transform-on-a-row-slice) test exists for
  `hybrid_orth_mi_auto_scorer_fe_with_recipes` specifically (the class of test that
  `test_orth_cluster_basis_replay_parity.py` / `test_orth_triplet_quad_replay.py` provide for the
  already-fixed families) -- this is precisely why ORTH_SCORING_B-2 was never caught. This
  reaffirms the prior c7b report's P1 test-gap finding, now narrowed to the one site that is
  still actually broken.
* No concurrency/thread-safety test exists for `_ROWS_CACHE` (ORTH_SCORING_B-3).

## Proposals

1. **Fix ORTH_SCORING_B-1 / B-2 at the root**: add a shared `_coerce_y_int64` import (reuse the
   sibling files' helper rather than re-copy-pasting it) to `_orth_auto_scorer_fe.py`'s
   `_score_plug_in`, and thread `_evaluate_basis_column(..., return_params=True)` through
   `hybrid_orth_mi_auto_scorer_fe_with_recipes` exactly as the other 7 already-fixed files do.
   Given this is the SECOND time a monolith split has silently un-done a fix-propagation pass
   (the first time being the original 16-way scorer-zoo duplication the 2026-07-20 audit found),
   consider a grep-based CI gate (as the prior report's proposal already suggested) that also
   greps newly-carved sibling files, not just files present at audit time.
2. **Fix ORTH_SCORING_B-3**: guard `_ROWS_CACHE`'s read-check-evict-write sequence with a
   `threading.Lock()` (cheap; this path is not a hot loop) or switch to a bounded
   `collections.OrderedDict` with `move_to_end`/`popitem(last=False)` under the same lock for a
   true LRU instead of insertion-order eviction.
3. Add a `test_no_file_over_1k_loc.py`-style *proactive* 700-LOC soft-warning tier (distinct from
   the current 900-1000 hard gate) so a file like `_orthogonal_cluster_basis_fe.py` (809 lines)
   gets flagged for a split BEFORE it crosses the hard limit under the next feature addition,
   rather than only after.
4. Once ORTH_SCORING_B-1 is fixed, add the missing fractional-y regression test directly to
   `test_auto_scorer_batch_identity.py` or `test_orthogonal_scorer_bugfixes.py` (parametrize
   over `_score_plug_in` the same way the existing `_coerce_y_*` parametrized test already does
   for the sibling scorer families), and add `_orth_auto_scorer_fe`'s
   `hybrid_orth_mi_auto_scorer_fe_with_recipes` to the row-slice replay-parity suite alongside
   the triplet/quadruplet/cluster-basis families.
