# Audit: fs_boruta_root

**Cluster:** `fs_boruta_root`
**Scope:** `src/mlframe/feature_selection/boruta_shap/**` (package + `_benchmarks/`), plus the loose `*.py`
files directly inside `src/mlframe/feature_selection/` (excluding `filters/`, `shap_proxied_fs/`,
`wrappers/`, `_benchmarks/`). `filters/**` and `shap_proxied_fs/**` are OUT OF SCOPE (separate, closed
audit cycle 2026-07-25) and were not reviewed even where referenced from in-scope files.

**Files reviewed:** 34 (27 loose files in `feature_selection/` + 7 files under `boruta_shap/`)
**LOC reviewed:** ~8,469 (full-file reads, not sampled)

Every file in scope was read in full. No file was skipped or spot-checked.

## Findings table

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| FS_BORUTA_ROOT-1 | P1 | `boruta_shap/__init__.py:753-757` | `TentativeRoughFix()` computes accept/reject medians over `self.history_x` **including the phantom all-zero row 0** (the pre-loop `np.zeros` initializer), biasing the decision for every tentative feature; the sibling `_io_plot.py` explicitly skips this row via `.iloc[1:]` in the same class. | Change `hx[self.tentative].median(...)` / `hx["Max_Shadow"].median(...)` to `hx.iloc[1:][...]` / `hx.iloc[1:]["Max_Shadow"]`, matching `_io_plot.py`'s existing pattern. | Scanner: find every `history_*`/`_history` attribute initialized via `np.zeros(...)` and grown via repeated `np.vstack`; flag any consumer of the resulting DataFrame that does not slice off row 0 (or grep all `self.history_x` reads and diff their row-0 handling against each other). |
| FS_BORUTA_ROOT-2 | P2 | `boruta_shap/_shadow_stats.py:284-309` (`find_sample`) | The KS-sample search starts at `element = 1` (the array's SECOND entry, ~10% of rows) instead of `0` (~5%), contradicting the docstring ("Starts of a 5%"); on frames with `<= 2` rows (`sample=True`) `size[element]` raises `IndexError` because `get_5_percent_splits` can return an array with fewer than 2 elements. | Start at `element = 0`; guard `element` against `len(size)` before the first `size[element]` access (or clamp `element = min(element, len(size) - 1)`). | Property test: for every function indexing a precomputed step/grid array from a non-zero literal offset, fuzz over tiny input sizes (n=0,1,2,3) and assert no `IndexError`/off-by-one vs. the documented starting fraction. |
| FS_BORUTA_ROOT-3 | P2 | `general.py:206-210` (`estimate_features_relevancy`) | `num_randomized_permutations += int(np.ceil((min_permuted_mi_evaluations - expected_evaluations_num) / len(feature_columns)))` divides by `len(feature_columns)` with no zero guard; a `bins` frame that contains only target columns (no candidate features) raises `ZeroDivisionError` instead of returning an empty/graceful result. | Guard with `if feature_columns: ... else: return [], ...` (or skip the ceil-adjustment when `len(feature_columns) == 0`, since there is nothing to screen). | Grep-based scanner for `/ len(<expr>)` / `// len(<expr>)` not preceded by a truthiness/length check in the same function; fuzz each such function with an empty collection for the divisor. |
| FS_BORUTA_ROOT-4 | P3 | `boruta_shap/_fit_explain.py:521-527` | The "log every 5 trials" block (`if new_ncols != last_ncols or trial % 5 == 0: ...`) sits OUTSIDE the `for trial in pbar:` loop (dedented to the loop's sibling level), so it runs exactly once, after the LAST trial, using whatever `trial`/`last_ncols` happened to be at loop exit — it never actually logs periodically as the code shape implies. | Either move the block inside the loop body (if periodic logging was intended) or delete it as redundant with the `calculate_rejected_accepted_tentative`/`pbar.set_description` logging that already runs. | AST checker: flag a `for` loop immediately followed by a conditional block that (a) references the loop variable and (b) compares against a variable only assigned before the loop and once after — a strong signal the block was meant to be loop-body code but got left outside it. |
| FS_BORUTA_ROOT-5 | P3 | `boruta_shap/__init__.py:766-774` (`TentativeRoughFix`) | After resolving tentative features into `accepted`/`rejected`, `self.tentative` is never cleared or narrowed — it still lists the just-resolved features, so `len(bs.tentative)` (or any caller reading `.tentative` post-call) reports stale, non-zero counts for features that are no longer undecided. The box-plot color mapping happens to self-correct only because `dict(zip(...))` lets the later-concatenated `accepted`/`rejected` keys overwrite the stale `tentative` entry — any other consumer of `.tentative` has no such protection. | After building `newly_accepted`/`newly_rejected`, set `self.tentative = np.array([], dtype=self.tentative.dtype)` (or equivalent) so the attribute reflects the post-call state. | Regression test: call `TentativeRoughFix()` on a fixture with a nonempty tentative set and assert `len(bs.tentative) == 0` afterward. |
| FS_BORUTA_ROOT-6 | P3 | `unanimous_permutation_prune.py:94-104` | No validation that `cv_splits` is non-empty; an empty sequence makes `per_fold_deltas = []`, and `np.stack([], axis=0)` raises an opaque `ValueError: need at least one array to stack` instead of a clear, actionable error naming the bad argument. | Raise `ValueError("unanimous_permutation_prune: cv_splits must contain at least one (train_idx, val_idx) pair")` at entry when `not cv_splits`. | Generic fuzz pass: call every public FS entrypoint taking a `Sequence`/`list` of folds/candidates with an empty sequence and assert a clear `ValueError` naming the parameter, not a raw numpy/library exception. |

## Severity counts

- P0: 0
- P1: 1
- P2: 2
- P3: 3

## Narrative

### FS_BORUTA_ROOT-1 (P1) — `TentativeRoughFix` biased by the phantom zero-row

`create_importance_history()` initializes `self.history_shadow = np.zeros(self.ncols)` and
`self.history_x = np.zeros(self.ncols)` as 1-D arrays *before* the trial loop runs. Each trial's
`update_importance_history()` then `np.vstack`s the new per-column importances onto these, so after `N`
trials `self.history_x` has `N + 1` rows: row 0 is the leftover all-zero initializer, rows 1..N are real
per-trial z-scored importances. `store_feature_importance()` promotes this ndarray to a DataFrame and adds
`Max_Shadow`/`Min_Shadow`/`Mean_Shadow`/`Median_Shadow` columns computed the same way (`[max(i) for i in
self.history_shadow]`), so those columns also carry a spurious `0.0` in row 0.

`_io_plot.py`'s `results_to_csv` and `plot` both read `self.history_x.iloc[1:]` explicitly — i.e., the
codebase already knows row 0 must be excluded and does so consistently in those two call sites. But
`TentativeRoughFix()` (in `boruta_shap/__init__.py`) reads `hx[self.tentative].median(axis=0)` and
`hx["Max_Shadow"].median(axis=0)` directly off `self.history_x`, with **no** `.iloc[1:]`. Because
`normalize=True` (the default) z-scores importances per trial (mean 0, std 1), both the real-feature and
shadow distributions are centered near 0, so splicing in one literal `0.0` sample measurably shifts the
median for any feature whose true population median is not exactly 0 — most acutely with small `n_trials`
(where row 0 is a larger fraction of the sample) but present at any trial count. `TentativeRoughFix` is the
only place in the codebase that decides tentative features by comparing `median_tentative` against
`median_max_shadow`, so this directly changes accept/reject outcomes for a documented, user-facing public
method (`BorutaShap.TentativeRoughFix()`, the canonical Boruta workflow step for resolving leftover
tentative features). Found by comparing every `self.history_x` read-site in the package and noticing
`_io_plot.py`'s two call sites deliberately strip row 0 while `TentativeRoughFix`'s two call sites do not —
the same DataFrame, two different row-0 conventions in the same class. No existing test exercises this
(the one `TentativeRoughFix` test, `test_boruta_shap_logger_warn_not_print.py`, only checks that the
decision is logged via `logger` instead of `print`, not that the decision itself is unbiased).

### FS_BORUTA_ROOT-2 (P2) — `find_sample` starts one grid step too late, crashes on tiny frames

`find_sample()`'s docstring says "Starts of a 5% however will increase to 10% and then 15% etc.", but the
implementation sets `element = 1` before the search loop and immediately indexes `size[element]` on the
first iteration. `size = self.get_5_percent_splits(self.X.shape[0])` returns
`np.arange(step, length, step)`, so `size[0] == step` (~5% of rows) and `size[1] == 2*step` (~10%) — the
loop therefore always starts its very first draw at ~10%, never trying the documented ~5% bracket at all.
Beyond the docstring mismatch, this is a real crash risk: for `self.X.shape[0] <= 2` (with
`sample=True`), `get_5_percent_splits` can return an array of length 0 or 1 (e.g. `length=1` →
`np.arange(1,1,1)` is empty; `length=2` → `np.arange(1,2,1)` has exactly one element), so `size[1]` raises
`IndexError` on the very first line of the search loop, before the existing 20-miss/size-growth safety net
(added by the `test_boruta_find_sample_terminates.py` regression test) ever gets a chance to run. Verified
by hand-tracing `get_5_percent_splits` for `length` in `{1, 2, 3}`; the existing regression test only
covers `length=400` (plenty of grid points), so it does not exercise this path.

### FS_BORUTA_ROOT-3 (P2) — `estimate_features_relevancy` divides by zero on an all-target frame

In `general.py`'s `estimate_features_relevancy`, `feature_columns` is the set of `bins` columns that are
not in `target_columns`. When `feature_columns` is empty (a `bins` frame consisting solely of target
columns — a plausible state if an earlier binning/pre-screen stage dropped every candidate feature), the
line `num_randomized_permutations += int(np.ceil((min_permuted_mi_evaluations - expected_evaluations_num)
/ len(feature_columns)))` divides by zero. `expected_evaluations_num` is `0` in this situation (its only
non-permuted-cache term is `min_randomized_permutations * len(feature_columns) == 0`), so with the default
`min_permuted_mi_evaluations=500` the `if expected_evaluations_num < min_permuted_mi_evaluations:` branch
is always entered, guaranteeing the crash whenever this state is reached — it is not a rare corner of the
condition. `run_efs` (this file) calls `estimate_features_relevancy` directly with whatever `bins`
`bin_numerical_columns` produced, so a strict-enough binning config that filters out all raw feature
columns propagates straight into this crash instead of a clean "nothing left to screen" result.

### FS_BORUTA_ROOT-4 (P3) — dead post-loop "log every 5 trials" block

At the end of `fit()` (`_fit_explain.py`), immediately after `self.calculate_rejected_accepted_tentative(...)`
and `pbar.set_description(...)`, the code reads:

```python
new_ncols = len(self.columns)
if new_ncols != last_ncols or trial % 5 == 0:
    logger.info("Undecided features: %s", f"{len(self.tentative):_}")
    last_ncols = new_ncols
```

This block is indented at the same level as `for trial in pbar:` (i.e. it is a sibling of the loop, not
inside it), and `last_ncols` is initialized to `0` once, before the loop, and never touched anywhere else.
Because it runs only once (after the loop exits) and `last_ncols` is still `0` at that point, `new_ncols !=
last_ncols` is true for any fit that decided at least one feature — meaning the `trial % 5 == 0` disjunct
is functionally irrelevant and the whole block reduces to "log once, unconditionally, at the very end,
using the terminal `trial` value". The shape of the code (an `if ... % 5 == 0` gate with a running
`last_ncols` comparison) strongly signals it was intended as periodic per-trial progress logging that
never made it inside the loop body — as written it neither logs periodically nor adds information beyond
what `calculate_rejected_accepted_tentative`'s own `logger.info` calls already emit two lines earlier.

### FS_BORUTA_ROOT-5 (P3) — `TentativeRoughFix` leaves `self.tentative` stale after resolving it

`TentativeRoughFix()` computes `newly_accepted`/`newly_rejected` from `self.tentative`, appends them to
`self.accepted`/`self.rejected`, but never updates `self.tentative` itself — so after the call,
`self.tentative` still contains every feature that was just resolved, even though none of them are
undecided anymore. `create_mapping_of_features_to_attribute` (used by `plot`/`box_plot`) happens to render
correctly regardless, purely as a side effect of `to_dictionary`'s `dict(zip(keys, values))` letting later
duplicate keys (accepted/rejected, concatenated after tentative) silently overwrite the stale tentative
entry — that is incidental, not a designed safeguard, and any other code reading `bs.tentative` directly
(e.g. to report "N features remain undecided") would see a stale, non-empty count.

### FS_BORUTA_ROOT-6 (P3) — `unanimous_permutation_prune` gives an opaque error on empty `cv_splits`

`unanimous_permutation_prune` validates `min_fold_agreement_fraction`'s range at entry but never checks
that `cv_splits` (a required, no-default `Sequence` argument) is non-empty. If a caller passes `[]` (e.g. a
`TimeSeriesSplit` that yields no valid splits for a tiny dataset), `per_fold_deltas` stays `[]` and
`np.stack(per_fold_deltas, axis=0)` raises `ValueError: need at least one array to stack` — a numpy-internal
message that does not name `cv_splits` or explain the actual precondition, unlike the function's other
validated preconditions.

## Dimension coverage notes

- **Correctness bugs (crashes / silently-wrong results):** covered above (FS_BORUTA_ROOT-1 through -3, -6).
  No mutable-default-argument bugs were found in this cluster (dataclass `field(default_factory=...)` is used
  correctly throughout; no bare `def f(x=[])`-style signatures exist in scope).
- **ML correctness (leakage / reproducibility / calibration / sample_weight):** no leakage found —
  `BorutaShap.explain()` has an explicit train/basis-row-count assertion guarding against val/test leaking
  into the SHAP background; `HybridSelector` and `hetero_vote.py` seed every RNG explicitly
  (`np.random.default_rng(random_state)`), never touch the global `np.random` state, and document their
  sample-weight non-threading as a deliberate, consistent (not silent) simplification. `ridge_forward_prefilter`,
  `unanimous_permutation_prune`, and `stochastic_bandit_selection` all correctly seed CV splits from the
  caller's `random_state` rather than a hardcoded literal (the latter has an explicit comment noting a prior
  fix for exactly this class of bug). No new instances found.
- **Computational efficiency:** no unnecessary `.copy()`/materialization issues found beyond what is already
  documented and justified in-line (e.g. `BorutaShap.fit`'s single documented `self.X = X.copy()`,
  `HybridSelector.corr_clusters`'s O(n*p)-vs-O(p^2) blocked path). `remove_features_if_rejected`,
  `create_shadow_features`'s fast numeric path, and `_binom_test_cached`'s LRU are all already-optimized
  hot paths with measured justification in comments.
- **Edge cases / robustness:** covered above (FS_BORUTA_ROOT-2, -3, -6). Beyond those, empty-input and
  single-row/single-column guards are unusually thorough throughout this cluster (e.g. `compare_selectors`
  rejects zero-column `X`; `hetero_vote` short-circuits `P == 0` and single-class `y`; `pre_screen.py` has
  explicit float-noise / sparse-column variance handling). No additional gaps found.
- **Test coverage gaps:** the three bugs above (FS_BORUTA_ROOT-1, -2, -5) are each an UNTESTED code path
  in a file whose neighbors are otherwise heavily regression-tested (27 dedicated test files under
  `tests/feature_selection/boruta_shap/`) — this cluster is not undertested in aggregate, but these specific
  branches (`TentativeRoughFix`'s row-0 handling, `find_sample`'s tiny-frame path, `TentativeRoughFix`'s
  post-call `.tentative` state) fell through the otherwise-dense coverage. No test in this repo asserts
  against the code's own output instead of an independent oracle within this cluster's scope.
- **Code quality / architecture:** no duplication or misleading naming found beyond FS_BORUTA_ROOT-4 (dead
  code) and FS_BORUTA_ROOT-5 (stale state). Docstrings and comments are unusually rigorous and accurate
  throughout this cluster (multiple modules explicitly document REJECTED alternatives, measured benchmark
  numbers, and known limitations in-line) — no stale/misleading docstrings were found other than
  `find_sample`'s (folded into FS_BORUTA_ROOT-2).
- **OSS/hygiene (comment cruft, mojibake, dashes, mypy):** no stray audit-wave markers, mojibake, or
  double-dash prose were found in this cluster's comments (the one Cyrillic reference in `ace.py`'s
  docstring, "Дьяконов", is an intentional, correctly-encoded citation of a named lecture source, not
  mojibake). Type hints are present and consistent with the rest of the codebase; no bare `object` params
  or implicit-Optional signatures were found in scope.
