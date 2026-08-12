# feature_selection/ non-MRMR selectors + boruta_shap -- mlframe audit

## Scope

All 26 files listed for the cluster, directly under `src/mlframe/feature_selection/` (no recursion into
excluded subdirectories), plus every `.py` file under `src/mlframe/feature_selection/boruta_shap/**`
(recursively, including its `_benchmarks/` subfolder). Every file below was read in full.

- `src/mlframe/feature_selection/hybrid_selector.py` (784 LOC)
- `src/mlframe/feature_selection/structure_discovery.py` (410 LOC)
- `src/mlframe/feature_selection/general.py` (401 LOC)
- `src/mlframe/feature_selection/importance.py` (380 LOC)
- `src/mlframe/feature_selection/ace.py` (366 LOC)
- `src/mlframe/feature_selection/functional_adapters.py` (362 LOC)
- `src/mlframe/feature_selection/mi.py` (325 LOC)
- `src/mlframe/feature_selection/compare_selectors.py` (276 LOC)
- `src/mlframe/feature_selection/registry.py` (254 LOC)
- `src/mlframe/feature_selection/pre_screen.py` (218 LOC)
- `src/mlframe/feature_selection/ridge_forward_prefilter.py` (189 LOC)
- `src/mlframe/feature_selection/forward_select.py` (184 LOC)
- `src/mlframe/feature_selection/varying_size_top_k_subsets.py` (184 LOC)
- `src/mlframe/feature_selection/hetero_vote.py` (167 LOC)
- `src/mlframe/feature_selection/stochastic_bandit_selection.py` (165 LOC)
- `src/mlframe/feature_selection/greedy_backward_elimination.py` (163 LOC)
- `src/mlframe/feature_selection/unanimous_permutation_prune.py` (126 LOC)
- `src/mlframe/feature_selection/drop_raw_after_embedding.py` (125 LOC)
- `src/mlframe/feature_selection/cascade_select.py` (121 LOC)
- `src/mlframe/feature_selection/zero_importance_pruning.py` (121 LOC)
- `src/mlframe/feature_selection/stochastic_bandit_selection_ensemble.py` (111 LOC)
- `src/mlframe/feature_selection/__init__.py` (100 LOC)
- `src/mlframe/feature_selection/drop_noninformative_vs_reference.py` (98 LOC)
- `src/mlframe/feature_selection/cascade_select_stability.py` (96 LOC)
- `src/mlframe/feature_selection/drop_near_noise_univariate_auc.py` (96 LOC)
- `src/mlframe/feature_selection/optbinning.py` (95 LOC)
- `src/mlframe/feature_selection/boruta_shap/__init__.py` (888 LOC)
- `src/mlframe/feature_selection/boruta_shap/_fit_explain.py` (665 LOC)
- `src/mlframe/feature_selection/boruta_shap/_shadow_stats.py` (409 LOC)
- `src/mlframe/feature_selection/boruta_shap/_io_plot.py` (168 LOC)
- `src/mlframe/feature_selection/boruta_shap/_auto_dispatch.py` (131 LOC)
- `src/mlframe/feature_selection/boruta_shap/_benchmarks/bench_shadow_min_pad_narrow_frames.py` (107 LOC)

**Total files reviewed: 32. Total LOC reviewed: 8285** (5917 in the 26 top-level files + 2261 in the 5
non-benchmark boruta_shap package files + 107 in the boruta_shap benchmark script).

No file was skipped or partially read. This cluster's code is, on the whole, unusually well-documented and
already shows extensive prior audit remediation (many inline comments reference "Wave N" fixes, prior
correctness traps, and rejected-optimization notes) -- most files yielded zero or few new findings.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P0 | correctness | `pre_screen.py:187-196` | Sparse-column variance approximation ignores the fill-value mass, so any sparse column whose few explicitly-stored (non-fill) values happen to be few or identical is unconditionally treated as zero-variance and dropped by the unsupervised pre-screen, even when it carries strong signal (e.g. a rare-event/fraud flag). |
| F2 | P1 | correctness/edge-case | `boruta_shap/_fit_explain.py:450-524` | `fit()` crashes with `NameError: name 'trial' is not defined` at line 523 when `n_trials<=0`, because the loop variable `trial` (bound only inside `for trial in pbar:`) is read unconditionally after the loop; `n_trials` has no validation anywhere in `__init__`/`fit`. |
| F3 | P1 | reproducibility | `stochastic_bandit_selection.py:50-51` | When `cv` is left at its default (`None`), the CV splitter is built as `KFold(n_splits=3, shuffle=True, random_state=0)` -- a hardcoded seed that ignores the function's own `random_state` parameter. Every call (including every independent seed in `stochastic_bandit_selection_ensemble`) that doesn't pass `cv=` explicitly gets bit-identical fold splits regardless of `random_state`, so only the bandit's subset-sampling RNG actually varies across "independent" seeds. |
| F4 | P2 | docs | `hybrid_selector.py:474-476` | `_run_boruta_premerge`'s docstring says `"permutation" held-out by default`, but `BorutaShap.__init__`'s (and `HybridSelector.__init__`'s own `boruta_driver`) actual default is `"gini"` (confirmed at `hybrid_selector.py:161` and its own `__init__` comment at 241-247, which explicitly says "gini stays the default"). The two docstrings on the same class directly contradict each other. |
| F5 | P2 | docs | `zero_importance_pruning.py:8-9` vs `zero_importance_pruning.py:98-117` | The module docstring claims the batch-drop loop keeps "tracking CV score only once per round and **stopping on degradation**", but the implementation never breaks on a CV-score decline -- it always adopts `candidate_remaining` as the new `remaining` set every round (line 113) and only remembers the best-scoring round separately, running to `max_rounds`/convergence regardless of how many consecutive rounds regress. |
| F6 | P2 | docs | `cascade_select_stability.py:30-32` | Docstring says "with `n_bootstrap` left at its default of **1** run's worth of behavior disabled", but the actual signature default is `n_bootstrap: int = 20` (line 23) -- a stale docstring from an earlier default. |
| F7 | P2 | api/discoverability | `hetero_vote.py` (whole file) vs `__init__.py:11-49` | `heterogeneous_relevance_vote` is a complete, well-tested (see `tests/feature_selection/stability/test_hetero_vote*.py`, excluded from this audit but confirmed to exist) primitive referenced from `ace.py`'s own module docstring, yet it is the only "top-level" selection primitive in this cluster not re-exported from `feature_selection/__init__.py`'s `__all__`, and it has no `registry.py` factory / training-suite wiring (unlike `ace_select`, which got both). |
| F8 | P2 | docs/validation | `stochastic_bandit_selection_ensemble.py:63-75` | Docstring states "At least 2 seeds are required for a meaningful stability diagnostic", but the only enforced check is `len(seeds) < 1` (line 74) -- a single seed silently runs and reports every selected feature at stability 1.0, contradicting the documented requirement. |
| F9 | P2 | ml-best-practice/edge-case | `ace.py:353-366`, `functional_adapters.py:23-39` | `_default_estimator` (and its verbatim mirror `_default_tree_estimator`) route to a regression estimator whenever `y.dtype.kind` is float, even for a float-typed binary/low-cardinality target (e.g. `0.0`/`1.0` labels) -- unlike `HybridSelector`, which explicitly validates/warns on exactly this ambiguity (`hybrid_selector.py:552-559`), these two default-estimator heuristics silently fit a `RandomForestRegressor` on what may really be a classification target, with no warning. |
| F10 | P2 | code-quality | `boruta_shap/_io_plot.py:71-72` | `data = self.history_x.iloc[1:]; data["index"] = data.index` mutates a row-sliced view/copy of `self.history_x` in place -- a classic pandas chained-assignment anti-pattern that is fragile across pandas versions (can raise `SettingWithCopyWarning`, and its copy-vs-view semantics are not guaranteed). Currently works because `.iloc[1:]` happens to return an independent frame on the tested pandas version, but it is not a safe pattern to rely on. |

**F1** (`pre_screen.py`): `compute_unsupervised_drops`'s pandas branch special-cases `pd.SparseDtype` columns for
the variance check: `sp_vals = np.asarray(col.values.sp_values)`; if `sp_vals.size <= 1` it hardcodes
`var_val = 0.0`, and otherwise computes `np.nanvar(sp_vals)` -- the variance of only the explicitly-stored
(non-fill) values, never the full reconstructed column (fill cells + stored cells). Concretely: a sparse
column with `fill_value=0.0`, 999 fill cells and one stored value `5.0` has a true population variance of
≈0.025 (`>> 1e-24`, i.e. genuinely informative), but this code reports `var_val=0.0` and drops it. Worse,
*any number* of identically-valued stored entries (e.g. a rare binary flag stored as three `1.0`s among a
thousand `0.0`s -- exactly the TF-IDF/rare-flag production pattern this same function's null-handling logic
was explicitly hardened for a few lines above) makes `np.nanvar(sp_vals) == 0.0`, so the column is dropped
regardless of how strong its true signal is. Because this is a **train-only, silent, pre-model** filter with
no downstream check that can re-admit the column, a genuinely predictive rare-event indicator stored as a
sparse column is permanently and silently removed before any selector or model ever sees it. Suggested fix:
compute variance over the reconstructed full-length array (or the closed-form sparse variance
`n_stored/n * var(sp_vals) + (n_stored*n_unfilled/n^2) * (mean(sp_vals) - fill_value)^2`-style formula that
accounts for the fill-value mass) instead of `nanvar` over the stored subset alone.

**F2** (`boruta_shap/_fit_explain.py`): `fit()`'s trial loop is `pbar = tqdmu(range(self.n_trials), ...)` /
`for trial in pbar: ...`; after the loop, `new_ncols = len(self.columns); if new_ncols != last_ncols or trial
% 5 == 0:` reads `trial` unconditionally. `self.n_trials` (constructor default `150`, `__init__.py:148`) is
never validated to be `>= 1` anywhere in `__init__` or `fit`. If a caller passes `n_trials=0` (or a
dynamically-computed budget that evaluates to `<=0`), `range(0)` never executes the loop body, `trial` is
never bound, and the function crashes with `NameError: name 'trial' is not defined` instead of a clear
validation error or a graceful zero-trial short-circuit. Suggested fix: validate `n_trials >= 1` in `fit`
(or `__init__`) with a clear `ValueError`, or seed `trial = -1` before the loop so the post-loop logging is
safe.

**F3** (`stochastic_bandit_selection.py`): `_stochastic_bandit_selection_core`'s only use of the caller's
`random_state` is `rng = np.random.default_rng(random_state)`, which drives the per-epoch weighted subset
sampling. The CV splitter used to *score* each epoch's subset is either the caller-supplied `cv`, or -- when
`cv is None` -- `KFold(n_splits=3, shuffle=True, random_state=0)`, a literal `0` independent of the function's
own `random_state` argument. This matches the checklist's flagged pattern of "a caller-supplied
random_state/random_seed never actually reaches the call": running `stochastic_bandit_selection(..., cv=None,
random_state=1)` and `stochastic_bandit_selection(..., cv=None, random_state=2)` scores every epoch's
candidate subset against the *exact same* 3 folds; only the subset draws differ. This also silently weakens
`stochastic_bandit_selection_ensemble`'s stability diagnostic when `cv` is left `None`: every "independent"
per-seed run in the ensemble is scored on identical folds, so cross-seed disagreement undercounts a real
source of noise (fold variance) that the ensemble's own docstring implies each seed explores independently.
Suggested fix: default `cv = KFold(n_splits=3, shuffle=True, random_state=random_state)` when `cv is None`.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| P1 | code-quality/reuse | `ace.py:353-366`, `functional_adapters.py:23-39` | `_default_tree_estimator` in `functional_adapters.py` is a byte-for-byte duplicate of `ace._default_estimator`'s classification-heuristic + RandomForest defaulting (the docstring even says "mirrors `ace._default_estimator` verbatim"); worth hoisting into one shared helper (e.g. `mlframe.feature_selection._sklearn_defaults`) so the float-binary-target edge case in F9 only needs fixing once. |
| P2 | memory | `hybrid_selector.py:590`, `777-784` | `HybridSelector.fit` stashes the full augmented training frame and target on `self._Xaug_`/`self._y_` "for combine-rule diagnostics"; `__getstate__` drops them before pickling, but a live (non-pickled) fitted `HybridSelector` retains a full copy of the augmented (raw + engineered) training data for its whole in-process lifetime, which conflicts with this repo's 100+GB-frame memory-discipline convention for any caller that keeps the fitted selector object around (e.g. in a long-lived service) rather than immediately pickling it. Consider gating the stash behind an opt-in flag (default off) or storing only a shape/hash-level diagnostic. |
| P3 | test-coverage | `boruta_shap/_fit_explain.py` | No test exercises `n_trials<=0` (the crash in F2) or `stability_subsamples>1` combined with `n_trials<=0`; a regression test for F2 would also double as the first `n_trials` boundary test for this module. |
| P4 | test-coverage | `stochastic_bandit_selection.py` | Every existing test (`tests/feature_selection/test_biz_val_stochastic_bandit_selection.py`) passes `cv=` explicitly, so the `cv=None` default path (including the F3 hardcoded-seed bug) is entirely untested. A test asserting that two `stochastic_bandit_selection(..., cv=None, random_state=seed)` calls with different seeds see different fold assignments (not just different subset draws) would catch this class of bug going forward. |
| P5 | ml-best-practice | `hetero_vote.py:94-167` | `heterogeneous_relevance_vote`'s `weight_by_cv_skill` option is documented as measured-neutral on the one benched bed ("changed the selection in 0/12 cells"); since the module is otherwise a strong, isolated primitive (F7), consider whether it's worth wiring into `registry.py` + the training suite the same way `ACE` was, given it already has full test coverage and a clear precision-oriented use case documented in its own module docstring. |
| P6 | architecture | `hybrid_selector.py` (784 LOC) | Sits right at this repo's own "carve before ~800-900 LOC" convention threshold (`mlframe/CLAUDE.md` "New code goes in focused submodules from the start"). Not yet a violation, but the `_run_mrmr` / `_run_shap` / `_run_boruta_premerge` / `_tree_signals` / `_admit_tree_products` / `_augment` member-orchestration methods (lines 286-444) are already a fairly self-contained "shared-artifact" concern that could move to a sibling module (e.g. `hybrid_selector_members.py`) ahead of the next feature addition pushing it over the limit. |
| P7 | code-quality | `hybrid_selector.py:290-325,378-379,431,649-656` | The five broad `except Exception` sites inside `HybridSelector.fit` (MRMR/tree-signal/FE-augment/shap-member/boruta-member stages) are each explicitly reasoned as intentional graceful-degradation (documented at the `_run_mrmr` site: "MRMR is shared infra under active development... failure degrades... rather than crashing"). This is a defensible, explicitly-documented design choice (not flagged as a Finding), but it does mean a genuine programming bug inside any member stage (e.g. a typo introduced in a future edit) would silently present as a "member degraded" warning + reduced-member vote rather than a test failure, which is worth keeping in mind if a member's test coverage ever thins out. |

## Coverage notes

- Every file in the assigned scope (including the `boruta_shap/_benchmarks/` subfolder, which the task's
  recursive-glob instruction technically includes) was opened and read in full; nothing was skipped.
- `pre_screen.py`'s polars branch (lines 80-119) was reviewed and found free of the sparse-variance bug (F1)
  found in the pandas branch, since polars has no sparse-dtype equivalent in this codebase's usage -- this
  is a pandas-only bug.
- Symbols imported *from* the excluded MRMR/shap_proxied_fs packages (e.g. `hybrid_selector.py`'s use of
  `mlframe.feature_selection.filters.MRMR`, `mlframe.feature_selection.shap_proxied_fs.ShapProxiedFS`/
  `restrict_artifacts`, `boruta_shap/_fit_explain.py`'s reference to `filters/group_aware.GroupAwareMRMR` via
  `registry.py`) were read only at the call-site signature/contract level, per the exclusion instructions --
  their internals were not analyzed and no findings were attributed to them.
- Test files under the excluded `tests/feature_selection/{mrmr,mrmr_api,filters,shap_proxied,gpu,clustering,
  discretization,info_theory,screening,stability,robustness,fe,golden,contracts,biz_val,core,_artifacts}/`
  directories (which include the `hetero_vote` and `pre_screen` test suites) were not opened; their existence
  was confirmed only via filename search (`find`/`grep`), consistent with the task's exclusion of those
  directories -- so F1's and F7's "well-tested" / "test coverage exists" claims rest on filenames only, not
  on having read the excluded test bodies.
