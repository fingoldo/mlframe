# votenrank/ (incl. votenrank/leaderboard) -- mlframe audit

## Scope

Every file under `src/mlframe/votenrank/**` was read in full (36 files, 4226 LOC total, per `wc -l`). This is more files/LOC than the task's ~24-file/~3.2k-LOC estimate because the recursive glob also picked up the `_benchmarks/` subpackage (12 cProfile/timing harness scripts), which the estimate apparently excluded; I reviewed those too since the scope statement says "every .py file, recursively."

Files reviewed:
- `__init__.py`
- `utils.py`
- `data_processing.py`
- `fairness_computation.py`
- `iia_exp.py`
- `rank_splice.py`
- `rank_percentile_stacking.py`
- `knn_fallback_predictor.py`
- `constrained_weight_blend.py`
- `geometric_weight_blend.py`
- `_confidence_gated_blend_ktc_dispatch.py`
- `confidence_gated_blend.py`
- `stability_exp.py`
- `similarity_blend.py`
- `hill_climb.py`
- `adversarial_stochastic_blend.py`
- `dual_optimizer_blend.py`
- `shapley_blend.py`
- `correlation_diversity_ablation.py`
- `leaderboard/__init__.py`
- `leaderboard/settings.py`
- `leaderboard/_cw.py`
- `leaderboard/_rules.py`
- `leaderboard/leaderboard_impl.py`
- `_benchmarks/bench_minimax_winning_votes.py`
- `_benchmarks/bench_rank_splice.py`
- `_benchmarks/bench_rank_percentile_stacking.py`
- `_benchmarks/bench_constrained_weight_blend.py`
- `_benchmarks/bench_knn_fallback_predictor.py`
- `_benchmarks/profile_shapley_blend.py`
- `_benchmarks/bench_dual_optimizer_blend.py`
- `_benchmarks/bench_similarity_blend.py`
- `_benchmarks/bench_adversarial_stochastic_blend.py`
- `_benchmarks/bench_geometric_weight_blend.py`
- `_benchmarks/bench_hill_climb.py`
- `_benchmarks/bench_correlation_diversity_ablation.py`
- `_benchmarks/bench_confidence_gated_blend.py`

Every file was reviewed in full depth (none were too large to read completely -- the biggest, `leaderboard/leaderboard_impl.py`, is 283 LOC). I also cross-referenced `tests/votenrank/*`, `tests/training/test_ensembling_votenrank_votenrank.py`, and `tests/models/test_votenrank_import_reorg.py` to assess test coverage, and read the signature/docstring of `mlframe.feature_engineering.transformer.knn_search` (external to this cluster) purely to confirm its k-clamping contract for two callers in scope.

## Findings

| ID | Severity | Category | File:Line | Summary |
|----|----------|----------|-----------|---------|
| F1 | P1 | correctness | leaderboard/leaderboard_impl.py:54-57 | `Leaderboard.__init__` silently accepts a `weights` dict key that doesn't match any table column, extending `self.weights` via pandas `.loc` enlargement instead of raising, which inflates the `weights.sum()` denominator used by `mean_ranking`/`optimality_gap_ranking`. |
| F2 | P1 | correctness | leaderboard/leaderboard_impl.py:48-66 | `Leaderboard.__init__` never validates that `table.index` (model names) is unique; every `_rules.py`/`_cw.py` method relies on `.loc[model]` returning a single `Series`, which breaks silently/crashes for a duplicated model name -- a scenario `data_processing.py`'s own `preprocess_glue`/`preprocess_sglue` already dedupe against for exactly this reason. |
| F3 | P1 | correctness | data_processing.py:94-102 | `preprocess_value` hardcodes a fixed 7-row model-name relabeling with zero verification that the scraped table's row order actually matches the assumed roster order -- a reordered/changed source table would silently mislabel every model. |
| F4 | P2 | edge-case/docs | leaderboard/_rules.py:20-22 | `mean_ranking(mean_type="geometric")` feeds the (possibly zero/negative) score table straight into `scipy.stats.gmean`, which silently returns 0/NaN rather than raising when a task score is `<=0`; the domain constraint is undocumented. |
| F5 | P2 | edge-case | leaderboard/_rules.py:19,160 | `mean_ranking` (arithmetic) and `optimality_gap_ranking` both divide by `self.weights.sum()`; an all-zero-weights `Leaderboard` (a valid caller input) silently produces `inf`/`NaN` scores instead of raising. |
| F6 | P2 | edge-case | leaderboard/_rules.py (borda/dowdall/plurality/threshold/baldwin) | `borda_ranking`, `dowdall_ranking`, `plurality_ranking`/`_approval_ranking`, `threshold_election`, `baldwin_election` have no partial-table (NaN) guard of their own -- they are only ever routed away from a partial table because `elect_all`/`rank_all` filter to `PARTIAL_METHODS`; called directly on a partial `Leaderboard`, `pandas.sum(skipna=True)` silently drops each model's NaN tasks, so different models end up scored over different numbers of tasks with no warning. |
| F7 | P2 | docs | shapley_blend.py:169,186-189 | `shapley_blend`'s docstring says `prune_below=0.0` "keeps every non-negative-value member", but the code's `keep_mask = weights > threshold` is a strict `>`, so a model with an exact-zero Shapley value (weight `0.0`) is pruned, not kept -- the docstring should say "positive-value member". |
| F8 | P2 | silent-failure | confidence_gated_blend.py:185-190 | The `cupy` backend's `try/except Exception: pass` (falling through to numpy) swallows the failure with zero logging -- not even `logger.debug` -- unlike the sibling dispatch module (`_confidence_gated_blend_ktc_dispatch.py`), which logs every backend-probe failure at debug level; a silently-failing GPU backend here is undiagnosable from logs alone. |
| F9 | P2 | robustness | knn_fallback_predictor.py:47-65 | `KNNFallbackPredictor.predict()` calling before `fit()` (or after `fit()` with an empty `y`) crashes with a bare `IndexError` from `self._y_train[ids]` (via `knn_search`'s documented empty-training-set sentinel `ids=0`) rather than a clear "must call fit() first" `ValueError`. |
| F10 | P2 | edge-case | dual_optimizer_blend.py:41-57 | `dual_optimizer_weight_blend(..., include_coord_descent=True)` on a single-model pool (`n_models=1`) crashes inside `_coordinate_descent_simplex_search`'s `rng.choice(n_models, size=2, replace=False)`, which requires a population of >= 2; there is no guard anywhere in the module for this degenerate-but-reachable case. |
| F11 | P2 | GPU/CPU-parity | fairness_computation.py:26-64 | Every `naive_*_score` function calls `.cuda()` unconditionally with no `MLFRAME_DISABLE_GPU`/`CUDA_VISIBLE_DEVICES=""` CPU-fallback path, unlike the rest of the package's GPU-touching code (`confidence_gated_blend.py`, `_confidence_gated_blend_ktc_dispatch.py`); the module docstring does document this as an intentional "CUDA-only" simplification, so this is a repo-convention gap rather than a surprise, but it means the module cannot be exercised at all on a CPU-only CI/dev box. |
| F12 | P2 | code-quality | stability_exp.py:19 | `sns.set(style="whitegrid")` runs as an import-time side effect, silently changing the global seaborn/matplotlib style for any other code in the same interpreter session that happens to import this module. |
| F13 | P2 | code-quality | utils.py:26, iia_exp.py:51, leaderboard/leaderboard_impl.py:219 | Three comments embed a "Wave N (2026-05-20):" audit/phase marker + date stamp, which the repo's own `CLAUDE.md` explicitly forbids ("no process/audit metadata in code comments: no phase/wave markers ... date stamps"); the technical content of each comment is otherwise good and worth keeping, just not the marker/date prefix. |
| F14 | P2 | test-coverage | leaderboard/_rules.py, leaderboard/_cw.py | `condorcet_election`, `baldwin_election`, `threshold_election`, `copeland_ranking`'s `"lower"`/`"upper"`/`"difference"` slice types, `Leaderboard.two_step_ranking`/`get_meta_leaderboard`, and `_cw.py`'s `_find_weights_for_majority_graph`/`find_weights_for_condorcet`/`split_models_by_feasibility` have no dedicated correctness-asserting test in `tests/votenrank/`; they are only exercised incidentally (order-reproducibility only, no value assertions) by `test_reproducibility_method_order.py`. |
| F15 | P2 | architecture | leaderboard/_cw.py:20,90-93 | `_find_weights_for_majority_graph` returns either a `Dict[str, float]` or the literal string `"infeasible"` -- a stringly-typed sentinel instead of `None`/an exception/a small result type -- forcing every caller to compare against a magic string (`== "infeasible"`, see `leaderboard_impl.py:278`) that mypy can't meaningfully check. |

**F1** (`leaderboard/leaderboard_impl.py:54-57`): `weight_dict = weights or {}` then `for task, weight in weight_dict.items(): self.weights.loc[task] = weight`. If a caller passes e.g. `weights={"taskA": 2.0, "taskB_typo": 3.0}` where `"taskB_typo"` isn't one of `self.tasks`, pandas `.loc` enlargement silently adds a new row to `self.weights` for the nonexistent task instead of raising `KeyError`. Because `table * self.weights` aligns on columns (a nonexistent weight entry just doesn't multiply anything, contributing an all-NaN extra column that `.sum(axis=1, skipna=True)` ignores), the *numerator* is unaffected, but `self.weights.sum()` -- the *denominator* in `mean_ranking(mean_type="arithmetic")` and `optimality_gap_ranking` -- now silently includes the bogus weight, deflating every model's reported score by a wrong, silent factor. This is a uniform scalar rescale, so it does not flip which model wins (`mean_election`/`optimality_gap_election` are order-invariant to a positive scalar), but any caller reading the raw score magnitude (e.g., the "AM: 0.73" style values in `rank_all()`'s output columns, or a downstream consumer that isn't purely rank-based) gets a silently wrong number. Suggested fix: validate `weight_dict.keys() <= set(self.tasks)` in `__init__` and raise a clear `ValueError` naming the offending key(s).

**F2** (`leaderboard/leaderboard_impl.py:48-66`, exercised throughout `leaderboard/_rules.py` and `leaderboard/_cw.py`): nothing in `Leaderboard.__init__` checks `table.index.is_unique`. Every ranking method that does `self.ranks.loc[model]` (minimax, condorcet/copeland's majority graph, CW's `_find_weights_for_majority_graph`) assumes that expression returns a 1-D `Series`; for a duplicated model name it returns a 2-D `DataFrame` instead, and the subsequent boolean comparison against the full `ranks`/`majority_graph` table either raises an opaque pandas alignment error or (depending on pandas version/comparison operator) silently mis-aligns and produces wrong pairwise-win counts for every model. `data_processing.py`'s `preprocess_glue`/`preprocess_sglue` both explicitly dedupe repeated `"Model"` values before constructing the table specifically because scraped leaderboards routinely contain duplicate names -- so a caller who forgets that step (or merges two of these tables themselves) will feed `Leaderboard` exactly the input class of data this codebase already knows is common. Suggested fix: raise a clear `ValueError` in `__init__` when `not table.index.is_unique`, pointing at `preprocess_glue`/`preprocess_sglue` as the fix.

**F3** (`data_processing.py:94-102`): `preprocess_value` does `value.index = ["Human", "craig.starr", "DuKG", "HERO 1", ..., "HERO 4"]` -- a hardcoded 7-entry list assigned positionally, with no assertion that `len(value) == 7` (a length mismatch would at least raise `ValueError: Length mismatch`) and, more importantly, no check that the actual row order in the scraped table matches this assumed roster order. If the VALUE leaderboard source table is ever re-scraped in a different row order (new submission inserted mid-table, site re-sorts by score, etc.), every downstream row is silently mislabeled with the wrong model's name, exactly the "silently wrong results" bug class this audit specifically hunts for. Suggested fix: read the actual model-name column (if the source table's `"Model"` column, dropped nowhere else in this function, still carries the real names) instead of overwriting with a hardcoded list, or at minimum assert the hardcoded list matches the source `"Model"` column before overwriting.

**F4** (`leaderboard/_rules.py:20-22`): `gmean(table, axis=1, weights=self.weights)` with no domain check. A negative or exactly-zero score in any single task column (plausible for a raw margin/log-loss-style metric) drives that model's entire geometric-mean aggregate to `0`/`NaN`, silently and without a warning that the geometric mean's positivity requirement was violated. Suggested fix: validate `(table > 0).all().all()` (post-fill) before calling `gmean`, or document the constraint prominently in the docstring.

**F5** (`leaderboard/_rules.py:19,160`): both formulas divide by `self.weights.sum()`. A `Leaderboard` constructed with e.g. `weights={t: 0.0 for t in tasks}` (all-zero, a legal `Dict[str, float]`) makes every model's arithmetic-mean/optimality-gap score `NaN`/`inf` with no error. Suggested fix: raise in `__init__` (or in these two methods) when `self.weights.sum() <= 0`.

**F6** (`leaderboard/_rules.py`, multiple functions): `borda_ranking`, `dowdall_ranking`, `plurality_ranking` (`_approval_ranking`), `threshold_election`, and `baldwin_election` all read `self.ranks`/`self.max_ranks` directly without the `fillna(table.median())` step `mean_ranking`/`optimality_gap_ranking` apply, and none of them are in `PARTIAL_METHODS` (`leaderboard/settings.py:19`) -- so `elect_all`/`rank_all` correctly skip them on a partial table. But nothing stops a caller from invoking `lb.borda_ranking()` directly on a partial `Leaderboard`; `pandas.sum(axis=1)` silently skips the `NaN` entries per model, so two models with different amounts of missing data end up compared on effectively different numbers of tasks, with no error or warning surfaced. Suggested fix: add an `if self.is_partial: raise ValueError(...)` guard (or an explicit fill strategy) to each of these methods, matching the protection `elect_all`/`rank_all` already apply at the dispatch layer.

**F7** (`shapley_blend.py:169` vs `:186-189`): docstring: `"members with value <= prune_below * values.sum() are pruned (prune_below=0.0 keeps every non-negative-value member)"`. Code: `weights = np.clip(values, 0.0, None); ... keep_mask = weights > threshold` where `threshold = prune_below * total = 0.0`. `weights > 0.0` is `False` for any model whose (clipped) weight is exactly `0.0` -- including every model with a genuinely negative raw Shapley value (clipped to 0) AND any model whose raw Shapley value happened to be exactly `0.0`. So the actual behavior is "keeps every *positive*-value member", not "non-negative". Suggested fix: reword the docstring (cheap, no behavior change needed -- the code's strict-pruning behavior is the more defensible one for an ensemble-pruning function).

**F8** (`confidence_gated_blend.py:185-190`):
```python
if backend == "cupy":
    try:
        return _blend_cupy(...)
    except Exception:
        pass  # GPU unavailable/failed -> fall through to numpy.
```
The comment documents *that* it falls through, but the `except` block does not log anything (not even `logger.debug`), unlike `_confidence_gated_blend_ktc_dispatch.py`'s own cupy probe (`logger.debug("confidence_gated_blend tuner: cupy unavailable/failed (%s)", exc)`) for the identical failure class. A real GPU failure (OOM, driver mismatch, stale context) at call time is now completely invisible to anyone reading logs -- the function just quietly runs 25x slower via numpy with zero trace. Suggested fix: add the same `logger.debug(...)` call as the dispatch module for consistency and debuggability.

**F9** (`knn_fallback_predictor.py:47-65`): `_X_train`/`_y_train` default to `np.empty((0,0))`/`np.empty(0)` in `__init__`. If `predict()` is called before `fit()` (or `fit()` was called with 0 rows), `knn_search` hits its documented `n_sub == 0` branch and returns sentinel `ids` filled with `0` "so callers can downstream-process without an empty-set branch" -- but that contract assumes the caller's own target array isn't *also* empty. Here `self._y_train[ids]` with an empty `self._y_train` and `ids` full of `0` raises `IndexError: index 0 is out of bounds for axis 0 with size 0`, a confusing failure mode for what is really just "you forgot to call fit()". Suggested fix: an explicit early check (`if self._y_train.size == 0: raise ValueError("KNNFallbackPredictor.predict() called before fit()")`) in `predict()`.

**F10** (`dual_optimizer_blend.py:41-57`): `_coordinate_descent_simplex_search`'s inner loop does `i, j = rng.choice(n_models, size=2, replace=False)` every iteration; for `n_models == 1` `np.random.Generator.choice` raises `ValueError: Cannot take a larger sample than population when 'replace=False'`. `dual_optimizer_weight_blend` never validates `len(oof_preds) >= 2` before optionally calling this path (`include_coord_descent=True` is opt-in but unguarded), so a legitimate (if degenerate) single-candidate blend crashes only when the opt-in flag happens to be set. Suggested fix: guard at the top of `dual_optimizer_weight_blend` (or inside the helper) with a clear message when `n_models < 2` and `include_coord_descent=True`.

**F11** (`fairness_computation.py:26-64`): every scorer (`naive_masking_score`, `naive_t5_score`, `naive_gpt2_score`) unconditionally moves tensors to `.cuda()`; there is no CPU fallback and no check of the repo's own `MLFRAME_DISABLE_GPU`/`CUDA_VISIBLE_DEVICES=""` convention documented elsewhere in this same package (e.g. `_confidence_gated_blend_ktc_dispatch.py`). The module docstring is upfront about this ("CUDA-only... unconditionally"), so it reads as a deliberate simplification for this legacy fairness-benchmark research code rather than an oversight -- flagging it because it is a real functional gap (module cannot run at all without a GPU) and because the rest of the package treats GPU-availability handling as a first-class concern. Suggested fix (if this module is meant to stay usable going forward): add a `device` parameter defaulting to `"cuda" if torch.cuda.is_available() else "cpu"`.

**F12** (`stability_exp.py:19`): `sns.set(style="whitegrid")` executes at module import time, at module scope, unconditionally. Any other code (a notebook, a different plotting module, a test) that happens to `import mlframe.votenrank.stability_exp` for its `spearman_exp`/`count_and_plot` functions gets its global seaborn style silently rewritten as a side effect of the import, even if it never calls a plotting function from this module. Suggested fix: move the `sns.set(...)` call inside `create_exp_pic`/`count_and_plot` (the actual plotting entry points) instead of at import time.

**F13** (`utils.py:26`, `iia_exp.py:51`, `leaderboard/leaderboard_impl.py:219`): three comments read `"# Wave 60 (2026-05-20): ..."`, `"# Wave 49 (2026-05-20): ..."`, `"# Wave 31 (2026-05-20): ..."`. `CLAUDE.md`'s "Comment style" section explicitly bans this pattern ("no phase/wave markers ... date stamps ... that belongs in git history / the PR description"). The WHY content of each comment is genuinely useful and should stay; only the `"Wave N (date):"` prefix should go. Suggested fix: strip the wave/date prefix, keep the rest of the sentence (e.g. `"# Use a local Generator instead of mutating the process-global np.random RNG."`).

**F14** (`leaderboard/_rules.py`, `leaderboard/_cw.py`): grepping `tests/` for `condorcet|copeland|minimax|baldwin|two_step_ranking|get_meta_leaderboard|find_weights_for_condorcet|split_models_by_feasibility` turns up only `tests/votenrank/test_regression_minimax_winning_votes.py` (minimax only) and `tests/votenrank/test_regression_build_ranks_resets_majority_graph.py` (majority-graph invalidation only, not a specific election method's correctness) as real votenrank-cluster tests. `test_reproducibility_method_order.py` does call `elect_all()`/`rank_all()` (which touches every method including condorcet/baldwin/copeland/threshold), but only asserts the *column/row order* is `PYTHONHASHSEED`-stable, never the *values* -- so a correctness regression in e.g. `copeland_ranking(slice_type="upper")` or `_cw.py`'s LP-based `_find_weights_for_majority_graph` would not be caught by any test in the repo today. Suggested fix: add small hand-computable fixtures (3-4 models, 2-3 tasks with a known pairwise-majority structure) with expected winners/weights for each of these paths, per this repo's own testing convention.

**F15** (`leaderboard/_cw.py:20,90-93`): `_find_weights_for_majority_graph(self, edge_list, restrictions=None)` returns `{task: weight, ...}` on success or the literal string `"infeasible"` on failure (`leaderboard_impl.py:278`, `:373` -- i.e. `find_weights_for_condorcet`/`split_models_by_feasibility` compare against the magic string). This is a legitimate, working pattern but a fragile one: nothing prevents a typo'd comparison string at a future call site, and the return type can't be expressed cleanly for a type checker. Suggested fix: return `None` on infeasibility (more idiomatic, and `is None`-checkable) or raise a dedicated `InfeasibleWeightsError`.

## Proposals

| ID | Category | File:Line | Summary |
|----|----------|-----------|---------|
| PR1 | feature-parity | leaderboard/leaderboard_impl.py:111-142 vs :144-207 | `rank_all` supports `task_groups`/two-step grouped ranking; `elect_all` has no equivalent, so a caller wanting a grouped ELECTION (not just a ranking) has to hand-roll it via `get_meta_leaderboard` + a manual `*_election` call. |
| PR2 | docs/tests | leaderboard/leaderboard_impl.py:94-109 | `_ensure_majority_graph`'s NaN handling (a task is symmetrically excluded from a pairwise comparison whenever either side is missing it) looks correct on inspection but is undocumented; two models with completely disjoint task coverage end up recorded as a full "tie" (`0.5` in the majority graph) purely by construction (0 wins == 0 losses on both sides). Worth a one-line comment plus a regression test pinning this as intended, not accidental, behavior. |
| PR3 | test-coverage | leaderboard/leaderboard_impl.py:48-66 | Add regression tests for F1 (typo'd weight-dict key) and F2 (duplicate model names) once those are fixed, using the real failure signature as the assertion target per this repo's bug-fix convention. |
| PR4 | perf/CI | _benchmarks/*.py | All 12 benchmark scripts are well-written `__main__`-only harnesses (good identity checks, real cProfile output) but none are wired into any CI perf-regression gate -- they only run when a human invokes them manually. Worth considering a lightweight "assert measured speedup direction hasn't inverted" smoke test for at least the ones with an embedded correctness assertion (`bench_minimax_winning_votes.py`'s `assert a.equals(b)`), since that one currently doubles as an un-collected regression test. |
| PR5 | robustness | knn_fallback_predictor.py:35-39 | `KNNFallbackPredictor.__init__` accepts any `k`/`metric` without validation (e.g. `k=0` or `k=-1`); the error only surfaces later and indirectly inside `knn_search`. An eager `if k < 1: raise ValueError(...)` in `__init__` would give a much clearer failure at construction time. |

## Coverage notes

Nothing in this cluster's own 36 files was skipped or only partially read -- every file was small enough to read in full (largest is 283 LOC) and I read every line of every one. The one external boundary I intentionally did not go deeper than: `mlframe.feature_engineering.transformer.knn_search` (imported by `knn_fallback_predictor.py` and `similarity_blend.py`) and `mlframe.training.composite.ensemble.stacking.residual_correlation_matrix` (imported by `correlation_diversity_ablation.py`) -- per the audit instructions, I read only their signatures/docstrings for calling-convention context and did not analyze their internals or report findings about them, since both live outside this cluster's scope. `mlframe.metrics.fast_roc_auc` (imported by `shapley_blend.py`'s default score function) was likewise only referenced by name, not opened.

```json
{"cluster": "votenrank", "report_file": "votenrank.md", "files_reviewed": 36, "loc_reviewed": 4226, "p0": 0, "p1": 3, "p2": 12, "proposals": 5, "headline": "Leaderboard.__init__ never validates weights-dict keys or duplicate model names, letting a typo'd weight key silently corrupt the mean_ranking/optimality_gap_ranking score denominator and duplicate model names silently break every pairwise (.loc[model]) comparison in _rules.py/_cw.py -- exactly the kind of malformed input data_processing.py's own dedup logic shows this codebase already expects to see."}
```
