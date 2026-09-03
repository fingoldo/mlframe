# Audit: votenrank

**Scope**: `src/mlframe/votenrank/**` (rank-aggregation / ensemble-blending / voting utilities). Excludes
`feature_selection/filters/**` and `feature_selection/shap_proxied_fs/**` (out of scope, separately audited
2026-07-25).

**Files reviewed**: 38 (.py files, `__pycache__`/compiled artifacts excluded) — every file read in full, including
all 13 `_benchmarks/` harness scripts.

**LOC reviewed**: 4,276 (per `wc -l` over all in-scope `.py` files).

**Context**: this module already carries visible scars from a prior, narrower audit pass — comments tagged
`F1`/`F2`/`F3`/`F5`/`F6`/`F10` and a `tests/votenrank/test_votenrank_fixes.py` regression suite pinning 15
previously-found bugs (F1–F15). This report only lists *new* findings not already covered by that suite (each
candidate finding was cross-checked against `test_votenrank_fixes.py` before being included; one candidate —
`confidence_gated_blend`'s cupy-fallback debug-level logging — was DROPPED because F8 already pins that exact
behavior as intentional).

## Findings

| ID | Severity | File:Line | Summary | Suggested fix | Meta-test idea |
|----|----------|-----------|---------|----------------|-----------------|
| VOTENRANK-1 | P1 | `shapley_blend.py:192-205` | Degenerate-fallback branch of `shapley_blend` can silently return an all-zero `ensemble_pred` (and all-zero `weights`) while still reporting a "selected" model, when every model's raw Shapley value is <= 0. | In the fallback branch, set `survivor_weights[best_idx] = 1.0` directly instead of relying on `weights * keep_mask` (which is 0 when the clipped weight of the argmax model is 0). | Property test: for any input where every Shapley value is forced <= 0 (mock `score_fn` to always return a constant), assert `ensemble_pred` is NOT all-zero and equals (or is close to) the argmax model's own raw prediction; generalize as a scanner rule "any function with an explicit degenerate-fallback branch documented as 'falls back to X' must be asserted, via a forced-degenerate unit test, to actually return X's own values, not a zero/derived-from-clipped-value substitute." |
| VOTENRANK-2 | P2 | `shapley_blend.py:118-134` (`_permutation_shapley`) | `v_prev = float(score_fn(y, zeros))` (the empty-coalition score) is recomputed once per permutation (up to `n_permutations`=200 times by default) even though it is a constant already computed once by the caller as `v_empty` — wastes `1/n_models` of the total `score_fn` call budget. | Thread `v_empty` (already computed in `shapley_model_values`) into `_permutation_shapley`/`_msr_banzhaf` as a parameter instead of recomputing `score_fn(y, zeros)` inside the permutation loop. | Grep/AST meta-test: flag any function that recomputes an expression identical to one already computed in its caller and passed nowhere, inside a loop bounded by a "reps"/"n_iterations"/"n_permutations" parameter — a generic "loop-invariant score/metric-fn call" checker. |
| VOTENRANK-3 | P2 | `shapley_blend.py:23-35` (`_default_score_fn`) | Silently falls back to negative-RMSE (treating `y` as a continuous regression target) for ANY `y` whose cardinality is not exactly 2 — including a degenerate single-class `y` (e.g. from a bootstrapped/permuted coalition-scoring subsample) or an integer-coded multiclass target, both of which produce a numerically meaningless "score" with no warning. | Either raise a clear `ValueError` for cardinality > 2 (multiclass classification needs an explicit multiclass-aware `score_fn`, not a silent RMSE-on-labels fallback), or add a genuine multiclass branch (e.g. macro-OVR AUC). | Unit test: call `shapley_model_values`/`shapley_blend` with an integer-coded 3-class `y` and no explicit `score_fn`; assert it raises rather than silently returning a garbage "RMSE on class-label" score. Meta-test: grep for any `score_fn`/`metric_fn` default implementation that branches on `len(np.unique(y))==2` without an explicit `else: raise` for higher cardinalities. |
| VOTENRANK-4 | P2 | `similarity_blend.py:141-160` (`region_similarity_weights`) | Docstring promises the returned per-row weight matrix "rows summing to 1", but for a query row whose k-NN mean-distance to EVERY region underflows `exp(-mean_dist/similarity_scale)` to exactly `0.0` (plausible with the default `similarity_scale=1.0` on unscaled features and a genuinely far-OOD row), the `row_sums > 1e-12` guard falls back to dividing by `1.0`, silently returning an all-zero weight row instead of e.g. a uniform or nearest-region fallback — `predict_multi_region` then returns `0.0` for that row instead of any region's actual prediction. | When `row_sums <= 1e-12` for a row, fall back to uniform weights (`1/n_regions`) or route to the single nearest region, rather than leaving the row at all-zero. | Property test: construct a query row with `mean_dist` orders of magnitude past the `similarity_scale` underflow threshold (e.g. `similarity_scale=0.01`, row far from every region) and assert `region_similarity_weights(...).sum(axis=1)` is still `1.0` (not `0.0`) for that row. |
| VOTENRANK-5 | P2 | `correlation_diversity_ablation.py:147-149` | `np.nanmax(np.abs(off_diag))` returns `NaN` when a candidate model's entire correlation row against every other model is `NaN` (Pearson correlation is undefined for a constant/zero-variance prediction column) — the subsequent `max_corr < correlation_threshold` comparison is always `False` against `NaN`, so a genuinely constant-prediction model is silently EXCLUDED from the diversity-ablation report rather than flagged, and `np.nanmax` on an all-NaN slice additionally emits a noisy `RuntimeWarning`. | Explicitly detect an all-NaN correlation row (e.g. `np.isnan(off_diag).all()`) and either treat it as `max_corr = 0.0` (unambiguously "low correlation", since a constant predictor is trivially uncorrelated with anything) with a logged warning, or skip it with an explicit documented reason instead of silent NaN-driven exclusion. | Unit test: include one constant-prediction model (e.g. `np.full(n, 0.5)`) in `oof_preds` with an individual score below the best model's, and assert it appears (or is explicitly, loudly skipped) in `diversity_ablation_report`'s output rather than silently vanishing with no warning captured. |
| VOTENRANK-6 | P2 | `hill_climb.py:34-82, 145-150` | Neither `hill_climb_ensemble` nor `_hill_climb_single_path` validates that every array in `oof_preds` shares `y_true`'s shape before use — a mismatched-shape prediction array (e.g. `(n,)` vs `(n,1)`, or a different `n`) silently broadcasts (`running_sum + preds[j]`) into a wrong-shaped result instead of raising a clear error, producing a silently-wrong blend/score. | Validate `all(p.shape == y.shape for p in preds)` right after the `np.asarray` conversions in `hill_climb_ensemble`, raising a `ValueError` naming the offending index. | Property test: pass one `oof_preds` entry with an extra trailing axis of size 1 (`(n, 1)` instead of `(n,)`) alongside otherwise-valid `(n,)` arrays; assert `hill_climb_ensemble` raises rather than returning a `(n, n)`-shaped `ensemble_pred`/silently-wrong score. Same shape-guard is applicable to `constrained_weight_blend`, `geometric_weight_blend`, `dual_optimizer_weight_blend`, `adversarial_stochastic_blend` (all `np.stack`/`np.asarray` over caller-supplied prediction arrays without an explicit shape check) — a generic "every public ensemble-blend entry point validates all its `(n,)` array inputs share one shape before use" scanner rule would catch the whole family at once. |
| VOTENRANK-7 | P2 | `leaderboard/_rules.py:167` (`minimax_ranking`) | `models_scores.drop(model).max()` is `NaN` when the leaderboard has exactly 1 model (`.drop(model)` empties the Series) for every `score_type`. The resulting single-entry `NaN` ranking makes `ranking2top`'s `ranking == ranking.max()` comparison (`NaN == NaN` is always `False`) return an EMPTY list, so `minimax_election()` on a trivially-single-model leaderboard silently returns no winner instead of the one model. | Special-case `n_models == 1` in `minimax_ranking` (return a 1-entry Series with any finite sentinel score, e.g. `0.0`, for the sole model) rather than letting `.drop(model).max()` degrade to `NaN`. | Unit test: build a `Leaderboard` with exactly 1 model row and 2+ task columns, call `minimax_election()`, assert it returns `[the_one_model]` (not `[]`). Generalize: a parametrized `n_models in (1, 2, 3)` sweep over every `*_election` method already exists in spirit for F14's hand-computable tests — extend that sweep to `n_models=1` specifically, since `.drop(model)`/`.iloc[:2]`-shaped code is exactly where a single-row leaderboard breaks silently. |
| VOTENRANK-8 | P3 | `data_processing.py:100`, `dual_optimizer_blend.py:162`, `leaderboard/leaderboard_impl.py:64,72,80` | Five comments embed literal prior-audit finding-ID markers (`# F1:`, `# F2:`, `# F3:`, `# F5:`, `# F10:`) directly in production source — exactly the "phase/wave markers, finding IDs" pattern the repo's own `CLAUDE.md` explicitly forbids in code comments ("belongs in git history / the PR description"). The existing `test_f13_no_wave_date_markers_remain` regression test only greps for the literal strings `"Wave "` and `"(2026-05-20)"`, so it gives false confidence that all audit-wave markers were purged — it does not catch this `F<N>:` pattern at all. | Strip the `F<N>:` prefixes from all 5 comments, keeping the substantive WHY-explanation text (e.g. `# F1: a typo'd/nonexistent...` -> `# A typo'd/nonexistent...`). Broaden `test_f13_...` (or add a sibling test) to grep for `re.search(r"#\s*F\d+[:.]", src)` across the whole `votenrank` package, not just 3 named modules and 2 literal strings. | Scanner rule: a repo-wide regex/AST check (`#\s*(F|WAVE|FINDING)[-_ ]?\d+\b`) run over every `.py` file, flagged as a hygiene CI gate — catches this exact pattern anywhere it recurs, not just in votenrank. |
| VOTENRANK-9 | P3 | `shapley_blend.py:154-156` (`_bootstrap_stderr`) | Function is named `_bootstrap_stderr` but its own docstring says "not a true bootstrap" — it's `abs(values)/sqrt(n_permutations) + eps`, a plain analytic standard-error-of-the-mean proxy, not a resampling-based bootstrap estimate. The name actively misleads a reader skimming for where bootstrap resampling happens in this module. | Rename to `_analytic_stderr_proxy` (or similar) to match the docstring's own honest description. | Meta-test: grep for `def _?bootstrap_\w+` whose docstring contains the substring "not a (true |real )?bootstrap" — a self-contradicting name/docstring pair, generically detectable. |
| VOTENRANK-10 | P3 | `shapley_blend.py:137-151` (`_msr_banzhaf`) | The `n_coalitions` parameter is passed in from `shapley_model_values` under the name `n_permutations` (shared call-site signature with `_permutation_shapley`), but `_msr_banzhaf` samples independent coalitions, not permutations — reading `_msr_banzhaf` standalone, the parameter name is actively wrong for what it represents (each of the `n_coalitions` draws is one random coalition mask, never an ordering). | Rename the local parameter to `n_coalitions` inside `_msr_banzhaf` (keep the outer `shapley_model_values(..., n_permutations=...)` public name, since it's a shared, estimator-agnostic knob) or add a one-line docstring clarification at the top of the function. | N/A (naming-only; no generic scanner applies beyond a manual per-function name/semantics review). |
| VOTENRANK-11 | P3 | `confidence_gated_blend.py:140-146` | `per_sample_gate_calibration=True`'s docstring says `calibration_confidence`/`calibration_reliability` "must be disjoint from ... to avoid leaking test-set reliability into the gate" but nothing in the code checks for overlap — the leakage-avoidance contract is entirely caller-enforced, undocumented at the call site (no runtime check, no logged warning), and easy to violate silently (e.g. accidentally passing the same array object for both the blended rows and the calibration set). | Add a cheap heuristic guard (e.g. warn if `calibration_confidence` is the same object as / element-wise-equal to `auxiliary_confidence`) — can't detect general statistical overlap, but catches the common accidental-reuse mistake. | Unit test: call `confidence_gated_blend(..., per_sample_gate_calibration=True, calibration_confidence=auxiliary_confidence, calibration_reliability=...)` (deliberately reusing the exact blended array as its own calibration set) and assert a warning is logged. |
| VOTENRANK-12 | P3 | `stability_exp.py:43-44` (`spearman_exp`) | `for idx in nan_idxs_prod: table_nan.iloc[idx % rows, idx // rows] = np.nan` sets NaN cells one at a time via per-cell `.iloc` assignment inside a Python loop — each call pays pandas' per-cell-setitem overhead (block manager lookup + possible dtype-upcast check) instead of a single vectorized write. | Convert to a numpy view once (`arr = table_nan.to_numpy(); arr.flat[idxs] = np.nan` with a matching flatten order, or `np.unravel_index`), assign back once, rather than looping `.iloc[...] = ` per index. | Micro-benchmark regression: time `spearman_exp` with `nan_number` in the low thousands before/after the vectorized rewrite; assert the vectorized path is not slower and produces bit-identical NaN placement. |
| VOTENRANK-13 | P3 | `iia_exp.py:32,35` (`compute_iia_for_fixed_models`) | `Leaderboard(table.loc[models_order[:2]], weights)` and the subsequent `range(3, len(models_order) + 1)` loop assume `models_order` (i.e. `table.index`) has at least 3 entries; a leaderboard with fewer than 3 models silently produces a degenerate/misleading result (e.g. a 2-row or 1-row `Leaderboard`, an empty-range loop returning `result=0` unconditionally) rather than a clear error explaining IIA analysis needs >= 3 models to be meaningful. | Add an explicit `if len(table.index) < 3: raise ValueError(...)` guard at the top of `compute_iia`/`compute_iia_for_fixed_models`. | Unit test: call `compute_iia(method, a_2_row_table, weights, num_repetitions=5)` and assert it raises a clear error instead of silently returning `(0.0, 0.0, [0]*5)`. |
| VOTENRANK-14 | P2 | `data_processing.py:63-64` (`get_tracker_table`) | `model, task, _ = f.split("_")` assumes every experiment-impact-tracker output directory name has EXACTLY 3 underscore-separated components; any real model or task name that itself contains an underscore (common — e.g. `"meta-llama_Llama-3"`, `"cola_dev"`) makes `f.split("_")` return more (or fewer) than 3 parts, raising an unpacking `ValueError` and aborting the whole tracker-table load, or (for names with exactly one embedded underscore) silently misassigning fields with no validation. | Split with an explicit `maxsplit` bound anchored to the known trailing `_0` suffix from `tracker_filename` (e.g. `f.rsplit("_", 2)` if the third component is always the literal `"0"`), or store/parse a structured separator instead of relying on plain `"_"` splitting of caller-controlled names. | Unit test: call `get_tracker_table` (or its filename-parsing helper, factored out) against a synthetic `dirpath` containing a directory name like `"my_model_cola_dev_0"` (task with an embedded underscore) and assert it either parses correctly or raises a clear, named error — not a bare `ValueError: too many values to unpack`. |

## Findings by dimension

- **Correctness bugs**: VOTENRANK-1 (silently-wrong all-zero ensemble output), VOTENRANK-7 (empty election result for a valid single-model input), VOTENRANK-14 (fragile filename parsing that can crash or misparse on realistic model/task names).
- **ML correctness (leakage/reproducibility/calibration)**: VOTENRANK-3 (silently-wrong metric on a plausible target-cardinality mismatch), VOTENRANK-11 (leakage contract is caller-trusted, no runtime guard). No unseeded-RNG or hidden-global-state issues were found — every stochastic function in scope (`hill_climb`'s bagging, `shapley_blend`'s permutation/MSR-Banzhaf estimators, `adversarial_stochastic_blend`, `dual_optimizer_blend`'s coordinate descent, `stability_exp`, `iia_exp`) already threads an explicit `np.random.Generator`/`random_state` end to end, with the process-global-RNG anti-pattern already fixed and commented on in `stability_exp.py`/`iia_exp.py` (visible prior-audit remediation). sample_weight is correctly threaded through `SimilarityBlendEnsemble.fit`/`fit_multi_region`.
- **Computational efficiency**: VOTENRANK-2 (redundant loop-invariant recomputation), VOTENRANK-12 (per-cell pandas writes instead of a vectorized array assignment). No O(n^2)-should-be-O(n log n) algorithmic issues were found in the core blending/aggregation math; `correlation_diversity_ablation.py`'s own docstring already documents a prior perf fix (O(n_flagged * n_models * n_samples) -> O(n_models * n_samples)) and `_rules.py`'s `minimax_ranking` already documents a prior redundant-pass removal (see `bench_minimax_winning_votes.py`).
- **Edge cases and robustness**: VOTENRANK-4 (extreme-distance underflow breaking a documented invariant), VOTENRANK-5 (all-NaN correlation row silently excludes a constant-prediction model), VOTENRANK-6 (no shape validation across the whole blend-function family), VOTENRANK-7, VOTENRANK-13 (no minimum-model-count validation for IIA analysis). No crashes were found for empty/all-NaN/all-constant TABLE input to `Leaderboard` itself — the constructor already validates duplicate model names, unknown weight keys, and all-zero weights (F1/F2/F5, prior audit).
- **Test coverage gaps**: every P1/P2 finding above (VOTENRANK-1, -2, -3, -4, -5, -6, -7, -14) represents an untested code path — confirmed via cross-reference against `tests/votenrank/*` (18 test files, ~120+ test functions) before inclusion; none of the biz_val/regression suites exercise a degenerate all-non-positive Shapley pool, a single-model `Leaderboard`, a shape-mismatched `oof_preds` pool, an all-NaN correlation row, or an extreme-underflow similarity-blend row.
- **Code quality/architecture**: VOTENRANK-8 (stale audit-ID comments), VOTENRANK-9, VOTENRANK-10 (misleading names). No dead code, no overly-broad bare `except:` clauses were found in scope (all `except Exception` sites are narrowly justified with a comment and either re-raise-equivalent behavior or an explicitly-tested fallback — see F8 discussion above). No missing type hints / implicit-Optional patterns were found; the module is consistently `from __future__ import annotations`-typed with explicit `Optional[...]` throughout.
- **OSS/hygiene**: VOTENRANK-8 covers the only comment-cruft/stale-marker issue found. No mojibake, no em-dash-in-prose issues, no stale/misleading docstrings beyond VOTENRANK-9/-10's naming mismatches were found; docstrings throughout are unusually thorough and mostly accurate against the code they describe (a notable positive — several docstrings explicitly document known limitations, e.g. `mean_ranking`'s geometric-mean domain constraint, `hill_climb_ensemble`'s bagged-vs-single-path score semantics).

## Narrative

**VOTENRANK-1** (`shapley_blend.py`): Traced by hand. When every model's raw Shapley value is `<= 0` (a
realistic scenario — e.g. a model pool that all individually hurt the blend relative to the zero-baseline
under RMSE), `weights = np.clip(values, 0.0, None)` zeroes every entry, so `keep_mask.any()` is `False` and
the code enters its documented "fall back to the single best model" branch: `keep_mask[argmax(values)] =
True`. But `survivor_weights = weights * keep_mask` still multiplies by the ALREADY-CLIPPED `weights` array,
whose entry at `argmax(values)` is `0.0` (clipped, since the value itself was negative) — so
`survivor_weights` stays all-zero, `renormalize`'s `survivor_weights.sum() > 0` guard skips the rescale, and
the `ensemble_pred` accumulation loop (`if survivor_weights[m] > 0: ...`) never fires, leaving `ensemble_pred
= 0`. The function's own comment says this branch exists "rather than returning an empty, unusable ensemble"
— but that is exactly what it returns (`ensemble_pred` identical to the empty-coalition baseline), while
`selected_indices` misleadingly reports one model as chosen. No existing test in `test_biz_val_shapley_blend.py`
constructs this all-non-positive-values scenario.

**VOTENRANK-2** (`shapley_blend.py`): `_permutation_shapley`'s inner loop starts every permutation by
recomputing `v_prev = float(score_fn(y, np.zeros(...)))` — but this score is a pure function of `(y, zeros)`,
identical across all `n_permutations` iterations, and is ALREADY computed once by `shapley_model_values` as
`v_empty` (line 103) before dispatching to the estimator. The estimator functions never receive `v_empty`, so
this constant is silently recomputed `n_permutations` times (default 200), each an `O(n_rows)` AUC-sort/RMSE
call — a real, easily-avoidable fraction of the module's own documented cost model
(`n_permutations * n_models` evals; this adds `n_permutations` more, i.e. `1/n_models` extra overhead).

**VOTENRANK-3** (`shapley_blend.py`): `_default_score_fn` branches solely on `len(np.unique(y)) == 2`; any
other cardinality (0, 1, or >= 3 distinct values) falls through to `-sqrt(mean((y - blended) ** 2))`, i.e.
literal RMSE treating `y` as a continuous target. A caller passing an integer-coded 3+-class classification
target (a very natural mistake, since nothing about the function name signals a binary-only assumption beyond
the docstring) gets a numerically well-defined but semantically meaningless "score" with zero warning —
Shapley values computed from it would still "look" valid (finite floats, efficiency-axiom-satisfying sums)
while being uninterpretable as ensemble-selection guidance.

**VOTENRANK-4** (`similarity_blend.py`): `region_similarity_weights`'s docstring explicitly promises "rows
summing to 1". The implementation computes `sims[:, i] = exp(-mean_dist / similarity_scale)` per region, then
`row_sums = sims.sum(axis=1); row_sums = np.where(row_sums > 1e-12, row_sums, 1.0)`. `exp(-x)` underflows to
exactly `0.0` in float64 once `x` exceeds ~745.13 — entirely plausible with the class's own default
`similarity_scale=1.0` if a query row's k-NN mean-distance to every region's training set is even moderately
large (unscaled real-world features routinely have Euclidean distances in the hundreds to thousands). When
that happens for every region simultaneously, `row_sums` is `0.0`, the guard divides by the fallback `1.0`
instead, and the returned row is all-zero — silently violating the stated invariant and making
`predict_multi_region` return a flat `0.0` for that row (not even the nearest region's own prediction). No
test in `test_biz_val_similarity_blend.py` probes a row this far from every region.

**VOTENRANK-5** (`correlation_diversity_ablation.py`): `off_diag = np.delete(corr_matrix[i], i); max_corr =
float(np.nanmax(np.abs(off_diag))) if off_diag.size > 0 else 0.0`. This correctly guards the *empty* case but
not the *all-NaN* case: `residual_correlation_matrix` (Pearson correlation) produces `NaN` for any pairing
involving a zero-variance (constant) prediction column, so a model whose predictions happen to be constant
gets an all-NaN row here. `np.nanmax` on an all-NaN slice returns `NaN` (with a `RuntimeWarning`), and
`NaN < correlation_threshold` is `False` under IEEE-754 comparison semantics, so `is_low_correlation` is
`False` for that candidate — it never gets flagged for ablation at all, with zero indication to the caller
that this happened for a NaN-driven reason rather than a genuine high-correlation reason.

**VOTENRANK-6** (`hill_climb.py`): Confirmed by reading `_hill_climb_single_path` and
`hill_climb_ensemble` end to end — the only validation performed is `n_models == 0` (empty pool) and the
per-array `np.asarray(p, dtype=np.float64)` dtype coercion; no code path checks `p.shape == y.shape` for each
prediction array. `trial_sum = running_sum + preds[j]` (and the analogous `np.tensordot`/`np.stack` calls in
the sibling blend functions `constrained_weight_blend`, `geometric_weight_blend`,
`dual_optimizer_weight_blend`, `adversarial_stochastic_blend`) will happily broadcast a shape mismatch
(e.g. one `(n,)` array and one `(n, 1)` array) into an unintended `(n, n)` result rather than raising — this
is the same "silent broadcasting" bug class the project's own memory notes call out (dimension #1's "type
coercion bugs" / #4's "malformed input types"), just not yet closed in this module family.

**VOTENRANK-7** (`leaderboard/_rules.py`): Hand-traced with `self.models = ["m1"]` (a 1-row `Leaderboard`).
`self.ranks.loc["m1"] < self.ranks` compares the single row to itself, giving all-`False`; both `models_scores`
and the `does_win` comparison's RHS are therefore `0`, so `does_win` is `False` and `models_scores` (after
`*= does_win`) is `0` for the one row. `models_scores.drop("m1")` then empties the Series entirely, and
`.max()` on an empty `Series` returns `NaN` (pandas). The outer `-pd.Series(data=[NaN], ...)` stays `NaN`,
`ranking2top`'s `ranking == ranking.max()` compares `NaN == NaN` (always `False` under IEEE-754), so
`minimax_election()` returns `[]` for a leaderboard where the trivially-correct answer is `["m1"]`. This is
a real, silently-wrong output for a legitimate (if edge-case) input shape the constructor never rejects, and
is exactly the "single-row input" edge case this audit's dimension #4 calls out explicitly. Verified this is
specific to `minimax_ranking`'s `.drop(model)` pattern — `condorcet_election` and `copeland_ranking` were
traced through the same n=1 scenario and handle it correctly via the majority-graph's diagonal `+ 0.5` term.

**VOTENRANK-8** (multiple files): A grep across the whole votenrank tree for the literal-finding-ID comment
pattern (`# F\d+[:.]`) found 5 live occurrences, all clearly artifacts of a prior audit-fix pass (each reads
like `# F1: a typo'd/nonexistent task key would otherwise...`). The repo's own `CLAUDE.md` states this exact
pattern is forbidden ("no phase/wave markers, finding IDs, date stamps ... that belongs in git history / the
PR description"), and `tests/votenrank/test_votenrank_fixes.py::test_f13_no_wave_date_markers_remain` shows a
prior attempt was made to purge audit markers from votenrank specifically — but that test's assertion
(`"Wave " not in src or "(2026-05-20)" not in src`) checks only 3 named modules for 2 literal substrings,
missing this different-but-analogous marker pattern (`F<N>:`) entirely, including in a 4th module
(`dual_optimizer_blend.py`) the test doesn't even scan.

**VOTENRANK-9/-10** (`shapley_blend.py`): Both are pure naming issues found while reading the estimator
internals closely: `_bootstrap_stderr`'s own docstring immediately disclaims being a real bootstrap (it's an
analytic `|value|/sqrt(n)` proxy), and `_msr_banzhaf`'s `n_permutations` parameter name (inherited from the
shared `shapley_model_values(..., n_permutations=...)` call site) describes permutations even though the
Banzhaf estimator samples independent random coalition masks, never orderings — reading the function in
isolation, the parameter name actively misdescribes what it controls.

**VOTENRANK-11** (`confidence_gated_blend.py`): The docstring for `per_sample_gate_calibration` is explicit
and correct about the leakage risk ("Must be disjoint from ... to avoid leaking test-set reliability into the
gate") but this is purely an unenforced caller contract — nothing in `confidence_gated_blend`'s body checks
whether `calibration_confidence`/`calibration_reliability` overlap with the rows being blended. General
statistical disjointness can't be verified from arrays alone, but the common accidental mistake (passing the
exact same array reference/values for both) is cheaply detectable and currently silent.

**VOTENRANK-12** (`stability_exp.py`): `spearman_exp`'s per-cell NaN injection (`table_nan.iloc[idx % rows,
idx // rows] = np.nan`, once per index in `nan_idxs_prod`) is a real, if modest, perf inefficiency — pandas'
`.iloc` scalar setitem pays block-manager/dtype-check overhead per call rather than one vectorized write. This
is an experimentation/plotting utility (not a hot production path), which is why it's rated P3 rather than
P2, but it is a genuine "missed vectorization" instance per this audit's dimension #3.

**VOTENRANK-13** (`iia_exp.py`): `compute_iia_for_fixed_models` unconditionally does `table.loc[models_order[:2]]`
and then `range(3, len(models_order) + 1)`; for `len(models_order) < 3` the loop body never executes and the
function silently returns `result = 0` (looking like "zero IIA violations measured", not "not enough models
to measure IIA at all") — a misleading zero rather than a clear error, for a legitimately-possible small-table
input.

**VOTENRANK-14** (`data_processing.py`): `get_tracker_table`'s directory-name parser
(`model, task, _ = f.split("_")`) assumes exactly 3 underscore-delimited components; `tracker_filename` (the
function that WRITES these directory names) constructs them as `f"{model}_{task}_0/"`, so any real model or
task identifier containing its own underscore — extremely common in practice (HuggingFace model IDs, GLUE
task variants like `"cola_dev"`) — breaks the round-trip: 4+ components raises an unpacking `ValueError` and
aborts the entire tracker-table load for every row, not just the malformed one.

## Positive observations (no findings)

- `Leaderboard.__init__`/`build_ranks` already validates duplicate model names, unknown weight-dict keys, and
  all-zero weight sums with clear `ValueError`s (visible remnants of a prior audit pass, confirmed still
  correct on re-read).
- Every stochastic function in the module threads an explicit seed/`Generator` end to end; no
  process-global `np.random` state mutation was found anywhere in scope.
- `sample_weight` is correctly threaded through `SimilarityBlendEnsemble.fit`/`fit_multi_region` (conditionally
  passed only when not `None`, avoiding a `TypeError` on estimators that don't accept the kwarg).
- No mutable-default-argument bugs were found (`grep`-verified across the whole tree).
- No bare/overly-broad `except:` clauses; every `except Exception` site in scope is narrow, commented, and
  either covered by an existing regression test (`test_f8_cupy_fallback_logs_debug`) or genuinely defensive
  against an optional third-party import (`numba`, `cupy`).
