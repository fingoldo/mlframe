# Cross-cutting audit: guards that pass because they looked at nothing

Date: 2026-09-05. Scope: `src/mlframe` (2,466 files) + `tests/` (3,493 files).

Method: three AST scanners (`getattr(obj, x, <empty>)` feeding a check; assert-loops over
filtered comprehensions with no non-empty pin; `if <precond>: assert` with no `else`;
production validation loops over an empty-fallback collection), then manual reading of the
ranked hits. Shape 1 (negative membership over a derived collection) was re-run over `src/`
only, per the prior sweep: **zero** `assert x not in <call/comprehension>` sites exist in `src/`.
No test suite was run; three tiny targeted snippets were executed under `PYTHONPATH=src`.

---

## CONFIRMED

### VG-01 [P1] metric-over-time-direction-tests-assert-nothing
**File:** tests/reporting/test_metric_over_time_direction.py:51 (and :59)

**Summary:** Two regression tests exist to pin the shipped bug named in the module docstring:
`diagnostics_dispatch` computed `higher_is_better = metric not in ("mse","brier")`, so
`rmse` rendered an inverted `(higher=better)` title on the default-ON temporal drift chart.
Each test believes it is reading the rendered panel title and asserting the direction token.

**Failure scenario:** Both assertions sit under `if "over time" in title:`. `metric_over_time`
(src/mlframe/reporting/charts/drift.py:641-646) returns an `AnnotationPanelSpec` titled just
`metric` whenever no time bucket clears `min_samples`. The fixture builds `n=400` rows with
`ts = np.arange(400).astype("datetime64[D]")`, i.e. **400 distinct days, one sample each**,
against the default `freq="D", min_samples=100`. No bucket ever clears the threshold, so the
title is the bare metric name, the guard is False, and **both tests execute zero assertions
today**. An inverted direction label would ship green.

**Evidence:** Ran the test's own `_direction_in_title` body under `PYTHONPATH=src`:
`'rmse' TITLE= 'rmse' -> over time in title? False` and
`'roc_auc' TITLE= 'roc_auc' -> over time in title? False`.
Source of the fallback title: `drift.py:644`, `title=metric if title is None else title`.

**Suggested fix:** Give the fixture buckets that clear `min_samples` (e.g. repeat each day
~200 times, or pass `min_samples=1`), then assert unconditionally:
`assert "over time" in title, title` followed by the direction assertions. Emptiness then
becomes a failure instead of a silent pass.

### VG-02 [P1] cache-key-id-scan-reads-only-the-call-header-line
**File:** tests/training/test_dataset_cache_fingerprint.py:239 (and :256)

**Summary:** Two tests police the 2026-05-23 fix that removed `id(train_df)` / `id(val_df)`
from the CatBoost Pool cache keys (CPython reuses a freed object's address, so an id-keyed
cache returns another frame's Pool). They collect
`key_assign_lines = [ln for ln in src.splitlines() if ln.lstrip().startswith("key = ")]`
and assert `"id(train_df)" not in ln` per line.

**Failure scenario:** Both real assignments are **multi-line calls**. The only matching
physical line is `key = compute_signature(`, a line that by construction contains no
arguments at all, so `"id(train_df)" not in ln` is trivially true. The arguments live on the
continuation lines (`_cb_pool_build.py:171`, `train_df,`; `_cb_pool.py:738`, `val_df,`), which
the scan never looks at. Re-introducing `id(train_df)` as an argument leaves both tests green.
The collection is not empty, but it is *content*-empty, which is the same defect.

**Evidence:** Snippet run: each module yields exactly 1 matching line,
`['    key = compute_signature(']` and `['        key = compute_signature(']`.
`grep -A8` shows the arguments on lines 171-173 and 738-740.

**Suggested fix:** Parse the module with `ast` (the file already does AST work elsewhere),
find the `Assign` whose target is `key`, and assert `"id(" not in ast.unparse(node.value)`.
Pin with `assert key_assigns, "no key= assignment found; this test needs updating"`.

### VG-03 [P1] enum-exhaustiveness-police-has-a-hardcoded-module-allowlist
**File:** tests/test_meta/test_enum_exhaustiveness.py:83-96 (skip at :113)

**Summary:** `test_every_enum_value_is_dispatched_on` claims to check that *every* `Literal`
or `Enum` value across *every* training config class is actually dispatched on somewhere in the
consumer corpus. `_enum_literal_fields()` walks `mlframe.training.configs` but drops any class
whose `__module__` is not in a hardcoded 8-entry `_accepted_modules` set.

**Failure scenario:** `configs.py` has since been split further. Two sibling modules are **not**
in the allowlist, so 5 config classes are silently excluded from the police, with no skip and no
warning; the test still reports green over a shrunken corpus. An undispatched Literal value
added to `TrainingBehaviorConfig` (or the other four) ships unnoticed. The `if not fields:
pytest.skip(...)` at :113 is the same shape one level up: a rename of the whole package would
skip the test rather than fail it.

**Evidence:** Snippet run under `PYTHONPATH=src`:
`EXCLUDED mlframe.training._model_configs_behavior literal_fields= 1 ['LearningToRankConfig', 'MultilabelDispatchConfig', 'QuantileRegressionConfig', 'TrainingBehaviorConfig']`
and
`EXCLUDED mlframe.training._model_configs_ensembling literal_fields= 0 ['EnsemblingConfig']`.
Also present on disk and unlisted: `_composite_target_discovery_config_base.py`,
`_helpers_training_configs.py`.

**Suggested fix:** Replace the allowlist with a prefix test
(`obj.__module__.startswith("mlframe.training.")`) or derive it from the package's own
submodule listing, and turn `if not fields: pytest.skip` into
`assert fields, "no Literal/Enum config fields discovered; the discovery walk broke"`.
Additionally assert a floor on `total_values` so a future shrink is loud.

### VG-04 [P1] wilcoxon-serial-vs-parallel-equivalence-is-empty-vs-empty
**File:** tests/training/composite/discovery/test_composite_discovery_parallel.py:239

**Summary:** `test_wilcoxon_per_seed_matches_serial` claims the per-seed Wilcoxon RMSE arrays
produced by the tiny-rerank stage are identical between `n_jobs=1` and `n_jobs=4`.

**Failure scenario:** Both sides are read as
`getattr(<disc>, "_wilcoxon_per_seed_composite", {})`, the key-set assertion is
`set(ser) == set(par)` (true for two empty dicts), and the value comparison is a `for` loop over
`ser_per_seed.items()` (zero iterations). Nothing in the test pins either dict non-empty. The
attribute is populated only at `_tiny_rerank.py:520-529`, inside the reduce over
`_rerank_results`, and only for families that `_rerank_one_spec` returned a `_per_seed` entry
for. If discovery keeps zero specs on this fixture (MI screening rejecting everything), or the
Wilcoxon path is not taken, both dicts stay empty, the comparison is symmetric, and the test is
100% vacuous, while being the *only* guard against a serial/parallel numeric divergence.

**Evidence:** Read `src/mlframe/training/composite/discovery/_tiny_rerank.py:505-531`
(population is per-`kept_specs`, per-family, conditional) and the test body at :234-250. The
sibling test in the same file at :230 has the same unpinned structure.
Note the contrast: `tests/training/composite/discovery/test_biz_val_discovery_fit.py:222`
already applies exactly the right fix
(`assert flags_a, "alpha-drift gate did not fire; test is not exercising P14"`) and :237 does
too (`assert pool, "auto-base pool empty; ..."`), so the pattern is known in this codebase and
just was not carried to this file.

**Suggested fix:** Add `assert ser_per_seed, "tiny-rerank produced no per-seed Wilcoxon arrays;
the serial/parallel comparison would be vacuous"` before the key-set assertion.

### VG-05 [P2] pd-view-memo-ordering-test-skips-exactly-when-it-should-fail
**File:** tests/training/test_single_slot_memo_write_order.py:78

**Summary:** Asserts the pandas-view memo publishes the value (`result`) before the key
(`id_key`), so a torn read cannot pair a fresh key with a stale view.

**Failure scenario:** `if not rec.set_order: pytest.skip("pd-view memo store path not exercised
on this build/config")`. If `get_pandas_view_of_polars_df` stops writing to
`_PD_VIEW_LAST_CACHE` (the memo being removed, short-circuited, or gated behind a config that
flips), `rec.set_order` is empty and the test skips instead of failing. The write-ordering
contract then goes entirely unverified with no red signal.

**Evidence:** The **sibling** test 14 lines above does it correctly:
`assert rec.set_order, "memo store path did not run"` (line 60). The asymmetry is the finding.

**Suggested fix:** Use the sibling's line verbatim:
`assert rec.set_order, "pd-view memo store path did not run"`.

### VG-06 [P2] unary-spec-base-column-contract-skipped-when-nothing-survives
**File:** tests/training/composite/discovery/test_discovery_unary_base_free.py:175

**Summary:** Regression test for a crash where `iter_transform` extracted
`df[base_column]` and blew up on the empty-string sentinel a unary spec carries. Asserts every
kept unary spec has `base_column == ""` and that `iter_transform` emits it.

**Failure scenario:** `unary_specs = [s for s in disc.specs_ if s.transform_name in
_UNARY_TRANSFORMS]`, then `if not unary_specs: pytest.skip(...)`. Two `for s in unary_specs`
loops then carry every assertion, so zero iterations is a pass even without the skip. The test
already forces survival with `eps_mi_gain=-1e9, top_k_after_mi=50`, so any change that stops
unary transforms from reaching `specs_` (a new gate, a rename in `_UNARY_TRANSFORMS`, a
screening reorder) silently retires the regression test.

**Evidence:** Read lines 168-190. Note the sibling above at :162 asserts `counts[unary] == 1`
per transform, so the fixture is *expected* to produce them; the skip contradicts that
expectation rather than covering a legitimate optional path.

**Suggested fix:** Replace the skip with
`assert unary_specs, "no unary spec reached specs_; the base-free regression is not exercised"`.

### VG-07 [P2] composite-spec-schema-test-skips-on-empty-specs
**File:** tests/training/composite/discovery/test_biz_val_training_composite_discovery.py:133

**Summary:** `test_..._returns_valid_spec_schema_on_linear_target` pins the documented spec
serialisation schema (`name`, `target_col`, `transform_name`, `base_column`, `fitted_params`,
`mi_gain`) and the allowed transform names.

**Failure scenario:** `if not specs: pytest.skip("discovery rejected all candidates; gain
semantics; covered by other tests")`. A regression in the discovery gain gate (or in
`export_specs` returning `[]`) makes the fixture emit nothing, and the entire schema contract
goes unchecked. The fixture is `_linear_residual_target(n=2500, seed=42)`, deliberately built so
`linear_residual` wins, so emptiness there is itself a defect, not a legitimate variation.

**Evidence:** Read lines 118-140.

**Suggested fix:** `assert specs, "linear-residual fixture produced no specs; schema contract
unexercised"`. If flakiness is genuinely feared, mark the emptiness case xfail-strict rather
than skip.

### VG-08 [P2] fe-auto-userwarning-test-asserts-nothing-when-no-warning-fires
**File:** tests/feature_selection/mrmr/core/test_mrmr_error_messages_ux_audit.py:101

**Summary:** `test_fe_auto_enabled_generators_emits_userwarning`; the name claims the notice
is a user-visible `UserWarning`, not a log-only message.

**Failure scenario:** `matches = [w for w in caught if "fe_auto=True enabled" in
str(w.message)]`, then `for w in matches: assert w.category is UserWarning`. If the message
wording changes at all, or `fe_auto` enables nothing on this fixture (which the inline comment
explicitly permits), `matches` is empty and the test asserts nothing. Regressing the channel
back to `logger.info`, the exact defect the test is named for, leaves it green.

**Evidence:** Read lines 90-102; the comment "fe_auto may legitimately choose to enable nothing
for this fixture; only assert the channel exists and fires with the right category when it does"
documents the vacuity as intentional, which makes it decoration rather than a guard.

**Suggested fix:** Pick a fixture where `fe_auto` provably enables a generator and assert
`matches, "fe_auto enabled generators but emitted no UserWarning"`. If the fixture cannot be
made deterministic, assert on the emitting call site (monkeypatch `warnings.warn` and assert it
was reached) so the channel, not the fixture's luck, is what is measured.

### VG-09 [P2] multibase-alphas-validation-skipped-on-domain-check
**File:** tests/training/test_multibase_spec.py:152

**Summary:** Pins that a multi-base spec whose `fitted_params` lack `alphas` raises a clear
`ValueError` ("alphas") rather than crashing inside the suite.

**Failure scenario:** `valid = transform.domain_check(y_full, base_primary)`, then
`if not valid.any(): pytest.skip("domain_check disqualified all rows for the synthetic data")`.
The `pytest.raises(ValueError, match="alphas")` block is never entered when the synthetic
fixture drifts out of the transform's domain (a change to `domain_check`'s bounds, or to
`_build_multi_base_spec`'s rng draw). The clear-error contract then goes unverified.

**Evidence:** Read lines 135-156.

**Suggested fix:** `assert valid.any(), "domain_check disqualified every row; the alphas
validation is not being exercised"`, and construct `base_primary` inside the transform's domain
by construction rather than by rng draw.

### VG-10 [P2] missing-fit-column-raise-skipped-on-empty-support
**File:** tests/feature_selection/stability/test_stability_transform_validation.py:99

**Summary:** `test_transform_missing_fit_column_raises`; dropping a fit-time column that is in
`support_` must raise `ValueError` matching "missing".

**Failure scenario:** `selected_names = [sel.feature_names_in_[i] for i in sel.support_]`, then
`if not selected_names: pytest.skip("Empty support_; cannot test missing-column drop")`. A
selector regression that selects zero features skips this test rather than failing it, and
zero-selection is itself a known failure mode the repo tracks (there is a whole
`tests/training/test_fs_empty_selection_and_observability.py`). Two failures for the price of
one silent skip.

**Evidence:** Read lines 93-105. `_fit()` is the module's shared fixture, so nothing else in
this test pins the support non-empty.

**Suggested fix:** `assert selected_names, "selector chose zero features; the missing-column
contract is unexercised"`.

### VG-11 [P2] ks-stability-filter-returns-a-clean-report-when-it-inspected-no-columns
**File:** src/mlframe/feature_selection/filters/_ks_stability.py:64-82

**Summary:** Production drift guard. `ks_stability_filter` reports, per numeric feature, whether
the train/test distributions diverge (`stable=False` marks a drop candidate).

**Failure scenario:** When `feature_cols is None` the column list is derived as
`[c for c in train_df.columns if c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c])]`,
a **double filter**. If the two frames disagree on column names (a rename, a prefixed test
frame, a re-ordered pipeline that emits engineered names on one side only), or if the numeric
columns arrive as `object`/`string`/non-pandas-numeric extension dtypes, the intersection is
empty. The `for col in feature_cols` loop then runs zero times, `rows` stays empty, and the
function returns an empty report. Every caller reads "no unstable features"; a fully drifted
feature set is indistinguishable from a clean one. There is no log line and no exception.
A second, narrower instance of the same shape sits at line 82: a column whose finite values are
empty on either side is recorded `"stable": True`, so an all-NaN column is reported as *stable*
rather than as unmeasurable.

**Evidence:** Read lines 60-100. The only up-front validation in the function is the
`split_frac` range check at :69-73; nothing validates that any column survived the filter.

**Suggested fix:** After `feature_cols = list(feature_cols)`, raise or warn loudly when it is
empty: `if not feature_cols: raise ValueError("ks_stability_filter: no shared numeric columns
between train_df and test_df; nothing to test for drift")`. Record all-NaN columns as
`"stable": None` (unmeasurable) rather than `True`.

---

## LEADS (shape present; emptiness path plausible but not demonstrated here)

### VG-12 [P3] leaky-bases-negative-assert-is-true-by-construction
**File:** tests/training/test_base_leakage_guard.py:60

**Summary:** `test_leakage_guard_noop_without_time_ordering` asserts
`not getattr(disc, "_leaky_bases_dropped_", [])`; the guard must stay inert without
`time_ordering`.

**Failure scenario:** `_fit_temporal.py:53` sets the attribute *only* `if dropped:`, and the
whole call is gated on `config.detect_base_leakage` at `_fit.py:197-199`, with a further early
`return base_candidates` at `_fit_temporal.py:37-38`. So the assertion is satisfied by the
attribute simply never existing, which it also would be if `detect_base_leakage` stopped being
honoured entirely, or if `fit` never reached the guard. Mitigated: the sibling test at :46-47
does pin a positive (`assert "base_leaky" in dropped`), so a total regression is caught there.
Marked P3 for that reason, not because the shape is absent.

**Suggested fix:** Have the guard always set `discovery._leaky_bases_dropped_ = dropped`
(including the empty list) and assert `disc._leaky_bases_dropped_ == []`, so "ran and dropped
nothing" is distinguishable from "never ran".

### VG-13 [P3] fe-family-negative-assertions-cannot-tell-off-from-absent
**File:** ~25 sites, e.g. tests/feature_selection/biz_val/test_biz_val_filters_conditional_gate.py:188,
tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fe_encodings/test_count_freq_encoding.py:679-681,
tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_recipe_fe/test_recipe_fe_families.py:491-493,
tests/training/feature_selection/test_fuzz_mrmr_fe_grouped_lagged_coverage.py:81-82

**Summary:** The "family is OFF, so it produced nothing" half of each on/off pair, written as
`assert list(getattr(m, "<family>_features_", []) or []) == []`.

**Failure scenario:** The empty default makes attribute *absence* indistinguishable from
"present and empty". Renaming an artefact attribute (say `hybrid_orth_features_` to
`hybrid_orth_selected_`) turns every one of these into a vacuous pass while the corresponding
positive assertions fail loudly elsewhere. Mitigated in most files by an adjacent positive
assertion on the same attribute name, which is why this is P3, and why a rename would still be
caught somewhere. It is listed because the shape is systemic (~25 sites) and the mitigation is
incidental rather than designed. Note `_mrmr_setstate_defaults.py` seeds many of these
attributes to `[]` on unpickle, so *unpickled* estimators do have them; freshly-constructed ones
may not.

**Suggested fix:** Where the estimator contract says the attribute always exists after `fit`,
drop the default: `assert m.<family>_features_ == []`. Missing then raises `AttributeError`
instead of passing.

### VG-14 [P3] get-feature-names-out-contract-only-checks-selectors-that-have-it
**File:** tests/feature_selection/contracts/test_fs_selector_contract.py:286

**Summary:** `if callable(getattr(fs, "get_feature_names_out", None)): assert len(names) == n_out`,
the sklearn name-propagation contract inside a Pipeline.

**Failure scenario:** A selector that *loses* `get_feature_names_out` (a refactor moving off
`SelectorMixin`, a `__getattr__` change) silently exits the contract instead of failing it.
Currently safe: mlframe selectors inherit the method from `SelectorMixin`, so the branch is
taken for all of them today; hence P3, and hence a LEAD rather than a confirmed emptiness.

**Suggested fix:** Assert the method exists (`assert callable(getattr(fs,
"get_feature_names_out", None)), f"{name}: selector lost get_feature_names_out"`) with an
explicit exemption set, the way `_GFNO_EXEMPT` is already used at line 344 of the same file.

### VG-15 [P3] narrowed-except-check-unpinned
**File:** tests/test_meta/test_swallowed_failures_are_audible.py:122

**Summary:** `test_the_except_is_narrowed` filters AST handlers to those returning the literal
`"hermite"`, then asserts each does not catch bare `Exception`.

**Failure scenario:** The comprehension has no non-empty pin, so changing the fallback's return
value (a different default basis, a variable instead of a literal) empties the list and the
`for h in handlers` loop asserts nothing. Mitigated: the sibling
`test_the_route_fallback_warns` seven lines above uses the identical comprehension *with*
`assert handlers, "the basis-routing fallback was not found; this test needs updating"`, so the
same change fails loudly one test over.

**Suggested fix:** Copy the sibling's `assert handlers, ...` line into this test; the
comprehension is duplicated already, the pin should be too.

### VG-16 [P3] base-leakage-guard-silently-no-ops-on-a-short-time-ordering
**File:** src/mlframe/training/composite/discovery/_fit_temporal.py:36-38

**Summary:** Production. `apply_base_leakage_guard` computes
`_to_train = _to_all[train_idx] if _to_all.shape[0] >= int(np.max(train_idx)) + 1 else None`
and returns `base_candidates` unchanged when it is `None`.

**Failure scenario:** A caller that passes a `time_ordering` shorter than
`max(train_idx) + 1` (a subset frame, an `.iloc[]` slice whose ordering was not sliced with it)
gets the leakage guard silently disabled while `config.detect_base_leakage=True` says it is
on. Same-time re-encodings of `y` then enter discovery as bases. Not demonstrated against a real
caller here, so a LEAD.

**Suggested fix:** Log a warning (or raise) on the length mismatch instead of returning
silently; the config asked for the guard, so its inability to run is information the caller
needs.

### VG-17 [P3] missing-indicator-pairing-loop-over-a-self-excluding-filter
**File:** src/mlframe/preprocessing/missing_indicator_pairing.py:76

**Summary:** `fit_missing_indicator_imputation` iterates `[c for c in cols if c != group_col]`.

**Failure scenario:** When the caller passes a single-column `cols` that *is* `group_col`, the
list is empty and the fit loop performs no work, returning an empty/identity state rather than
signalling the degenerate input. Not traced to a real caller, so a LEAD.

**Suggested fix:** Validate the filtered list is non-empty before the loop and raise a named
`ValueError`.

---

## Summary

| ID | Sev | File:line | Shape | Status |
|----|-----|-----------|-------|--------|
| VG-01 | P1 | tests/reporting/test_metric_over_time_direction.py:51,59 | 4 | CONFIRMED, vacuous **today** |
| VG-02 | P1 | tests/training/test_dataset_cache_fingerprint.py:239,256 | 3 | CONFIRMED, vacuous **today** |
| VG-03 | P1 | tests/test_meta/test_enum_exhaustiveness.py:83-113 | 5/6 | CONFIRMED, 5 classes excluded **today** |
| VG-04 | P1 | tests/training/composite/discovery/test_composite_discovery_parallel.py:239 | 2+3 | CONFIRMED shape, empty-vs-empty symmetric |
| VG-05 | P2 | tests/training/test_single_slot_memo_write_order.py:78 | 5 | CONFIRMED (sibling asserts, this one skips) |
| VG-06 | P2 | tests/training/composite/discovery/test_discovery_unary_base_free.py:175 | 3+5 | CONFIRMED |
| VG-07 | P2 | tests/training/composite/discovery/test_biz_val_training_composite_discovery.py:133 | 5 | CONFIRMED |
| VG-08 | P2 | tests/feature_selection/mrmr/core/test_mrmr_error_messages_ux_audit.py:101 | 3 | CONFIRMED (documented as intentional) |
| VG-09 | P2 | tests/training/test_multibase_spec.py:152 | 5 | CONFIRMED |
| VG-10 | P2 | tests/feature_selection/stability/test_stability_transform_validation.py:99 | 5 | CONFIRMED |
| VG-11 | P2 | src/mlframe/feature_selection/filters/_ks_stability.py:64-82 | 6 | CONFIRMED (production) |
| VG-12 | P3 | tests/training/test_base_leakage_guard.py:60 | 2 | LEAD (sibling mitigates) |
| VG-13 | P3 | ~25 sites, mrmr/biz_val FE families | 2 | LEAD (systemic shape, incidental mitigation) |
| VG-14 | P3 | tests/feature_selection/contracts/test_fs_selector_contract.py:286 | 4 | LEAD (safe today via SelectorMixin) |
| VG-15 | P3 | tests/test_meta/test_swallowed_failures_are_audible.py:122 | 3 | LEAD (sibling mitigates) |
| VG-16 | P3 | src/mlframe/training/composite/discovery/_fit_temporal.py:36-38 | 6 | LEAD (production, no caller traced) |
| VG-17 | P3 | src/mlframe/preprocessing/missing_indicator_pairing.py:76 | 6 | LEAD (production, no caller traced) |

**Counts:** 11 CONFIRMED (4 P1, 7 P2), 6 LEADS (all P3). Three of the four P1s are vacuous
*right now*, not merely fragile.

**Shape 1 in src/:** zero instances. The AST scan found no
`assert x not in <call-or-comprehension>` anywhere under `src/mlframe`.

**Cross-cutting note:** the codebase already knows the correct remedy.
`test_biz_val_discovery_fit.py:222` and `:237` carry explicit non-emptiness pins with comments
naming the vacuity risk, and `test_swallowed_failures_are_audible.py:114` and
`test_infonet_weights_only_load.py:46` do too. Every CONFIRMED finding above is a site where
that established pattern was simply not applied, usually one function away from a sibling that
does apply it. A lint rule (an assert-loop or `pytest.skip` over a locally-derived collection
must be preceded by a truthiness assertion on that collection) would close the class.
