# xcut_nondiscriminating_asserts
Candidate functions regenerated: 1773 | hand-triaged: 430 | reported: 27

## Summary
I re-ran the AST pass from scratch over all 3,411 files / 24,528 test functions under `tests/` rather than
trusting the prior 903-item list, and got **1,773** candidates: 1,093 whose every assertion is
`is/is not None` / `isinstance` / `hasattr` / `callable` / `len(x) > 0`, plus **680 with no assertion at all**
(a class the prior pass did not count). My scanner also resolves one to three levels of intra-file helper
delegation, so a test whose real assertion lives in a module-local `_assert_*` helper is correctly excluded --
that alone removed ~110 false positives from the naive list (e.g. `test_biz_val_supervised_projection_ops.py`,
whose four "lifts_linear_auc" tests do assert a quantitative floor, inside `_augment_auc`). I then ran four
further shape scans the prior pass did not: 168 tests where **every** assertion sits inside an `if`, 884 where
every assertion sits inside a loop, 26 comparisons whose two sides are the identical AST, 7 assertions under a
`try` whose handler swallows `AssertionError`/`Exception`, 3 unconditional imperative `pytest.xfail(...)`
statements, and 4 `if ...: pass` bodies inside a test.

I hand-triaged 430 functions: the full 118-function priority band (`*bugfix*` / `*regression*` /
`test_biz_val_*` / `test_biz_value_*`, excluding the 394-test informational matrix), a 241-function band whose
docstring narrates a specific pre-fix defect, and every member of the four small auxiliary scans. About 71 were
read in full. **The prior agent's judgement holds for the large majority**: roughly 340 of the 430 are
legitimate -- a genuine `Optional[T]` contract (`_col_value_counts` on an array column really does return
`None`), a genuine crash regression where the call itself is the sensor (`MRMR().fit()` raised `AttributeError`
pre-fix, so reaching any assertion is the signal), or a type contract that really is the contract (the
sklearn-fallback path really must hand back a `pd.DataFrame`). The 27 below are the residue where the assertion
provably survives the defect the test names. The sharpest cluster is not weak assertions at all but **tests
that assert nothing**: a bf16 test that re-implements the production gating rule inline and then asserts its
own copy, a `caplog` fixture set up and never inspected, a Hypothesis property test that catches its own
`AssertionError`, and a dedicated fold-leakage regression file whose fixtures contain no NaN and therefore never
reach the whole-column median fill that is this audit PREPROCESSING_DATA-2 P0.

## Findings

### XCUT_NONDISCRIMINATING_ASSERTS-1 [P0] fixture-never-reaches-the-path
**File:** tests/preprocessing/test_auto_transform_select_fold_leakage.py:81
**Summary:** The dedicated CV-leakage regression file for `select_column_transforms` uses no NaN-bearing column
anywhere, so the whole-column median fill -- this audit's PREPROCESSING_DATA-2 P0 -- is never executed by the
file that exists to guard exactly that class of leak.
**What it would miss:** `src/mlframe/preprocessing/auto_transform_select.py:227-229` computes
`finite_fill[~np.isfinite(finite_fill)] = np.nanmedian(...)` over the **entire** column, three lines above the
`for train_idx, test_idx in fold_indices:` loop whose own comment reads "fitting on the full column (train+test)
before splitting would leak the test fold's own statistics into its held-out score". Every fold's held-out rows
are therefore imputed with a median that saw those rows. The end-to-end sensor at :65-81 builds
`pd.DataFrame({"noise": rng.normal(0, 1, n)})` -- no NaN, so the fill is a no-op and the leak cannot express
itself. Even if a NaN appeared, the assertion is `0.3 <= score <= 0.7` on a pure-noise column, which the test's
own comment concedes is "a robust proxy": a median-fill leak on a real signal column inflates the score of the
imputed rows, not of pure noise, so the leak is outside the assertion's reach in both directions.
**Suggested fix:** Add a fourth test that plants NaN in a column whose missingness is correlated with y, and
assert the fold-local fill: move the fill into `_fit_transform_fold`'s train-only scope and assert
`_fit_transform_fold(x_with_nan, "identity", train_idx, test_idx)[1]` equals a reference filled from
`np.nanmedian(x[train_idx])`, not from `np.nanmedian(x)`. Then parametrise the existing end-to-end test over
`nan_frac in (0.0, 0.2)` so the path is exercised at all.
**Evidence:** The module docstring at :1-8 states the contract as "must fit every transform on the TRAIN fold
only, never on the full column (train+test) before splitting ... any transform with fit statistics ... leaked
the test fold's own values". `grep -n nan` over both this file and
tests/preprocessing/test_biz_val_auto_transform_select.py returns nothing.

### XCUT_NONDISCRIMINATING_ASSERTS-2 [P1] asserts-the-test-s-own-reimplementation
**File:** tests/training/neural/test_bf16_auto_enable.py:98
**Summary:** `test_bf16_auto_enable_dispatcher_compute_capability_check` copies the production bf16 gating rule
into the test body and asserts on its own copy; no production code runs.
**What it would miss:** Everything. The body builds a local `trainer_params = {"accelerator": "cuda"}`, then
executes `if "precision" not in trainer_params and _resolved in ("cuda","gpu"): ... if _cc_major >= 8:
trainer_params["precision"] = "bf16-mixed"` -- the test's own transcription of the dispatcher -- and asserts
`trainer_params.get("precision") == "bf16-mixed"` (:123) and, for cc=7.5, `"precision" not in trainer_params`
(:139). The only import of production code is `from mlframe.training.neural.base import safe_accelerator
# noqa: F401`, which is never called. If the real dispatcher in `_fit_common` were deleted outright, inverted to
`_cc_major < 8`, or moved behind a flag defaulting off, both assertions still pass. The comment at :103-105
states the reason ("Direct import / probe of the gating logic isn't exposed as a function, so this test ASSERTS
via ... a synthetic `_fit_common-like` dispatcher block") -- an acknowledged stand-in that the file's green
status nonetheless reads as coverage of F-27.
**Suggested fix:** Extract the gate into a named pure function (e.g. `_resolve_default_precision(trainer_params,
resolved_accelerator) -> Optional[str]`), call **that** from `_fit_common`, and have this test call it with
`torch.cuda.get_device_capability` patched to `(8,0)` and `(7,5)`. Until the extraction lands, the honest
interim is to patch the three `torch.cuda` probes, run a real `reg.fit(...)`, and assert
`reg.trainer_params["precision"] == "bf16-mixed"` -- exactly what the sibling at :86-95 already does for the
explicit-precision case.
**Evidence:** Read tests/training/neural/test_bf16_auto_enable.py:98-139 in full; compare
`test_caller_precision_setting_is_not_overridden` at :95, which asserts on `reg.trainer_params["precision"]`
after a real fit and is the only test in the file that touches the production path.

### XCUT_NONDISCRIMINATING_ASSERTS-3 [P1] zero-assertion-negative-contract
**File:** tests/training/neural/test_bf16_auto_enable.py:76
**Summary:** `test_bf16_not_enabled_on_cpu_accelerator` has zero assertions; its whole body is `reg =
PytorchLightningRegressor(**_params()); reg.fit(X_tr, y_tr)`.
**What it would miss:** The negative contract in its own docstring -- "when accelerator resolves to CPU, bf16
must NOT be auto-set". A regression that drops the `_resolved in ("cuda","gpu")` guard and sets
`precision="bf16-mixed"` on every run is invisible: Lightning accepts `bf16-mixed` on CPU (it emits a
performance warning, not an error), so the fit completes and the test passes. The trailing comment at :82-83
makes the reasoning explicit and wrong -- "If we got here without crashing, the precision plumbing worked. The
CPU run cannot have bf16-mixed enabled (Lightning would warn or fail)" -- a warning is not a failure here.
**Suggested fix:** `assert "precision" not in reg.trainer_params` (the same dict
`test_caller_precision_setting_is_not_overridden` reads at :95), or
`assert reg.trainer_params.get("precision") != "bf16-mixed"` if a CPU default is legitimately stamped.
**Evidence:** Read tests/training/neural/test_bf16_auto_enable.py:76-83 in full -- six lines, no `assert`.

### XCUT_NONDISCRIMINATING_ASSERTS-4 [P1] fixture-captured-never-inspected
**File:** tests/training/test_permutation_fi_silent_skip_on_none_model.py:38
**Summary:** `test_permutation_fi_still_warns_on_genuine_failure` takes `caplog`, enters
`caplog.at_level(logging.WARNING, ...)`, calls the helper with a deliberately broken estimator -- and never
looks at `caplog.records`.
**What it would miss:** Precisely the regression named in its own docstring: "the silent-skip doesn't swallow
real bugs". If `_permutation_feature_importances` widens its `except` to cover the `RuntimeError` that
`_BrokenEstimator.predict` raises and returns silently with no WARN, the test passes. The whole point of the
file is that the `model is None` short-circuit must not also silence genuine failures; this half of the pair is
unasserted while its sibling at :22 is fully asserted.
**Suggested fix:** Mirror the sibling exactly -- add `warning_msgs = [r.getMessage() for r in caplog.records if
r.levelno >= logging.WARNING]` and `assert warning_msgs, "a genuine permutation-FI failure must WARN"`, plus a
substring check on the exception text.
**Evidence:** Read tests/training/test_permutation_fi_silent_skip_on_none_model.py:22-62.
`test_permutation_fi_silent_skip_when_model_is_none` does exactly this at :34; the `still_warns` test's body
ends at the call on :62.

### XCUT_NONDISCRIMINATING_ASSERTS-5 [P1] catches-its-own-failure
**File:** tests/training/neural/test_generate_mlp.py:538
**Summary:** A Hypothesis property test wraps all four of its assertions in `try: ... except AssertionError:
pass`, so no assertion in it can ever fail the suite.
**What it would miss:** Every property it claims to test across 50 generated examples: `isinstance(model,
nn.Sequential)`, `count_parameters(model) > 0`, `output.shape[0] == 2` and `output.shape[1] == num_classes`
(:517-527). A `generate_mlp` regression that emits the wrong output width for some `(num_features, num_classes,
nlayers)` combination -- exactly the combinatorial defect a property test exists to find -- is swallowed. The
justification comment at :539 ("Some parameter combinations might be invalid") argues for catching a
constructor `ValueError`, which the handler does **not** cover, while a wrong-shaped output is an
`AssertionError`, which it does.
**Suggested fix:** Narrow the handler to what construction can legitimately raise (`except ValueError:
pytest.skip(...)`, or better `assume(...)` so Hypothesis resamples the invalid combinations) and let
`AssertionError` propagate.
**Evidence:** Read tests/training/neural/test_generate_mlp.py:513-540. This is the only `except AssertionError`
in the whole `tests/` tree; the AST scan for assert-under-a-swallowing-`try` returned 7 sites and the other 6
re-assert the real invariant after the handler.

### XCUT_NONDISCRIMINATING_ASSERTS-6 [P1] assert-behind-an-if-plus-disjunction
**File:** tests/feature_selection/regression/test_regression_bug_fixes.py:154
**Summary:** The late-binding-closure regression test's single assertion sits behind `if result is not None:`
and is a three-way `hasattr`/`isinstance` disjunction, so it neither runs reliably nor checks the property the
docstring names.
**What it would miss:** The docstring's stated contract is "returns a result whose ``degree`` field is the BEST
observed -- not necessarily the final iteration" -- i.e. the exact symptom a reintroduced late-binding bug would
produce (every closure seeing the last loop value, so the reported degree becomes the final grid entry rather
than the argmax). Nothing compares `result.degree` to anything. Worse, `optimise_hermite_pair` returning `None`
("no degree beat the baseline") skips the assertion entirely, so a regression that makes the optimiser return
`None` for every input converts this test into a silent no-op. And `assert hasattr(result,"degree") or
hasattr(result,"best_degree") or isinstance(result, dict)` passes for any of three shapes, so it does not even
pin the return type.
**Suggested fix:** Run the grid twice with the degree order reversed and assert the returned `degree` is
identical (a late-bound closure yields the last grid entry, which differs between the two orders); or, cheaper,
record the per-trial scores via a spy and assert `result.degree == max(trials, key=score).degree`. Replace the
`if result is not None` with an explicit `assert result is not None` on a fixture where the optimiser reliably
returns a result.
**Evidence:** Read tests/feature_selection/regression/test_regression_bug_fixes.py:127-155.

### XCUT_NONDISCRIMINATING_ASSERTS-7 [P1] property-held-before-the-fix
**File:** tests/feature_selection/mrmr/core/test_mrmr_cluster_stability_categorical_regression.py:67
**Summary:** The test's own docstring says the pre-fix code "raised inside and silently fell back to classic" --
and the only assertion is that `support_` is populated, which the silent fallback also produces.
**What it would miss:** The entire defect. Post-fix and pre-fix both complete the fit and both set `support_`;
the difference is which path produced it. `assert getattr(fs, "support_", None) is not None` cannot distinguish
them, so a regression that reinstates the raise-and-fall-back-to-classic behaviour (or any future exception
inside the cluster path caught by the same fallback) leaves this test green while
`stability_selection_method='cluster'` is silently a no-op for every categorical-bearing frame.
**Suggested fix:** Assert the path, not the outcome: spy on the cluster routine (`monkeypatch.setattr(...,
raising=True)` around the cluster-stability entry point, assert `call_count == 1` and that no exception was
recorded), or assert a cluster-only artefact is populated on the fitted selector (e.g. `cluster_members_` or the
cluster-stability summary attribute) which the classic fallback leaves unset. A `caplog` check that no
"falling back to classic" warning was emitted is a valid second sensor.
**Evidence:** Read tests/feature_selection/mrmr/core/test_mrmr_cluster_stability_categorical_regression.py:60-67
-- the docstring at :61-62 states the pre-fix behaviour that makes the assertion non-discriminating.

### XCUT_NONDISCRIMINATING_ASSERTS-8 [P1] assertion-contradicts-its-own-docstring
**File:** tests/training/callbacks/test_biz_val_callbacks.py:110
**Summary:** A `test_biz_val_*` test whose docstring says the callback "must return True when the file exists"
asserts `result is None or isinstance(result, bool)`.
**What it would miss:** The stop-file mechanism itself. The stop file is created at :102
(`stop_path.write_text("stop")`), so the contract is `result is True`. The assertion admits `None`, `False` and
`True` equally: an inverted check, a stale cached `stop_file(fpath)` predicate, or a path-resolution regression
that makes `_check()` always return `False` -- i.e. training never stops when the operator drops the stop file,
the single behaviour this callback exists for -- all pass. The comment at :104-106 concedes it ("The exact
return semantics for the stop-file variant may differ; we just verify the call doesn't crash and returns
SOMETHING"), but the production method is unambiguous: `src/mlframe/training/callbacks/stop_file.py:125` is
`return bool(self._check())`, and that method carries `# pragma: no cover`. The surrounding
`try/except (AttributeError, NotImplementedError): pytest.skip(...)` at :107/111 adds a second escape.
**Suggested fix:** `assert cb.after_iteration(model=None, epoch=1, evals_log={}) is True` with the file present,
plus the negative leg -- a fresh `tmp_path` with no stop file, asserting the same call returns `False`. Drop the
`try/except`; `after_iteration` is a declared method of the class under test.
**Evidence:** Read tests/training/callbacks/test_biz_val_callbacks.py:95-112 against
src/mlframe/training/callbacks/stop_file.py:114-125.

### XCUT_NONDISCRIMINATING_ASSERTS-9 [P1] zero-assertion-memory-contract
**File:** tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:3155
**Summary:** `test_regression_validate_inputs_skips_integer_columns_before_copy` asserts nothing; it calls
`m._validate_inputs(X_int, y)` and ends.
**What it would miss:** Its entire stated contract -- "an all-integer numeric frame no longer gets
upcast-and-copied for the inf check -- only floating columns are selected before any array construction". The
pre-fix code also did not raise on an integer frame; it merely built a float64 copy of the whole frame. So the
implicit "does not raise" assertion held before the fix and holds after it, and a regression that reinstates the
whole-frame upcast is invisible. Under this repo's own memory-discipline rule ("Frames can be 100+ GB -- never
`.copy()`/reconstruct a frame to work around a bug"), silently reintroducing a full-frame float64
materialisation on the validation path is a first-order defect with a test that cannot see it.
**Suggested fix:** Spy on the allocation: `monkeypatch.setattr(np, "asarray", counting_wrapper)` (or patch
`pd.DataFrame.to_numpy` / `select_dtypes`) and assert zero float-array construction for an all-integer frame;
or measure directly with `tracemalloc.get_traced_memory()` around the call on a 50k-row int frame and assert the
peak delta stays under a small multiple of one column, not of the frame.
**Evidence:** Read tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:3146-3155 -- ten
lines, the comment at :3154 reads "Must not raise", and there is no `assert`.

### XCUT_NONDISCRIMINATING_ASSERTS-10 [P2] asserts-the-mechanism-exists-not-that-it-is-used
**File:** tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:729
**Summary:** A 13-assertion, 9-test cluster guarding concurrency fixes asserts only that a module-level
`threading.Lock` object exists, never that any critical section actually acquires it.
**What it would miss:** Every one of the races the fixes closed. `assert isinstance(bng_gpu._DY_DEVICE_CACHE_LOCK,
type(threading.Lock()))` passes for a module that defines the lock at import and mutates the LRU dict outside
any `with` block -- which is exactly the pre-fix state plus one unused global. A refactor that moves a
`_cache[key] = value` line above the `with lock:`, or replaces `with lock:` with a comment, leaves all 13
assertions green while the get-or-insert sequence, the LRU eviction, the shared-mem property-set/launch pair and
the CUDA-teardown idempotency check are all unguarded again.
**Suggested fix:** Assert acquisition, not existence: wrap the lock in a counting proxy
(`monkeypatch.setattr(mod, "_X_LOCK", CountingLock(real_lock))`) and assert `proxy.acquired >= 1` after
exercising the guarded routine once; or hold the real lock from a second thread and assert the guarded call
blocks (with a timeout) rather than proceeding. A cheap repo-wide gate: forbid a test named `*_has_lock` whose
only assertion is `isinstance(..., type(threading.Lock()))`.
**Evidence:** 13 assertion sites across 3 files, all the same shape --
tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:491, 729, 730, 984, 985, 1572, 2221,
2222, 2349, 2350, 2512, 3784; tests/training/feature_selection/test_mrmr_identity_cache_and_monres_autoknot.py:281;
tests/training/test_cache_safety_global_state.py:21 and :179. Each docstring states the pre-fix condition as "no
lock guarded X's get-or-insert / LRU bookkeeping / eviction" -- a statement about the critical section, not about
the object's existence.

### XCUT_NONDISCRIMINATING_ASSERTS-11 [P2] imperative-xfail-discards-the-measurement
**File:** tests/feature_selection/biz_val/test_biz_val_synthesis_and_drop_matrix.py:549
**Summary:** Three tests compute a real metric and then call `pytest.xfail(...)` unconditionally as the last
statement, so the measured value is formatted into a message and never compared to anything.
**What it would miss:** Both directions of the gap each test documents. At :540-549 the test fits RFECV on a
`y = sign(x0**2 - median)` target, computes `auc`, then unconditionally xfails with the auc interpolated into the
reason string -- so if RFECV did start recovering the quadratic (auc 0.95) the test still xfails and nobody is
told the documented FS gap closed; if the linear downstream degraded further, nothing changes either. Same shape
at tests/feature_selection/biz_val/test_biz_val_mnar_missingness.py:432, which computes the imputed-value AUC
and xfails on it. `pytest.xfail()` called imperatively halts the test immediately and marks it xfailed
regardless of outcome -- it is not `xfail(strict=...)`, so an XPASS is structurally impossible. A third
instance, tests/feature_selection/stability/test_selector_contract_protocol_extra.py:133, converts a currently
passing contract into a silent xfail: `test_wrong_length_input_features_raises` returns cleanly when the
selector raises, but calls `pytest.xfail(...)` when it does not -- so a selector that raises today and stops
raising tomorrow flips from PASS to XFAIL with no failure.
**Suggested fix:** Replace the unconditional `pytest.xfail(msg)` with an assertion pinning the gap's current
magnitude in the direction that would change if the gap closed -- `assert auc < 0.6, f"RFECV now synthesises the
quadratic (auc={auc:.3f}); delete this gap marker"`. That keeps the documentation and fails loudly the day the
gap closes. For the protocol case, replace the trailing `pytest.xfail` with a module-level
`@pytest.mark.xfail(strict=True)` parametrised on the selectors known to lack the check, so a newly-compliant
selector XPASSes as an error.
**Evidence:** AST scan for `pytest.xfail(...)` as a statement (not a decorator) over all of `tests/` returns
exactly these 3 sites. Read all three in full.

### XCUT_NONDISCRIMINATING_ASSERTS-12 [P2] assertion-3x-looser-than-its-own-docstring
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_quality_metrics.py:315
**Summary:** The docstring sets the false-positive ceiling at 30% of 10 pure-noise features; the assertion is
`n_selected < 10`.
**What it would miss:** Any FP rate from 40% to 90%. The docstring reads "On 10 pure-noise features, FP rate
should be <= 30%", and the comment even reasons about the ~30-40% band the 3-permutation confirmation permits --
then asserts only that not all ten survive. A regression that weakens the confirmation permutation test until 9
of 10 pure-noise features are selected passes. The assertion is additionally gated on `if not fallback:` (:314),
so a regression that makes `fallback_used_` latch True on every all-noise frame -- itself a defect -- skips the
assertion entirely.
**Suggested fix:** `assert n_selected <= 3, f"all-noise FP rate {n_selected}/10 exceeds the documented 30%
ceiling"`, and replace the `if not fallback:` guard with `assert not fallback, "fallback engaged on a plain
all-noise frame"` (or, if fallback is a legitimate outcome, assert the disjunction explicitly rather than
skipping).
**Evidence:** Read tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_quality_metrics.py:289-315; the
docstring "<= 30%" against the assertion `< 10`.

### XCUT_NONDISCRIMINATING_ASSERTS-13 [P2] assert-skipped-by-the-regression-it-guards
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_quality_metrics.py:137
**Summary:** "Top-3 must be pure signals (precision = 1.0)" is asserted as `prec_at_3 >= 0.66`, and only when
`len(selected) >= 3`.
**What it would miss:** Two things at once. First, the docstring's contract is precision 1.0 on a fixture built
so all three signals are strongly separable (`y = (sig0 + 0.7*sig1 + 0.5*sig2) > 0` at n=2000 against 10 pure
noise columns); `>= 0.66` accepts one noise column in the top three, which on this fixture is a real selection
defect. Second, the `if len(selected) >= 3:` guard at :135 means the strongest possible regression -- MRMR
under-selecting to one or two features on a three-strong-signal frame -- runs zero assertions and passes green.
**Suggested fix:** `assert len(selected) >= 3, f"only {len(selected)} features selected on 3 strong signals"`
followed by an unguarded `assert prec_at_3 == 1.0`. If 0.66 is genuinely the sustainable floor across seeds, say
so in the docstring instead of claiming 1.0.
**Evidence:** Read tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_quality_metrics.py:113-137.

### XCUT_NONDISCRIMINATING_ASSERTS-14 [P2] the-comparison-in-the-name-is-a-pass-statement
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_contracts_robustness/test_mnar_missingness.py:272
**Summary:** `test_separate_bin_beats_fillna_zero_on_pure_mnar` fits both strategies and then reduces the
"beats" half of the comparison to `if "x_mnar" not in names_fz: pass`.
**What it would miss:** The entire comparative claim, which under this repo's biz_value convention ("compared
against the closest baseline") is the load-bearing part. Only `assert "x_mnar" in names_sep` (:265) is real. If
`nan_strategy="fillna_zero"` started recovering the MNAR signal too -- meaning `separate_bin` buys nothing and
the whole `nan_strategy` axis is dead weight -- the test passes unchanged. The comment at :266-271 talks itself
out of the assertion ("either x_mnar absent, or if present, separate_bin has a different (better) overall
composition") and then writes `pass`.
**Suggested fix:** Assert the difference: `assert "x_mnar" not in names_fz, "fillna_zero also recovered the MNAR
signal; separate_bin buys nothing on this fixture"`. If the fixture is genuinely marginal, compare a downstream
metric instead -- fit a logistic model on each selection and assert `auc_sep - auc_fz >= <floor>` -- which is the
quantitative form the convention asks for.
**Evidence:** Read the test at :243-274. The AST scan for `if <cond>: pass` inside a test function returns only 4
sites tree-wide; this is one of them.

### XCUT_NONDISCRIMINATING_ASSERTS-15 [P2] assert-skipped-on-the-expected-outcome
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_pre_distortion.py:815
**Summary:** "must return None on a pair independent of y" is asserted as `if res is not None: assert res.uplift
<= 1.10` -- so the documented outcome runs zero assertions, and the fallback tolerance is 10x the configured
gate.
**What it would miss:** The gate being removed rather than loosened. If `optimise_hermite_pair` stops applying
`baseline_uplift_threshold` altogether but happens to return `None` for another reason (a crash caught upstream,
an empty candidate set, a degree grid that never populates), the test passes with nothing asserted. And when a
result is returned, the test accepts an uplift of 1.10x against a configured `baseline_uplift_threshold=1.01` --
a warm start manufacturing a 9% uplift on a target provably independent of `(a, b)` is admitted.
**Suggested fix:** Assert the documented outcome directly -- `assert res is None, f"noise pair produced a result
with uplift {getattr(res,'uplift',None)}"` -- and if a marginal pass-through is genuinely tolerable, tighten the
fallback bound to the configured threshold (`res.uplift <= 1.01`) rather than 1.10.
**Evidence:** Read tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_pre_distortion.py:787-818; the
docstring "must return None" against the `if res is not None` guard, and `baseline_uplift_threshold=1.01` passed
at :811 against the `<= 1.10` at :817.

### XCUT_NONDISCRIMINATING_ASSERTS-16 [P2] non-emptiness-as-proof-of-a-leakage-contract
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fe_hybrid_orth/test_hybrid_fe_sklearn_ecosystem.py:407
**Summary:** A per-fold leakage contract ("the recipe builder runs ON THE TRAINING SLICE of every fold, no
shortcut that re-uses a leaky fit from outside the fold") is asserted as `len(m.hybrid_orth_features_) > 0`.
**What it would miss:** The shortcut it names. A non-empty `hybrid_orth_features_` is produced identically by a
fresh per-fold fit and by a cached recipe built on the full `X` -- and this codebase explicitly memoises fits by
content hash (CLAUDE.md: "MRMR memoizes fits by content-hash"), so a caching regression keyed on something
coarser than the row slice would return the same non-empty list on all five folds and pass. The docstring itself
concedes the weaker point ("y-leakage from the holdout slice into the recipe is structurally impossible because
recipes are pure functions of X, but we still pin that the FE pipeline engaged") -- the assertion does not even
pin engagement, only non-emptiness of an attribute.
**Suggested fix:** Assert the recipes actually differ across folds where the training slices differ: collect
`m.hybrid_orth_features_` (or the recipe fingerprints) per fold and assert `len(set(map(tuple, per_fold))) > 1`
on a fixture with fold-sensitive signal; or spy on the recipe-builder entry point and assert `call_count == 5`.
**Evidence:** Read
tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_fe_hybrid_orth/test_hybrid_fe_sklearn_ecosystem.py:389-409.

### XCUT_NONDISCRIMINATING_ASSERTS-17 [P2] dead-knob-invisible
**File:** tests/feature_selection/mrmr/core/test_mrmr_basic.py:256
**Summary:** The only test for `permutation_subsample` asserts `mrmr.support_ is not None` and explicitly
declines to check that the knob does anything.
**What it would miss:** The knob being ignored. The docstring states the mechanism -- "the kernel sees 200-row
arrays, ii_obs sees full 1000" -- and then says "We don't assert speedup at this tiny size ... only that the
path activates cleanly". Nothing verifies activation either: if `permutation_subsample` were dropped from the
`CatFEConfig` plumbing entirely (a plausible casualty of any config refactor), the fit still completes and
`support_` is still set, so the parameter silently becomes a no-op with full green coverage. This repo already
recognises the class -- tests/training/test_biz_val_training_preproc_backend_dead_knobs.py exists precisely to
catch knobs that do nothing -- and this one has no such sensor.
**Suggested fix:** Spy on the permutation kernel and assert the array it receives has 200 rows, not 1000:
`monkeypatch.setattr(<perm kernel module>, "<kernel>", recording_wrapper)` then `assert recorded_shapes[0][0] ==
200`. That is load-independent and needs no timing assertion.
**Evidence:** Read tests/feature_selection/mrmr/core/test_mrmr_basic.py:221-256.

### XCUT_NONDISCRIMINATING_ASSERTS-18 [P2] both-branches-are-no-ops
**File:** tests/training/pipeline/test_pre_pipeline_applied_to_test.py:245
**Summary:** A ~100-line test that fits a full Lightning MLP inside a `TransformedTargetRegressor` pipeline ends
in `if has_nan: pass  # This is the bug` -- zero assertions.
**What it would miss:** Both directions of the behaviour it exists to pin. The docstring says "This test proves
the MLP NaN behaviour so we know to handle it differently (pre-pipeline MUST be applied)". It proves nothing: if
the MLP started raising `ValueError` on NaN input (a torch/Lightning upgrade, or a guard added upstream), the
`has_nan` computation at :244 would be reached with `has_nan == False` and the test would still pass, silently
invalidating the premise the rest of the pre-pipeline design rests on. Conversely a regression that makes
`predict` return NaN for finite input is equally invisible.
**Suggested fix:** `assert has_nan, "MLP no longer returns NaN silently on NaN input -- the pre-pipeline premise
has changed; re-derive the NaN guard"`. One line, and it turns a ~100-line inert fixture into a real premise
sensor.
**Evidence:** Read tests/training/pipeline/test_pre_pipeline_applied_to_test.py:147-247; the terminal `if
has_nan: pass` at :245-247 and no `assert` anywhere in the function.

### XCUT_NONDISCRIMINATING_ASSERTS-19 [P2] biz_val-file-that-asserts-nothing
**File:** tests/feature_engineering/transformer/test_biz_val_real_datasets.py:1342
**Summary:** 394 of the 396 test functions in this `test_biz_val_*` file have zero assertions; they run boosting
matrices on real datasets, print the numbers, and always pass -- a 10-15 minute always-green block under a
`biz_transformer` marker.
**What it would miss:** Every transformer-FE regression on real data. Each test delegates to `_per_dataset_test`
-> `_run_matrix` -> `_print_matrix`, none of which assert; the loader-failure path additionally `pytest.skip`s
(:1307), so a dataset that stops loading also reads as green-ish. The docstrings advertise specific expected
effects that are never checked -- `test_matrix_kin8nm` says "Expect RFF to lift all three boostings
substantially (~5-12% R^2)"; if RFF's lift went to zero or negative, the number would be printed and the test
would pass. The module docstring is honest that set 1 is "INFORMATIONAL (always passes)", and that honesty is
why I rate this P2 rather than P1 -- but the file is named `test_biz_val_*`, is collected as tests, and its green
status is read as coverage of the transformer-FE value claim, against a repo convention stating in capitals that
every ML trick gets a quantitative biz_value test with a threshold 5-15% below the measured value.
**Suggested fix:** Split the file. Keep the informational matrix as a script or a `@pytest.mark.report`-gated
entry point not named `test_biz_val_*`, and promote the handful of datasets where a lift is genuinely claimed
(kin8nm/RFF, phoneme/row-attention) into real floors: `assert lift_rff_r2 >= <measured * 0.85>`. The same applies
to the 87 zero-assertion tests in the sibling
tests/feature_engineering/transformer/test_validation_records_at_scale.py (e.g. :113
`test_scale_iter69_year_100k_cb_r2`, whose docstring asks "Does it scale to ...?" and never answers in an
assertion).
**Evidence:** AST count over the file: 394 of 396 test functions contain no `Assert` node anywhere, including
through the `_per_dataset_test` / `_run_matrix` / `_print_matrix` helper chain. Read the module docstring at
:1-17 and `_per_dataset_test` at :1338-1352.

### XCUT_NONDISCRIMINATING_ASSERTS-20 [P3] presence-instead-of-absence
**File:** tests/training/neural/test_neural_medium_severity_regressions.py:362
**Summary:** Two tests pin "no per-call lazy import" by asserting that module-top names exist.
**What it would miss:** The per-call import. `assert hasattr(ce, "_HAS_XXHASH")` and `assert hasattr(ce,
"_hashlib")` (:362-363, same shape at :373 for the MLP-ranker fit) are satisfied by a module that hoists the
names to the top and still executes `import xxhash` inside `_compute_cache_key` on every call -- which is
precisely the cost the fix removed. The defect is the presence of an import statement in a hot function; the
test checks the presence of a module attribute instead.
**Suggested fix:** Assert the absence at runtime: wrap `builtins.__import__` in a counter, call
`_compute_cache_key(...)` twice, and assert zero import events for `xxhash`/`hashlib` after the first call. That
is behavioural and survives renames of the module-level constants.
**Evidence:** Read tests/training/neural/test_neural_medium_severity_regressions.py:355-373 against the
docstring claim ("Hash backend ... was imported per-call inside `_compute_cache_key`").

### XCUT_NONDISCRIMINATING_ASSERTS-21 [P3] type-instead-of-value
**File:** tests/estimators/test_custom_bugfixes.py:15
**Summary:** `test_create_dummy_lagged_predictions_negative_lag_uses_np_nan` asserts only `isinstance(out,
np.ndarray)` -- never that the fill value is NaN.
**What it would miss:** A wrong fill value. The test name and the comment name the contract as the NaN fill
("`np.NaN` was removed in numpy 2 -> AttributeError on the lag<=0 branch"); the fix at
src/mlframe/estimators/custom.py:460 is `cval = np.nan`. A regression to `cval = 0.0` (a plausible "just fix the
crash" edit) silently turns the dummy baseline's out-of-range region into hard zeros -- for a regression
baseline a wrong-results defect, not a crash -- and this test passes. Only the crash is caught.
**Suggested fix:** `out = create_dummy_lagged_predictions(np.array([1.,2.,3.,4.]), strategy="constant_lag",
lag=-1); assert np.isnan(out[-1]) and not np.isnan(out[0])`, mirroring the exact-array comparison the ndarray
legs of `test_helpers_ensure_no_infinity_bugfix.py` already use.
**Evidence:** Read tests/estimators/test_custom_bugfixes.py:12-16 against
src/mlframe/estimators/custom.py:454-464.

### XCUT_NONDISCRIMINATING_ASSERTS-22 [P3] type-asserted-content-not
**File:** tests/reporting/test_ux_crosscutting_regressions.py:38
**Summary:** Two of the five tests in `TestNoFabricatedNumbersOnEmptyInput` assert only the panel type, while
the other three in the same class also assert the annotation text.
**What it would miss:** An empty or meaningless annotation. The class's stated contract is "a builder with no
usable row must SAY so". `assert isinstance(panel, AnnotationPanelSpec)` (:38, and :64 for the
spectral-embedding degenerate graphs) is satisfied by `AnnotationPanelSpec(text="")` -- a blank box where a
"no rows" message should be, which reads to a user exactly like the fabricated zero the fix removed. The
siblings show the right shape: :43 asserts `all(p.text for p in panels)` and :59 asserts `"rows" in panel.text`.
**Suggested fix:** Add `assert panel.text` at both sites, and ideally a substring check naming the reason
(`"rows"` / `"no data"`), matching :59.
**Evidence:** Read tests/reporting/test_ux_crosscutting_regressions.py:31-64 -- the five tests side by side.

### XCUT_NONDISCRIMINATING_ASSERTS-23 [P3] every-assert-behind-a-hasattr-guard
**File:** tests/training/test_all_models.py:1049
**Summary:** Five prediction-sanity tests wrap their only assertion in `if hasattr(entry, attr)` + `if preds is
not None` + `if len(preds) > 0`, so a suite that stops producing predictions passes all five.
**What it would miss:** The regression these tests are the last line of defence against -- the suite silently
returning model entries with no `test_preds` at all. `test_predictions_not_all_nan` (:1049-1053),
`test_predictions_not_all_same` (:1079-1082), `test_predictions_in_valid_range_regression` (:960),
`test_probabilities_sum_to_one` (:992) and `test_prediction_shape_matches_input` (:1085) all degenerate to
no-ops in that case, and each costs a full `train_mlframe_models_suite` run to do so.
**Suggested fix:** Turn the guards into assertions: `assert hasattr(model_entry, "test_preds") and
model_entry.test_preds is not None and len(model_entry.test_preds) > 0` first, then the sanity check unguarded.
The suite is expected to produce predictions for a ridge regression on the shared fixture; if it legitimately
may not, the guard should be a `pytest.skip` with a reason, not a silent fall-through.
**Evidence:** Read tests/training/test_all_models.py:1025-1082; the AST scan for "every assertion inside an
`if`" flagged all five members of this class.

### XCUT_NONDISCRIMINATING_ASSERTS-24 [P3] documented-expectation-written-as-pass
**File:** tests/calibration/test_pick_best_calibrator.py:59
**Summary:** `if out["rule"] == "default_beta": pass` with the comment "Default beta should NOT trigger at
n=2000 (only when n_oof<1000)".
**What it would miss:** `default_beta` firing at n=2000 -- i.e. the n-threshold in the selection policy being
inverted or dropped -- which the comment identifies as wrong and the code then permits. The preceding line (:55)
already admits `default_beta` into the allowed rule set, so no other assertion catches it.
**Suggested fix:** `assert out["rule"] != "default_beta", f"default_beta fired at n_oof={out['n_oof']}; it is
reserved for n_oof < 1000"`. If the betacal-absent case really can reshuffle the candidate pool (the comment's
stated reason for the `pass`), gate on `pytest.importorskip("betacal")` rather than disarming the check.
**Evidence:** Read tests/calibration/test_pick_best_calibrator.py:31-64.

### XCUT_NONDISCRIMINATING_ASSERTS-25 [P3] one-attribute-as-proxy-for-a-liveness-contract
**File:** tests/feature_selection/shap_proxied/test_shap_proxied_fs_split_memory.py:99
**Summary:** "the selector must not retain a handle to the (large) parent X" is asserted as
`sel._deferred_holdout is None`.
**What it would miss:** Any other retaining attribute. The docstring states the contract as a liveness property
of the parent block ("so the parent block can be garbage-collected as soon as the caller drops its reference"),
and the assertion checks one named private field. A refactor that clears `_deferred_holdout` but stashes the
same frame on `self._X_`, in a closure held by a fitted sub-estimator, or in a cached splitter, passes -- while
the wide block stays alive for the lifetime of the selector, which on this repo's 100+ GB frames is the failure
this lever exists to prevent.
**Suggested fix:** Assert the liveness directly: `ref = weakref.ref(X); sel.fit(X, y); del X; gc.collect();
assert ref() is None`. That is attribute-agnostic and cannot be defeated by a rename.
**Evidence:** Read tests/feature_selection/shap_proxied/test_shap_proxied_fs_split_memory.py:71-101.

### XCUT_NONDISCRIMINATING_ASSERTS-26 [P3] tautology
**File:** tests/models/test_biz_val_lgbm_defaults.py:143
**Summary:** The "opt-in" contract is asserted as `default_lgbm_params() == default_lgbm_params()` -- both sides
are the same call with the same arguments.
**What it would miss:** The contract in the docstring, "omitting `auto_extra_trees` must leave the static
default unchanged". Comparing a function to itself can only fail if the function is nondeterministic; it says
nothing about whether omitting the new parameter changed the returned defaults. The two preceding assertions
(`baseline_small["extra_trees"] is True`, same for large) do carry the real contract, so this line contributes
nothing but a misleading comment ("bit-identical across calls, new params untouched"). Same shape at
tests/models/test_biz_val_lgbm_defaults_dart_heuristic.py:110.
**Suggested fix:** Compare against the other side of the switch -- `assert default_lgbm_params() ==
default_lgbm_params(auto_extra_trees=False)` -- or against a pinned dict of the pre-flip defaults. Otherwise
delete the line; the two assertions above it are the contract.
**Evidence:** AST scan for `assert <X> <op> <X>` with byte-identical sides returns 26 sites tree-wide; 24 are
legitimate (NaN checks written as `v != v`, and same-seed determinism checks such as
tests/test_rng_determinism_sweep.py:41). These two are the exceptions.

### XCUT_NONDISCRIMINATING_ASSERTS-27 [P3] output-never-inspected
**File:** tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:1281
**Summary:** `test_regression_mdlp_recurse_oos_validated_no_dead_present_parent_branch` passes a `splits` list
into the recursion and never looks at it.
**What it would miss:** The "still works correctly" half of its own docstring. The call is made with both
`counts_parent` and `present_parent` supplied, which is the combination that worked before the dead-branch
removal too, so the implicit "does not raise" assertion held pre-fix. And because `splits` is never inspected, a
regression that makes the recursion return without emitting any cut on a perfectly separable `y = (x > 0)`
fixture -- the strongest possible failure of MDLP -- passes.
**Suggested fix:** `assert len(splits) >= 1` plus a check that the recovered cut sits near zero for this fixture
(`assert abs(x[splits[0]]) < 0.3`), which is what "still works correctly" means here.
**Evidence:** Read tests/feature_selection/regression/test_regression_mrmr_audit_2026_07_22.py:1268-1284.

## Coverage
**Reached.** The prior list of 903 is superseded, not sampled: I regenerated from source and got 1,773
candidates (1,093 weak-only + 680 zero-assertion) over all 3,411 files / 24,528 test functions, which is a
superset of the prior 903 on the weak-only axis (my pass adds `hasattr`/`callable`, `BoolOp` conjunctions of
weak terms, and zero-assertion functions; it subtracts ~110 helper-delegating false positives the prior pass
would have counted). Of these I hand-triaged **430**:

- **118 / 118** of the priority band -- every candidate in a file matching `bugfix|regression|biz_val|biz_value`,
  excluding the 394-test informational matrix (reported as finding 19). Read as docstring-versus-assertion;
  ~40 read in full.
- **241 / 241** of the defect-narrative band -- every candidate whose docstring contains `pre-fix|previously|
  used to|before the fix|bug|regression|must not|silently|defect|broke|crash|leak|wrong`. This is the shape-2
  band the brief calls highest-value; read by docstring, ~25 read in full.
- **All 40** members of the four auxiliary shape scans, read in full: 7 assertions under an exception-swallowing
  `try`, 3 unconditional imperative `pytest.xfail`, 26 identical-both-sides comparisons, 4 `if ...: pass` bodies.
- **~31** of the 168 "every assertion inside an `if`" band, selected by biz_val/regression naming and by whether
  the `if` condition is derivable from the code under test (the dangerous case) rather than from a fixture
  capability flag (the benign case).

**Left: 1,343 candidates, in three groups, all lower yield.**

1. **~980 weak-only candidates in ordinary unit-test files with a neutral docstring** (auto-generated
   `"""Foo bar baz."""` summaries), concentrated in `test_*_coverage*.py`, `test_*_fill.py`, `test_*_smoke.py`
   and the `test_meta/test_broad_except_logging_*.py` family. Lower yield for a structural reason, not a
   convenience one: the shape-2 signal comes from comparing a docstring's defect claim against the assertion,
   and these have no claim to compare against. Spot-reading ~15 found exactly the pattern the prior agent
   described -- `_estimate_slot_nbytes` really must return an `int`, `_has_scikit_survival` really must return a
   `bool`, `get_process_rss_mb` really must return a `float` on probe failure. Their residual defect class is
   "asserts a type where a value is the contract", a coverage-quality issue rather than a
   defect-survives-the-test issue, and it is uniform enough across ~980 sites to warrant a lint rule rather than
   980 findings.
2. **~250 zero-assertion tests that are legitimate crash sensors** -- `MRMR().fit()` after an
   `AttributeError`-in-`get_params` fix, `_warmup_numba_kernels` after a recursion fix, `fix_random_seed(large)`
   after a `ValueError` fix. For these the call is the assertion, and I verified by reading the docstring that
   the pre-fix code raised. I read ~35; the remainder share the naming pattern `*_does_not_crash`, `*_no_raise`,
   `*_survives_*`, `*_still_importable`.
3. **The two shape bands I did not exhaust**: 137 of the 168 "assert only inside an `if`" (most are
   `if spec.has_gfno:` / `if backend_available:` capability gates on a parametrised fixture, which is correct
   usage), and the 884 "assert only inside a loop" band, which I sampled but did not triage. The dangerous
   sub-case there is a loop over a collection the code under test produces -- an empty collection makes the body
   never run -- and separating those from loops over literal parameter lists needs per-site reading. I flag that
   884-item band as the largest single surface still open; a cheap narrowing filter is "loop iterable is derived
   from a call on the object under test AND no assertion on the iterable's length precedes the loop".

**Not verified by execution.** I did not run the suite. Every "would still pass" claim above is derived by
reading the assertion against the production code or against the test's own docstring; the cases where
production source was read to confirm are findings 1 (`auto_transform_select.py:227-229`), 8
(`stop_file.py:114-125`) and 21 (`custom.py:454-464`). Findings 2, 3, 5, 9, 18 and 19 need no execution to
confirm -- they assert nothing.
