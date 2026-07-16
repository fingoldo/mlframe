# MRMR test coverage quality audit

This audit split into three sub-passes over the test tree (the biz_val/fe directories are large):
A) `biz_val` remaining subdirs (contracts_robustness, regression_union, provenance_streaming, dcd, flat root)
B) `fe/` + `biz_val/test_biz_value_mrmr_scorers/`
C) `core/` + `caching/` + `regression/`

---

## A) biz_val test coverage (contracts_robustness, regression_union, provenance_streaming, dcd, flat root — excluding scorers)

METHOD: Full reads of ~10 large/central files across contracts_robustness, regression_union, provenance_streaming, dcd, and the flat root; directory listing + targeted greps (sample_weight, groups=, multi-output, provenance/streaming, GPU/cupy, inspect.getsource/read_text) across all ~130 files in scope.

**Source-inspection anti-pattern**: No occurrences of `inspect.getsource()` or reading a `.py` file as raw text to grep source strings were found anywhere in scope. The only `inspect.*` usages found are legitimate signature/param introspection (behavioral-adjacent, not text-scraping):
- `test_biz_value_mrmr_fe_mechanisms/test_layer80.py:412` - `inspect.signature(_build_pool_mapping)`
- `test_biz_value_mrmr_provenance_streaming/test_layer53.py:121` - `inspect.signature(MRMR.partial_fit)` checks param names/order - fine, pins an API contract, not source text.
- `test_biz_val_mrmr_adversarial_medium_n.py:355` - `inspect.signature(MRMR.__init__).parameters` checks a ctor param exists.
- `test_biz_val_mrmr_default_filtering.py:197` - `inspect.signature(...).parameters["fe_min_polynom_degree"].default` checks a default value.
None of these are the anti-pattern this audit was asked to flag; codebase is currently clean on this specific axis in the audited scope.

**GROUP-AWARE MI / `groups=` PARAMETER: WEAK/MISSING**
grep for `groups=|group_id|feature_groups` only hit test_layer17.py, test_layer101.py, test_layer79.py, test_biz_val_mrmr_underselection.py (all read in full). In every one of these, "group_id"/"group" is a DATA COLUMN used to build a fixture (e.g. per-group mean leakage in test_layer17.py, `fe_grouped_delta_group_col="group_id"` FE feature in test_layer79/101), NOT an actual `groups=` selection-time parameter of MRMR controlling group-aware relevance/redundancy. No test in the audited scope exercises an actual `groups` constructor/fit argument that changes selection behavior in a directly observable, behavioral way. Genuine coverage gap.

**SAMPLE_WEIGHT: ONE GOOD BEHAVIORAL TEST, otherwise thin**
- `test_biz_val_filters_mrmr.py:1343` `test_biz_val_mrmr_sample_weight_flips_top_feature_under_recency_vs_uniform` - GENUINE behavioral test. Builds two features A/B where A drives y on the recent half and B on the older half, fits with `sample_weight=None` vs a recency-weighted array, and asserts the actual selected top-1 feature name FLIPS (uniform picks B, recency-weighted picks A). Well-reasoned, FE families explicitly disabled to isolate the raw-feature ranking flip. Good.
- `test_biz_value_mrmr_provenance_streaming/test_layer53.py` uses sample_weight-adjacent machinery only indirectly (partial_fit_decay described as "implemented atop the existing sample_weight resample contract") but never calls `.fit(..., sample_weight=...)` directly.
Aside from these, no other file in the ~130 exercises sample_weight at all - single-file, single-test coverage area. Adequate as a positive/negative behavioral pin but thin in breadth (no zero-weight-collapses-rows test, no mrmr_gains_ value check, no regression/continuous-target sample_weight test - only classification).

**MULTI-OUTPUT y: MISSING**
grep for `multi.output|multioutput|y_multi|n_outputs` returned zero hits across the entire scope. No test anywhere constructs a multi-column y (DataFrame y) or checks MRMR's behavior/API surface for multi-output targets — nor is there a test asserting correct REJECTION of multi-output.

**PROVENANCE/STREAMING CORRECTNESS: STRONG, GENUINELY BEHAVIORAL**
- `test_biz_value_mrmr_provenance_streaming/test_layer53.py` (partial_fit) - excellent behavioral suite: first-call support_ set-equality with plain fit (C1), buffer growth counts and exact row counts after N batches (C2), an actual target-driver FLIP under decay=1.0 vs decay=0 (C3), exact buffer truncation counts under `partial_fit_window` (C4), full pickle round-trip with `assert_frame_equal`/`assert_series_equal` on buffers plus `assert_array_equal` on support_ (C5), and clone() blank-slate verification. Model example of the requested behavioral bar.
- `test_biz_value_mrmr_provenance_streaming/test_layer54.py` (fe_provenance_) - also strongly behavioral: exact DataFrame schema/column order, row-count identity, exact ordering match against transform() output order, origin-label correctness per mechanism kind, numeric alignment of mrmr_gain against mrmr_gains_ via assert_allclose, strictly-increasing support_rank invariant, full pickle-preserves-DataFrame check, clone-drops-then-refit-repopulates semantics.
- `test_biz_value_mrmr_regression_union/test_layer101.py TestMegaFixtureAllOn` - checks fe_provenance_["origin"] diversity (>=4 distinct origins), duplicate-name detection, downstream-AUC recovery contract with an honestly-documented (not padded) floor.
Streaming/provenance is the best-covered dimension in this whole sample.

**GPU/CPU DISPATCH PARITY: MISSING (in-scope)**
grep for `cupy|use_gpu|device=|GPU\b` hit 6 files; the only one read in full (`test_biz_value_mrmr_contracts_robustness/test_layer20.py`) uses "GPU" only in a docstring aside describing wall-time budgets - it is a CPU-only wide-matrix/embedding-scale test, not a GPU/CPU parity test. No test in this scope actually runs MRMR/an MI kernel on both backends and asserts equal/near-equal output. GPU/CPU dispatch parity is untested in this slice (may be covered by test_biz_value_mrmr_scorers, see part B).

**OTHER NOTABLE WEAK/SMOKE-ONLY TESTS**
- `test_biz_val_filters_mrmr.py` contains many parametrized "completes without raising" tests (e.g. `test_biz_val_mrmr_fe_max_pair_features_completes`, `test_biz_val_mrmr_cv_int_parametrize_completes`, `test_biz_val_mrmr_cv_shuffle_parametrize_completes`, `test_biz_val_mrmr_only_unknown_interactions_completes_smoke`) - only assert `1 <= len(support_) <= p`. Weak, largely redundant coverage of "doesn't crash" alongside adjacent tests that DO assert signal recovery.
- `test_biz_value_mrmr_regression_union/test_layer101.py TestPriorLayerRoster` and `test_layer79.py TestPriorLayerDiscoverability` are pure file-count/glob "silent-delete" guards - legitimate as anti-regression tripwires but zero MRMR-correctness coverage.
- `test_biz_val_mrmr_adversarial_medium_n.py:355` and `test_biz_val_mrmr_default_filtering.py:197` are single-assertion signature/default-introspection tests - fine as narrow API-contract pins but trivially weak in isolation.

FILES SAMPLED BUT NOT DETAILED (skimmed via directory listing/grep only, no anomalies beyond above): test_biz_value_mrmr_contracts_robustness/test_layer{6-16,18,19}.py, test_biz_value_mrmr_dcd/{test_accessors,test_anchor_refinement,test_distance_autotune,test_hierarchical,test_kitchen_sink,test_recipe_pool}.py, test_biz_value_mrmr_fe_encodings/*, test_biz_value_mrmr_fe_hybrid_orth/*, test_biz_value_mrmr_fe_mechanisms/*, test_biz_value_mrmr_grouped_cat_fe/*, test_biz_value_mrmr_linear_preselect/*, test_biz_value_mrmr_param_oracle/*, test_biz_value_mrmr_recipe_fe/*, and most loose root files (adaptive_chirp/fourier, gate_med, gate_vs_elementary, hard_cases, interaction_info_prefilter_speedup, multiway_synergy, order2_maxt_floor, pair_prewarp, pre_distortion, prefer_engineered, quality_metrics, rc2_undersampled, selection_stability_report, sufficient_summary, ultra, usability_raw_retention). None surfaced a groups=, multi-output-y, or GPU-dispatch hit, reinforcing those three areas are gaps suite-wide.

**SUMMARY VERDICT (part A)**
- Strong/behavioral: partial_fit (test_layer53.py), fe_provenance_ (test_layer54.py), leakage-detection (test_layer17.py), mega-fixture composite (test_layer101.py/test_layer79.py), sample_weight top-1 flip (test_biz_val_filters_mrmr.py:1343).
- Weak but harmless padding: numerous "completes without raising" parametrized smoke tests in test_biz_val_filters_mrmr.py.
- Genuine gaps: groups=/group-aware MI (behavioral test absent), multi-output y (absent entirely), GPU/CPU dispatch parity (absent in this scope - check test_biz_value_mrmr_scorers).
- No source-inspection anti-pattern instances found.

---

## B) fe/ + biz_val/test_biz_value_mrmr_scorers/ test coverage

**Regression tests (campaign #2 fixes):**

`tests/feature_selection/test_mrmr_critique_sf2_jmim_confirm.py:13-16` (`test_confidence_chain_threads_use_jmim`) - SOURCE-INSPECTION/weak: uses `inspect.signature()` to check "use_jmim" is a parameter name on two functions; never exercises the value flowing through the confirmation path. A signature can carry a param that is silently ignored inside the body - this test cannot catch that.

Same file, lines 19-28 (`test_jmim_fit_completes_with_confirmation`) - WEAK: only asserts `hasattr(m,"support_")` and `len(...)>=1`, no exception. Does not verify that the confirmation now uses the JMIM statistic instead of CMIM (the actual bug being fixed) - a pure "didn't crash" smoke test for a scoring-correctness bug.

`tests/feature_selection/test_mrmr_critique_sf3_jmim_exponent.py:12-18` - WEAK guard: checks an env var default and a module-level boolean constant (`_JMIM_EXPONENT_DISCOUNT_ONLY` is False). Legitimate as a "don't silently flip the default" tripwire, but doesn't touch actual selection output; real evidence lives in an external bench script (bench_sf3_jmim_exponent_selection.py), not in a runnable regression assertion here.

`tests/feature_selection/info_theory/test_jmim_cache_parity.py:56-112` - BEHAVIORAL, the strongest of the three: hard parity gate comparing `get_feature_names_out()`/`support_` between cache-enabled and forced-cache-miss ("killed") runs, plus asserting `sum_hits>0` (cache actually engages) and `killed_hits==0` (kill path verified effective). Real byte-identical selection-equivalence test with a monkeypatch-based negative control - good practice, no weak spots.

**fe/ directory (26 files, all read):**

Behavioral, transform-value-checking (strong):
- `fe/test_biz_val_filters_mrmr_fe_hermite_l2_penalty.py` - numeric penalty thresholds.
- `fe/test_biz_val_filters_mrmr_fe_pair_prewarp_basis_degree.py` - transform() correlation floors (corr>=0.85).
- `fe/test_biz_val_filters_mrmr_fe_polynomial_basis.py` - transform() correlation floors (corr>=0.80).
- `fe/test_biz_val_filters_mrmr_fe_triple_maxt.py` - MI-floor/gate-consumer set checks; no transform-value check.
- `fe/test_biz_val_filters_mrmr_fe_warp_linear_margin.py` - exact selected-vars pin; no transform-value check.
- `fe/test_biz_val_mrmr_sample_weight_fe.py` - Jaccard/set-overlap thresholds; one sub-test xfailed (np.allclose transform check) documenting a known real gap rather than hiding it - acceptable per project convention.
- `fe/test_biz_value_mrmr_fe_canonical.py` - transform() correlation (rho>0.5) and Ridge r2>=0.72, deterministic drop-set checks.
- `fe/test_biz_value_mrmr_fe_downstream_delta.py` - real downstream Ridge R2/LogReg AUC deltas on actual transform() output (delta>0.05) - strongest transform-value test in fe/.
- `fe/test_biz_value_mrmr_fe_form_selection.py` - exact winner pin plus binned-MI floor on transform() output; also pins a CPU/GPU fp-jitter tiebreak (w_cpu==w_gpu) - simulated, not a real GPU run.
- `fe/test_biz_value_mrmr_univariate_basis_fe.py` - transform() correlation floors 0.70-0.85 per basis case.
- `fe/test_fe_batch_parity.py` - np.allclose(cpu,gpu,atol=1e-9) - genuine GPU/CPU dispatch parity, gated @pytest.mark.gpu.
- `fe/test_fe_edge_mi_parity.py` - CPU-edge vs GPU-resident-edge parity, also gated @pytest.mark.gpu; asserts divergence-is-real check too.
- `fe/test_fe_multi_gpu.py` - multi-GPU vs single-GPU MI table parity (np.allclose atol=1e-9), gated @pytest.mark.gpu.
- `fe/test_mrmr_append_engineered_no_repeated_assign.py` - assert_frame_equal no-mutation check + assert_array_equal on actual transform values + perf assert (dt<5.0).
- `fe/test_mrmr_cat_fe_integration.py` - recipe-source membership + replayed-value range check + pickle round-trip.
- `fe/test_mrmr_engineered_replay.py` - np.testing.assert_allclose(transform_output, ground_truth_formula, rtol=1e-5) on disjoint test data - exact formula match, excellent.
- `fe/test_mrmr_fe_composite_feedforward.py` - np.allclose(out, exact formula) plus Spearman rho>=0.95 on replayed held-out data, plus pickle equality.
- `fe/test_mrmr_fe_fixes_adversarial.py` - byte-exact transform-value comparison (np.array_equal) between slice-fit and full-fit; CPU forced via env var, never compared to GPU.
- `fe/test_mrmr_fe_on_polars.py` - structural: engineered feature-NAME-set equality between polars/pandas paths, not raw numeric values.
- `fe/test_mrmr_fe_vote_dropped_no_recipeless_select.py` - real log-message + feature-name/transform-column membership checks.
- `fe/test_mrmr_feature_engineering.py` - mostly behavioral (exact substring/name composition pins); two weaker cells: `test_unary_transform_detection` (membership only, no threshold) and `test_no_false_positives_independent_features` (count ceiling only, borderline weak).

Weak (no-exception / smoke only):
- `fe/test_fe_pairs_cpu_fallback_buffer_overflow.py` - only hasattr(support_) and sum>=1; pure crash-regression smoke, no values checked.
- `fe/test_mrmr_append_engineered_warns_on_name_mismatch.py` - only a log-substring check, no numeric assertions.

Mixed/weakened-by-skip:
- `fe/test_fe_fusion_scoring_subsample.py` - real stride/shape assertions but end-to-end test has a `pytest.skip(...)` escape hatch if no compound feature is admitted, which can silently blank the test on a given run - flagged per "don't accept documented skips" convention.
- `fe/test_fe_gate_scoring_subsample.py` - same pattern: stride-formula sub-test solid, but fit-based sub-test has a pytest.skip bailout if the gate never fires.
- `fe/test_fe_path_perf_identity.py` - behavioral (exact selected-indices/names pinned against a hardcoded reference list, bit-identical numpy check) but brittle by construction (hardcoded reference arrays) rather than weak.

Source-inspection occurrences found:
- `biz_val/test_biz_value_mrmr_scorers/test_conditional_routing.py:528-533` (`test_default_routing_criterion_is_corr`) - uses `inspect.signature(gen).parameters["routing_criterion"].default == "corr"` instead of exercising behavior. Borderline legitimate (pinning a constructor default) but an inspection substitute - if the default is read but never wired through, this test would still pass.
- `test_mrmr_critique_sf2_jmim_confirm.py:16` - inspect.signature usage as already noted above.
No other inspect.getsource()/source-as-text patterns found anywhere in fe/ or biz_val/.

**biz_val/test_biz_value_mrmr_scorers/ (16 files, all read):**

All 16 are predominantly BEHAVIORAL with real numeric thresholds in the project's biz_value style:
- `test_adaptive_degree.py` - exact per-source degree recovery pins (x1==2, x2==4, x3==6), noise floor <0.05 nats, AUC lift >=+0.05.
- `test_bootstrap_mi.py` - uplift_lcb/mean/std numeric comparisons, cross-seed floor divergent_seeds>=5; includes a sort-order optimization gate that is NOT GPU/CPU parity despite superficially resembling it.
- `test_cluster_basis.py` - exact cluster-count/size pins, MI-ratio floor agg_mi>1.3*best_member_mi.
- `test_cmi_greedy.py` - family-collapse counts, AUC-parity tolerance <=0.07, pickle-replay allclose.
- `test_cmim.py` - margin novel>max_dup+0.02, Spearman floor >=0.5, AUC lift >=+0.005.
- `test_conditional_routing.py` - per-column basis/degree recovery pins, recipe-replay allclose, held-out generalization check.
- `test_copula_mi.py` - cm>0.3, invariance approx(abs=1e-12), AUC lift.
- `test_dcor.py` - dcor>0.40, Pearson blind-spot <0.20-0.25, symmetry approx(abs=1e-12).
- `test_diff_basis.py` - uplift>1.0, auc_aug>=0.85 and delta>=0.15.
- `test_ensemble_uplift.py` - Jaccard/AUC non-regression floors, borda agreement counts.
- `test_hsic.py` - independence floor <0.001, signal floors explicitly recalibrated with documented rationale - good practice, not silent weakening.
- `test_jmim.py` - jmim_picks_x2>=3/5 seeds, dup-count gate, AUC floor >=+0.005; this is the direct scorer-level counterpart to the "jmim-taming" campaign fix and does test real scoring behavior numerically - but has no docstring/commentary tying it to the campaign #2 fix (that context lives only in the dedicated regression files above).
- `test_ksg_mi.py` - per-seed ksg_he3>pi_he3, aggregate mean*1.10, non-negativity floor.
- `test_three_gate_oof.py` - majority-vote bias check n_seeds>=4, CMI-collapse ratio <0.10*marginal, documented threshold calibration.
- `test_total_correlation.py` - tc>=0.3 and tc>=3.0*max_pairwise+0.1 (headline higher-order contract).
- `test_triplet_cross_basis.py` - exact top-ranked column name pin, MI floor, AUC lift auc_aug>auc_raw+0.20.

**Specific gap checks:**
- **fast_search profiles: MISSING**. Not one file across fe/ (26) or biz_val/ (16) references fast_search; test_mrmr_fe_composite_feedforward.py explicitly sets `fe_fast_search=False` to pin the exhaustive path — the fast path itself is never exercised or asserted on.
- **cv-based stability selection: MISSING**. No file references cross-validation-based stability selection; closest analog is test_bootstrap_mi.py's bootstrap-resample uplift/CI ranking, a different mechanism (resampling, not k-fold-CV).
- **GPU/CPU dispatch parity for scorers (jmim/cmim/cmi): MISSING at the scorer level**. None of the 16 biz_val scorer files compare a GPU-dispatched score against its CPU counterpart - all pure CPU/numpy/sklearn synthetic evaluations. Real GPU/CPU parity exists only in the fe/ MI-table layer (test_fe_batch_parity.py, test_fe_edge_mi_parity.py, test_fe_multi_gpu.py), all gated behind @pytest.mark.gpu and skipif(not cuda-available) - meaning on a CPU-only CI run (documented default per project memory), these parity assertions never execute at all, leaving jmim/cmim/cmi scorer-level GPU path with zero executed parity coverage in ordinary test runs.
- **fe correctness on actual transform() values: PRESENT and reasonably strong** - test_mrmr_engineered_replay.py, test_mrmr_fe_composite_feedforward.py, test_biz_value_mrmr_fe_downstream_delta.py, test_mrmr_append_engineered_no_repeated_assign.py, test_mrmr_fe_fixes_adversarial.py all check real numeric transform() output; weaker fe files are the minority.

---

## C) core/ + caching/ + regression/ test coverage

**Source-inspection anti-pattern hunt** (explicit grep across all 85 files): Only one file references `inspect.getsource` at all:
- `tests/feature_selection/mrmr/core/test_mrmr_setstate_ctor_defaults_no_drift.py:10` - docstring explicitly disclaims doing source-text inspection. Verified by reading the full body: resurrects a legacy pickle via `MRMR.__new__(MRMR); m.__setstate__({})`, reads `MRMR._ctor_defaults()`, compares live post-setstate attribute VALUES against a fresh MRMR() instance and against `_SETSTATE_LEGACY_OVERRIDES`. Genuinely behavioral, not string-grepping. No anti-pattern found anywhere else in the three directories.

**Coverage-depth verdicts for 14 requested behaviors:**

1. **fit() classification/regression, actual feature identity/count checked** — REAL. `test_mrmr_classification.py`, `test_mrmr_regression.py` use `signal_recovery_count` and assert `overlap >= 1`; `test_mrmr_create_keep_drop.py` + `test_mrmr_distribution_profiles.py` do full create/keep/drop verification of exact feature identity across 39+ synthetic formulas with a documented xfail registry (no green-by-relaxation). `test_mrmr_edge_cases.py::test_perfect_feature_detection` checks `'perfect' in selected_features`. Strong.

2. **transform() shape AND values match selected columns** — REAL. `test_mrmr_basic.py::test_transform_shape` checks shape only (weak alone), but `test_mrmr_edge_cases.py::test_transform_polars_input` checks `out.shape == (df.shape[0], len(names))` AND `list(out.columns) == names`, and `test_regression_categorical_factorize_replay_not_constant` (in `test_mrmr_cov_sweep_four_paths.py`) checks actual transformed cell VALUES bit-exact against a `pair_to_code` map on disjoint holdout data. Strong.

3. **get_support() actual boolean mask** — WEAK/MISSING dedicated coverage. `test_mrmr_tree_rescue.py::test_rescue_transform_pickle_and_support_consistency` checks `m.get_support().sum() == len(m.support_)` (count check, not full mask-content comparison). No test found asserting `get_support()` element-wise against a hand-computed boolean array. Genuine gap.

4. **get_feature_names_out() actual names checked** — REAL. Extensively covered: `test_mrmr_edge_cases.py::test_get_feature_names_out_includes_engineered`, `test_mrmr_nan_strategy.py`, `test_mrmr_polars_transform_*` (three files) all assert `list(out.columns) == names` and specific name content. Strong.

5. **pickling/__getstate__/__setstate__ round-trip: state equality** — REAL, well covered. `test_mrmr_diagnostics_pickle_parity.py` (byte-identical explain_selection(), array-equality of `_stability_replay_state_`, pickle-size guard), `test_internals.py::TestMrmrPickle` (assert_array_equal on support_), `test_mrmr_edge_cases.py::test_pickle_round_trip_smoke`, `test_mrmr_setstate_ctor_defaults_no_drift.py`, `test_mrmr_setstate_mro.py`, `caching/test_fingerprints_cache_isolation.py` (asserts replayed instance's fe_provenance_ is NOT the same object as cache source, support_ is writeable). One of the best-covered areas.

6. **multi-output y** — MISSING. No file in any of the three directories constructs a 2-D/multi-column y target. All targets single-column. Real gap if MRMR's public contract claims multi-output support.

7. **group-aware MI / groups parameter** — REAL but narrow. `regression/test_regression_mrmr_strict_groups.py` covers the `strict_groups` toggle (raises NotImplementedError when True+groups given, warns when False, no-op when groups=None) — but only tests the warn/raise gating, not that groups actually changes MI estimation to avoid cross-group leakage.

8. **sample_weight changes actual selection/scores** — REAL. `core/test_mrmr_sample_weight_unit.py` is thorough: None-vs-omitted byte-identical, uniform weight == unweighted, shape/NaN/negative/zero-sum validation raises, non-uniform weight actually triggers the resample branch, plus a polars-input regression. Never demonstrates non-uniform weights actually FLIPPING which feature is selected (only that the mechanism engages) — real-but-incomplete.

9. **fast_search profiles** — Not found under this name; closest is `_mrmr_sis_screen.py` tests covering the SIS front-gate cascade. Treat as MISSING under the exact name.

10. **cv-based stability selection** — REAL, present in multiple places: `test_mrmr_diagnostics_pickle_parity.py` (`selection_stability_report(n_boot=5)` post-pickle), `test_mrmr_cluster_stability_categorical_regression.py` (cluster/complementary-pairs stability selection end-to-end). Note: `test_biz_value_mrmr_selection_stability_report.py` (per earlier grep) only checks `isinstance(txt, str) and "selection-stability" in txt` — WEAK.

11. **GPU/CPU dispatch path parity** — REAL. `test_mrmr_basic.py::TestMRMRPermKernelGPU` directly calls both CPU and cupy kernels and checks numeric parity within tolerance, plus dispatcher threshold logic. Gated behind `pytest.importorskip("cupy")` so likely never runs on this dev box.

12. **Edge cases**:
    - all-constant column: REAL — `test_mrmr_edge_cases.py::test_constant_feature`, `test_mrmr_degenerate_frames.py`.
    - single-class y: REAL — `test_mrmr_edge_cases.py::test_validate_inputs_constant_y_raises`.
    - NaN in X: REAL, deep — `test_mrmr_nan_strategy.py`, `test_mrmr_cov_sweep_four_paths.py`.
    - NaN in y: REAL — `test_mrmr_degenerate_frames.py::test_y_nan_raises_valueerror`.
    - empty selection: REAL — `test_mrmr_edge_cases.py::test_no_features_selected_transform`.
    - n_features=1: not explicitly found in these three directories; may live in a shared-fixture file elsewhere.

13. **jmim taming regression test** — Not found by filename in core/caching/regression (a test_jmim.py exists under biz_val/scorers, outside this scope, likely weak per grep). No "jmim taming" regression test found in these three directories.

14. **cmiperm / cmi permutation regression test** — Not found under this name in core/caching/regression. MISSING in these directories (test_cmim.py exists under biz_val/scorers, outside scope).

**Weak tests found (real weak tests, not source-inspection):**
- Almost all of `test_mrmr_basic.py`, parts of `test_mrmr_classification.py`/`test_mrmr_regression.py`, `test_mrmr_parameters.py`, `test_mrmr_integration.py::test_pipeline_compatibility` assert only `hasattr(mrmr, 'n_features_')`/no-crash — smoke tests, not behavioral checks of which features got selected.

**Summary (part C):** Core/caching/regression directories are overall strong on: pickle/setstate round-trip fidelity, transform correctness, sample_weight validation, NaN/degenerate-column handling, GPU/CPU numeric parity, feature-identity recovery. No source-inspection anti-pattern found. Genuine gaps: no multi-output-y test, no dedicated get_support() mask-content test, groups= tested only for warn/raise gating, no jmim-taming or cmi-permutation regression test inside core/caching/regression (both likely exist only under biz_val/, outside these three directories — worth confirming whether that's intentional scope-splitting or a real gap).
