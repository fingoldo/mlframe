# FS tests audit 2026-06-10 — KEY=training_integration

Scope: integration-level FS testing in the training pipeline (`train_mlframe_models_suite` + predict path).

Sampling statement (for the verifier): read FULLY — tests/training/test_feature_selection.py (817),
test_composite_x_feature_selection.py (431), test_bizvalue_feature_selection.py (424),
test_metadata_feature_selection_report_observability.py (323), test_unfit_pre_pipeline_recovery_i59.py (167),
src/mlframe/training/_feature_selection_config.py (195), core/_setup_helpers_pre_pipelines.py (178),
core/_phase_train_one_target_helpers.py (298), pipeline/_pipeline_helpers.py:534-720, core/_phase_finalize.py:385-510,
core/_phase_train_one_target_schema.py:160-250, tests/training/conftest.py:175-415,
tests/inference/test_predict_passthrough_replay.py (198), heads of test_predict_round_trip_parity.py /
test_predict_from_models_unfitted_pre_pipeline.py / test_shortlist_transformer_adapter_suite.py,
test_integration_prod_like_polars.py:600-770, test_suite_coverage_gaps.py:1770-1800+2140-2230, test_core.py:380-420+2370-2510,
test_fuzz_metamorphic.py:200-263, _fuzz_suite_helpers.py:845-875, _fuzz_combo/axes.py FS axes. Grepped (not read) the rest of
tests/training (200+ files) for FS-knob/predict/load co-occurrence; excluded .claude/worktrees. Cross-checked
audit_disposition.md + the six sibling reports in this directory to avoid duplicates (dupes are cross-referenced inline).

---

### [training_integration-01] No save -> load -> predict round-trip parity test exists with FS enabled — the one known-recurring FS prod-bug class is exactly here
Severity: P0
Kind: gap-mask
Files:
- tests/inference/test_predict_round_trip_parity.py:1-12 (docstring: "Trains a deliberately mixed suite (polars input + preprocessing_extensions enabled + multiple base models)" — zero FS mentions in the whole file, verified by grep)
- tests/training/_fuzz_suite_helpers.py:851-871 ("This is a smoke check, not a bit-for-bit equivalence ... metadata/preprocessing trail is not hydrated by joblib.load alone")
- tests/inference/test_predict_passthrough_replay.py:39-67 (`_SelectFirstNCols` "Stand-in for a fitted MRMR/RFECV selector" — never a real selector)
- tests/training/test_unfit_pre_pipeline_recovery_i59.py:7-16 ("The per-model pre_pipeline contains a feature selector (MRMR / BorutaShap) that survived save/load with state that triggers NotFittedError on transform" — the bug class is documented as having happened, fuzz iter-59/301/326/318)
Evidence: the canonical round-trip parity harness (`test_in_memory_and_disk_predict_agree_on_simple_suite` and 3 siblings) never sets `FeatureSelectionConfig`; the fuzz disk check is `joblib.load` of the first `.dump` only; every FS-related predict test uses stand-in selector classes. So the full chain train(FS) -> save -> `load_mlframe_suite` -> `predict_from_models` -> parity is exercised NOWHERE, while the repo's own regression history (iter-59 family) shows fitted selectors losing fit-state across save/load, and user memory flags runtime-cache pickle exclusion as a recurring class (MRMR carries `_FIT_CACHE`/identity-cache overrides stamped via `_mlframe_identity_cache_override_`, `_setup_helpers_pre_pipelines.py:130-131` — a ctx-scoped dict stamped onto the instance, i.e. exactly the kind of live-object reference that must not survive pickling).
Proposal: extend the EXISTING harness in test_predict_round_trip_parity.py with FS cells rather than a new file. `@pytest.mark.parametrize("fs_cfg", [mrmr_simple, mrmr_fe_on, rfecv_cb])` x `@pytest.mark.parametrize("frame", ["pandas", "polars"])`: n=600, 6 informative + 4 noise, seed=0; train with `use_ordinary_models=True` + the FS config (mrmr_kwargs: use_simple_mode=True, n_workers=1, max_runtime_mins=1; fe cell: fe_max_steps=1, fe_ntop_features=3); capture `preds_mem = predict_from_models(df_test, models, metadata)`; `models2, meta2 = load_mlframe_suite(models_path)`; `preds_disk = predict_from_models(df_test, models2, meta2)`; assert per-model `np.allclose(preds_mem[k], preds_disk[k], atol=1e-9)` for the FS-branch entries specifically (key contains "MRMR "/"cb_rfecv"), AND assert via caplog that NO "Skipping pre_pipeline" / "falling back to fit_transform" recovery warning fired (the recovery branch firing on a fresh round trip IS the bug). Also assert the reloaded selector is fitted: `check_is_fitted(_unwrap_selector(models2[...][0].pre_pipeline))` raises nothing, and that pickling did not capture the ctx identity-cache dict (`not hasattr(sel, "_mlframe_identity_cache_override_") or sel._mlframe_identity_cache_override_ is None` — pin whichever exclusion contract io.py implements). Reuse the file's `_save_threads_zero` Windows workaround. Fast mode: keep the matrix at 6 cells, n=600, iterations=10.
Effort: M

### [training_integration-02] The predict-time FS recovery branch is tested only via a local re-implementation plus an AST source-presence scan — zero behavioral coverage of the production code
Severity: P1
Kind: quality|gap-mask
Files:
- tests/training/test_unfit_pre_pipeline_recovery_i59.py:35-53 (`_subset_recovery` — "Mirror of the recovery block at predict.py:1323-1370 (post-fix). Keep this in sync with that block")
- tests/training/test_unfit_pre_pipeline_recovery_i59.py:104-166 (`test_recovery_branch_lives_in_predict_py` — AST-parses predict.py + siblings and asserts the strings `feature_names_in_`, `feature_names_`, "Skipping pre_pipeline" appear; docstring openly says the ast route exists "to avoid the behavioural-tests memory rule")
Evidence: all 4 behavioral tests in the file call `_subset_recovery` (the test-local copy), never `predict_from_models`. The AST test is a source-presence sensor: production could keep the strings but break the logic (wrong branch order, swallowed exception, subset applied to the wrong frame) and every test stays green. Same anti-pattern the sibling report flagged at selector level (shared_lift-23); this is the training-predict instance.
Proposal: replace/augment with a behavioral test. Build a real fitted suite cheaply OR construct the models dict directly (the pattern in tests/inference/test_predict_passthrough_replay.py:_build_minimal_suite): inner LGBMRegressor fitted on cols ["x0","x1"]; attach as `pre_pipeline` a real fitted MRMR whose `transform` is then broken by `monkeypatch.setattr(type(sel), "transform", raise_notfitted)` (or simply `del sel.support_` to fake the iter-59 stale state). Call `predict_from_models(df_with_extra_cols, models, metadata)`; assert (a) the model's predictions are present and `np.allclose` to `inner.predict(df[["x0","x1"]])`, (b) caplog captured the "Skipping pre_pipeline" warning. Delete the `_subset_recovery` mirror once the behavioral test fails-on-prefix is verified (per repo rule: validate the new test fails when the recovery block is commented out). Keep no AST scan.
Effort: S

### [training_integration-03] Engineered-FE-through-the-suite predict path unpinned: selected engineered features (MRMR FE default-on) are asserted at fit time but never predicted on
Severity: P1
Kind: gap-mask|integration
Files:
- tests/training/test_bizvalue_feature_selection.py:340-353 (comment: "the downstream model trains on them. selected_features therefore reports ... engineered features reproducible at predict time via MRMR's stored recipes" — but the file never calls a predict entrypoint)
- tests/inference/* — grep "MRMR|RFECV|BorutaShap" hits only test_predict_passthrough_replay.py (stand-in class, numeric/text passthrough scenario, no FE recipes)
Evidence: when `use_mrmr_fs=True` with FE on (production default `fe_max_steps>=1`), the model's input columns include recipe-generated names (`info_3__sin1`, `add(...)`). Replay of those recipes inside the suite's predict path (`predict_from_models` -> `_apply_pre_pipeline_with_passthrough` -> MRMR.transform recreating engineered columns on NEW rows) has zero integration coverage — the sibling finding gaps_fe_masking-14 covers selector-level replay rigor; nothing covers the TRAINING-suite predict wiring (passthrough re-attach + engineered columns + downstream model width contract together).
Proposal: synthetic where an engineered feature is reliably selected: y = (x1*x2 > 0) binary with x1,x2 ~ N(0,1) (zero marginal, strong pair signal) + 3 noise cols, n=800; train suite with `use_mrmr_fs=True`, mrmr_kwargs enabling binary FE (fe_max_steps=1, fe_ntop_features=3, fe_binary_preset="minimal", seeds fixed); assert metadata["selected_features"] contains at least one non-raw name (engineered); then `predict_from_models` on a held-out 200-row slice: assert all finite, AUROC >= 0.75 (measured floor minus margin — an FE-replay bug that zeroes/garbages the engineered column drops this to ~0.5), and column-width errors absent from caplog. Chain into finding 01's reload cell (the FE-recipe pickle is the highest-risk artifact).
Effort: M

### [training_integration-04] Weight-aware FS (`use_sample_weights_in_fs=True`) has only marker-stamp unit tests — no test that the weights ever reach a selector or change a selection through the suite
Severity: P1
Kind: gap-mask|bizvalue
Files:
- src/mlframe/training/_feature_selection_config.py:79-86 (contract: "MRMR.fit and RFECV.fit receive the suite's sample_weight via fit_params, so the selected features reflect the active weighting ... FS is computed ONCE per target and reused across weight schemas" when False)
- tests/training/test_fs_config_use_sample_weights_in_fs_default_off.py:17-67 (4 tests: config default + `_build_pre_pipelines` marker stamping only)
- tests/feature_selection/test_sample_weights_fs.py:26-113 (4 tests: passthrough forwards/skips weight by marker; cache key folds weight — all helper-level)
- src/mlframe/training/pipeline/_pipeline_helpers.py:684-704 (the actual forwarding into `fit_transform`)
Evidence: every existing test stops at "the marker attribute is True/False" or "the helper forwards kwargs when marker set". Nothing runs `train_mlframe_models_suite` with a sample_weight-producing extractor and the flag ON and asserts (a) the selector's fit actually received the weights, (b) selection responds to weighting, (c) with the flag OFF the FS result is reused across weight schemas (the documented cache invariant). A wiring break anywhere between config -> `_build_pre_pipelines` -> trainer -> `_apply_pre_pipeline_transforms` would pass all current tests. Sibling gaps_selection_masking-05 covers selector-level weight support; the suite-level thread is untested.
Proposal: dataset with two subpopulations (80% rows: y driven by feature A; 20% rows: y driven by feature B), extractor supplying sample_weight = 50.0 on the 20% slice; run suite twice (flag False / True) with `use_mrmr_fs=True`, fixed seeds, simple-mode MRMR. Assert: flag=True selection contains B (weighted signal dominates); flag=False selection contains A; AND (mechanism pin, robust to selection noise) monkeypatch-spy `MRMR.fit` recording `sample_weight is not None` — True only under the flag. For the reuse invariant: two weight schemas in one suite run with flag=False, spy asserts exactly ONE MRMR fit per target.
Effort: M

### [training_integration-05] groups / time-series CV never co-tested with FS through the suite (strict_groups injection + groups forwarding are unit-tested only)
Severity: P1
Kind: gap-mask|integration
Files:
- src/mlframe/training/core/_setup_helpers_pre_pipelines.py:119-123 (fs_use_groups -> `mrmr_kwargs["strict_groups"] = True`, "raises loudly rather than computing group-naive MI")
- src/mlframe/training/pipeline/_pipeline_helpers.py:548,690,702 (`groups` threaded into selector fit — audit row FS-P1-1 fix)
- tests/feature_selection/test_regression_S30_mrmr_strict_groups.py, tests/feature_selection/test_fs_pre_screen_group_seed.py (selector/unit level only — grep confirms no `train_mlframe_models_suite` + groups + FS test anywhere)
- tests/training/conftest.py:222-244 (`sample_timeseries_data` fixture exists; grep shows zero co-occurrence of TimeSeriesSplit/timeseries fixtures with FS config in tests/training)
Evidence: the suite-level chain (split config produces groups -> `fs_use_groups` -> strict_groups default -> groups kwarg reaches `selector.fit`) is covered only in fragments at unit level; no test constructs a grouped or time-ordered suite run with `use_mrmr_fs=True`/`rfecv_models` and asserts either the loud-raise (strict_groups with group-naive estimator) or the grouped path being taken. RFECV's TimeSeriesSplit auto-detect on polars (disposition FS-L-4) is similarly never exercised from the suite.
Proposal: (a) groups: panel data with 20 groups x 50 rows, leaky within-group feature (per-group constant equal to group mean of y) + honest feature; suite run with group-aware split config + `use_mrmr_fs=True`; assert either ValueError with "strict_groups" message (if estimator group-naive) or — with a group-capable config — that the leaky feature is NOT selected while a random-split run selects it (the discriminating recipe: group-naive MI rates the group-constant feature highly; group-aware does not). (b) time series: `sample_timeseries_data`-shaped frame with a 1-step-lagged-target feature (near-leak) + honest features, suite with time-ordered split + `rfecv_models=["cb_rfecv"]`; assert the run completes and RFECV's internal splitter respected order (spy on the splitter class or assert the selector attribute the auto-detect sets).
Effort: M

### [training_integration-06] FS x ensembling/stacking never co-tested deterministically — every FS integration test pins `use_mlframe_ensembles=False`
Severity: P1
Kind: gap-mask|integration
Files:
- tests/training/test_feature_selection.py:336,370,395,432,470,517,545,566,602,633 (`use_mlframe_ensembles=False` in all 10 suite calls)
- tests/training/test_bizvalue_feature_selection.py:177 (same), test_suite_coverage_gaps.py:2172 (same in the MRMR+RFECV stack test), test_integration_prod_like_polars.py:664,727 (same)
- grep: no tests/training file contains both `use_mlframe_ensembles=True` and an FS knob in the same test (verified via per-file co-occurrence grep; only the random fuzz axes can combine them)
Evidence: the ensembling stage consumes predictions from entries trained on DIFFERENT post-FS feature sets (ordinary=all cols, MRMR branch=subset, RFECV branch=other subset). Member-collection, uniformity gating and stack-matrix building across heterogeneous pre_pipelines is exactly the cross-subsystem seam integration tests exist for, and it currently has only stochastic fuzz coverage (a given CI run may never draw the combo).
Proposal: n=600 binary, 5 informative + 10 noise; suite with `mlframe_models=["cb","lgb"]` (iterations=10), `use_ordinary_models=True`, `FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs=<simple fast>)`, `use_mlframe_ensembles=True`. Assert: suite returns ensemble entries (ctx.ensembles / models dict per disposition ENS-P0-5 wiring); ensemble members include at least one FS-branch model (member names carry the "MRMR " prefix); ensemble val metric >= min(member val metrics) - 0.02 (sanity, not a win claim); no exception swallowed (caplog free of ensemble-skip errors). One test, ~30 s, no fast-mode split needed.
Effort: S

### [training_integration-07] Empty-selection (0 features kept) suite contract is pinned by a vacuous assert
Severity: P2
Kind: quality|gap-mask
Files:
- tests/training/test_core.py:2370-2397 (`min_relevance_gain: 10.0  # Very high threshold to ensure no features selected` ... final assert: `assert isinstance(models, dict)` with comment "(may have empty models if no features selected)")
Evidence: the only suite-level empty-selection test (a) never verifies that 0 features were in fact selected (if MRMR still kept features the test silently tests nothing), and (b) asserts a tautology — `train_mlframe_models_suite` returns a dict on every code path. The graceful-degradation contract (branch skipped? model trained on 0 cols? log emitted? metadata stamped?) is undefined and unprotected.
Proposal: keep the forcing config but pin the contract: after the run, read `metadata.get("selected_features_per_model", {})`; assert the MRMR-branch key is absent OR its list is empty (confirming the forcing worked — if neither, fail with "forcing config no longer produces empty selection; rewrite fixture"); then assert the explicit degrade behavior: with `use_ordinary_models=False`, `models` contains no trained entry for the target AND caplog matches a "no features selected|skipping" record (add the log line in prod if missing — fix on the spot per repo rules, do not assert around its absence). Mirror cell for RFECV with `min_features_to_select` floor: assert the floor PREVENTS empty selection (n_features_ >= floor).
Effort: S

### [training_integration-08] Three cannot-fail tests in test_feature_selection.py: `or True` asserts, `except: pass`, and a "training suite" test that never calls the suite
Severity: P2
Kind: quality
Files:
- tests/training/test_feature_selection.py:669-674 (`assert hasattr(selector, "selected_features_") or hasattr(selector, "support_") or True` — tautology; plus `except Exception ... pytest.skip("Constant features raised expected error")` — banned documented-skip shape)
- tests/training/test_feature_selection.py:695-700 (`try: selector.fit(X, y) except Exception: pass` — NaN test literally cannot fail)
- tests/training/test_feature_selection.py:754-800 (class `TestModelCloningInTrainingSuite` "Tests that models are properly cloned in train_mlframe_models_suite" — the test fits two CatBoost clones directly, never calls the suite; final assert at 798: `assert not np.array_equal(pred1, pred2) or True` — tautology)
Evidence: quoted verbatim above. (test_mrmr_different_quantization_methods in the same file is already dispositioned as param_axes-11 — excluded here.)
Proposal: (a) constant-feature: MRMR has a defined contract (disposition FS-P2-2: all-constant cols NOT rejected, MI=0 surfaces) — assert it: fit succeeds AND "constant" not in selected set on a fixture where 2 informative features exist; drop the skip. (b) NaN: MRMR handles NaN natively via nan_strategy (per `_build_pre_pipelines` comment at _setup_helpers_pre_pipelines.py:107-110) — assert fit SUCCEEDS and the NaN-bearing informative column is still selectable (make feature_0 informative: y = 2*feature_0 + noise); no try/except. (c) cloning: either rewrite to run the suite with two pre_pipeline branches and assert `models[...][0].model is not models[...][1].model` (the actual regression it claims to guard), or fold into finding 06's test; delete the tautological assert either way.
Effort: S

### [training_integration-09] FS-report observability tests run against duck-typed stub selectors only, and the end-to-end smoke self-disarms (pytest.skip on any exception + a final assert that passes without MRMR)
Severity: P2
Kind: quality|gap-mask
Files:
- tests/training/test_metadata_feature_selection_report_observability.py:139-230 (local `class MRMR/RFECV/BorutaShap` stubs feed `_build_feature_selection_report` — the real classes never touched)
- tests/training/test_metadata_feature_selection_report_observability.py:292-293 (`except Exception as exc: pytest.skip(f"suite call failed in test environment: {exc!r}")` — banned escape hatch: any suite regression becomes a skip)
- tests/training/test_metadata_feature_selection_report_observability.py:322 (`assert "MRMR" in _kinds or None in _kinds` — since the ordinary branch always stamps selector_name=None, this passes even when the MRMR branch produced no report at all)
Evidence: the report builder dispatches on attribute surfaces (`support_` int-indices for MRMR, `feature_importances_` dict keyed "<nfeatures>_<fold>" for RFECV, `history_x`/`accepted` for BorutaShap — _phase_train_one_target_helpers.py:87-163). If any real selector changes its attribute shape, prod reports silently degrade to all-None while every stub test stays green — precisely the unit-vs-integration drift the mission flags.
Proposal: (a) add one consistency test per real selector: fit real MRMR (use_simple_mode, n=200x6) and real RFECV (cv=2, max_refits=2) on a 2-informative fixture, call `_build_feature_selection_report(pre_pipeline=fitted, ...)`, assert kept_features non-empty, dropped_features non-empty, reason_per_feature covers all input columns, and for RFECV scores is a non-empty dict (these fail the day the attribute surface drifts). (b) e2e: delete the try/except-skip (the suite call must work — if it does not, that IS the failure) and tighten line 322 to `assert "MRMR" in _kinds` (the test trains with use_mrmr_fs=True; an MRMR-branch report is the whole point).
Effort: S

### [training_integration-10] Report fields `friend_graph` / `cluster_aggregate` and the schema-level failure stamp are never asserted anywhere
Severity: P2
Kind: gap-mask
Files:
- src/mlframe/training/core/_phase_train_one_target_helpers.py:167-183 (conditional `_report["friend_graph"]` / `_report["cluster_aggregate"]` keys)
- src/mlframe/training/core/_phase_train_one_target_schema.py:238-248 (on builder exception: warning + six-None fallback record)
- grep: "friend_graph" / "cluster_aggregate" appear in zero assertions under tests/training (only selector-level tests in tests/feature_selection touch the attributes themselves)
Evidence: the observability contract docstring (test file lines 1-13) enumerates selector_name/kept/dropped/hash/scores/reasons but the builder ships two more documented fields plus a failure-path shape; none asserted. A regression that breaks `to_meta()` serialization or the exception fallback would surface only as a prod log warning.
Proposal: (a) MRMR fitted with DCD/cluster aggregation active on a fixture with a duplicated-feature cluster (x, x+eps noise copy): assert `"cluster_aggregate" in report` and the entry names members + combiner (skip-free; if the selector did not build an aggregate, the fixture is wrong — tighten until deterministic). (b) failure path: monkeypatch `_build_feature_selection_report` to raise inside a tiny suite run; assert every model_schemas entry still carries `feature_selection_report` with all six keys None and caplog has the "build failed" warning — pins the never-abort contract at the integration site, not just the helper.
Effort: S

### [training_integration-11] `metadata["selected_features"]` union semantics under mixed ordinary+FS branches are untested, and the per-model key has zero consumers in tests
Severity: P2
Kind: gap-mask|integration
Files:
- src/mlframe/training/core/_phase_finalize.py:391-415,507-509 (union over EVERY entry exposing `.columns`; `selected_features_per_model` keyed `f"{_ttype}/{_tname}/{_mn}"`)
- tests/training/test_bizvalue_feature_selection.py:111-114 (helper probes `metadata['selected_features']` FIRST), 167-181 (`_run_suite` always sets `use_ordinary_models=True`)
- tests/training/test_suite_coverage_gaps.py:2146-2190 (`test_mrmr_and_rfecv_stack_runs` docstring: "the metadata captures something for both selectors" — actual assert: `assert trained` only)
Evidence: with `use_ordinary_models=True` + MRMR in one run, whether the ordinary (all-columns) entry contributes `.columns` to the union determines whether the bizvalue noise-rejection assert is meaningful or accidentally green — and no test pins it. `selected_features_per_model` (the unambiguous surface) is asserted by nothing; the stack test's docstring promise is unbacked.
Proposal: one test, stack run (ordinary + use_mrmr_fs + rfecv_models, n=500, 3 informative + 8 noise): read `metadata["selected_features_per_model"]`; assert (a) keys exist for both the "MRMR " and "cb_rfecv " branches (and the ordinary branch, per whatever the production intent is — pin it explicitly), (b) the MRMR branch list excludes >=6 of the 8 noise cols, (c) document/pin the union: `metadata["selected_features"] == sorted(set().union(*per_model.values()))`. If the ordinary entry turns out to pollute the union (making the union useless as an FS summary), that is a prod fix (exclude `pre_pipeline=None` entries from the union), not a test workaround — flag and fix per repo rules.
Effort: S

### [training_integration-12] Suite-level determinism with FS is untested — the metamorphic battery explicitly skips every `use_mrmr_fs` combo
Severity: P2
Kind: gap-mask|param
Files:
- tests/training/test_fuzz_metamorphic.py:257-258 (`if c.use_mrmr_fs: continue` in `_curated_metamorphic_combos`, justified only by runtime: "Prefer simple combos (no OD, no PCA, no MRMR) so the dual-run stays under ~30s")
- src/mlframe/training/core/_setup_helpers_pre_pipelines.py:89-97,114-118 (fs_random_seed defaulting for RFECV/MRMR — "the whole pipeline (split + FS + model) is reproducible from one seed"; covered only by unit tests in tests/feature_selection/test_fs_pre_screen_group_seed.py)
Evidence: grep finds no test that runs `train_mlframe_models_suite` twice with the same seed and FS enabled and compares the selected sets or metrics. The seed-defaulting code exists specifically to make the FS-enabled suite reproducible from one seed; the claim has no end-to-end sensor, and the one battery designed for exactly this class of check excludes FS by curation.
Proposal: dedicated test (cheaper than a fuzz dual-run): n=500x12, fixed split seed, `use_mrmr_fs=True` with simple-mode MRMR and NO explicit mrmr seed (so the defaulting path is the thing under test); run the suite twice into separate tmp dirs; assert `metadata1["selected_features_per_model"] == metadata2["selected_features_per_model"]` and the FS-branch model's val metric identical to 1e-12. Additionally add ONE curated FS combo to `_curated_metamorphic_combos` (simple-mode MRMR keeps the 30 s budget), so the metamorphic perturbation contracts also cover FS.
Effort: S

### [training_integration-13] Categorical-survival-through-FS is promised by docstrings but asserted nowhere; pandas-categorical x suite-FS has zero deterministic coverage
Severity: P2
Kind: gap-mask|integration
Files:
- tests/training/test_integration_prod_like_polars.py:639-677 (`test_polars_enum_with_mrmr_feature_selection` docstring: "Polars+Enum cats must survive the MRMR.transform call ... and reach the tree model in a trainable state" — sole assert: `assert trained`)
- tests/training/test_integration_prod_like_polars.py:686-746 (kitchen-sink: same `assert trained` + target-type presence)
- tests/training/test_feature_selection.py:322 ("Remove categorical for simplicity"), :419 ("Use only numeric features") — the dedicated FS integration file deliberately strips cats from both suite-level MRMR tests
- tests/training/conftest.py:266-292 (`sample_categorical_data` fixture exists with a cat-driven signal `(cat_1 == "cat_A_10") * 5` — never combined with FS config, per co-occurrence grep)
Evidence: if MRMR (or the passthrough wiring) silently dropped every categorical column, all current suite-level tests stay green. The fixture that makes a cat column the dominant signal already exists and is unused for FS.
Proposal: parametrize (pandas-category, polars-Enum): use `sample_categorical_data` (cat_1 carries a 5-sigma signal); suite with cb (native cats) + `use_mrmr_fs=True` (cat handling on); assert `"cat_1"` (or a cat-derived engineered/encoded name matching `re.compile(r"(?<![A-Za-z0-9])cat_1(?![A-Za-z0-9])")`) appears in `metadata["selected_features_per_model"]` for the MRMR branch, and the branch model's metric beats a numeric-only run by a measured margin (floor ~0.05 R2, set 10% below measured). This converts "trainable state" from no-crash into the documented survival contract.
Effort: M

### [training_integration-14] BorutaShap has no deterministic suite-level integration test (unit marker tests + random fuzz axis only); regression-target derivation untested end-to-end
Severity: P2
Kind: gap-mask|integration
Files:
- tests/training/test_use_boruta_shap_config_default_off.py:34-82 (3 tests, all on `_build_pre_pipelines` directly)
- tests/training/_fuzz_combo/axes.py:423-432 (`use_boruta_shap_cfg`: (False, True) — stochastic draw only)
- src/mlframe/training/core/_setup_helpers_pre_pipelines.py:148-153 (target_type -> `classification` derivation: "raises ValueError('Unknown label type: continuous') inside sklearn.multiclass on regression targets" pre-fix)
Evidence: no test anywhere calls `train_mlframe_models_suite(use_boruta_shap=True)` deterministically; whether a given CI run exercises the branch depends on the fuzz draw. The regression-target auto-derivation (the very bug the code comments document) has no pin: a refactor reintroducing classification=True-on-regression would only fail in fuzz, eventually.
Proposal: `pytest.importorskip("shap")`; two cells via parametrize (binary, regression): n=300x6 (2 informative), `use_boruta_shap=True, boruta_shap_kwargs={"n_trials": 20}`, cb model iterations=10, `use_ordinary_models=False`. Assert: trained dict non-empty; the entry's `feature_selection_report["selector_name"] == "BorutaShap"`; for the regression cell specifically assert no "Unknown label type" in caplog and the run completes (the derivation pin). Mark `@pytest.mark.slow` and provide fast mode via `fast_subset(["binary","regression"], representative="regression")` (regression is the derivation-bearing cell).
Effort: M

### [training_integration-15] Early-stopping x FS interaction asserted only as `hasattr(n_features_in_)` — no behavioral pin that ES reaches the RFECV inner fits
Severity: LOW
Kind: gap-mask
Files:
- tests/training/test_feature_selection.py:282-305 (`test_rfecv_with_early_stopping` — passes `early_stopping_rounds=5` and asserts only `hasattr(selector, "n_features_in_")`, which is true after any fit)
- tests/feature_selection/test_regression_w5_p2low.py:106 (`test_w5_fs_f16_deepcopy_splitter_in_early_stopping_path` — unit-level splitter handling only)
Evidence: if early_stopping_rounds were silently dropped on its way into the inner CB fits (a real bug class: kwargs filtered by a wrapper), the current test cannot notice.
Proposal: behavioral delta at the selector level (cheap, no suite needed): same data/seed, RFECV(CatBoostRegressor(iterations=200), early_stopping_rounds=2, cv=2, max_refits=2) vs early_stopping_rounds=None; capture the fitted inner estimators' `best_iteration_` (RFECV stores fold estimators / expose via spy on CatBoostRegressor.fit kwargs); assert the ES run received `early_stopping_rounds` in fit kwargs and stopped before 200 trees while the None run did not. The kwargs-spy form is robust to selection noise.
Effort: S

### [training_integration-16] test_feature_selection.py module hygiene: dead docstring (import precedes it), obfuscated `__import__` RNG, legacy global `np.random.seed` in 7 tests
Severity: LOW
Kind: quality
Files:
- tests/training/test_feature_selection.py:1-7 (line 1 is `from mlframe.training import ...`; the triple-quoted block at lines 3-7 is therefore a no-op string expression, not the module docstring)
- tests/training/test_feature_selection.py:24 (`_W53_RNG = __import__('numpy').random.default_rng(0)` — numpy is imported normally at line 10)
- tests/training/test_feature_selection.py:493,580,613,651,678,704,723 (`np.random.seed(42)` global-state seeding; same class as test_code_quality-13, which scoped only tests/feature_selection — these training-side sites were outside that finding's file list)
Evidence: quoted; verified line numbers this session.
Proposal: move the docstring above the import; replace `__import__('numpy')` with the existing `np` alias; convert the seven `np.random.seed(42)` + `np.random.randn/randint` blocks to a local `rng = np.random.default_rng(42)`. Pure hygiene, no behavior change; do it in one pass per the grep-all-instances rule.
Effort: S

### [training_integration-17] Bizvalue Test 2 runs the baseline suite TWICE per seed (separate timing-only and models-only runs) and has no fast mode — 12 suite runs at iterations=80 every CI pass
Severity: LOW
Kind: quality
Files:
- tests/training/test_bizvalue_feature_selection.py:268-269 (`_, _, t_baseline = _run_suite(df, tmp_path / "A", use_mrmr=False, iters=80)` immediately followed by `models_a, _, _ = _run_suite(df, tmp_path / "A2", use_mrmr=False, iters=80)` — identical config, two runs, one consumed only for `t_baseline`, the other only for AUROC)
- tests/training/test_bizvalue_feature_selection.py:263 (3-seed parametrize; no slow marker, no fast_subset — file-level grep confirms)
Evidence: `_run_suite` already returns `(models, metadata, elapsed)` — a single call provides both the timing and the models; the duplicate halves nothing and doubles ~25% of the file's wall time. (The weakness of the test's speed ASSERTION is already dispositioned as bizvalue_value_proofs-05; this finding is the orthogonal cost/structure issue.)
Proposal: collapse to one baseline run per seed (`models_a, _, t_baseline = _run_suite(df, tmp_path / "A", use_mrmr=False, iters=80)`); wrap the seed list with the repo-standard `fast_subset([42, 7, 99], representative=42)` so default runs execute one seed and the full sweep stays available under the env flag.
Effort: S

### [training_integration-18] ShortlistTransformerAdapter suite test: module-level `except Exception: skip` can hide a training-package import regression; needs_y=True (target-aware, leakage-relevant) path never goes through the suite
Severity: LOW
Kind: gap-mask|quality
Files:
- tests/training/test_shortlist_transformer_adapter_suite.py:14-20 (`try: from mlframe.training... except Exception as exc: pytest.skip(f"suite not importable ({exc!r})", allow_module_level=True)` — broader than importorskip: ANY import-time bug in mlframe.training silently skips both tests)
- tests/training/test_shortlist_transformer_adapter_suite.py (2 tests total; both use `needs_y=False` + compute_rff_features; grep confirms no needs_y=True suite usage anywhere)
Evidence: the adapter's needs_y branch is the one with leakage semantics (target passed into the compute function on the train fold only); it has unit coverage in the transformer package but the opt-in suite path — the documented purpose of `_suite_adapter.py` (memory: "Opt-in ShortlistTransformerAdapter ... lets you wire one via FeatureSelectionConfig(custom_pre_pipelines=...)") — is only exercised target-free. The module-level catch-all skip violates the no-documented-skips rule for first-party imports.
Proposal: (a) replace the try/except with plain imports (mlframe.training is first-party; a broken import must FAIL the file). (b) add `test_adapter_needs_y_runs_through_suite`: adapter over a target-aware shortlist function (e.g. compute_local_lift) with needs_y=True via custom_pre_pipelines; assert trained + the adapter's transform on the held-out val produced finite added columns (probe entry.columns for the adapter-generated names) + permuted-y control: same run with shuffled target yields no val-AUC lift from the added features (leakage gate, floor delta <= 0.02). (c) chain one predict_from_models call so the adapter's transform-at-predict is covered.
Effort: M

---

## Summary

| Severity | Count |
|---|---|
| P0 | 1 |
| P1 | 5 |
| P2 | 8 |
| LOW | 4 |
| **Total** | **18** |

Verdict: suite-level FS testing is broad on "does it crash" (pandas/polars, MRMR/RFECV/custom, multi-target,
kitchen-sink) but thin on contracts: the single most important integration chain — train-with-FS -> save -> load ->
predict parity — is untested anywhere despite being the documented origin of a real prod bug family (iter-59), and the
predict-side recovery for exactly that family is pinned only by a test-local re-implementation plus an AST string scan.
Weight-aware FS, groups/TS-CV threading, FS x ensembling, determinism, and empty-selection are each covered only at the
marker/unit level or by tautological asserts, so a wiring break between config and selector would pass today's suite;
the proposals above are mostly S/M-effort extensions of harnesses that already exist (round-trip parity file,
selected_features_per_model metadata, fast_subset).
