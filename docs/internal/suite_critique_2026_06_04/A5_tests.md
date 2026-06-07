# A5 — Test-suite critique (train_mlframe_models_suite + subsystems)

READ-ONLY review, 2026-06-04. Scope: `tests/training/`, `tests/feature_selection/`,
`tests/feature_engineering/`, `tests/calibration/`, `tests/evaluation/`, plus the
`tests/inference/` predict-path tests that gate suite output. 1336 test files total.

## Executive summary

The suite is, overall, **mature and disciplined**. Strong points verified directly:
- Deep end-to-end coverage: **364 live `train_mlframe_models_suite(...)` calls across 70 files**
  (`test_core.py` alone has 105). This is not a shallow-mock suite.
- A **7790-line pairwise fuzz-combo enumerator** (`_fuzz_combo.py`) whose axis space already
  covers binary / regression / multiclass / multilabel / LTR / multi-target-regression, polars
  utf8/enum/nullable carriers, numpy-vs-native target carrier, OD, imbalance, drift injection,
  degenerate cols, PySR, RFECV, and ~70 config knobs. The anti-masking contract (no
  canon/xfail/skip to silence prod bugs) is documented at the top of the file and in CLAUDE.md.
- Excellent **regression-sensor discipline** for known-fixed bugs (OOF-vs-in-sample divergence,
  composite-discovery target-leak into caller frame, S-numbered sensors).
- **Meta-tests already enforce** several anti-patterns: `test_no_inspect_getsource.py`,
  `test_no_unsafe_module_reload.py` (snapshot/restore for `sys.modules`/`reload`),
  marker registration, no-bare-except, no-mutable-defaults.
- Calibration/bootstrap biz_value tests are **quantitative** and exemplary
  (`tests/evaluation/test_bootstrap.py`, `test_bizvalue_calibration_ensemble.py`).

The findings below are the actionable deltas — mostly **P1/P2 coverage gaps and quality
tightening**, not P0 correctness holes. The single most systemic issue is the large
**source-text-read assertion** footprint (the `inspect.getsource` ban was satisfied by
switching to `Path.read_text() + "x" in src`, which is the same brittle source-inspection
anti-pattern under a different name).

---

## Findings

### A5-01 — `test_target_type_combinations.py` asserts nothing about the entrypoint it claims to cover
- **Severity:** P1
- **Category:** weak-assertion / coverage-gap
- **File(s):** `tests/training/test_target_type_combinations.py`
- **What/Why:** The module docstring says "Parametrized smoke tests over (task_type, target_dtype)
  combos that **the training entrypoint must accept without raising**." But none of the three
  parametrized tests ever call `train_mlframe_models_suite` (or any mlframe code). They only assert
  on the locally-built synthetic frame: `assert x.height == len(y)`, `assert s.dtype.is_numeric()`,
  `assert len(uniq) == 2`. The tests would pass even if the suite rejected every target type. This
  is a hollow test masquerading as target-type coverage (binary_float / multiclass_str combos in
  particular are never exercised against the suite).
- **Proposed change:** Make each combo actually call `train_mlframe_models_suite` (or at minimum
  `select_target` / target-type detection) with a tiny iteration budget and assert it returns a
  models dict + metadata without raising, plus a behavioral check (e.g. n_classes inferred
  correctly for `multiclass_str`, regression head for float target). Reuse `cpu_only_hyperparams`
  + `fast_iterations` fixtures so it stays <5s.
- **Confidence:** High (read the full file; zero suite calls present).

### A5-02 — `predict_from_models` round-trip parity is regression-only AND self-skips on Windows
- **Severity:** P1
- **Category:** coverage-gap / anti-pattern
- **File(s):** `tests/inference/test_predict_round_trip_parity.py`
- **What/Why:** This is the canonical in-memory-vs-disk predict parity gate (the comment lists the
  exact bug classes it guards: dropped extensions stack, ignored ensemble flavour, missing
  CT_ENSEMBLE persistence). Two problems: (1) **It only covers a regression target** (`y` float).
  No multiclass or multilabel parity. (2) The disk-side assertion has **two `pytest.skip` paths**
  (lines 160, 164) gated on "known Windows zstd quirk (B2#20/#45)". On Windows — the dev/CI
  platform per env — the disk save can silently produce no metadata/dump file and the test skips,
  so the **disk round-trip is effectively never enforced where it runs**. The in-memory sibling
  test does NOT exercise save/load, so a real save/load regression on Windows would go uncaught.
- **Proposed change:** (a) Fix or work around the zstd write so the disk path is exercised
  (the test already ships a single-threaded `_save_threads_zero` shim — make the disk assertion
  unconditional once the shim is in effect, and `pytest.fail` rather than `skip` if the file is
  missing after the shimmed write). (b) Add multiclass and multilabel variants of
  `test_in_memory_and_disk_predict_agree_*` asserting per-class probability parity.
- **Confidence:** High (read full file; both skip branches verified).

### A5-03 — No `predict_from_models` parity test for multiclass / multilabel suite output
- **Severity:** P1
- **Category:** coverage-gap
- **File(s):** `tests/inference/` (whole dir); only `test_predict_from_models_lgb_cat_dtype.py`
  mentions multiclass/multilabel at all
- **What/Why:** `predict_from_models` / `predict_mlframe_models_suite` is a P0 production path
  (it replays the fitted pipeline + extensions + chosen ensemble flavour + per-class probability
  combine). The inference tests overwhelmingly use regression `y` or binary. The only multilabel
  save/load coverage (`tests/training/test_save_load_multi_output.py`) round-trips the raw
  `_ChainEnsemble` / `MultiOutputClassifier` estimator via `joblib`, NOT the full-suite
  `predict_from_models` replay — so the `_combine_probs` / argmax / per-class shape canonicalisation
  in the predict path is untested for K>2 and multilabel end-to-end.
- **Proposed change:** Add `tests/inference/test_predict_multiclass_multilabel_parity.py`: train a
  3-class and a 3-label suite (lgb+xgb), call `predict_from_models` twice + after load, assert
  `ensemble_probabilities` shape is `(n, K)` / per-label `(n, 2)` and that probabilities sum to 1
  per row (multiclass) and round-trip to float precision.
- **Confidence:** High.

### A5-04 — Source-text-read assertions are a systemic side-channel (the getsource ban was renamed, not fixed)
- **Severity:** P2
- **Category:** anti-pattern
- **File(s):** 147 files use `.read_text(...)` (272 sites); 73 files assert `"..." in src`
  (435 sites). Clear behavioral-proxy offenders, e.g.
  `tests/training/test_regression_w4c_perf_p2low.py:138` (`assert "if verbose:" in src`),
  `test_regression_w2a_bridge_zerocopy.py:172`, `test_tvt_round5_4_followups.py` (15 string-presence
  asserts), `test_default_via_or_trap.py` (11), `test_future_disposition_codeeff_conv.py` (11).
- **What/Why:** `test_no_inspect_getsource.py` forbids `inspect.getsource`, and the codebase
  migrated those tests to `Path(mod.__file__).read_text() + "<token>" in src`. That is the *same*
  anti-pattern the rule (`feedback_behavioral_tests`) exists to prevent — it asserts an
  implementation string, not behavior. It is brittle (renaming a local var, reformatting a guard,
  or moving the line breaks the test without any behavioral regression) and gives false confidence
  (the guard could be present in source but bypassed at runtime). NOTE: many `read_text` uses are
  *legitimate* (LOC-budget checks in monolith-split sensors like
  `test_monolith_split_w10e_helpers.py:28`, CHANGELOG cross-walks) — the issue is specifically the
  behavioral-proxy `"token" in src` subset.
- **Proposed change:** Triage the 73 `"x" in src` files. For each behavioral-proxy assertion,
  replace with a runtime check (call the function with `verbose=False`, spy/monkeypatch the
  `null_count()` call site and assert it isn't invoked; assert the actual transform output rather
  than that a code line exists). Keep `read_text` only for LOC budgets, doc/CHANGELOG cross-walks,
  and meta-linters. Extend `test_no_inspect_getsource.py` to also flag `"<token>" in <src_var>`
  where `<src_var>` came from a module `.read_text()` (with an allowlist for LOC/meta uses).
- **Confidence:** High on the pattern's prevalence; Medium on per-file conversion difficulty.

### A5-05 — `test_coverage_fill.py` (FE) has genuinely weak `is not None` / `hasattr` assertions
- **Severity:** P2
- **Category:** weak-assertion / biz-value-gap
- **File(s):** `tests/feature_engineering/test_coverage_fill.py` (18 `is not None` sites:
  `:265 assert res is not None and res.height > 0`, `:276`, `:293 assert fig is not None`,
  `:909`, `:955`, `:971`, `:1007 assert model is not None and hasattr(model, "equations_")`,
  `:1439`)
- **What/Why:** The file is self-described as coverage-filler ("touch a branch with the cheapest
  input that lights it up, not exhaustive correctness"). For `compute_mps_targets`, `run_pysr_fe`,
  and the chart builders the assertion is "didn't return None / has the right height" — it catches
  a crash but not a wrong result. PySR FE in particular is a headline ML trick; `run_pysr_fe`
  returning a model with `equations_` present is not a win-assertion. (The dedicated PySR
  biz_value at `test_biz_val_pysr_fe_upgrade.py` covers the *suite-level* win, so this is a
  quality-tightening not a true biz_value gap for PySR.)
- **Proposed change:** For the MPS/PySR/chart fillers, add at least one behavioral assertion per
  function (e.g. `compute_mps_targets` output column values match a hand-computed SMA on a
  3-row fixture; `run_pysr_fe` recovers a known `x0*x1` equation form on a noiseless synthetic).
  Leave the pure numeric-kernel tests (`welford_mean_var_seq`) as-is — those already assert
  `np.testing.assert_allclose`.
- **Confidence:** High.

### A5-06 — Outlier-detection / imbalance biz_value tests downgrade to `pytest.xfail` on miss instead of asserting the win
- **Severity:** P2
- **Category:** biz-value-gap / anti-pattern
- **File(s):** `tests/training/test_bizvalue_outliers_earlystop.py:229-239,323-326`,
  `tests/training/test_bizvalue_imbalance_grid.py:242`
- **What/Why:** Per CLAUDE.md, a biz_value test must "QUANTITATIVELY assert the win… by FAILING the
  win". These files instead call `pytest.xfail("...RMSE-lift below 10%...")` when the lift is not
  observed — converting a quality regression into a non-failing xfail. The file even self-documents
  "(xfailing) RMSE-lift assertion" and "Mark as xfail when below 10%". That means a silent
  regression in outlier handling / imbalance grid would xfail-pass, not fail CI. (Note: `xfail`
  here is conditional/runtime, distinct from the legitimate strict-xfail uses elsewhere.) This
  contradicts the project rule "Fuzz/biz_value tests are bug DETECTORS — never masked with xfail".
- **Proposed change:** Re-tune the synthetic so the win is stable across seeds (the calibration
  biz_value test is the model: large n, structured miscalibration), then assert a hard floor with a
  5-15% margin. If the trick genuinely cannot be made to win reliably on synthetic data, that is
  itself a finding about the trick — document it, don't xfail-mask it.
- **Confidence:** High (verified the runtime `pytest.xfail` calls).

### A5-07 — `assert x is not None` cluster: most are fine, but verify the FS/FE long tail
- **Severity:** Low
- **Category:** weak-assertion
- **File(s):** 123 `assert … is not None` sites across 60 files; mrmr-layer files
  (`test_biz_value_mrmr_layer47.py`, etc.) and `test_biz_val_mrmr_dcd_wave9.py` (4)
- **What/Why:** Spot-checked `test_biz_value_mrmr_layer47.py`: the `assert m.dcd_ is not None`
  lines are immediately followed by structural assertions (`"tau_calibration" in m.dcd_`,
  `0.30 <= tau <= 0.75`), so they are guards before a stronger check, not the whole test. The
  cluster is mostly benign. The exceptions worth a pass are the FE coverage-fillers (A5-05) and any
  mrmr-layer test where `is not None` is the *only* assertion in the function body.
- **Proposed change:** Low-priority sweep: grep each `is not None`-only test body; where it is the
  sole assertion, add a behavioral check. Do not bulk-rewrite the guard-before-check uses.
- **Confidence:** Medium (sampled, not exhaustive).

### A5-08 — Fast-mode (`@pytest.mark.fast` + `fast_subset`) adoption is thin relative to suite size
- **Severity:** P2
- **Category:** speed
- **File(s):** conftest `fast_subset`/`is_fast_mode` (`tests/conftest.py:148-177`); marker defined
  `pyproject.toml:309`; `@pytest.mark.fast` used in only a handful of files
  (`evaluation/test_bootstrap.py`, `calibration/test_quality.py`, scattered)
- **What/Why:** The marker docstring promises "Every new ML feature ships with a representative
  subset marked @pytest.mark.fast so the full code path runs in <15s total." In practice the
  `fast` marker is sparsely applied across the 70 suite-calling files, and `fast_subset` is used in
  very few parametrize lists. The 364 full-suite calls means `--fast` does not meaningfully shrink
  the heavy training tests (only `slow`/`slow_only` get deselected). There is no quick
  representative-path smoke run for the training subsystem.
- **Proposed change:** Audit the 70 suite-calling files; mark one representative test per file
  `@pytest.mark.fast`, and route the heaviest parametrize lists (model matrix in `test_all_models.py`,
  target-type sweeps) through `fast_subset`. Add a CI smoke job `pytest -m fast` with a wall budget.
- **Confidence:** Medium (marker scarcity confirmed via grep; exact <15s target unverified).

### A5-09 — Sleep-based timing tests for cache/LRU/mtime ordering
- **Severity:** P2
- **Category:** speed / flakiness
- **File(s):** `tests/training/test_phases_registry.py:115,129,141,142` (`time.sleep(0.005–0.02)`),
  `test_feature_handling_medium_feature_handling.py:179` (`time.sleep(0.005)` "ensure distinct
  mtimes"), `test_suite_artefact_cache.py:133`, `test_setup_timing_2026_05_30.py:53`
  (`time.sleep(2.05)`), `tests/utils/test_disk_cache.py:232` (`sleep(1.05)`)
- **What/Why:** Wall-clock sleeps to coax distinct mtimes / LRU ordering are flaky on
  coarse-precision filesystems (FAT/some network FS show 1-2s mtime granularity) and slow. The
  codebase already knows the better pattern — `test_cache_lru_eviction.py` injects a fake clock
  (`fake_time_mod = types.SimpleNamespace(time=_tick, ...)`, see its header note), and
  `test_preprocessing.py:472` deliberately avoids a flaky `sleep(0.01)`. `test_setup_timing`'s
  `sleep(2.05)` is a ~2s-per-test tax.
- **Proposed change:** Replace mtime-ordering sleeps with explicit `os.utime(path, (t, t))` to set
  distinct mtimes deterministically, and replace timer-based sleeps with the fake-clock injection
  pattern already used in `test_cache_lru_eviction.py`. The concurrency sleeps in
  `test_audit_concurrent_exception_swallow.py` (genuine thread interleaving) can stay.
- **Confidence:** High.

### A5-10 — Remaining unsafe-ish `sys.modules` / reload sites rely on a heuristic guard
- **Severity:** P2
- **Category:** anti-pattern (pollution risk)
- **File(s):** `tests/training/test_automl.py` (5× `importlib.reload(automl_module)`),
  `tests/training/test_phase_temporal_audit_polars_fastpath.py:178` (`importlib.reload(_mod_ref)`),
  `tests/training/test_lazy_mrmr_import_l5.py` (`del sys.modules` + `sys.modules.pop` ×4),
  `tests/feature_engineering/test_histogram_fallback_l55.py:44` (`importlib.reload(mod)`),
  `tests/feature_selection/test_biz_value_mrmr_layer31.py` / `layer39.py` (`del sys.modules`)
- **What/Why:** `test_no_unsafe_module_reload.py` passes these because it detects a restore *marker*
  via **string matching** (e.g. presence of `"_mod_ref.__dict__.update("` or `"subprocess.run("`
  anywhere in the file). That is a coarse heuristic: a file can contain the marker string in an
  unrelated place, or reload a module whose top-level singletons (caches/registries/locks) are not
  restored by a `__dict__` swap. `test_automl.py` reloads the automl module under a stubbed
  `autogluon` — low risk because it's a leaf, but `test_phase_temporal_audit_polars_fastpath.py`
  reloads an mlframe internal. The canonical 2026-05-22 MRMR fit-cache incident is exactly this
  class.
- **Proposed change:** Harden `test_no_unsafe_module_reload.py` to scope the restore-marker check to
  the same function/fixture as the reload (AST, not whole-file string match), and to flag reloads of
  `mlframe.*` modules that own module-level mutable singletons. Migrate the
  `test_phase_temporal_audit_polars_fastpath.py` reload to subprocess isolation or env-var/monkeypatch.
- **Confidence:** Medium (the guard's string-match weakness is verified in the meta-test source;
  per-site pollution risk is module-dependent).

### A5-11 — Multi-target-regression (K-independent-target) lacks a dedicated biz_value win test
- **Severity:** P2
- **Category:** biz-value-gap / coverage-gap
- **File(s):** `tests/training/test_multi_target_regression_*.py` (smoke + dispatch),
  `test_mtr_nnls_ensemble.py`, `test_extended_multi_target_coverage.py`
- **What/Why:** MTR is a recent estimator code path (num_classes=K head sharing a trunk, MSE on
  (N,K)) and is wired as a fuzz axis (`_fuzz_combo.py:133 "multi_target_regression"`). The existing
  tests are smoke (`_smoke`), dispatch-construction (`_dispatch`), and NNLS-ensemble mechanics —
  none assert the quantitative win the CLAUDE.md rule requires (e.g. shared-trunk MTR beats K
  independent single-target models, or NNLS ensemble beats mean, on a synthetic with correlated
  targets where the trunk-sharing should pay off).
- **Proposed change:** Add `test_biz_val_training_mtr.py`: synthetic with K=3 correlated targets
  sharing latent structure; assert mean test RMSE of the MTR head ≤ K-independent baseline − margin,
  and NNLS-ensembled RMSE ≤ best-single-member − margin. Keep n≤2000, iterations≤10.
- **Confidence:** Medium (verified the MTR tests are smoke/dispatch via filenames + the fuzz axis
  comment; did not read every MTR test body).

### A5-12 — Massive mrmr "layerNN" file proliferation (≈95 files) — structure / discoverability
- **Severity:** Low
- **Category:** structure
- **File(s):** `tests/feature_selection/test_biz_value_mrmr_layer6.py` … `layer104.py`
  (≈95 files), plus ~30 other `test_biz_value_mrmr_*.py`
- **What/Why:** The MRMR biz_value coverage is split across ~95 sequentially-numbered "layer"
  files. Each header reads "NEVER xfail … real LogReg AUC numbers" (good — they are quantitative).
  But the per-layer numbering encodes audit-wave history (the exact junk CLAUDE.md bans inside
  *source comments*; here it's in filenames + module docstrings). Discoverability is poor (which
  layer tests which parameter?), and the per-file fixed import/prewarm cost is paid ~95×, inflating
  collection time. The CLAUDE.md convention is per-CLASS files (`test_biz_val_filters_mrmr.py`),
  which already exists alongside the layer sprawl.
- **Proposed change:** Do NOT delete (they pin real numbers). Instead: (a) add a one-line index
  (module docstring or a `LAYER_INDEX.md`) mapping layerNN → the parameter/contract it pins, so the
  set is navigable; (b) over time, fold parameter-scoped layer tests into the canonical
  per-class `test_biz_val_filters_mrmr.py` as named functions, retiring the standalone file once its
  contract is absorbed (verify the number is preserved). Treat as slow cleanup, not urgent.
- **Confidence:** Medium (file count from listing; sampled headers confirm they are quantitative).

### A5-13 — `addopts = "-x"` globally aborts the run on first failure (CI signal vs full picture)
- **Severity:** Low
- **Category:** speed / structure
- **File(s):** `pyproject.toml:298` (`addopts = "-ra -x -s --no-cov --strict-markers --timeout=60 …"`)
- **What/Why:** `-x` (stop on first failure) is baked into the default `addopts`. For a 1336-file
  suite this means one early failure hides the full failure set, forcing iterative
  fix-rerun-fix-rerun cycles and obscuring whether a change broke 1 test or 200. It pairs oddly with
  the `--instafail` plugin (which exists precisely to surface failures immediately *without*
  stopping). The project memory rule "Fix ALL test failures in one pass" is harder to honor when the
  default config hides all but the first.
- **Proposed change:** Drop `-x` from the persisted `addopts`; rely on `--instafail` for fast
  feedback and let developers pass `-x` ad hoc. Keep `--timeout=60`, `--no-cov`, `-s`,
  `--strict-markers`, `--strict-config` (all good and aligned with the Windows/CI rules).
- **Confidence:** Medium (config verified; the tradeoff is a judgment call — `-x` is defensible for
  a local dev loop, less so as the committed default).

### A5-14 — Single-target-only assertions in the suite-level smoke tests miss per-target plumbing
- **Severity:** Low
- **Category:** coverage-gap
- **File(s):** session fixtures `trained_suite_regression` / `trained_suite_binary`
  (`tests/training/conftest.py:435-493`)
- **What/Why:** The two shared heavy fixtures both train a *single* target. Multi-target /
  multilabel suite paths (per-target metric dicts, per-target ensemble flavour stamping, per-target
  composite discovery) have no equally-cheap shared fixture, so tests that want to inspect those
  re-train ad hoc or skip them. The `_session_fixture_immutability_sensor` (good pattern) only
  guards the single-target fixtures.
- **Proposed change:** Add a session-scoped `trained_suite_multi_target` fixture (2–3 regression
  targets, tiny budget) so per-target inspection tests can consume it without re-fitting, and so the
  immutability sensor covers the multi-target metric_dict too.
- **Confidence:** Medium.

---

## Things that are GOOD (do not "fix")

- `tests/evaluation/test_bootstrap.py` — textbook quantitative assertions (CI brackets true AUC;
  bit-identical reproducibility; DeLong p-thresholds; degenerate→NaN not crash). Keep as the model.
- `tests/training/test_oof_stacking_replaces_in_sample.py` — asserts OOF preds *materially diverge*
  from in-sample on a memorizing tree (real OOF discipline check), not just shape.
- `tests/training/test_regression_S04_composite_discovery_no_leak.py` — pins the exact
  copy(deep=False)+setitem leak path on the production helper. Strong regression sensor.
- `_fuzz_combo.py` / `test_fuzz_suite.py` — anti-masking contract is explicit and the axis space is
  genuinely broad (all target types incl. LTR + MTR, all polars carriers). This is the suite's
  crown jewel; protect it.
- conftest hygiene — per-xdist-worker CatBoost train_dir, fake-clock LRU test, session-fixture
  immutability sensor, pytest-randomly seed-overflow shim, heavy-marker-gated memory cleanup.
- Meta-tests (`test_no_inspect_getsource`, `test_no_unsafe_module_reload`, marker registration) —
  the right idea; A5-04 and A5-10 are about *tightening* them, not adding them.

---

## Suggested priority order

1. **A5-01** (hollow target-type test — actively misleading) and **A5-02/A5-03** (predict-path
   parity gaps on the P0 inference path, regression-only + Windows self-skip).
2. **A5-06** (xfail-masked biz_value wins) and **A5-11** (MTR biz_value gap) — quality-regression
   blind spots.
3. **A5-04** (source-text-read side-channel) — large but mechanical; do as a tracked sweep.
4. **A5-08 / A5-09 / A5-13** (speed/flakiness/config) — cheap, high quality-of-life.
5. **A5-05 / A5-07 / A5-10 / A5-12 / A5-14** — lower-severity tightening / structure.
