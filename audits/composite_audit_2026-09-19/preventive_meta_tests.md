# Composite targets audit 2026-09-19: preventive meta-tests and contract checks

**Input**: the six direction reports in this directory. They hold 138 findings: `transforms.md` (TRF-01..26), `discovery.md` (DSC-01..30), `estimator_ensemble.md` (EST-01..22), `suite_integration.md` (INT-01..19), `performance.md` (PRF-01..24) and `tests.md` (TST-01..17).

**Checkout**: `.claude/worktrees/agent-ab1823c081b446d9e` (origin/master a950f7e47). I read the code and changed nothing. The existing infrastructure these proposals reuse:
- `tests/test_meta/`: `_scan_guard.assert_scanned_enough`, `_shared_ast_cache.parsed_ast`, the `_*_baseline.json` ratchets with `regen_baselines.py` and `BASELINES_README.md`, the known-violator ratchet idiom of `test_scanning_gates_fail_closed.py` (`UNGUARDED`), and `test_shared_checks_wired.py` as the single place where py-ci-shared checks are consumed.
- `py_ci_shared`: `baseline_ratchet.Baseline`, `gate_population_canary` (`_CANARY`, `_candidate_files`), `content_hash_version_bump_gate`, `effect_assertion_parity.build_import_map`, `uncalled_functions`, `source_text_claims`, `vacuous_loop_assertions`, `config_call_site_parity`, `mutation_teeth.assert_revert_fails_tests` and `teeth_sweep`.
- pyutilz: `pyutilz.dev.code_audit` with its `non_neutral_except_fallback` / `default_via_or` rules, which `test_code_audit_baseline.py` consumes through `py_ci_shared.code_audit_meta`.

**Method**: I sorted each finding into one primary root-cause class; the table below names the class. For each class I then asked what would have failed at commit time on the code as it stands today. Every finding is still OPEN, so the current master is the pre-fix defect. I ran five one-line probes (single process, `OMP_NUM_THREADS=2`, `LOKY_MAX_CPU_COUNT=1`) to measure how precise the proposed static rules are. Their numbers are quoted as "measured" in the relevant PMT. Everything else labelled "expected" is inferred from the direction reports' own repro numbers.

**Proof protocol, common to every PMT**: (1) run the check on the current master; it must report the listed sites or fail on the listed transforms. (2) When the fix lands, add a one-line revert case to `scripts/teeth_cases/composite_2026_09_19.json` and confirm with `py_ci_shared.teeth_sweep` (or `mutation_teeth.assert_revert_fails_tests`) that re-introducing the defect turns the check red. (3) Every new scanner declares `_candidate_files()` and a `_CANARY` for `gate_population_canary` and calls `assert_scanned_enough`, so it cannot pass green after scanning nothing.

**No xfail anywhere**: registry-driven property tests that fail today on known transforms keep a named known-violator set, following the `UNGUARDED` idiom. Each listed violator must STILL fail, so a fix forces the entry to be removed and a stale entry turns red. A new transform can never join the set silently.

**Ordering**: PMT-01..41 are ordered by (findings caught) x (cheapness weight). The weights are: static meta scan 3, light contract under 10 s 2, contract with tiny fits of 10-60 s 1, suite-level run 0.5. Severity is set separately: P1 when the check catches a P0/P1 finding or at least 6 findings.

**Repo column**: "py-ci-shared" marks a scanner that is cross-project (no mlframe types in the rule) and belongs in `py_ci_shared`, consumed from mlframe's `test_shared_checks_wired.py`. "mlframe" marks a check tied to mlframe's registries or types.

---

## Root-cause taxonomy

Every finding appears once as the primary class. Secondary classes are listed where they drive a second check.

| # | Root-cause class | Findings (primary) | Count | Secondary members | Covered by |
|---|---|---|---|---|---|
| C1 | Predict is not a pure per-row function: batch composition, cold recurrence state, batch-derived fills, instance mutation during predict | TRF-04, TRF-05, TRF-12, TRF-22, EST-04, EST-15, EST-19, EST-20 | 8 | TST-05 | PMT-05, PMT-24, PMT-06 |
| C2 | Unit-dependent constants: a floor or eps in the wrong units, raw-unit margins, uncentred numerics | TRF-01, TRF-02, TRF-08, TRF-09, TRF-18 | 5 | TRF-16 | PMT-08, PMT-01, PMT-12, PMT-17 |
| C3 | Edge-regime blind spots: small n, ties/discrete y, constant or zero-variance input, missing group labels, wrong base dimensionality, over-restrictive domain | TRF-03, TRF-06, TRF-07, TRF-10, TRF-13, TRF-16, TRF-20 | 7 | EST-10 | PMT-01, PMT-17 |
| C4 | Statistical selection design: no complexity penalty, wrong centre, truncating threshold, estimator that ignores n, wasted sampler budget | TRF-11, TRF-14, TRF-15, TRF-19, DSC-15, DSC-30 | 6 | DSC-20 | PMT-19, PMT-17, PMT-06 |
| C5a | Leakage through fit statistics: a row's own y, eval rows or test rows feed the params used to score them | TRF-24, TRF-26, DSC-01, DSC-04, DSC-13, EST-17 | 6 | EST-16 | PMT-18, PMT-11 |
| C5b | Selection-on-holdout: the split used to select is also the one reported or used for the verdict | DSC-03, DSC-18, EST-05, EST-22, INT-14 | 5 | DSC-07 | PMT-29, PMT-13 |
| C6 | The evaluated object is not the deployed object: finite-subset scoring, a tautological watchdog, pre-MoE metrics, inherited statistics, a fallback never evaluated in its regime | DSC-08, DSC-28, EST-02, EST-08, EST-13 | 5 | INT-03 | PMT-39, PMT-07, PMT-24, PMT-22, PMT-31 |
| C7 | CV and split construction contracts violated: time order, groups, alignment convention, holdout size, mismatched splitters | DSC-05, DSC-11, DSC-19, DSC-21, DSC-22 | 5 | EST-15 | PMT-06 |
| C8 | Scale/unit mixing in comparisons and rankings: y vs T, nats vs RMSE fraction, val vs OOF, dedup vs non-dedup feature sets, honest vs optimistic | DSC-06, DSC-07, DSC-10, DSC-17, EST-11, INT-10 | 6 | DSC-21 | PMT-07 |
| C9a | Persistence/replay loses wrapper or runtime state: unwrapped dump, save before post-processing, runtime-only registry, skipped materialisation, metadata/model drift, non-rebuildable columns | DSC-02, INT-01, INT-02, INT-05, INT-06, INT-11 | 6 | TST-02 | PMT-21, PMT-22, PMT-32, PMT-33, PMT-30 |
| C9b | Cache key incomplete: inputs, fitted params or code version missing from the key | DSC-12, INT-13, INT-17 | 3 | - | PMT-15 |
| C9c | Alternate constructors and update paths drift from `fit()` | EST-09, EST-14, EST-21 | 3 | - | PMT-23, PMT-20 |
| C10 | One input serves two pipeline stages (raw base vs pre-pipelined features) | EST-01, INT-03 | 2 | EST-08 | PMT-31, PMT-22 |
| C11 | Silent or misleading failure handling: fail-open gate, debug-level substitution, wrong exception granularity, misleading reason or advice text | TRF-17, DSC-09, DSC-23, DSC-26, INT-09 | 5 | TRF-25, EST-03, EST-08, EST-16 | PMT-03, PMT-27, PMT-30, PMT-14 |
| C12 | Declared knob or contract not honoured: unread ctor params, dropped kwargs, default drift, env parsing, local-only effective config, doc policy not implemented | TRF-21, DSC-14, DSC-25, DSC-27, EST-12, EST-16, EST-18, INT-04, INT-07, INT-15, INT-16, INT-19 | 12 | - | PMT-10, PMT-14, PMT-34..38, PMT-27, PMT-30 |
| C12b | Default-ON corrective mechanism inert on the default path | DSC-16, DSC-24, DSC-29, PRF-04 | 4 | EST-08, EST-09 | PMT-20, PMT-19 |
| C13 | Hand-maintained parallel list or name heuristic instead of the authoritative source (registry, spec set) | TRF-25, DSC-20, INT-08, INT-18 | 4 | TRF-17, EST-08, TST-03, TST-10 | PMT-25, PMT-01 |
| C14 | polars/pandas divergence and frame copies | EST-07, INT-12, PRF-15, PRF-24 | 4 | TST-09 | PMT-16, PMT-26 |
| C15 | Combiner/output invariants not enforced: weight mass lost on drop/cap, crossed or zero-width quantiles | EST-03, EST-06, EST-10 | 3 | - | PMT-13, PMT-12 |
| P-A | Perf: loop-invariant work recomputed per spec, fold, base or split | PRF-01, PRF-03, PRF-05, PRF-07, PRF-08, PRF-10, PRF-11, PRF-13, PRF-14, PRF-16, PRF-17, PRF-18, PRF-20, PRF-21, PRF-22, PRF-23 | 16 | INT-17 | PMT-02 |
| P-B | Perf: memory layout, full-matrix copies, unbounded param size, GIL-bound per-iteration loop | TRF-23, PRF-02, PRF-06, PRF-09, PRF-12, PRF-19 | 6 | PRF-01, PRF-05 | PMT-09, PMT-01 |
| T-A | Tests: non-discriminating assertion shapes (range an envelope guarantees, median error, finiteness, isinstance-only) | TST-01, TST-03, TST-10, TST-11 | 4 | TST-07 | PMT-04 |
| T-B | Tests: fixture sits in the regime where the defect is silent, or asserts only a flag | TST-05, TST-06, TST-08, TST-09, TST-12 | 5 | - | PMT-01, PMT-05, PMT-06, PMT-16, PMT-24 |
| T-C | Tests pin a defect as the contract | TST-04 | 1 | - | PMT-41 |
| T-D | Tests exercise a surrogate (stub ctx, hand-written harness, source text) instead of production | TST-02, TST-07, TST-14, TST-15 | 4 | - | PMT-11, PMT-32, PMT-40 |
| T-E | Data-dependent skips and conditional asserts | TST-13 | 1 | - | PMT-04 |
| T-F | Test timing and cost hygiene | TST-16, TST-17 | 2 | - | PMT-28 |
| | **Total** | | **138** | | |

Per-report check: TRF 26, DSC 30, EST 22, INT 19, PRF 24, TST 17 = 138. Every finding is caught by at least one PMT below; the per-PMT lists add up to more than 138 because some findings are caught twice.

---

## Proposals

### PMT-01 [P1] Registry-driven transform property matrix: max-error round trip across regimes, degenerate legs, and registry metadata completeness
- **Asserts**: for every name in `list_transforms()` (a new registry entry is covered automatically):
  - (a) `max|inverse(forward(y)) - y| <= 1e-9 * max(1, max|y|)` over a fixture matrix: scale {1e-3, 1, 1e3, 1e6}; offset base `c + N(0,1)` with c in {0, 1e4}; n in {50, 300, 5000}; binary and 5-level y; zero-crossing y where the domain allows. The only allowed exceptions are transforms declared lossy (`Transform.loss_bound`, a new field), and for those the measured loss must stay within the declared bound.
  - (b) Degenerate legs: constant base gives an inverse independent of the predict base and equal to mean(y) (TRF-07). Constant y inverts `T + 1e-6` back to the constant (TRF-16). Groups `["a", None, "b", np.nan, "a"]` produce finite output, and the missing/unseen rows equal the global-params inverse (TRF-10). A transform with `n_bases >= 2` (new field) handed fewer base columns must raise or equal its documented fallback (TRF-13).
  - (c) Over-restriction probe: every row that `domain_check` rejects is run through forward/inverse. If a rejected row round-trips exactly and finitely, the domain is over-restrictive (TRF-20).
  - (d) Params size: the pickled `fitted_params` at n=1e5 is at most 2x its size at n=1e4 (TRF-23).
  - (e) Registry uniqueness: no two names resolve to the same `(fit, forward, inverse)` function triple, and every auto-chain proposal name maps onto the existing registry entry for that composition (DSC-20).
  - (f) Side-table completeness: any dict or set literal in `src/` or `tests/` whose keys are at least 80% registry names (and at least 5 keys) must equal the registry, or be derived from a `Transform` attribute (TST-03 tolerance table, TST-10 `_MULTI_BASE`, EST-08 `_ADDITIVE_TRANSFORMS`).
- **Implementation**:
  - New `tests/training/composite/transforms/test_transform_registry_properties.py` with a sibling fixture module `_registry_property_fixtures.py`.
  - Add three `Transform` fields: `n_bases: int = 1`, `loss_bound: float | None = None`, `additive_in_base: bool = False`. `_soft_shrink.ADDITIVE_BASE_TRANSFORMS` and the wrap-pass watchdog set then derive from the field.
  - Leg (f) is an AST meta-test `tests/test_meta/test_registry_name_literals_are_derived.py` using `_shared_ast_cache.parsed_ast` and `assert_scanned_enough`.
  - Violators today go into a named known-violator set per leg (the `UNGUARDED` idiom), not xfail.
  - Replaces the median-error body of `test_composite_transforms_registry_contract.py`. The fast subset (one cell per axis) runs by default; the full matrix sits behind the `slow` marker, per the CLAUDE.md fast-mode rule.
- **Would have caught** (12): TRF-01, TRF-06, TRF-07, TRF-10, TRF-13, TRF-16, TRF-20, TRF-23, DSC-20, TST-03, TST-10, TST-12.
- **Proof it fires**: on master, expected from the transforms.md repros:
  - `reciprocal_residual` at scale 1e3 has max train error 1714 (TRF-01).
  - `quantile_normal_y` at n=5000 fails on the clipped tail rows; `gaussian_copula_residual` on binary y has error 0.477 (TRF-06).
  - Constant base gives y_hat 196.2 against mean 100 (TRF-07).
  - Mixed None/str groups raise `TypeError` (TRF-10).
  - `second_diff` with a 1-D base gives `T = -y` (TRF-13).
  - Constant y: `inverse(T + 1e-6)` gives 8.0 instead of 7.0 (TRF-16).
  - `geometric_mean_residual` rejects `y <= 0` rows that round-trip exactly (TRF-20).
  - `rank_residual` params grow about 10x (TRF-23).
  - The `chain_linear_residual_cbrt` and `chain_linres_cbrt` pair share one function triple (DSC-20).
  - Leg (f) flags the 4 names missing from `_TRANSFORM_RTOL` and the `_MULTI_BASE` omissions.
- **False-positive risk**: low for (a)-(e), because each rests on a declared property. Leg (f) could hit an intentional subset, so each such literal needs a one-line reason in the allowlist.
- **Runtime**: fast subset about 5-10 s (51 transforms x about 8 cells, n <= 5000, no model fits). Full matrix about 60 s under `slow`. Leg (f) under 2 s.
- **Repo**: mlframe.
- **Disposition**: COMPLETED. Leg (a) is the strict max-error round trip in test_composite_transforms_registry_contract.py (TST-03). New tests/training/composite/transforms/test_transform_registry_properties.py runs legs (b) (constant base, constant y, missing/unseen group labels), (c) (restricted to y-side rejections, because a rejected base can divide and multiply back exactly while T means nothing), (d) and (e) over every registry name. They found three real defects, all fixed: `volatility_normalized_residual` on a constant base used an absolute 1e-12 volatility floor (y_hat ~2.5e12 against a train mean of 100; it now falls back to the residual's own scale); `rank_ecdf_residual`/`gaussian_copula_residual` ECDF tables are thinned to 2048 knots; `smoothing_spline_residual` stores the fitted (t, c, k) and no longer the raw bucket means (pickled params had grown 10x, to 3.2 MB at 1e5 rows). Second pass: `Transform` gained `additive_in_t`, `linear_in_base` and `n_bases`. The property module verifies the first two numerically on every declaring transform. The soft base shrink and the wrap-pass `MAE_T == MAE_y` watchdog now read them. The hand-kept watchdog set had held `quantile_residual` (`y = T * IQR + median`, so the watchdog false-fired) and the soft-shrink set had missed `causal_anchor_residual`. The three test-side multi-base sets, which disagreed, now derive from `n_bases`. Leg (f) is tests/test_meta/test_registry_name_literals_are_derived.py, scanning `src/` with a reason per remaining literal. Test-side subsets are selections, not properties, so they are out of its scope. `loss_bound` was not added: the only lossy transform, `y_quantile_clip`, has its exact loss asserted (inverse == clip), which is stronger than a bound.

### PMT-02 [P2] Call-budget harness: expensive primitives are invoked at most their ideal count per discovery fit and per post-phase
- **Asserts**: one tiny default-config discovery fit (n=600, 2 bases, groups on) plus one wrap pass and one xt-ensemble build, run under a counting harness. Each primitive must stay within a declared budget formula:
  - `lgb.Dataset.__init__` at most bases x seeds x folds (PRF-07). `Pipeline.fit` in the linear family at most bases x seeds x folds (PRF-08).
  - `_build_feature_matrix` exactly 1 per fit (PRF-18). `near_collinear_keep_mask` at most 1 correlation build per target (PRF-01).
  - Honest-holdout matrix gather 1 per fit, not per spec (PRF-05). Honest/gate raw-baseline fits 1 (PRF-10).
  - Per-spec tiny CV fits 0 for specs already measured by honest-OOF (PRF-03). WAIC fits only for multi-member bands (PRF-11).
  - Inner predict at most 1 per (component, split) across the hook, report and MoE (PRF-13), and at most 2 per (entry, split) in the wrap pass (PRF-14).
  - `pp.transform` 1 per (shared pp, fold) (PRF-16). `data_signature` 0 on a plain fit (PRF-17). `generate_interaction_bases` 1 (PRF-20).
  - `mi_y_b` replicates computed once per base (PRF-21, opt-in config cell). Prebin hash calls 0 on a cache-miss path (PRF-22). Region-adaptive full fits = number of regions (PRF-23).
  - The cache-signature computation imports no booster module (`sys.modules` check, INT-17).
- **Implementation**:
  - New `tests/training/composite/perf/test_composite_call_budgets.py`, plus a `CallBudget` context manager in `tests/training/composite/perf/_call_budget.py` that wraps targets with counting spies through `monkeypatch`.
  - Budgets are written as the IDEAL formula. Current excess goes into `_composite_call_budget_baseline.json` through `py_ci_shared.baseline_ratchet.Baseline`, recording the measured count and a note. The count may only go down, and a new primitive over budget fails.
- **Would have caught** (17): PRF-01, PRF-03, PRF-05, PRF-07, PRF-08, PRF-10, PRF-11, PRF-13, PRF-14, PRF-16, PRF-17, PRF-18, PRF-20, PRF-21, PRF-22, PRF-23, INT-17.
- **Proof it fires**: on master the counts exceed the formula. For example, PRF-07's rerank builds about 288 Datasets over at most 9 distinct fold matrices per base, and PRF-14 does 4 inner predicts per (entry, split). The baseline records these as the starting debt, and the proof is that deleting the baseline entry turns the test red.
- **False-positive risk**: low. Counts are deterministic for a seeded fixture. The risk is budget formulas going stale as the design changes, which the note on each entry mitigates.
- **Runtime**: about 20-40 s (one tiny discovery fit plus the post phases). Mark it `perf`, excluded from the local default run, and run it in CI.
- **Repo**: mlframe (the `CallBudget` helper is generic and could later move to py-ci-shared).
- **Disposition**: PARTIAL. New `tests/training/composite/perf/_call_budget.py` (`CallBudget`: counts a primitive wherever an mlframe module bound it, and restores on exit) and test_composite_call_budgets.py (`slow`). One default single-threaded discovery fit (n=600, two bases, a group column) must hold each primitive to its ideal count or its recorded excess in `_composite_call_budget_baseline.json`, which may only go down: `data_signature` 0 and `generate_interaction_bases` 1 (at ideal); `_build_feature_matrix` 4 vs 1 (PRF-18), `near_collinear_keep_mask` 5 vs 1 (PRF-01) and `lgb.Dataset` 54 (PRF-07; 18 shared-fold, 18 per-spec CV, 9 honest-gate, 8 WAIC, 1 baseline), each with a note. The fixture is serial because the shared fold-dataset cache is per thread by design (`set_label` mutates it); in parallel the count varies with scheduling (141-156). Not built: the post-phase budgets (inner predicts per component/split, wrap-pass predicts, `pp.transform` per fold, honest-holdout gather, region-adaptive fits) and INT-17's import check, which belongs to INT-17.

### PMT-03 [P1] Fail-open and below-WARNING substitution handlers in gates (shared AST scanner)
- **Asserts**: in scoped packages, an `except` handler must not:
  - (a) append the loop's current element to a list that the function returns or assigns as survivors/kept/selected (admit-on-error);
  - (b) `return True` or `continue` past a reject inside a function named `*gate*|*filter*|*check*|*_ok`;
  - (c) substitute a value (assign a constant or a fallback call) while logging only below WARNING.
  - (d) Separately, a reject predicate of the shape `if isfinite(x) and x >= thr: reject` is flagged, because NaN/inf skip the rejection.
- **Implementation**:
  - New `py_ci_shared/fail_open_handlers.py` exporting `find_fail_open_handlers(paths, gate_name_re=...)` and `assert_no_new_fail_open_handlers(..., baseline_path)` on `baseline_ratchet`.
  - Wired in `test_shared_checks_wired.py`, scoped to `src/mlframe/training/composite` and `src/mlframe/feature_selection`, with baseline `_fail_open_handlers_baseline.json` and `_CANARY = ("survivors.append(spec)  # cannot evaluate",)`.
  - It complements pyutilz's `non_neutral_except_fallback`. Measured: none of the gate modules appears in `_code_audit_baseline.json`, so that rule does not see these shapes.
- **Would have caught** (5): DSC-09 (rules a and d), TRF-17 (c), TRF-25 (c), EST-08 (c: per-split failures at DEBUG), EST-16 (c: fallback to global params at DEBUG).
- **Proof it fires**: measured with `grep`: rule (a) matches `_honest_rmse_gate.py:142` and `_yscale_holdout_gate.py:178,363,377,384,397` (6 sites). Rule (d) matches `_tiny_rerank.py:767`. Rules (b)/(c) match `extended.py:383-392`, `unary.py:384`, `_phase_composite_wrapping.py:759-770` and `ensemble/__init__.py:578-582` per the direction reports.
- **False-positive risk**: medium for (c), because some best-effort diagnostics legitimately log at debug. The scope is gate/transform packages only, and a `# best-effort:` marker is honoured (the convention the `test_log_only_except_*` files already use). (a) and (d) are low: two further `kept.append` sites (`_eval_stats.py:328`, `_fit_temporal.py:67`) need triage.
- **Runtime**: under 2 s (cached AST).
- **Repo**: py-ci-shared.
- **Disposition**: COMPLETED. The scanner is `py_ci_shared.fail_open_handlers` (py-ci-shared fdbed25, 13 unit tests), which covers rules (a)-(d). A `# best-effort: <reason>` marker exempts a fallback that cannot change a result, and appends into a list named for failures (`failed.append(k)`) are exempt from (a). It is wired as `test_no_new_fail_open_handlers` in tests/test_meta/test_shared_checks_wired.py over the composite and feature-selection packages, with the pin bumped in requirements-dev.txt. Composite backlog: the six gate sites were fixed with DSC-09. The scan also found three fail-open sites, now fixed: the base-leakage guard kept a base whose check raised; the fragility gate kept a spec whose base column it could not read; auto-chain kept a winning chain whose final fit failed, with empty params. Eleven fallbacks that can change a result now log at WARNING (the cache-key version and config fallbacks, the auto-base corr gate, the multi-base pool-corr guard, fold-refit and gate domain checks, the monotonic orientation, the Box-Cox lambda, the unary T envelope, the quantile alpha probe). Nine reporting/repr/perf fallbacks carry a best-effort reason. Two composite entries remain in `_fail_open_handlers_baseline.json`, each with its reason. The 331 feature-selection entries are recorded untriaged: that package is being changed by another session. Tests: tests/training/composite/discovery/test_guard_failures_fail_closed.py (fails pre-fix).

### PMT-04 [P1] Non-discriminating test-assertion shapes: literal wide ranges, median-of-error, isinstance-only biz tests, data-dependent skips
- **Asserts**: in test functions, flag:
  - (a) a chained comparison `lo < x < hi` with literal bounds where `hi/lo >= 20`, or `lo <= 0` together with `hi >= 10`;
  - (b) an envelope assert `pred.min() > k * y.min()` / `pred.max() < k * y.max()`;
  - (c) `np.median(np.abs(a - b))` compared to a tolerance in a test whose name contains `round_trip|roundtrip|inverse`;
  - (d) a `test_biz_val_*` function whose only asserts are `isinstance`, `is None`, `len(...) > 0` or boolean flags, with no numeric comparison;
  - (e) `pytest.skip(...)` reached after the test body has computed something (not the first statement, not `importorskip`, not a platform/env probe), including `if cond: assert ... else: pytest.skip(...)`.
- **Implementation**:
  - Extend `py_ci_shared` with `nondiscriminating_shapes.py` rules (a)-(c) and (e). Rule (d) is a naming convention local to mlframe, so it goes into the existing `tests/test_meta/test_no_nondiscriminating_assert.py` as reason 6.
  - Baselines keyed `file:function:reason` without line numbers, following that file's convention, and refreshed with `--refresh-nondiscriminating-assert-baseline`.
- **Would have caught** (5): TST-01 (`0 < RMSE < 100`, `0.5 < RMSE < 50`, the 0.5x/1.5x envelope), TST-03 (median round trip), TST-10 (envelope assert in the fuzz), TST-11 (four `test_biz_val_*` asserting only isinstance), TST-13 (the five data-dependent skip sites).
- **Proof it fires**: expected hits at `test_composite_integration.py:301-352`, `:537-604`, `test_composite_transforms_registry_contract.py:165-184`, `test_biz_val_training_composite_discovery.py:99-114/117-122/177-191/199-225`, `test_stacked_discovery_fixes.py:319-323` and `test_composite_business_value_locks.py:186-194`.
- **False-positive risk**: medium for (a): some metrics really do have wide legitimate ranges (probabilities in (0,1)), so the rule exempts `0 <= p <= 1`. Low for (c)-(e).
- **Runtime**: about 3 s over about 3600 test files (cached AST).
- **Repo**: py-ci-shared for (a), (b), (c), (e); mlframe for (d).
- **Disposition**: COMPLETED. Rules (a), (b), (c) and (e) are `py_ci_shared.nondiscriminating_shapes.shape_reasons` (py-ci-shared d619ced, 17 unit tests; `0 <= p <= 1` exempt, and environment-probe and missing-dependency skips exempt from (e)). Rule (d) is local: `biz-val-no-numeric` flags a `test_biz_val_*` whose every assertion is a type, None, non-empty or truthiness check and that calls no checking helper. Both are wired into tests/test_meta/test_no_nondiscriminating_assert.py, one baseline key per new reason so earlier entries keep their keys. Suite-wide at wiring: 111 late-skip, 45 biz-val-no-numeric, 25 wide-literal-range, 3 envelope-assert, 2 median-roundtrip, all recorded in the baseline. The composite-test hits (TST-10, TST-11, TST-13 among them) are fixed under those findings. The refresh paths of this and three sibling baselines wrote through `write_text`, which turns every newline into CRLF on Windows; they now write bytes.

### PMT-05 [P1] Row-purity contract for every registered transform and every deployable component: batch-invariant, NaN-local, thread-safe
- **Asserts**: fit a `CompositeTargetEstimator` per registry transform with an oracle inner (the inner returns the exact `forward(y)` for the rows it is given), then predict a 200-row continuation.
  - (a) As one batch vs as 1-, 7- and 50-row chunks: `recurrent=False` transforms must be bit-equal. `recurrent=True` transforms must be equal when a warm-up prefix and `recurrence_continuation` are supplied.
  - (b) With one NaN base injected: every other row equals carry-forward semantics and the NaN row gets the fallback (EST-04).
  - (c) 8 threads x 50 predicts leave `runtime_stats_["predict_calls"] == 400` and `soft_shrink_info_` consistent with the caller's own batch (EST-20).
  - (d) The same batch-vs-chunk property over every component class the CT ensemble can ship (`lag_predict`, stackers, MoE wrapper): a shared NaN-lag row gets an identical fill in two batches that differ only in their other rows (EST-19).
- **Implementation**:
  - New `tests/training/composite/estimator/test_cte_row_purity.py`, parametrised over `list_transforms()` and a `DEPLOYABLE_COMPONENTS` registry in `composite/ensemble/__init__.py`. A meta-guard asserts every class `_phase_composite_post_xt_ensemble` can instantiate is listed there.
  - Current violators go into the named known-violator set.
- **Would have caught** (7): TRF-04, TRF-05, TRF-12, EST-04, EST-19, EST-20, TST-05.
- **Proof it fires**: expected from the reports: `frac_diff` single-row vs batch differs by up to 14.56 (TRF-04); `ewma_residual` by 11.94 (TRF-05); `seasonal_residual` 1-row chunks restart at phase 0 (TRF-12); one NaN base moves row 6 by 34.9 (EST-04); a 1-row `lag_predict` batch with a NaN lag returns 0.0 (EST-19); unsynchronised counters lose increments (EST-20).
- **False-positive risk**: low. Recurrent transforms are judged only under the warm-up contract they are supposed to honour.
- **Runtime**: about 5-8 s (51 CTE fits with an oracle inner on n=400; no model training).
- **Repo**: mlframe.
- **Disposition**: COMPLETED. New tests/training/composite/estimator/test_cte_row_purity.py. (a) Every pointwise transform, with an elementwise inner, predicts 1-, 7- and 50-row chunks equal to the whole batch to 1e-14 relative (the only residue is 1-2 ulp from a transform's own BLAS dot). The recurrent warm-up contract and (b) NaN-base locality were already pinned by TST-05's test_predict_batching_invariance.py. (c) Concurrency: counters and shrink flags under 8 threads; it found and fixed EST-20. (d) The `lag_predict` component: it found and fixed EST-19. The DEPLOYABLE_COMPONENTS registry was not built: the stackers are linear combinations of the components checked here, and `lag_predict` was the one component with its own batch-state.

### PMT-06 [P1] Splitter and sampler consistency: one splitter factory, time order and groups honoured everywhere, sampler returns usable rows
- **Asserts**:
  - (a) Static: in `composite/discovery`, `composite/ensemble` and `core/_phase_composite_post_xt_ensemble`, `KFold(shuffle=True)` / `StratifiedKFold(shuffle=True)` may only be constructed inside `make_discovery_splitter(groups, time_ordering, ...)` (a new helper).
  - (b) Dynamic: one tiny discovery fit on a SHUFFLED grouped autoregressive frame with `time_ordering`, with every splitter constructor spied. Every fold's train rows must precede its validation rows in time (DSC-05). Every splitter is group-aware when groups exist (DSC-11). Raw and every spec in one rerank receive the same splitter config (DSC-21). The honest holdout is within `honest_holdout_frac*(1 +/- 0.25)` of the train rows (DSC-19). The holdout is group-disjoint from the screen pool under a shuffled `train_idx` (DSC-22). `_sample_indices` returns no non-finite-y rows (DSC-30).
  - (c) Ensemble: recurrent components receive contiguous folds (EST-15).
- **Implementation**:
  - (a) is `tests/test_meta/test_no_ad_hoc_shuffled_kfold.py` (AST, `assert_scanned_enough`).
  - (b) and (c) are `tests/training/composite/discovery/test_splitter_contract.py`, using `monkeypatch` spies on `sklearn.model_selection` classes as imported into each module.
- **Would have caught** (8): DSC-05, DSC-11, DSC-19, DSC-21, DSC-22, DSC-30, EST-15, TST-06.
- **Proof it fires**: measured for (a): shuffled-KFold constructions at `_auto_chain.py:250`, `_eval_waic.py:219`, `forward_stepwise.py:158`, `ensemble/__init__.py:531`, `ensemble/feature_stacking.py:177`, `_phase_composite_post_xt_mtr_oof.py:119` and `_screening_tiny.py:64,355` (the last two sit behind a groups/time branch and are allowed after the move into the factory). For (b), expected: the folds follow row position, not time (DSC-05); a 90%/10% group split gives a 90% holdout (DSC-19).
- **False-positive risk**: low for (b). (a) is scoped to discovery/ensemble; generic composite estimators (`dual_direction.py`, `pseudo_labeling.py`, etc.) are out of scope.
- **Runtime**: (a) under 1 s; (b) about 8-15 s (one discovery fit at n=600 with `tiny_model_cv_folds=2`).
- **Repo**: mlframe.
- **Disposition**: COMPLETED. (a) is tests/test_meta/test_no_ad_hoc_shuffled_kfold.py. No module in discovery, ensemble or the cross-target builder may build a shuffled KFold / StratifiedKFold / ShuffleSplit outside `discovery/_splitter.py` (canary included). The remaining hand-built sites (forward_stepwise, feature_stacking, the MTR OOF, the tiny-CV split cache and fallback) now route through the factory, with identical behaviour. (b) and (c) are tests/training/composite/discovery/test_splitter_contract.py: group-disjoint folds in the chain and WAIC CVs, forward folds under time order, one fold scheme per rerank, the holdout size bound, frame-aligned holdout groups, no NaN-target rows in the sampler, and contiguous OOF folds for a recurrent component. Together they caught and fixed DSC-11, DSC-19, DSC-21, DSC-22, DSC-30 and EST-15.

### PMT-07 [P1] Units- and provenance-tagged scores: ranking helpers refuse mixed units, and every ranking scorer is invariant under an affine-rescaled twin transform
- **Asserts**:
  - (a) Every spec ranking or budget sort in `composite/discovery` and `core/_phase_composite_*` goes through `rank_specs(specs, key=...)`. The key returns a `Score(value, unit, split, estimator, measured_for)` NamedTuple, and the helper raises when units, splits or estimators are mixed, or when `measured_for != spec.transform_name`.
  - (b) Static: a bare `.sort(key=...)` / `sorted(...)` over spec lists outside `rank_specs` is flagged.
  - (c) Metamorphic: for every scorer in a `SPEC_SCORERS` registry (tiny-CV y-scale RMSE, WAIC, honest RMSE, MI gain, yscale gate), a transform `t` and its twin `scaled_twin(t, k=100)` (forward x k, inverse / k) get identical scores and ranks (DSC-10). A spec forced through the shrunk-domain MI path (one dropped row) scores within noise of the full-domain path (DSC-06).
  - (d) Metadata metric records carry `scale in {"y","T"}` and `split`, and the targets-performance table refuses T-scale rows in the y-scale section (INT-10).
- **Implementation**:
  - `composite/discovery/_score.py` (new, small) holds `Score` and `rank_specs`.
  - (b) is `tests/test_meta/test_spec_sorts_go_through_rank_specs.py`.
  - (c) and (d) are `tests/training/composite/discovery/test_scorer_invariance.py`, where `scaled_twin` is a registry-free wrapper. A meta-guard asserts that every function under `discovery/` returning a per-spec score dict is registered in `SPEC_SCORERS`, detected by the name pattern `*_rmse*|*waic*|*mi_gain*|*_score*`.
- **Would have caught** (7): DSC-06, DSC-07 (honest and CV estimators mixed in `agg_scores`), DSC-10, DSC-17, DSC-28 (`measured_for` is the seed transform), EST-11 (val-split dummy vs OOF components), INT-10.
- **Proof it fires**: expected: the WAIC of `T=(y-base)/100` exceeds the WAIC of `T=y-base` by about `log 100` per row (DSC-10); `_phase_composite_discovery.py:866-878` sorts RMSE fractions and nats together, so the helper raises (DSC-17); the multi-base upgrade copies `mi_gain` from its seed (DSC-28).
- **False-positive risk**: low once the helper exists. The work is the migration of the existing sort sites.
- **Runtime**: (b) under 1 s; (c) about 5 s (tiny data, 5 scorers x 3 transforms).
- **Repo**: mlframe.
- **Disposition**: PARTIAL. The seven findings it names are fixed at source: DSC-06, DSC-07, DSC-10, DSC-17, DSC-28, EST-11 and INT-10. tests/training/composite/discovery/test_scorer_invariance.py pins each one, including the WAIC scale invariance (c) and the y-scale table (d), and every case fails pre-fix. The cross-target budget sort, the one sort that mixed units, is now the named `rank_pending_composites`. Not built: the typed `Score` / `rank_specs` wrapper and the (b) scan over every spec sort. The remaining sorts each order a single unit (auto-chain candidates by y-RMSE, the screen by `mi_gain`, stability by frequency).

### PMT-08 [P1] Scale and shift metamorphic property over every registered transform
- **Asserts**:
  - For every registry transform, fit on `(s*y, s*base)` for s in {1e-3, 1e3, 1e6} (and `s*y` alone for unary transforms). Estimate the affine map `T_s ~= a*T_1 + b` by least squares. Its residual must be below 1e-6 of `std(T_s)`, unless `Transform.scale_equivariant=False` (new field, with a reason; for example `yeo_johnson_y`).
  - Then with `T_hat_1 = T_1 + 0.1*std(T_1)` and `T_hat_s = a*T_hat_1 + b`, require `inverse_s(T_hat_s) == s * inverse_1(T_hat_1)` to 1e-6 relative.
  - Base-shift leg: for transforms declaring `base_translation_invariant=True`, which is true of the whole linear/polynomial residual family, T is unchanged when `base` becomes `base + 1e4` or `base + 1e6`.
  - Optional static companion: an AST rule in `composite/transforms` flagging `np.linalg.solve(X.T @ X, ...)` normal-equation solves on uncentred designs, and float literals below 1e-2 compared against names containing `iqr|mad|std|scale|var` with no multiplying reference (baselined).
- **Implementation**: in `test_transform_registry_properties.py` (shared fixtures with PMT-01), plus two new `Transform` fields. The static companion is `tests/test_meta/test_transform_absolute_floors.py`.
- **Would have caught** (5): TRF-01 (the eps_z clamp breaks equivariance at s=1e3), TRF-08 (`mad_eff` floor in y units; the cap binds at s=1e-3 and never at 1e6), TRF-09 (base shift 1e4 changes `std(T)` from 0.0099 to 0.596), TRF-16 (the `+1.0` knot in raw units), TRF-18 (the absolute `1e-6` IQR floor).
- **Proof it fires**: expected from the transforms.md repros: all five show different outputs across scales or shifts. The static companion flags `nonlinear.py:198,225` (TRF-18) and `extended.py:175-200` (TRF-09).
- **False-positive risk**: low for the property, since non-equivariant transforms declare the exemption with a reason. Medium for the static companion (the "hinge-gate normal equations" perf win in CLAUDE.md is a legitimate allowlist entry).
- **Runtime**: about 3-5 s (51 transforms x 4 scales x n=500).
- **Repo**: mlframe. The normal-equations rule could later join py-ci-shared.
- **Disposition**: COMPLETED. Added `Transform.scale_equivariant` (default True) and `base_translation_invariant`, with tests in test_transform_registry_properties.py. Fitting on (s*y, s*base) for s in {1e-3, 1e3, 1e6} must give T_s as an affine map of T_1 (residual <= 1e-6 std). The inverse must commute with the scale (1e-6 relative). A base shifted by 1e4 or 1e6 must leave T unchanged for the 18 translation-invariant transforms. The test found two more raw-unit constants, both fixed. `log_y` added 1.0 to a strictly positive target: nearly the identity on a target of order 1e-3, a log at 1e6. It now takes a plain log for positive y and a spread-relative margin otherwise. `asinh_residual` / `asinh_residual_multi` took arcsinh in raw units; they now fit scales (median |x|) for y and each base. The two tests that pinned the old raw-unit asinh were reframed to fixtures in the transform's own units. The exempt transforms are yeo_johnson_y and the two chains built on it, whose (y + 1) ** lam form is unit-dependent by definition. The optional AST companion was not built: the property test catches the same defects on every registered transform.

### PMT-09 [P2] Memory layout, copy and GIL-loop scanners with tracemalloc budgets for discovery
- **Asserts**:
  - (a) A matrix allocated with `np.empty((n_rows, n_cols))` or `np.column_stack` and then read column-wise (`m[:, j]`) inside a loop or njit kernel must be allocated `order="F"` (PRF-02).
  - (b) `np.asarray(x, dtype=np.float64)` / `np.ascontiguousarray(..., float64)` on a whole feature matrix inside per-base/per-spec loops is flagged (PRF-01 copy, PRF-11, PRF-19).
  - (c) A Python `for` loop over permutations that calls an `@njit` kernel once per iteration with a fresh RNG draw is flagged (the CLAUDE.md "GIL-bound per-resample loop" class, PRF-12).
  - (d) Dynamic: `tracemalloc` peak of `_filter_features` at n=200k x F=50 stays at most 1.3x one sample matrix, and the resident extra at the `tiny_model_rerank_done` checkpoint stays at most `(n_screen x F x 2B)` (PRF-06, PRF-09).
- **Implementation**:
  - (a)-(c) in `tests/test_meta/test_discovery_layout_and_copies.py` (AST, baseline `_discovery_layout_baseline.json`).
  - (d) in `tests/training/composite/perf/test_discovery_memory_budgets.py`, reusing the pattern of `tests/test_meta/test_memory_budgets.py`.
- **Would have caught** (5): PRF-02, PRF-06, PRF-09, PRF-12, PRF-19.
- **Proof it fires**: expected hits at `screening.py:342,405`, `_auto_chain.py:440`, `_tiny_rerank_waic.py:93`, `_collinear_numba.py:324` and `_auto_base.py:512-561`. For (d), PRF-06 estimates peak at 2x full matrix plus mask, so the budget is exceeded.
- **False-positive risk**: medium for (b)/(c) (a one-off upcast outside a loop is fine, so the rule is scoped to loop bodies). (d) is low-risk but depends on the host, so it uses a ratio against the input size, not absolute bytes.
- **Runtime**: AST under 1 s; tracemalloc about 5-10 s.
- **Repo**: mlframe. Rule (c) is generic and could move to py-ci-shared later.
- **Disposition**: PARTIAL. New tests/test_meta/test_discovery_layout_and_copies.py scans the discovery package (AST) for three rules, with a ratchet baseline `_discovery_layout_baseline.json`: (a) a column read inside a loop of a matrix allocated C-order (`np.empty((n, m))` / `np.column_stack` without `order='F'`); (b) a float64 `asarray` / `ascontiguousarray` / `array` of a feature-matrix-named value inside a loop; (c) a loop that draws from an RNG and calls a module-level `@njit` kernel on every iteration. A canary pins each rule firing on its shape and not on the fixed form. Today there are 0 hits; the expected hits the audit listed are gone after the PRF fixes. Rule (a) is scoped to reads: the only candidate was `forward_stepwise_multi_base` writing columns into a C buffer that per-fold row gathers read, where C order is right. Not built: (d), the tracemalloc peak budgets for `_filter_features` and the rerank checkpoint.

### PMT-10 [P2] Kwarg forwarding: a variant wrapper accepts and forwards its base method's optional parameters; an in-scope argument is not silently omitted (shared scanner)
- **Asserts**:
  - (a) When a function `F` calls `self.G(...)` (or a same-module `G`), and F is a named variant of G (F's name starts with G's name, like `fit_stacked`/`fit_with_stability_check` -> `fit`, or it is registered as a delegate), then every optional keyword parameter of G must be accepted by F and forwarded (or F takes `**kwargs` and forwards it).
  - (b) Available-but-not-passed: in a call to a resolvable callee with an optional parameter `p` (default None), if the caller's scope has a parameter or local literally named `p` and the call omits `p`, flag it.
  - (c) Delegate state: `obj = SameClass(cfg)` created inside a method of `SameClass` and then `obj.fit(...)` must copy the private attributes `self._x` that `fit` reads before being called (for example `_group_ids_for_rerank`, `_hint_strengths_pct`).
- **Implementation**:
  - `py_ci_shared/kwarg_forwarding.py` with `find_dropped_variant_params`, `find_available_but_not_passed` and `find_delegate_state_loss`, baselined through `baseline_ratchet`.
  - Wired in `test_shared_checks_wired.py` for `src/mlframe/training` with baseline `_kwarg_forwarding_baseline.json`.
- **Would have caught** (3): DSC-14 (`fit_stacked` and friends lack `time_ordering`, `val_df`, `val_y`), DSC-27 (the per-group delegate loses `_group_ids_for_rerank` and `_hint_strengths_pct`), EST-12 (`from_nnls_stack` / `from_linear_stack` accept `sample_weight`; the builder has it in scope and omits it).
- **Proof it fires**: expected at `_stacked.py:126,219,286,384`, `_stability_check.py:151`, `_per_group.py:84-96` and `core/_phase_composite_post_xt_ensemble/__init__.py:841-852`.
- **False-positive risk**: medium for (b): a local named `groups` may carry a different grouping. It is baselined with notes. (a) and (c) are low.
- **Runtime**: under 3 s.
- **Repo**: py-ci-shared.
- **Disposition**: PARTIAL. The three findings this scanner targets are fixed directly, each with a behavioural regression test: DSC-14 (stacked and stability-check variants forward `time_ordering` / `val_df` / `val_y`, test_variant_fits_forward_kwargs.py), DSC-27 (per-group delegates get their group's val rows and inherit `_group_ids_for_rerank` / `_hint_strengths_pct`) and EST-12 (the CT stack path is weighted end to end). Not built: the shared `kwarg_forwarding` scanner (variant-parameter drop, available-but-not-passed, delegate state loss) that would catch the next instance of the class.

### PMT-11 [P1] Test-to-production reachability: no test certifies an uncalled production function, and every gate module has an importing test
- **Asserts**:
  - (a) Cross-check `py_ci_shared.uncalled_functions` with the test import map from `py_ci_shared.effect_assertion_parity.build_import_map`. A production function that has no production caller but IS imported and asserted on by a test fails the check. That test certifies a fix nothing runs.
  - (b) Every module matching `discovery/_*gate*.py|_*rerank*.py|_filter*.py|_per_group.py` has at least one test that imports it or one of its public functions directly.
  - (c) `_uncalled_functions_baseline.json` entries whose docstring contains `leak|optimis|guard|fix|gate` cannot be added without a note (`py_ci_shared.baseline_hygiene`).
- **Implementation**: new `py_ci_shared/test_reachability.py` (`assert_no_tested_but_uncalled`, `assert_modules_have_importing_tests`), wired in `test_shared_checks_wired.py`. Rule (c) extends the existing `test_shared_uncalled_functions.py` refresh path.
- **Would have caught** (3): DSC-04 and TST-07 (`refit_transform_on_fold` has no production caller and a biz_val test certifies it), TST-14 (`_tiny_rerank_waic.py`, `_filter_and_gate.py` and `_per_group.py` have no direct test).
- **Proof it fires**: measured: `_uncalled_functions_baseline.json:775` holds `src/mlframe/training/composite/discovery/_eval.py::refit_transform_on_fold`. The uncalled-function check found this defect, and the baseline refresh absorbed it. `tests/training/composite/eval/test_eval_perfold_refit.py` imports it, so rule (a) fires.
- **False-positive risk**: low. A utility kept for external API users is allowlisted with a reason.
- **Runtime**: under 5 s (both scans already run in the meta suite; this cross-joins their results).
- **Repo**: py-ci-shared.
- **Disposition**: PARTIAL. tests/test_meta/test_tested_but_uncalled.py covers two rules. (a) It joins the uncalled-functions baseline with the names tests reference: 68 composite functions are tested yet uncalled today, and they are recorded in `_tested_but_uncalled_baseline.json`. A new entry fails, and so does a stale one once a function is wired in. (b) Every discovery gate / rerank / filter / per-group module must have an importing test; all do today, so TST-14's gap is closed. The motivating `refit_transform_on_fold` is called by production since 870694a39. Remaining: triage of the recorded 68 entries into wire-in or justified. Most are public diagnostics (plot_*, winkler, bayesian fits); the gate-like ones were checked. `stability_select_specs`, `screen_base_pool`, `make_purged_cv` and `purged_oof_holdout` are exported public API; production's stability path reuses the first one's helpers. `calibration_adjusted_score` is a research ranking utility that no config enables. None is a fix that production silently skips. Rule (c) is not built.

### PMT-12 [P1] Out-of-range and perturbation leg: OOD bases stay sign-consistent, the inverse is Lipschitz in T_hat, and quantiles stay ordered
- **Asserts**: for every registry transform:
  - (a) On predict bases 5% below the train min and above the train max, the inverse of the train-median T is finite, keeps the sign of the train y when that sign is definite, and lies within [0.5, 2]x of the inverse at the nearest train edge (TRF-02).
  - (b) `|inverse(T + d) - inverse(T)| <= 1.1 * L * |d|` for d = 1e-3*std(T), where L is the pointwise `|d inverse / dT|` estimated at fixed base on a single row. A recurrence that amplifies perturbations fails (TRF-04: gain 9.75 against a pointwise 1).
  - (c) For each transform through `CompositeTargetEstimator.predict_quantile`, the quantile columns are non-decreasing in alpha on every row, and fallback rows are not zero-width unless declared NaN (EST-10).
- **Implementation**: `test_transform_registry_properties.py` (legs a, b) and `tests/training/composite/estimator/test_predict_quantile_contract.py` (leg c), both parametrised over `list_transforms()`.
- **Would have caught** (4): TRF-02, TRF-04, TRF-16 (a step of `+1.0` for `d=1e-6`), EST-10.
- **Proof it fires**: expected: `centered_ratio` at base 97 gives -13.04 against a true value of about 291 (TRF-02); `frac_diff` shifts y by 0.92 for d=0.1 (TRF-04); constant y jumps 7 -> 8 (TRF-16); `centered_ratio` with `base + c < 0` crosses q10 > q90 (EST-10).
- **False-positive risk**: low. The Lipschitz leg uses the transform's own pointwise derivative, so steep transforms are not penalised.
- **Runtime**: about 3-5 s.
- **Repo**: mlframe.
- **Disposition**: COMPLETED. (a) and (b) are in test_transform_registry_properties.py. (a): a base 5% beyond either train edge inverts the median T to the edge value's sign, within [0.5, 2]x of it, for every non-recurrent base transform. The sign is compared with the in-range edge: for `reciprocal_residual` the median T at an extreme in-range base is already negative. (b): nudging one row's T moves that row's y by at most 1.1x the pointwise derivative, for every transform. (c) is test_predict_quantile_contract.py (ordered quantiles on every row, a real interval on fallback rows), which found and fixed EST-10.

### PMT-13 [P1] Ensemble combiner invariants for every stacking strategy, including "the gate can fire"
- **Asserts**: for every strategy in the cross-target strategy registry (`nnls_stack`, `linear_stack`, `oof_weighted`, `mean`, meta-stacker), with stub components returning fixed OOF/test columns:
  - (a) Making one component raise at predict keeps the prediction mean within 5% of the surviving blend, and the log text matches the branch taken (EST-03).
  - (b) `cap_inference_components(k)` keeps the OOF mean within 2% of the uncapped blend, and the capped predictor is re-gated (EST-06).
  - (c) "Gate can fire" canary: a pool of 1 good component plus 20 pure-noise components at small OOF n must make the honest gate fall back to the best single component (EST-05).
  - (d) Cross-fitted scoring: the gate's score rows are disjoint from the rows the weights were fit on. This is asserted through the PMT-29 ledger when available, otherwise through a spy on the solver's row set.
- **Implementation**: `tests/training/composite/ensemble/test_combiner_invariants.py`, parametrised over the strategy registry. A meta-guard asserts every `from_*_stack` constructor is registered. It reframes `test_composite_ensemble_linear_stack_dropout.py` and `test_composite_medium_findings.py::test_m2_*` (TST-04) in the same change.
- **Would have caught** (4): EST-03, EST-05, EST-06, EST-10 (through the `predict_quantile` ordering leg shared with PMT-12).
- **Proof it fires**: expected: two components at 0.5/0.5 with one failing give a mean of 449.4 against 898.7 (EST-03); the cap drops weight mass without renormalising (EST-06); in-sample NNLS never loses to its best single component (EST-05).
- **False-positive risk**: low.
- **Runtime**: about 2 s (numpy stubs, no model fits).
- **Repo**: mlframe.
- **Disposition**: PARTIAL. (c) "the gate can fire" and (b) the cap keeping the blend's level are pinned in tests/training/composite/ensemble/test_combiner_invariants.py; they fixed EST-05 and EST-06. (a) the failing-component mean is already covered by EST-03's resolved tests. Not built: the strategy-registry parametrisation over every constructor, the meta-guard, and the (d) row-disjointness spy.

### PMT-14 [P2] Transform-call gateway: every registry-transform fit/forward/inverse call goes through one signature-gated helper, and weights are honoured
- **Asserts**:
  - (a) Static: in `src/mlframe/training`, calls to `.fit(`, `.forward(` and `.inverse(` on a receiver bound from `get_transform(...)`, `TRANSFORMS_REGISTRY[...]` or `_TRANSFORMS_REGISTRY[...]`, or on parameters named `transform|tr|t|_t|bivariate|unary`, must go through `call_transform(transform, "fit", y, base, groups=..., sample_weight=...)`. That helper uses the existing `_callable_accepts_param` signature gate.
  - (b) Registry property: for every transform whose `fit` accepts `sample_weight`, zero weights on half the rows equal a fit on the kept half. A chain whose first stage accepts `sample_weight` must accept it too.
- **Implementation**: helper in `composite/transforms/naming.py` (next to `get_transform`); (a) is `tests/test_meta/test_transform_calls_use_gateway.py` with baseline `_transform_gateway_baseline.json`; (b) goes into `test_transform_registry_properties.py`.
- **Would have caught** (3): INT-09 (`_eval.py:344` fits without `groups`), EST-16 (`ensemble/__init__.py:579,836` refit without `groups`/`sample_weight`), TRF-21 (the chain `_fit(y, base)` closures drop weights).
- **Proof it fires**: measured with `grep`: bare transform-fit calls at `_auto_chain.py:265,315`, `_eval.py:344`, `_region_adaptive.py:176,189,199,201`, `ensemble/__init__.py:579,836` and `sklearn_compat.py:253,255`. `_eval.py:207` already passes `fit_kwargs` (the compliant shape). Leg (b) is expected to fail on `chain_linres_cbrt/yj/cbrt_qn`.
- **False-positive risk**: medium. `quantile.py:166` is a probe fit on a fake y (allowlist); receiver-name heuristics could mis-bind (scope limited to composite/core).
- **Runtime**: AST under 1 s; property about 2 s.
- **Repo**: mlframe.
- **Disposition**: PARTIAL. (b) is built and found a real defect, now fixed. A zero sample weight did not remove the row: the grouped linear residual counted zero-weight rows in its group sizes and James-Stein shrinkage (8.8 off in y), and rank-ECDF, copula, robust, Theil-Sen, target-encoding, multi-base, causal-anchor and monotone-chain fits counted them in their grids, trims and counts (0.004-0.58 off). The fix is new `transforms/_zero_weight.ZeroWeightDroppingFit`, installed by `Transform.__post_init__` on every order-free fit that takes `sample_weight` (recurrent transforms are exempt). It drops zero-weight rows, with their `groups`, before fitting; it is a module-level class so transforms still pickle. `test_a_zero_weight_removes_the_row_from_the_fit` (17 weighted transforms) fails for 10 of them before the fix, and `test_the_weighted_set_covers_the_residual_family_and_its_chains` checks every `chain_linres*` accepts weights. (a): tests/test_meta/test_transform_calls_use_gateway.py (AST over composite and core, canary) records the 53 remaining bare `fit` / `forward` / `inverse` calls on transform-bound names in `_transform_gateway_baseline.json`, which may only shrink. `CompositeTargetEstimator.fit` now fits through `call_transform` (groups / sample_weight / row_index signature-gated), which replaces its hand-rolled gate (fit 360 -> 341 lines). Not done: converting the other 10 bare fits; none has groups or weights in scope today.

### PMT-15 [P2] Cache-key completeness by input perturbation, plus a code-version gate on discovery sources
- **Asserts**:
  - (a) For every cache-key builder (the discovery-phase key in `_phase_composite_discovery.py:521-541`, the model-cache path in `train_eval.py:400-470`, the specs-replay signature), a `CACHE_KEY_INPUTS` table lists each input of the cached computation with a perturbation. Perturbing any listed input must change the key.
  - (b) A meta-guard derives the required inputs automatically: every parameter of `CompositeTargetDiscovery.fit`, plus every private attribute that `fit` reads and that the phase sets before calling it (`_group_ids_for_rerank`, `_hint_strengths_pct`), must appear in the table or in a "does not affect the result" allowlist with a reason.
  - (c) The composite model cache key includes a digest of `transform_name + fitted_params`.
  - (d) `DISCOVERY_ALGO_VERSION` is gated by `py_ci_shared.content_hash_version_bump_gate.assert_version_bumped_with_content` over `src/mlframe/training/composite/discovery/**/*.py` and `transforms/**/*.py`, and the signature reads library versions via `importlib.metadata` (no booster import).
- **Implementation**: `tests/training/composite/cache/test_cache_key_completeness.py` covers (a)-(c) and uses key functions only, no fits. `tests/test_meta/test_discovery_algo_version_bumped.py` covers (d), with baseline `_discovery_algo_version_baseline.json` and `--refresh` registered in `conftest.py` through that module's `register_refresh_option`.
- **Would have caught** (3): DSC-12, INT-13, INT-17.
- **Proof it fires**: expected: two keys with identical data and config but different `group_ids` (or val frames) are equal (DSC-12); a spec differing only in `fitted_params` reuses the cached `.dump` (INT-13); any edit under `discovery/` leaves `mlframe.__version__` unchanged, so (d) fails on the first source edit (INT-17).
- **False-positive risk**: low. (d) makes a deliberate one-line bump per selection-affecting change, which is the purpose of the gate.
- **Runtime**: under 3 s.
- **Repo**: mlframe (the version gate reuses the py-ci-shared module).
- **Disposition**: PARTIAL. Built: (a) for the discovery key. `test_the_discovery_cache_key_changes_with_every_input_that_changes_the_specs` perturbs group ids, hint strengths, time order, val y and val frame one at a time, and each changes the key (DSC-12 fixed). (c): the composite model cache records `composite_spec_digest`, and test_composite_model_cache_digest.py pins that the digest follows the fitted params and that a stale or digest-less dump is invalidated (INT-13). (d): `DISCOVERY_ALGO_VERSION` is in the key, and test_discovery_algo_version_bumped.py gates it on a CRLF-normalised hash of composite/discovery + composite/transforms. Versions now come from `importlib.metadata`, and a subprocess test shows no booster import (INT-17 fixed). Not built: (b), the automatic derivation of required key inputs from `CompositeTargetDiscovery.fit`'s parameters and the private attributes it reads; and the train_eval model-cache key perturbation table.

### PMT-16 [P2] polars/pandas carrier parity over row-slicing helpers, plus an order-losing mask-filter scanner (shared)
- **Asserts**:
  - (a) Contract: every helper registered in `FRAME_ROW_SLICERS` (the OOF holdout slicer, `_phase_composite_post_xt_mtr_oof` slices, `_extract_groups`, the base extractor) returns identical rows for pandas and polars with `idx` monotone, reversed and shuffled. An identity component through `compute_oof_holdout_predictions` gives `max|pred - y| == 0` for each `oof_holdout_source`.
  - (b) Static: a function that has both an `.iloc[idx]` branch and a `.filter(mask)` branch, where `mask` is built from an index array (`mask[idx] = True`, `np.isin(..., idx)`), is flagged because the filter drops the index order. `X[idx.tolist()]` on a frame is flagged.
- **Implementation**: (a) is `tests/training/composite/ensemble/test_frame_carrier_parity.py`, with a meta-guard that every function in composite taking `(X, idx)` and branching on `isinstance(X, pl.DataFrame)` is registered. (b) is `py_ci_shared/order_losing_filters.py`, wired in `test_shared_checks_wired.py`.
- **Would have caught** (3): EST-07, PRF-24, TST-09.
- **Proof it fires**: expected: reversed timestamps give polars `max|pred - y| = 39.0` against pandas 0.0 (EST-07); (b) flags `ensemble/__init__.py:782-787` and `_phase_composite_post_xt_mtr_oof.py:43-45`.
- **False-positive risk**: low for (a). (b) is low, because a mask filter is only flagged when it sits next to a positional-index branch built from the same index.
- **Runtime**: (a) about 2 s; (b) under 1 s.
- **Repo**: (a) mlframe; (b) py-ci-shared.
- **Disposition**: PARTIAL. Leg (a) built: test_frame_carrier_parity.py checks OOF holdout alignment with an identity component (pandas vs polars; monotone / reversed / shuffled time; k=1 and 3) and three row slicers against pandas under all three index orders. It found and fixed EST-07 plus the same order-losing mask in `feature_stacking`, `_slice_frame_rows`, `_row_select` and `_subset_rows`. Not built: the meta-guard registering every `(X, idx)` function that branches on polars, and leg (b), the shared `order_losing_filters` scanner. A grep for mask-from-index next to `.iloc` found 7 functions, and the 5 with a real order dependence are fixed.

### PMT-17 [P1] Absorption and consistency on each transform's canonical DGP
- **Asserts**: every registry entry has a `canonical_dgp` factory in `_CANONICAL_DGP` (a meta-guard enforces `keys == registry`, so a new transform must declare one). On that DGP:
  - Residual transforms absorb the base relation: `var(T)/var(y) <= 0.1` at n in {300, 2000} and base offset c in {0, 1e4}.
  - Smoothers improve with n: RMSE(g - truth) at n=20k is at most 0.7x the value at n=2k.
  - Grouped transforms applied to identical per-group distributions give per-group level offsets averaging about 0 (below 0.1 sd).
- **Implementation**: `tests/training/composite/transforms/test_transform_canonical_dgp.py` plus `_canonical_dgp.py` beside it. Violators go into the named known-violator set.
- **Would have caught** (5): TRF-03 (`monotonic_residual` at n=300: var_explained 0.573), TRF-09 (offset base leaves a third of the signal), TRF-13 (`second_diff` 1-D: `T = -y`, ratio about 1), TRF-14 (NW RMSE 0.085 -> 0.076 for 100x data), TRF-15 (lognormal groups all shifted up).
- **Proof it fires**: expected from the transforms.md repro numbers above; each is below its threshold.
- **False-positive risk**: low, since thresholds are set on the DGP each transform is designed for. The cost is writing 51 small DGP factories once.
- **Runtime**: fast subset about 5 s; the NW n=20k cell about 3 s.
- **Repo**: mlframe.
- **Disposition**: PARTIAL. New tests/training/composite/transforms/test_transform_canonical_dgp.py. Every base transform declares a canonical DGP family in `_CANONICAL_DGP` (additive, linear, second difference, product, multiplicative, geometric or saturating), and a meta-guard keeps the keys equal to the base transforms in the registry. Absorption is measured scale-free, at n in {300, 2000} and, for the additive-type families, with the base offset by 1e4: inverting the constant median T at each row's base must explain at least 90% of y's variance (80% for the product family). The test found a TRF-03-class defect in `quantile_residual`, now fixed. At n=300 its ten bins of 30 rows sat under the 50-row minimum, so every bin fell back to the global median and T ignored the base (R2 0.00). The bin count is now `min(n_bins, n // min_bin_n)`, which gives R2 0.96. Not built: the smoother-improves-with-n leg and the grouped-level leg.

### PMT-18 [P1] Self-influence and fit-row disjointness canaries: no row's derived value depends on its own y, and scored rows are never in the params' fit rows
- **Asserts**:
  - (a) Self-influence: for every y-derived producer in `Y_DERIVED_PRODUCERS` (grouped causal bases, target-encoding forward, residual `g = y - T` of every registry transform), perturbing `y_i` by 1 changes the produced value at row i by at most `5/n`. Global in-sample fits are O(1/n); a group-level or own-row leak is O(1).
  - (b) Test-row influence: perturbing non-train rows leaves every train-derived parameter unchanged, and a mis-shaped `train_mask` raises.
  - (c) Fit/score disjointness: every registry transform's `fit` is replaced by a spy recording the fitted rows by tagging y with unique values. The spy runs through `_tiny_cv_rmse_y_scale`, the yscale-gate fallback, `compute_oof_holdout_predictions` and the OOF `pre_pipeline` path. For each scored fold, the params used to forward/inverse the scored rows must come from a fit whose rows are disjoint from them. The same spy applies to supervised selectors inside `pre_pipeline`.
- **Implementation**: `tests/training/composite/test_leakage_canaries.py` with the producer registry in `composite/discovery/__init__.py` (a meta-guard asserts every public function in `_grouped_causal_bases.py` and `_base_engineering.py` is registered).
- **Would have caught** (7): DSC-01 (first-in-group base == own y, sensitivity 1.0), DSC-04 (tiny CV reuses all-row params), DSC-13 (the gate fallback's params saw the eval groups), TRF-24 (the divisor eps reads test rows; a mis-shaped mask is ignored), TRF-26 (singleton encoding sensitivity 1/21 against a 5/n bound), EST-16 (the grouped refit falls back to global params that include the fold), EST-17 (the supervised FS pipeline was fit on the holdout rows).
- **Proof it fires**: expected: the `[5,6,6]` lag output for the solo row (DSC-01) gives sensitivity 1.0; the spy shows `_screening_tiny_perbin.py:203/220/338` scoring rows inside the `_eval.py:344` fit set (DSC-04).
- **False-positive risk**: low. The 5/n bound separates the O(1/n) influence of a global fit from the O(1) influence of a leak by orders of magnitude at n=2000.
- **Runtime**: about 10-20 s (tiny fits with spies; one discovery fit at n=600).
- **Repo**: mlframe.
- **Disposition**: RESOLVED (leg a). New tests/training/composite/test_leakage_canaries.py with the `Y_DERIVED_PRODUCERS` registry (kept in the test; a meta-guard requires every public function of `_grouped_causal_bases.py` / `_base_engineering.py` to be registered). (1) Every grouped and temporal causal base (lag, trailing/rolling mean and median, expanding mean, diff) is exactly unchanged at row i when `y_i` moves; canary: the legacy `first_fill='group_first'` scores 1.0 on the group heads (DSC-01). (2) The train target encoding over size-5 categories moves by at most 5/n with the row's own y; canary: `oof_folds=None` scores 1/(5+20) (TRF-26). (3) For every registry transform, refitting with `y_i` moved shifts row i's reconstruction `inverse(T_i, base_i)` by under 0.1 of the move. The spec's 5/n bound does not hold for legitimate local fits. Measured at n=1000/4000, the rank, ECDF, copula, quantile_normal, quantile/median-residual and smoothing-spline fits sit at 0.02-0.07, flat in n, while global fits shrink as 1/n. The spline's flat 0.04 comes from its knot count growing with n; the smoother `s = (m + sqrt(2m)) var` fixes that but fits the true curve worse in every case measured, so it is recorded as rejected in extended.py. Not built: legs (b) (TRF-24's mask checks have their own tests) and (c) the fit/score-disjointness spy, which targets the open DSC-13 and EST-17 and will be built with those fixes. COMPLETED (2026-09-24): leg (c): tests/training/composite/test_fit_score_disjointness.py replaces the registry transform's fit/forward/inverse with a spy; bases are unique, each fit stamps its params with its call id and records the bases it saw, each inverse records which fit's params scored which bases. The tiny CV and the OOF holdout path (linear_residual and linear_residual_grouped) score every row with params that never saw it; the y-scale gate's no-val fallback always includes a leak-free reconstruction. Each path has a canary that disables its per-fold refit (DSC-04, EST-16, DSC-13) and must show the overlap; EST-17's supervised pre_pipeline leg is covered by test_oof_refits_supervised_selection.py. Leg (b): test_leakage_canaries.py blows up every row outside train_mask by 1e3 and requires every train_mask taker to learn the identical result; a mis-shaped mask must raise; a meta-guard requires every public composite function with a train_mask parameter to be registered. Leg (b) found a live leak: score_interaction_pairs / discover_interaction_bases took their MI on every row (the mask reached only the div eps floor), so pairs were chosen by MI that read the test targets; they now score the mask rows only and reject a mis-shaped mask.

### PMT-19 [P2] Null-DGP selection canaries: every selection routine picks the null on pure noise
- **Asserts**: for every routine in `NULL_CANARIES`, run across 10 seeds on data with no signal, the null is chosen in at least 9 of 10:
  - `seasonal_residual` period selection picks no seasonality (or 1), and period-12 data picks 12, not 24.
  - `signed_power_y` fits `p == 1` on symmetric y.
  - The stability check with `n_bootstrap_runs=3` drops a spec that appears once.
  - `discover_chains` on a raw-beats-all DGP returns `[]`.
  - `discover_incremental` on appended rows with a permuted base returns `reuse=False`.
  - The default-config discovery on pure noise emits no spec, or reports finite FDR p-values.
- **Implementation**: `tests/training/composite/test_null_canaries.py`. A meta-guard asserts that every function under `composite/` containing `argmin`/`argmax` over a candidate grid, or named `*select*|*choose*|*pick*`, is registered or allowlisted.
- **Would have caught** (7): TRF-11 (noise picks 52 in 20/20 seeds), TRF-19 (identity never on the grid), DSC-15 (`int(0.6*3) = 1`), DSC-16 (drift cannot fire under `eps=-10`), DSC-20 (chains surface when worse than raw), DSC-24 (FDR inert with `bootstrap_n=0`), TST-11 (the pure-noise test never asserted `specs == []`).
- **Proof it fires**: expected from the quoted repros; each routine picks a non-null on noise today.
- **False-positive risk**: low with the 9/10 majority. The seeds are fixed, so the test is deterministic.
- **Runtime**: about 15-30 s (the discovery-on-noise cells dominate; one fit at n=600).
- **Repo**: mlframe.
- **Disposition**: RESOLVED. New tests/training/composite/test_null_canaries.py, deterministic seeds. The seasonal period is 1 on noise and 12 on a period-12 signal, `signed_power_y` fits p = 1 on symmetric y, and the stability check drops a spec found in one run of three (DSC-15 fixed here). `test_default_discovery_emits_no_spec_on_pure_noise` (slow) requires zero specs on all 10 noise seeds, and it found a real defect. The honest RMSE gate compared composites only with the raw tiny model, which overfits noise and loses to the constant mean, so seed 1 shipped 10 noise specs. The gate now also rejects a spec whose honest y-RMSE does not beat the constant train mean (`_no_better_than_constant`, ledgered); `DISCOVERY_ALGO_VERSION` was bumped to 2 because selection changed. The MI winner's-curse test, which needs noise specs to reach the report, now disables that gate explicitly. Covered elsewhere: drift on a permuted base (test_incremental_discovery.py, DSC-16), inactive FDR is reported (DSC-24). Not built: `discover_chains` returning [] when raw beats every chain (DSC-20 is open) and the meta-guard registering every argmin / select routine. COMPLETED (2026-09-24): discover_chains returns [] on a raw-beats-all DGP in 10/10 seeds (DSC-20 fixed); canaries added for the change-point scan, the count-weighted blend's k and from_train_metrics' fallback; and the meta-guard registers every composite function named like a selector or computing an argmin/argmax (27) as either a NULL_CANARIES routine with its canary or a _NOT_SELECTION entry with a reason, failing on unregistered and stale entries. The MoE canary found a live defect: with shrink_rtol=0 and min_group_rows=1 the gate chose per group by noise - experts 10% noisier than lag won half the groups and served 5% worse than lag on fresh rows. A non-lag expert now needs a paired-gain z >= 2 over lag in a group of >= 20 rows (new defaults moe_gate_min_gain_z=2.0, moe_gate_min_group_rows=20); on the same noise the gate serves lag's RMSE, a genuinely better expert still wins (0.70 of lag), and the choice-mechanics unit tests set min_group_rows=1, min_gain_z=0 explicitly.

### PMT-20 [P1] Liveness registry for default-ON mechanisms: every corrective default must change something on the default path
- **Asserts**: a `DEFAULT_ON_MECHANISMS` table maps each default-ON knob to a probe that runs the DEFAULT construction path (the suite's `from_fitted_inner`, the default `oof_holdout_source="kfold"`, the default discovery config). Each probe asserts an observable effect:
  - `soft_base_shrink`: `n_shrunk > 0` on a base 10 IQRs out.
  - `mi_gain_fdr_control`: finite p-values, or an INFO record saying FDR is inactive.
  - Drift detection: `reuse=False` on a destroyed relation.
  - OOF pre-screen: invoked under kfold (the refit spy count drops).
  - Wrap-pass watchdog: executed under `skip_wrap_pass_predict=True`.
  - knn budget guard: fires before the first Kraskov call.
  - Auto-enabled discovery: the effective config reaches post-processing (the unit-level phase function returns `enabled=True`).
  - A meta-guard requires every field of `CompositeTargetDiscoveryConfig` and ctor param of `CompositeTargetEstimator` that defaults to `True` and matches `*_enabled|*_control|soft_*|enable_*|*watchdog*|*guard*` to be in the table or allowlisted with a reason.
- **Implementation**: `tests/training/composite/test_default_on_liveness.py`. The field enumeration reuses the `_all_config_classes()` logic of `test_config_field_consumption.py`.
- **Would have caught** (7): EST-09, DSC-24, DSC-16, PRF-04, EST-08 (a), DSC-29, INT-04.
- **Proof it fires**: expected: `from_fitted_inner` gives `n_shrunk == 0` (EST-09, pinned by `test_from_fitted_inner_has_no_range_and_is_noop`); `_ext_X is None` under kfold, so the pre-screen is skipped (PRF-04); `_maybe_auto_enable_discovery`'s copy stays local (INT-04).
- **False-positive risk**: low. A knob that is legitimately inert on some path must say so in the allowlist, which is the point.
- **Runtime**: about 15-30 s (7 small probes, n <= 600).
- **Repo**: mlframe.
- **Disposition**: COMPLETED. New tests/training/composite/test_default_on_liveness.py. The meta-guard enumerates every default-True field of `CompositeTargetDiscoveryConfig` and every default-True ctor param of `CompositeTargetEstimator` that matches the corrective-name pattern (15 plus 1). Each must appear in `DEFAULT_ON_MECHANISMS`, which maps it to the test asserting its effect with the knob at its default, or in `INERT_BY_DESIGN` with a reason. The one inert entry is `enable_multiseed_early_stop`: the default `honest_oof_selection` skips the CV seeds it would cut short. Stale entries fail too. Each liveness test must exist and build at least one configuration with the knob on (an OFF control run beside it is allowed; a canary pins the check). `MECHANISMS_WITHOUT_A_KNOB` pins drift detection, the knn guard before auto-base, the OOF pre-screen under kfold, and auto-enabled discovery reaching post-processing. Found and fixed on the way: DSC-24 (inactive FDR now reported), DSC-16 (drift judged against prior-rows gain), DSC-29 (knn guard before auto-base), and five `CompositeTargetEstimator` MoE ctor params that were never read (removed, ea55262e4). Two mechanisms had no default-path test and got one: the honest-OOF floor (probe in this file) and the ct-ensemble dummy floor (carved into `apply_dummy_floor_gate`, 10e902b7e). The probes live in the tests where each fix landed rather than being duplicated here.

### PMT-21 [P1] Persist-after-mutate phase order: nothing mutates a persisted model or metadata after the last save (AST)
- **Asserts**:
  - (a) In the suite orchestrators (`core/_main_train_suite*.py`), every call to a persisting function (`finalize_suite`, `_finalize_and_save_metadata`, `_persist_ct_ensemble_entries`) comes after every call to a function that mutates `models[...]`, `metadata[...]` or `entry.model`. The mutators are derived by an AST walk one call level deep (functions assigning into those subscripts or attributes).
  - (b) Every assignment `entry.model = <wrapper>` in `training/core` is followed, in phase order, by a persist call covering that entry.
- **Implementation**: `tests/test_meta/test_persist_after_mutate.py` (`_shared_ast_cache`, `assert_scanned_enough` with a floor sized to the core package).
- **Would have caught** (2): INT-02 (`finalize_suite(ctx)` at `_main_train_suite_phases.py:394`, then `run_composite_post_processing` at `:396`), INT-01 (the wrap in `_phase_composite_wrapping.py:247,418` happens after `train_eval.py:541` has saved).
- **Proof it fires**: measured order from the source: line 394 persists, line 396 mutates, and nothing is saved afterwards.
- **False-positive risk**: low. A post-save mutation that is intentionally not persisted (a transient report) is allowlisted with a reason.
- **Runtime**: under 1 s.
- **Repo**: mlframe.
- **Disposition**: COMPLETED. New tests/test_meta/test_persist_after_mutate.py. An AST walk of `training/core` classifies mutators: functions that assign into `models[...]` / `metadata[...]` (bare or as an attribute) or set `.model`, closed transitively over calls, so the composite wrap in `_run_composite_target_wrapping` reaches `run_composite_post_processing`. Persisters are the save entry points plus any function whose last save follows its last mutation. In every `_main_train_suite*.py` function that saves, no mutator may be called after the last persister; `_ALLOWED_LATE` takes intentional exceptions with a reason (none today). Leg (b), the `entry.model = wrapper` assignments, is covered through that closure: the test asserts the wrap counts as a mutation. Proof: deleting the `persist_after_composite_post(ctx)` re-save (the INT-02 shape) fails the test at `run_recurrent_finalize_and_composite_post:397:run_composite_post_processing`; a canary pins the detector on the same shape.

### PMT-22 [P1] One module-scoped composite suite fixture with discriminating persistence, routing and reporting contracts
- **Asserts**: one tiny suite run (TVT fixture, `mlframe_models=["linear","lgb"]`, `transforms=["linear_residual"]`, `max_total_composite_targets=1`, `data_dir=tmp_path`), plus one precomputed-bundle rerun and one grouped variant with MoE on. After these runs:
  - (a) Every models entry loaded from disk has the in-memory type and predicts identically (INT-01).
  - (b) The on-disk metadata keys are a superset of the in-memory keys, and every in-memory model key has a dump (INT-02).
  - (c) The spec names in metadata equal the trained composite keys, and every capped spec appears in the failures (INT-11).
  - (d) The precomputed rerun trains the same composite keys (INT-06).
  - (e) Every spec's `transform_name` is in the whitelist (INT-07).
  - (f) For an additive spec, `|RMSE_y(recorded) - RMSE_T(inner)| < 1e-6` on every split, and `predict_from_models` / `predict_mlframe_models_suite` return finite y-scale values for every composite key (INT-03, EST-01 through the linear strategy's scaler).
  - (g) The composite rows of the targets-performance table equal `composite_target_y_scale_metrics` (INT-10).
  - (h) The tags come from the spec set: the chain spec gets `MTRESID` (INT-08).
  - (i) In the grouped variant, `cross_target_ensemble_metrics` test RMSE equals the shipped `entries[0].model.predict(test)` (EST-13).
  - (j) No composite phase logs "failed" at WARNING (TST-01 e).
- **Implementation**: rewrite `tests/training/composite/test_composite_integration.py` around a `scope="module"` fixture (TST-17). Add a composite-on variant of `tests/inference/test_predict_round_trip_parity.py`.
- **Would have caught** (13): INT-01, INT-02, INT-03, INT-06, INT-07, INT-08, INT-10, INT-11, EST-01, EST-13, TST-01, TST-02, TST-17.
- **Proof it fires**: the suite_integration.md repros were run on this exact fixture: the dumps deserialise to bare `Ridge` (INT-01); no `CT_ENSEMBLE.dump` is written (INT-02); y RMSE is 0.607 against T RMSE 0.334 (INT-03); the chain spec is trained under the whitelist (INT-07).
- **False-positive risk**: low. The assertions are identities, not tuned thresholds.
- **Runtime**: about 90-180 s for 3 suite runs. That replaces the 8 runs (620 s recorded) of the current file, so it is a net saving. Not a meta-test.
- **Repo**: mlframe.
- **Disposition**: PARTIAL. New tests/training/composite/test_composite_suite_contracts.py runs ONE module-scoped suite (TVT fixture, linear + lgb, `transforms=['linear_residual','additive_residual']`, `max_total_composite_targets=1`, 49 s for all seven tests). Built legs: (a) every trained slot reloads via `load_mlframe_suite` with the same model types, and `predict_from_models` on the reloaded suite equals the in-memory one key by key; (b) the saved metadata covers every returned key, and the CT-ensemble metrics match; (c) exported spec names equal the trained composite keys, and the capped spec is a failure with the cap reason; (e) every spec's transform is in the whitelist; (f) `predict_mlframe_models_suite` composite predictions are finite and on the y scale; (g) the targets-table composite rows carry `scale='y'` and the recorded y-scale test RMSE; (j) no composite logger emits a failure at WARNING. Proof: deleting `persist_after_composite_post(ctx)` fails (a) with `_CT_ENSEMBLE__target` missing on disk. Fixed to make the legs hold: INT-07, INT-08, INT-11, EST-13. Not built: (d) the precomputed-bundle rerun; the additive `RMSE_y == RMSE_T` identity in (f); (h) and (i) as suite variants (unit-level in test_extreme_ar_skip_decision.py and test_composite_post_moe_value_report.py); and the TST-17 rewrite of test_composite_integration.py onto this fixture.

### PMT-23 [P2] State parity across alternate constructors: fit() vs from_fitted_inner() vs update() vs unpickle
- **Asserts**: for every registry transform, fit a CTE on tiny data, then build `from_fitted_inner` from the same inner and spec.
  - (a) `set(fitted_params_)`, the public trailing-underscore attributes, and `get_params()` are equal, apart from a declared exemption list.
  - (b) After an `update()` refit, every data-derived key that `fit()` computes (`base_fit_range`, the T-clip envelope, the y-clip) equals a fresh `fit()` on the buffer. The spy compares each key's derivation helper.
  - (c) Grouped and continuation transforms constructed through `from_fitted_inner` predict on the y scale (they must not raise "requires groups").
- **Implementation**: `tests/training/composite/estimator/test_constructor_state_parity.py`, parametrised over `list_transforms()`.
- **Would have caught** (3): EST-09 (`base_fit_range` and `target_name_` are missing), EST-14 (`update` leaves `base_fit_range` stale and uses a different T-clip formula), EST-21 (a grouped spec cannot be expressed).
- **Proof it fires**: expected from `_from_fitted.py:114-121` (no `base_fit_range`) and `_update.py:160-197`.
- **False-positive risk**: low. Legitimately fit-only diagnostics go in the exemption list with reasons.
- **Runtime**: about 3-5 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED. New tests/training/composite/estimator/test_constructor_state_parity.py, parametrised over all 51 registry transforms. `from_fitted_inner` must match `fit` on the public fitted attributes (exempt: `inner_pre_pipeline_`, with its reason), the fitted-param keys, the T-clip band and the predictions. It found a real defect: without a stamped envelope, `from_fitted_inner` clipped T to `+/-10 std(y)` even when the train base was supplied. For 42 transforms that band was tens of times too wide (+/-48 vs [-0.17, 0.18] for asinh_residual) or off-centre (ratio [0.6, 4.8]), so the clip protected nothing. The fix is new `_reconstruct_t_train`: any transform whose inputs are present (y; plus `base_train` for base-dependent ones, 2-D for multi-base; plus the new `groups_train` for grouped ones) gets the exact train T through `call_transform` and the same MAD envelope as `fit`. The deployed wrapper and the OOF refit wraps pass groups. 42 of 51 cases fail before the fix. Not built: leg (b), `update()` parity (EST-14 is open), and the unpickle leg (the pickle round-trip suites cover pickling separately). COMPLETED (2026-09-24): leg (b): after a drift refit every data-derived key (alpha, beta, the y-clip, the median, the T-clip, the base range) equals a fresh fit() on the same rows to 1e-9, for linear_residual and linear_residual_robust. It found that update() refit linear_residual_robust with the drift check's closed-form OLS, so a robust spec served least-squares coefficients after every refit (beta 0.603 vs the robust 0.506); _apply_drift_refit now takes the coefficients from the transform's own fit on the live segment (from the reported change point) and derives every envelope from that segment. The unpickle leg pins public attributes, fitted-param keys and predictions for all 51 registry transforms.

### PMT-24 [P1] Unseen-key fallback property for every router and grouped component
- **Asserts**: for every class in `UNSEEN_KEY_ROUTERS` (the MoE gate, the OOD lag router, the volatility router, `per_group_router`, every `requires_groups` transform):
  - (a) Fitted on keys 0-29 and predicted on keys 100-129, the fallback's RMSE is at most 1.02x the pooled-best expert measured on the fit data.
  - (b) For grouped recurrent transforms under continuation, the seed of an unseen group equals the ungrouped continuation seed.
  - (c) Unseen-group rows equal `inverse(T_hat, base, global params)` to 1e-12.
- **Implementation**: `tests/training/composite/test_unseen_key_fallback.py`. A meta-guard asserts every class with an attribute named `global_choice_|fallback_*|_global_*` is registered.
- **Would have caught** (3): EST-02 (gated RMSE 8.00 against 0.98), TRF-22 (`tail_anchor = anchor`), TST-08 (unseen-group tests asserting only finiteness).
- **Proof it fires**: expected from estimator_ensemble.md's 30-group repro and `_grouped_extra.py:63,223`.
- **False-positive risk**: low.
- **Runtime**: about 3 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED. New tests/training/composite/test_unseen_key_fallback.py. The meta-guard requires every class that stores a group-keyed global fallback (`global_choice_|_global_idx|global_estimator_|_global_prior`) to be in `UNSEEN_KEY_ROUTERS` with the test that pins its unseen-key answer. The spec's `fallback_*` pattern was dropped because it matched the `fallback_predict` constructor parameter of six estimators, which is an OOD-row policy, not key routing. The OOD and volatility lag routers are excluded with the reason (they route on base range and local volatility, not a key table). Legs: (a) the MoE gate, fitted on groups 0-29 and served groups 100-129, is within 1.02x the pooled-best expert on the same rows. Proof: forcing unseen rows to lag reproduces EST-02 exactly (8.05 vs 0.98) and fails. (c) linear/monotonic/quantile grouped and target encoding reproduce the inverse from their stored global parameters to 1e-12. All seven grouped transforms give label-independent unseen-group inverses. `PerGroupCompositeRouter` serves unseen groups exactly `global_estimator_`, and `LeakageSafeEncoder` encodes unseen categories to `_global_prior` for three methods. Not built: leg (b), that a grouped recurrent transform's unseen-group seed equals the ungrouped continuation seed; only label-independence covers the three recurrent transforms. COMPLETED (2026-09-24): leg (b): for each recurrent grouped transform (ewma_residual_grouped, frac_diff_grouped, rolling_quantile_ratio_grouped), under recurrence continuation an unseen group's inverse must equal the ungrouped twin's continuation inverse to 1e-12, on a trended y where a cold seed differs (canary). A meta-check requires the recurrent grouped set to be exactly the twin table. The leg found rolling_quantile_ratio_grouped seeding an unseen group with an empty window (6/40 rows off, up to 1.8); it now continues from the ungrouped tail_base.

### PMT-25 [P2] Authoritative-source scanner: no name heuristics or unchecked target-slot writes where a registry or spec set exists
- **Asserts**:
  - (a) Calls to `is_composite_target_name(` outside `naming.py` and the legacy-pickle loader are flagged; consumers must use `is_composite_target(ctx_or_metadata, tt, name)` backed by the spec-name set.
  - (b) Direct subscript assignment `target_by_type[...][...] = ...` outside an `insert_target(...)` helper is flagged. The helper raises or suffixes on a name collision.
  - (c) Adapter/tuple objects built from the registry functions in more than one module are flagged, so there is one construction site.
- **Implementation**: `tests/test_meta/test_composite_authoritative_sources.py` (AST).
- **Would have caught** (4): INT-08, INT-18, TRF-25 (the six adapter tuples rebuilt in `_registry_extended.py:172-201`), TRF-17 (the registry description string duplicates a formula owned by the code: a registry `description` that states a formula must name the constant or function it reads, checked through a docstring/identifier parity rule reusing `py_ci_shared.doc_identifier_parity`).
- **Proof it fires**: measured: 6 production call sites of `is_composite_target_name(`; `_phase_composite_discovery.py:926-927` writes the slot directly.
- **False-positive risk**: low for (a) and (b). (c) and the description rule may need a couple of allowlist entries.
- **Runtime**: under 1 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED - (a) and (b) ship as tests/test_meta/test_composite_authoritative_sources.py: the name heuristic is allowed only in naming.py, and a double-subscript write into a target-slot dict only in _target_slots.py (the scan fails at HEAD on _phase_composite_discovery.py:859). (c) is not implemented: TRF-25 and TRF-17 are already fixed, and a rule against rebuilding registry adapter tuples has no second site left to anchor on, so it would be an allowlist with no subject COMPLETED (2026-09-24): (c) built in the same file: Transform(...) may be constructed only in registry.py, _registry_extended.py and the chain factories in nonlinear.py, and a registry fit function may be wrapped into a Transform in one module only (the adapter's identity is its fit; linear_residual_multi_robust legitimately shares the multi-base forward/inverse under a robust fit, which a first version keyed on every function flagged). A canary on a temp tree reports both a build outside the registry modules and a fit wrapped in two registry modules; a subject check requires 40+ builds in the allowed modules.

### PMT-26 [P2] Frame-copy scanner for per-target loops, plus a pandas-2.x shared-memory test
- **Asserts**:
  - (a) In `training/core` and `composite`, `pd.concat([df[cols], s], axis=1)`, `df[list_of_cols]` fed into `concat`, and `.copy()` / `.clone()` on a frame-typed name are flagged unless the line carries `# frame-copy: <reason>`.
  - (b) Dynamic, in the pandas-2.x CI matrix: `np.shares_memory` holds between a feature column of the discovery frame and `filtered_train_df`.
- **Implementation**: (a) is `tests/test_meta/test_no_frame_copy_in_target_loops.py` with baseline `_frame_copy_baseline.json`. (b) is `tests/training/core/test_discovery_frame_zero_copy.py`, marked for the pandas<3 matrix leg.
- **Would have caught** (2): INT-12, PRF-15 (the same site, `_phase_composite_discovery_helpers.py:88-111`).
- **Proof it fires**: expected hit at `_phase_composite_discovery_helpers.py:88-111`. (b) fails on pandas 2.x without copy-on-write.
- **False-positive risk**: medium. Legitimate small-frame copies exist, handled by the marker and the baseline.
- **Runtime**: (a) under 1 s; (b) about 1 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED - (a) is tests/test_meta/test_no_frame_copy_in_target_loops.py with _frame_copy_baseline.json, which is empty: composite and core hold no frame copy at all now, since feature_stacking and gated_regression_mixture were moved onto append_column too (a shallow copy(deep=False) is not counted, and a deliberate copy opts out with '# frame-copy: <reason>'). (b) is test_discovery_frame_zero_copy.py, unmarked because the assertion holds on every supported pandas, not only the 2.x leg

### PMT-27 [P3] Diagnostics truthfulness: report reasons come from the ledger, printed advice is executed, alert policy matches its docstring
- **Asserts**:
  - (a) For every stage in the rejection-ledger stage registry, forcing a rejection at that stage makes `report()` name that stage. Registry-driven, so a new gate is covered by default.
  - (b) Every log string that recommends an action (the regex `pass it via|set \w+=|increase \w+|disable \w+`) is listed in `PRINTED_ADVICE_TESTS` with a test that performs the advice and observes the promised effect (py-ci-shared WRITING_TESTS habit 6).
  - (c) The env-signature drift check warns on a major/minor difference and not on a patch-only one.
- **Implementation**: `tests/training/composite/discovery/test_report_reasons_from_ledger.py` for (a). (b) is a new `py_ci_shared/printed_advice.py`: it finds advice strings and fails when an advice literal has no registered test. (c) is a unit test next to `core/predict.py`.
- **Would have caught** (3): DSC-23, DSC-26, INT-19. It also flags the false "(re-normalising)" wording of EST-03.
- **Proof it fires**: expected: a yscale-gate rejection reports "dropped after the MI gate by a downstream filter (top_k_after_mi trim ...)" (DSC-23); the `_filter.py:251-262` advice has no working path (DSC-26); a `pandas 2.2.2 -> 2.2.3` difference warns (INT-19).
- **False-positive risk**: medium for (b): the regex will match some informational text, handled through an allowlist.
- **Runtime**: about 3 s.
- **Repo**: (a) and (c) mlframe; (b) py-ci-shared.
- **Disposition**: PARTIAL - (a) ships as tests/training/composite/discovery/test_report_reasons_from_ledger.py, parametrised over the RejectStage vocabulary so a new gate is covered as soon as it writes a ledger row, and (c) as tests/training/core/test_env_signature_drift.py. (b), the printed-advice scanner, belongs in py-ci-shared and is still owed; the two advice strings this audit named (DSC-26, INT-19) are fixed, so it has no failing subject left here

### PMT-28 [P3] Test timing and cost hygiene: relative timing races need real slack, and repeated heavy trainings share a fixture
- **Asserts**:
  - (a) Extend `test_no_single_shot_timing_assertion.py`: a relative timing comparison `t_a <= t_b * k` in a test needs `k >= 1.25`, interleaved legs, or the `perf` marker. A duration compared inside an assert with arithmetic on measured seconds (the HPO pruner case) is flagged.
  - (b) A test module where three or more test functions call `train_mlframe_models_suite` directly (not through a fixture) is flagged.
- **Implementation**: (a) extends the existing local scanner and its baseline. (b) is `tests/test_meta/test_heavy_training_shares_fixture.py`.
- **Would have caught** (2): TST-16 (`nj_t <= np_t * 1.05`), TST-17 (8 direct suite trainings in `test_composite_integration.py`).
- **Proof it fires**: measured: `_single_shot_timing_baseline.json` holds only one composite entry (`test_biz_val_grouped_causal_bases.py:380`), so the current scanner does not see the 1.05 race. Expected hits at `test_screening_mi_njit_bit_identity.py:136-163` and `test_composite_integration.py`.
- **False-positive risk**: low.
- **Runtime**: under 2 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED - (a) test_no_single_shot_timing_assertion.py now sees measurements taken through a local timing helper (the njit sentinel's shape) and flags a relative race t_a <cmp> t_b * k whose slack max(k, 1/k) is under 1.25, best-of-N or not, unless the test is marked perf; any ratio of durations read off a result object in seconds (the HPO pruner case) is flagged whatever k is. Canary cases cover each shape. Checked against the previous detector on the same tree: a strict superset (26 sites against 22, none lost). The refresh also dropped 41 stale baseline entries that other changes had fixed or shifted (the test only warned about those). In the composite scope the grouped-causal profile smoke moved to best of three; the njit sentinel and the HPO test were already fixed under TST-16. (b) test_heavy_training_shares_fixture.py records modules whose test functions train the full suite directly three or more times (_heavy_training_baseline.json, may only shrink); test_composite_integration.py drops out of it: the OOF-gate and dummy-baseline tests now read the shared fixture too, leaving only the two negative controls, which need configs of their own

### PMT-29 [P1] Split-role ledger: selection rows and report rows never overlap, and verdicts read test
- **Asserts**:
  - A lightweight production ledger (`composite/_row_roles.py`) records, when `MLFRAME_ROW_ROLE_LEDGER=1` or a test fixture enables it, each consumer that reads a named row set (`honest_holdout`, `val`, `oof_rows`, `test`) and its role (`fit`, `select`, `report`, `verdict`, `plot`).
  - The contract test runs one tiny grouped discovery fit plus the xt-ensemble builder on stubs and asserts:
    - `honest_holdout` has no `select` consumer when it has a `report` consumer (DSC-03);
    - no row set is both `fit` and `report` for the same decision (EST-05: the weights and the gate on one OOF matrix);
    - `verdict` consumers read only `test` (DSC-18, EST-22: val-chosen routers reported on val);
    - `plot` consumers in train-time diagnostics read only train rows (INT-14).
- **Implementation**: a ledger of about 60 LOC plus `tests/training/composite/test_row_role_ledger.py`. The ledger is a no-op unless enabled, so production cost is zero.
- **Would have caught** (5): DSC-03, DSC-18, EST-05, EST-22, INT-14.
- **Proof it fires**: expected: `honest_holdout_idx_` is read by `_honest_rmse_gate`, `_tiny_rerank` (rank) and `apply_honest_oof_floor` (select), and by `rescore_specs_on_holdout` (report) (DSC-03). The verdict at `_phase_composite_post_summary.py:190-257` reads val (DSC-18).
- **False-positive risk**: low once roles are annotated. The work is annotating the consumer call sites (about 15).
- **Runtime**: about 10-15 s (one tiny discovery fit plus the stub ensemble build).
- **Repo**: mlframe.
- **Disposition**: RESOLVED. New `composite/_row_roles.py`: `note_rows(row_set, role, consumer, rows)` records a read when `MLFRAME_ROW_ROLE_LEDGER=1` or a test forces it, and is a single boolean check otherwise. Annotated consumers: `apply_honest_rmse_gate` and `honest_oof_reconstruction_rmse` (honest_holdout, select), `apply_honest_holdout` (honest_holdout, report), `format_composite_vs_raw_block` (test, verdict) and the discovery chart (train, plot). tests/training/composite/test_row_role_ledger.py checks the contracts on one grouped discovery fit: the honest-holdout rows read to select are disjoint from the rows read to report; verdict consumers read only test; the discovery chart reads exactly its train rows; the ledger is off by default and rejects unknown roles. Proof: feeding the gate the whole holdout (the DSC-03 shape) fails with 450 rows both selected and reported. Fixed first so the legs could hold: DSC-18 (verdict on test), EST-22 (val-selected metrics flagged), INT-14 (charts on train). Not built: the `fit`-vs-`report` leg for the xt-ensemble OOF matrix (EST-05 is fixed by cross-fitting, not yet annotated); EST-22's val selection is carried as `val_selection_biased` in metadata rather than as ledger reads. COMPLETED (2026-09-24): the xt fit/report leg: the cross-fitted stack gate notes, per fold, the OOF rows its weights were fitted on (fit) and the rows it scores (report) under one decision id, and its in-sample fallback (too few rows to cross-fit) notes fit and report on the same rows. test_row_role_ledger.py checks that every fold's fit and report rows are disjoint and every OOF row is reported exactly once; the canary drives a 12-row matrix into the in-sample fallback and must see all 12 rows collide. EST-22's val selection stays carried as val_selection_biased in metadata: the MoE gate chooses on val by design, and the flag is what the report reads, so a ledger read would restate it without adding a check.

### PMT-30 [P1] Config-restriction and per-candidate isolation contract: every registry transform is accepted, isolated and honoured
- **Asserts**: for each registry transform, a tiny discovery fit with `transforms=[name]`, `group_column` set and `screening="mi"`. The fast subset is one transform per family; the full set runs under `slow`. Each fit must:
  - (a) never raise; an unsupported transform becomes a per-candidate ledger rejection (INT-09);
  - (b) emit specs only of the listed transform family (INT-07);
  - (c) give every exported spec a `base_column` that exists in the input frame or has a persisted rebuild recipe (DSC-02).
- **Implementation**: `tests/training/composite/discovery/test_config_restriction_contract.py`, parametrised over `list_transforms()` with the fast-mode helper.
- **Would have caught** (4): INT-07, INT-09, DSC-02, TST-08 (no discovery test listed a `*_grouped` transform).
- **Proof it fires**: expected: `quantile_residual_grouped` raises `ValueError: groups kwarg is required` (INT-09); `transforms=["linear_residual"]` yields `chain_linear_residual_cbrt` (INT-07); `y__gcausal_lag1` is not in the frame (DSC-02).
- **False-positive risk**: low.
- **Runtime**: fast subset about 15-25 s (about 10 fits at n=300); the full set about 90 s under `slow`.
- **Repo**: mlframe.
- **Disposition**: COMPLETED. New tests/training/composite/discovery/test_config_restriction_contract.py runs `run_composite_target_discovery` with `transforms=[name]` and a group column for every registry transform: 10 family representatives always, the other 42 under `slow` (all 52 pass in 64 s). Each run must (a) record no target-level failure (discovery did not abort), (b) export only specs of the listed family, and (c) name only base columns present in the frame. Found and fixed while building it: with a group column set, discovery aborted for 8 of the 10 representatives with `KeyError: 'y__gcausal_lag1'`. The y-scale gate evaluates on the val frame, which lacked the engineered grouped causal bases. New `grouped_causal_bases_for_frame` builds them on val from `val_y`; they are strictly causal within each group. INT-07 and INT-09 are fixed too.

### PMT-31 [P1] Stage-sentinel inner: wrappers and every predict entry point must feed the inner its own pipeline stage and the base its raw stage
- **Asserts**:
  - A `StageSentinelInner` records a fingerprint of the frame it was fit on (column set, per-column mean/std) and raises in `predict` when the incoming frame's fingerprint differs.
  - A `StageSentinelBase` asserts that the base column arrives at raw scale.
  - Each wrapper class (CTE, `PrePipelinePredictShim`, CT ensemble, `_MoEGatedDeployableModel`) is built around the sentinel behind a `StandardScaler` pipeline, and each entry point is called: `predict_from_models`, `predict_mlframe_models_suite` (in-memory), the wrap-pass metric block, the per-model hook, `_get_train_pred` and the OOF refits. No sentinel may raise.
  - This also replaces the watchdog's tautological check (EST-08 b) with an independent oracle.
- **Implementation**: `tests/training/composite/estimator/test_stage_routing_contract.py`. It reuses the stub-ctx approach of `tests/inference/test_predict_ct_ensemble_save_load.py`, but with real wrapper classes. A meta-guard asserts every class under `composite/` holding both `estimator_` and a base-extraction call is registered.
- **Would have caught** (3): EST-01, INT-03, EST-08.
- **Proof it fires**: expected: RMSE 165.8 (raw X to the inner) and 419.0 (pp X to the base), against the 0.99 oracle (EST-01); `ValueError: Feature names seen at fit time, yet now missing: row_extreme_top1_score` (INT-03).
- **False-positive risk**: low.
- **Runtime**: about 5-10 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED. New tests/training/composite/estimator/test_stage_routing_contract.py. `StageSentinelInner` fingerprints the frame it was fit on (columns, per-column mean/std) and raises on any other stage. The base is checked through the y-scale RMSE against a large-scale raw base: reading it scaled lands 50+ off. Covered entry points: a wrapper carrying `inner_pre_pipeline` (`predict(raw)`); a pipeline-less wrapper through `composite_predict`, the route `predict_from_models` uses; `PrePipelinePredictShim`; and a CT ensemble of shims. Each must reach RMSE < 1 on a y with noise sd 0.5. A canary pins both misroutes. Proof: disabling `composite_predict`'s stage computation (the EST-01 shape) fails with the inner seeing a 152.9-sd mean drift. Not built: the MoE wrapper, the wrap-pass metric block, the per-model hook, `_get_train_pred` and the OOF refits as entry points, and the meta-guard registering every class that holds `estimator_` plus a base read. COMPLETED (2026-09-24): the remaining entry points now run against the sentinel: build_composite_wrapper (the wrap pass and the per-model hook both build through it), run_wrap_watchdog (which predicts the inner itself and would surface a sentinel trip as its warning), emit_per_model_composite_y_scale_test (the entry is left wrapped, predicts at RMSE < 1, and logs no warning), the deployed MoE wrapper over a stage-carrying composite and a shimmed raw model, and the OOF refits (a cloned sentinel refit per fold; the component must survive and its holdout RMSE stay < 1). _get_train_pred predicts through PrePipelinePredictShim, already covered. A meta-guard registers every class under training/ holding self.estimator_ and calling a base extractor (today CompositeTargetEstimator) against its test.

### PMT-32 [P1] Fresh-process persistence round trip for every registry transform and the whole auto-chain name space
- **Asserts**:
  - For every name in `list_transforms()` plus every name the auto-chain proposer can generate (residual x unary grid), build a CTE through `from_fitted_inner` with a trivial inner and save it through `save_mlframe_model`. Also save tiny `CompositeCrossTargetEnsemble` and `_MoEGatedDeployableModel` instances.
  - In ONE subprocess, load everything through the production `_SafeUnpickler` path and predict. Predictions must equal the in-process values.
- **Implementation**: `tests/inference/test_composite_fresh_process_roundtrip.py`, one `subprocess.run` over a temp dir, following the CLAUDE.md "true isolation needs a subprocess" rule.
- **Would have caught** (2): INT-05 (`UnknownTransformError` for `chain_linear_residual_cbrt` in a new interpreter), TST-02.
- **Proof it fires**: expected from the suite_integration.md `dill.load` repro in a new interpreter.
- **False-positive risk**: low.
- **Runtime**: about 10-20 s (one interpreter start with mlframe import, plus 51 tiny builds).
- **Repo**: mlframe.
- **Disposition**: RESOLVED. New tests/inference/test_composite_fresh_process_roundtrip.py wraps every registry transform plus every auto-chain name (`_RESIDUAL_STAGE_NAMES` x `_TAIL_UNARIES`, 57 wrappers in all) around a picklable linear inner via `from_fitted_inner`. Each is saved with `save_mlframe_model`, as is a CT ensemble of a plain and a chain wrapper. ONE subprocess loads everything through `load_mlframe_model` and predicts, and predictions must equal the in-process ones to 1e-9. Proof: disabling the wrapper's chain re-registration (the INT-05 shape) fails 7 of 58 with `UnknownTransformError`. Not built: `_MoEGatedDeployableModel` in the round trip. COMPLETED (2026-09-24): the round trip now also saves the deployed _MoEGatedDeployableModel over the CT ensemble, a raw-y pipeline and the lag column, with a gate fitted per group of g that routes groups to at least two different experts, and the same subprocess loads it and must reproduce its predictions to 1e-9.

### PMT-33 [P1] Runtime registry mutation must have a load-time replay (shared scanner)
- **Asserts**: a function-scope write (`X[k] = ...`, `X.setdefault`, `X.update`, `X.pop`) to a module-level dict named `*REGISTRY*` is flagged unless the enclosing function is:
  - an import-time registration helper (a decorator applied at module scope), or
  - listed as a replay writer that is reachable from `__setstate__` or a `load_*` function in the same package.
- **Implementation**: `py_ci_shared/runtime_registry_mutation.py` (`find_runtime_registry_writes`, `assert_writes_have_replay`), wired in `test_shared_checks_wired.py` with an allowlist of replay writers.
- **Would have caught** (1): INT-05.
- **Proof it fires**: measured: exactly 2 function-scope writes in composite. One is `_opt_in_steps.py:248` (discovery-time registration, flagged). The other is `_auto_chain.py:200` inside `reregister_auto_chain_transforms`, which is only reachable from the cache-replay branch, not from `__setstate__` or a loader, so it is flagged as well until the replay is wired.
- **False-positive risk**: low (2 sites in the package).
- **Runtime**: under 1 s.
- **Repo**: py-ci-shared.
- **Disposition**: COMPLETED. New py-ci-shared module `runtime_registry_mutation` (py-ci-shared 9117175, 6 unit tests, README section). It flags every function-scope write (subscript store, `setdefault`, `update`, `pop`, `del`) to a module-level `*REGISTRY*` dict defined in any scanned file, counting writes through an import. Helpers used as a module-scope decorator or called in a module-scope statement are exempt, since they run at import. `assert_writes_have_replay` requires a reasoned `replay_writers` entry for every other writer and fails on stale entries. mlframe wiring: `test_runtime_registry_writes_have_a_replay` in test_shared_checks_wired.py over all of src (pin bumped to 9117175). Five writers are listed, each with a reason: `reregister_auto_chain_transforms` (the replay itself), `_run_auto_chain` (replayed on load; proven in a fresh process by PMT-32), the provider cache's `_register_or_get` / `_do_load` (per-process, never named by a pickle) and the `register_metric` plugin API.

### PMT-34 [P2] getattr default parity: `getattr(cfg, "field", literal)` must match the pydantic field default (shared scanner)
- **Asserts**: for every `getattr(<config-ish receiver>, "<field>", <literal>)` in scope, the literal equals the pydantic default of that field on the resolved config class. The receiver is resolved by name convention (`config`, `self.config`, `cfg`, `*_config`) and the class by the field set.
- **Implementation**: `py_ci_shared/config_getattr_default_parity.py`, a sibling of `config_call_site_parity.py` that reuses its constant resolution. Wired in `test_shared_checks_wired.py` with `schema_classes=[CompositeTargetDiscoveryConfig, ...]` and baseline `_getattr_default_parity_baseline.json`.
- **Would have caught** (1): DSC-25 (`_eval_stats.py:285 reject_on_alpha_drift` False vs default True).
- **Proof it fires**: measured: 231 resolved sites in `composite` + `core`, with 46 mismatches, DSC-25 among them. Beyond DSC-25 the probe found latent drift not in the 138 findings:
  - `require_beats_raw_baseline` True vs False at `_tiny_rerank.py:159,235,620,677`;
  - `auto_chain_discovery_enabled` / `interaction_base_discovery_enabled` False vs True at `_fit.py:677-678` and `_opt_in_steps.py:302-303`;
  - `multi_base_enabled` False vs True at `_fit_multibase.py:48,73`;
  - `transform_waic_validation_enabled` False vs True at `_tiny_rerank.py:929`;
  - `tiny_model_n_seed_repeats` 1 vs 3 at `_tiny_rerank.py:216,243`;
  - `mi_sample_strategy` "random" vs "stratified_quantile" at 5 sites;
  - `max_total_composite_targets` None vs 25 at `_phase_composite_discovery.py:912`;
  - `random_state` 0 vs 42 at about 12 sites.

  Like DSC-25, these matter only for duck-typed configs, but each needs a disposition. I list them here so they are not dropped.
- **False-positive risk**: medium. `random_state` and similar names shared across config classes can resolve to the wrong class; the scanner restricts receivers to discovery-config call sites and baselines the rest.
- **Runtime**: under 2 s.
- **Repo**: py-ci-shared.
- **Disposition**: RESOLVED - py-ci-shared config_getattr_default_parity (commit b68ed93, 7 unit tests, README section) compares each getattr fallback with the field's declared default, skipping required, default_factory and cross-schema-conflicting fields, and takes the receiver names of the config being checked so a shared field name is not read off another config. Wired as test_getattr_defaults_match_the_discovery_config over composite and the composite core phases with an EMPTY allowlist: all 49 sites it found were aligned to the config's own defaults, the listed drifts included (require_beats_raw_baseline x4, auto_chain/interaction/multi_base enabled, transform_waic_validation_enabled, tiny_model_n_seed_repeats x2, mi_sample_strategy x4, max_total_composite_targets, min_honest_gain_to_train, cross_target_ensemble_strategy, skip_wrap_pass_predict, oof_holdout_frac, oof_max_train_rows, random_state x17 and the rest). The two ``enabled`` hits the probe reported were calibration and conformal configs, not this one

### PMT-35 [P3] Unread constructor parameters in estimator classes (shared scanner)
- **Asserts**: every `__init__` parameter of a class is either used in the `__init__` body beyond `self.p = p`, or read as `self.p` / `getattr(self, "p")` (or via an `est`/`estimator`/`wrapper` receiver) somewhere in the class's package.
- **Implementation**: `py_ci_shared/unread_init_params.py`, wired in `test_shared_checks_wired.py` for `src/mlframe`, with an allowlist for sklearn meta-params consumed by `get_params` only.
- **Would have caught** (1): EST-18.
- **Proof it fires**: measured. The package-scoped rule on `composite/` returned 10 hits: the 5 EST-18 params (`moe_gate_enabled`, `moe_shrink_rtol`, `moe_tie_rtol`, `moe_min_group_rows`, `moe_failsafe`) plus 5 more. Of the 5 extra, `conformal_ood_adaptive` is read via `getattr(self, ...)` in `conformal.py:599` (outside the package, so the scope must be the composite root). `rolling_rmse_window` and the three `LgbFoldCache` params are consumed inside `__init__`. With the two refinements (whole composite root; use inside `__init__` counts), exactly the 5 EST-18 params remain.
- **False-positive risk**: low after the refinements (measured 0 false positives).
- **Runtime**: under 2 s.
- **Repo**: py-ci-shared.
- **Disposition**: RESOLVED - py-ci-shared unread_init_params (commit 39e9b49, 6 unit tests, README section) reports every __init__ parameter that is only stored on the instance, following renamed stores (self._p = p) and counting a read through any receiver, getattr or a string literal. Wired in mlframe as test_no_unread_constructor_parameters over all of src; the two allowlist entries are a vendored upstream model's signature, and the one live hit outside the audit, FeatureCache.content_fingerprint, was removed rather than allowlisted

### PMT-36 [P3] Environment flags parsed through one shared parser (shared scanner)
- **Asserts**: an `os.environ.get("<PREFIX>_...")` used as a boolean (truthiness, `not`, `== "1"`, membership in an ad-hoc tuple) is flagged. Boolean env flags must go through `env_flag(name, default)`, which accepts `{"1","true","yes","on"}` and rejects `{"0","false","no","off",""}`, taking the project's own prefix, never mlframe's.
- **Implementation**: `py_ci_shared/env_flag_parsing.py`, wired in `test_shared_checks_wired.py` with `prefixes=("MLFRAME_",)`. The helper lives in pyutilz or mlframe utils.
- **Would have caught** (1): INT-16.
- **Proof it fires**: measured sites: `reporting/_reporting_regression/__init__.py:467` (`not os.environ.get(`), `core/_setup_helpers_pipeline_cache.py:158,202` (`== "1"`), `core/_predict_pre_pipeline.py:119` (ad-hoc tuple), and `core/_phase_config_setup.py:343` per INT-16.
- **False-positive risk**: low. Non-boolean env reads (paths, numbers) are not in boolean context.
- **Runtime**: under 1 s.
- **Repo**: py-ci-shared.
- **Disposition**: RESOLVED - py-ci-shared env_flag_parsing (commit 56b7fe0, 6 unit tests, README section) reports every prefixed env read used as a boolean: truth test, not, comparison with a string literal, membership in a literal collection. Wired as test_composite_env_flags_go_through_one_parser over composite, core and reporting, with two allowlisted numeric-override presence checks; the wider tree still carries hand parses outside this audit's scope, and the scanner is there for them

### PMT-37 [P3] Deferred-dead config fields must warn when set, and no allowlisted field may advertise how to enable it
- **Asserts**: for every entry in `_USER_DEFERRED_DEAD` (in `test_config_field_consumption.py`):
  - (a) constructing the config with a non-default value emits a `UserWarning` / `DeprecationWarning`;
  - (b) the field's description does not contain enabling instructions (`Enable by|set to .* to enable|> 0.0`).
- **Implementation**: two tests added to `tests/test_meta/test_config_field_consumption.py`, reusing `_all_config_classes()`.
- **Would have caught** (1): INT-15. It also forces the INT-15 part of TST-04 to be reframed.
- **Proof it fires**: expected: `CompositeTargetDiscoveryConfig(force_inject_diff_on_top_ablation_pct=50.0)` validates silently, and its comment says "Enable by setting > 0.0".
- **False-positive risk**: low.
- **Runtime**: under 1 s.
- **Repo**: mlframe.
- **Disposition**: RESOLVED - (a) and (b) ship as two parametrised tests over _USER_DEFERRED_DEAD in test_config_field_consumption.py. To make (a) pass, every deferred-dead field now warns: InertFieldsWarningMixin (training/_inert_fields.py) carries the composite config's mechanism to the other ten config classes, which declare their dead fields in INERT_FIELDS next to the class rather than only in a test's ledger. The probe walks candidate values of the field's own shape, with a full constructor call for the two fields whose validation reads the rest of the config. The scan also found that _all_config_classes() skipped _model_configs_ensembling and _model_configs_behavior, so EnsemblingConfig, MultilabelDispatchConfig and QuantileRegressionConfig were audited by nothing; both modules are in the set now

### PMT-38 [P1] A config rebuilt with `model_copy(update=...)` must reach its consumers (shared scanner)
- **Asserts**: a function that binds `<cfg>.model_copy(update=...)` to a local must return it, store it on `ctx`/`self`/metadata, or pass it to a callee. Otherwise, when the caller keeps using the original object, the effective config is local-only and is flagged.
- **Implementation**: `py_ci_shared/discarded_model_copy.py`, wired in `test_shared_checks_wired.py`.
- **Would have caught** (1): INT-04.
- **Proof it fires**: expected at `core/_phase_composite_discovery.py:81-116` (`_maybe_auto_enable_discovery`'s copy is used only inside `run_composite_target_discovery`, while `_main_train_suite.py:766` passes the original config onward).
- **False-positive risk**: low to medium. A copy intentionally scoped to one call is allowlisted with a reason.
- **Runtime**: under 1 s.
- **Repo**: py-ci-shared.
- **Disposition**: COMPLETED. New py-ci-shared module `discarded_model_copy` (py-ci-shared b8aaa18, 10 unit tests, README section). It flags a function-scope `name = <expr>.model_copy(update=...)` whose `name`, or a plain alias of it, is never returned or yielded, stored into an attribute or subscript, passed to a call, or used as its own method's receiver. `assert_no_discarded_model_copy` takes reasoned allowlist entries and fails on stale ones. Wired in mlframe as `test_no_discarded_model_copy` over all of src with an empty allowlist (pin bumped to b8aaa18). The one initial hit, `_disc_cfg_base` in `run_composite_target_discovery`, was a false positive: the copy reaches its consumers through the alias `_disc_cfg = _disc_cfg_base`, so the scanner learned to follow aliases. The INT-04 shape (a copy only read locally) is pinned by the scanner's own unit test for a copy used only in a local read (py-ci-shared).

### PMT-39 [P2] Survivorship-scored metrics: a metric computed only on rows where the prediction is finite (shared scanner)
- **Asserts**: a call to a metric function (`rmse|mae|mse|r2|mean_squared_error|...`) whose y_true and y_pred arguments are both indexed by the same mask, where that mask is derived from `np.isfinite(<prediction>)`, is flagged, unless the enclosing function also scores the non-finite rows (fills and rescores) or records the dropped fraction in the returned verdict.
- **Implementation**: `py_ci_shared/survivorship_scoring.py`, wired in `test_shared_checks_wired.py` with baseline `_survivorship_scoring_baseline.json`.
- **Would have caught** (1): DSC-08.
- **Proof it fires**: expected at `_yscale_holdout_gate.py:403-416`, `_honest_rmse_gate.py:180-193`, `_honest_oof_select.py:163-169` and `_auto_chain.py:288-291`.
- **False-positive risk**: medium. Diagnostic helpers that intentionally score finite rows only must return the dropped fraction to pass.
- **Runtime**: under 1 s.
- **Repo**: py-ci-shared.
- **Disposition**: RESOLVED - py-ci-shared survivorship_scoring (commit 41cbadc, 8 unit tests, README section) reports every metric call whose two arguments are indexed by the same isfinite(prediction) mask, accepting a function that fills the dropped rows or reports the dropped fraction in its verdict; counting finite rows for a floor does not count, which is what the four gates did. Verified against the pre-DSC-08 sources: it names all four sites (_yscale_holdout_gate, _honest_rmse_gate, _honest_oof_select and its inner scorer). Wired over all of src with an empty allowlist, where it is now silent

### PMT-40 [P2] `source_text_claims` misses source text accumulated with `+=`: close the taint gap and drain the composite allowlist
- **Asserts**: the shared detector's assertion mode propagates taint through `AugAssign` (`src += path.read_text()`), so an assert on `src` counts as a claim. The composite files currently allowlisted as a whole (`training/composite/discovery/test_training_composite_discovery_fixes.py`, `training/composite/test_training_composite_loose_a_fixes.py`) are converted to behavioural tests and removed from `_ALLOWLIST` in `test_no_source_text_claims.py`.
- **Implementation**: patch `py_ci_shared/source_text_claims.py` (`_Detector._tainted` handles `ast.AugAssign` like `ast.Assign`) with a unit test in `py-ci-shared/tests/test_source_text_claims.py` using the exact shape. Refresh mlframe's `_source_text_baseline.json`.
- **Would have caught** (1): TST-15.
- **Proof it fires**: measured: `find_source_text_claims(tests/inference/test_predict_cte_raw_x.py)` returns `[]` in the default assertion mode, while `mode="read"` reports `line=35, function='test_predict_module_has_cte_raw_x_dispatch', kind='reads a source file'`. The claim reaches its assert only through `src += _p.read_text(...)`.
- **False-positive risk**: low (same semantics as the existing Assign taint).
- **Runtime**: unchanged.
- **Repo**: py-ci-shared (the detector fix); mlframe (allowlist drain).
- **Disposition**: RESOLVED - py-ci-shared source_text_claims taints an AugAssign target like a plain assignment (commit 9ea5fb4, unit test on the accumulate-then-assert shape), which made five previously invisible claims visible; they are recorded in _source_text_baseline.json. Both composite files left _ALLOWLIST: the njit-kernel wiring check is now a monkeypatched kernel whose substitute output has to reach the caller, and the shallow-copy check keeps only its np.shares_memory assertion

### PMT-41 [P3] Advisory scan for tests that pin a conceded defect
- **Asserts**: a test whose docstring or adjacent comment concedes the behaviour is wrong or degenerate (`does not perfectly|degenerates|is inert|no-op|lossy|by design|not renormal|known (bug|defect)`) while asserting exact equality to that behaviour is listed for review. The list is advisory, a reading list rather than a gate, like `mutation_teeth` survivors. The companion convention: a deliberately pinned known defect lives in a test named `test_known_defect_<id>_*`, which the scan accepts and which `disposition_test_references` ties to an OPEN finding.
- **Implementation**: `py_ci_shared/conceded_defect_pins.py` (report-only), wired in `test_shared_checks_wired.py` as a non-failing report with a baseline count ratchet.
- **Would have caught** (1): TST-04. By the phrases it matches, it covers 4 of TST-04's 7 sub-items (EST-03 dropout docstring "does not perfectly reconstruct y", EST-09 `..._is_noop`, TRF-13 `..._degenerates_...`, TRF-06 "tail eps-clip loss").
- **Proof it fires**: expected at `test_composite_ensemble_linear_stack_dropout.py:89-121`, `test_biz_val_soft_base_shrink.py:355-373`, `test_biz_val_second_diff.py:81` and `test_composite_transforms_registry_contract.py:91`.
- **False-positive risk**: high (the phrases are common in honest tests), which is why it only reports and does not gate.
- **Runtime**: under 2 s.
- **Repo**: py-ci-shared.
- **Disposition**: RESOLVED - py-ci-shared conceded_defect_pins (commit 8cebcfd, 6 unit tests, README section) lists test functions whose docstring or leading comment matches the concession phrases and whose body pins a value exactly (== , assert_array_equal, assert_allclose with zero tolerances); a test named test_known_defect_<id>_... is accepted. Wired as test_conceded_defect_pins_in_the_composite_tests_do_not_grow with a two-sided count ratchet at 26: growth fails, and a drop must be recorded. As predicted the list is noisy - most hits say 'degenerate' or 'no-op' about the input of an honest edge-case test - which is why it ratchets a count instead of failing per site. The four TST-04 pins it was built for were fixed with their defects; the dropout test it still lists documents the deployed no-refit policy (EST-03, resolved), not a defect

---

## Summary of placement

- **py-ci-shared (14)**: PMT-03, PMT-04 (rules a/b/c/e), PMT-10, PMT-11, PMT-16 (b), PMT-27 (b), PMT-33, PMT-34, PMT-35, PMT-36, PMT-38, PMT-39, PMT-40, PMT-41. PMT-15 (d) reuses the existing `content_hash_version_bump_gate`.
- **mlframe meta-tests (static, under 3 s each)**: PMT-01 (f), PMT-06 (a), PMT-07 (b), PMT-08 (static companion), PMT-09 (a-c), PMT-14 (a), PMT-21, PMT-25, PMT-26 (a), PMT-28, PMT-37.
- **mlframe registry-driven contract tests** (a new transform, strategy, router or scorer is covered by default through the registry, with a meta-guard that the side registry is complete): PMT-01, PMT-05, PMT-08, PMT-12, PMT-13, PMT-14 (b), PMT-17, PMT-19, PMT-20, PMT-23, PMT-24, PMT-30, PMT-32.
- **mlframe canaries and harnesses with tiny fits**: PMT-02, PMT-06 (b), PMT-18, PMT-29, PMT-31. There is one suite-level fixture, PMT-22, which replaces the existing 8-run integration file.

New transform metadata these proposals need on `Transform` (all defaulted, so existing entries keep working): `n_bases`, `loss_bound`, `additive_in_base`, `scale_equivariant`, `base_translation_invariant`, plus the side tables `_CANONICAL_DGP`, `SPEC_SCORERS`, `DEPLOYABLE_COMPONENTS`, `UNSEEN_KEY_ROUTERS`, `Y_DERIVED_PRODUCERS`, `DEFAULT_ON_MECHANISMS`, `NULL_CANARIES`, `CACHE_KEY_INPUTS` and `FRAME_ROW_SLICERS`. Each side table gets a completeness meta-guard.

New items surfaced while measuring (not among the 138; each needs a disposition): the 45 getattr-default drift sites beyond DSC-25 (PMT-34); the `source_text_claims` AugAssign blind spot in py-ci-shared (PMT-40); `refit_transform_on_fold` admitted into `_uncalled_functions_baseline.json` although it is a leak fix with a test certifying it (PMT-11); and the absence of the gate modules from `_code_audit_baseline.json`, which means the shared `non_neutral_except_fallback` rule does not model the admit-on-error shape (PMT-03).
