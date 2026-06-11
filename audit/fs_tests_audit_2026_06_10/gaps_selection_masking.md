# FS test-suite audit — adversarial gap-finder for SELECTION algorithms (KEY=gaps_selection_masking, 2026-06-10)

Sampling method (so the verifier can replicate): mapped the 468-file suite via `tests/feature_selection/LAYER_INDEX.md` (97 MRMR layer contracts), then ran a scenario-token x selector-family grep matrix and deep-read only the discriminating files: `test_selectors_shared.py` (full), `conftest.py` (fixtures), `LAYER_INDEX.md`, `test_biz_value_mrmr_layer6/11/13/17/19/27.py` (headers+key tests), `test_biz_value_mrmr_quality_metrics.py:300-340`, `test_biz_value_mrmr_hard_cases.py` (test names), `test_biz_val_filters_boruta_shap.py` (full), `test_hetero_vote.py` (full), `test_hybrid_selector*.py` (test names), `test_biz_val_shap_proxied_fs.py` (test names), `test_sample_weights_fs.py` + `test_mrmr_sample_weight_unit.py` + `test_rfecv_sample_weight_unit.py`, `test_wrappers_audit_p1_extra.py` (XOR block), `test_wrappers_audit_p1_stress.py` (B6 dedup), `test_wrappers_rfecv_fi_semantics.py:131-149`, `test_edge_cases_robustness.py` (names), `test_biz_value_mrmr_pre_distortion.py` (header), `test_noise_floor.py` (header), plus src reads of `hetero_vote.py`, `registry.py`, `wrappers/_rfecv.py` (coef_scale_source), `_feature_engineering_pairs/_pairs_core.py` (subsample draw), `boruta_shap` / `shap_proxied_fs` fit signatures. Already-dispositioned items in `audit_disposition.md` (and the known-blocked orth-basis recipe parity / GPU SU gate / shap binning) are NOT re-reported. Expected-to-fail underselection sensors (`test_biz_value_mrmr_underselection.py`) are known open prod bugs, not re-reported.

Key greps run (from `tests/feature_selection/`, recurse, worktrees excluded by cwd):

- G1: `grep -ln " \^ \|xor\|XOR" test_boruta*.py test_hetero_vote.py test_hybrid_selector*.py test_shap_proxied*.py test_biz_val_shap_proxied_fs.py test_mrmr_shap_proxied_pipeline.py` -> EMPTY
- G2: `grep -li "imbalan\|rare_\|minority\|class_weight"` over the same family files -> EMPTY (27 hits suite-wide, all MRMR/wrappers-incidental/conftest)
- G3: `grep -l "sample_weight"` -> 12 files, none of them boruta/shap_prox/hetero/hybrid
- G4: `grep -rln "cauchy\|pareto\|lognormal\|standard_t(\|heavy.tail"` -> 20 files, ALL MRMR/FE-side (none in test_wrappers*/test_boruta*/test_shap_prox*/test_hetero*)
- G5: `grep -rn "pure_noise\|all_noise\|no_signal\|zero_signal\|null dataset"` -> MRMR-side only (layer27, quality_metrics, FE noise controls, permutation MI)
- G6: `grep -rln "columns\[::-1\]\|iloc\[:, ::-1\]\|reversed(.*columns\|permutation(.*columns\|reindex(columns"` -> 5 files; layer19 + shap contract are transform-time reorder, layer89/90/94 are one-hot reindex; NO fit-time selection-invariance test
- G7: `grep -rn "warped.*dup\|monotone.*dup\|rank(x"` -> comments about ENGINEERED near-monotone-duplicate pruning only (layer64:454, layer66)
- G8: `grep -rln "1e9\|e\+09\|1_000_000_000"` -> 9 files, all GPU/perf/byte-cap contexts, no adversarial-scaling selection contract
- G9: `grep -rn "chain"` -> recipe-replay/np.unique-chain contexts only; no redundancy-chain dataset
- G10: `grep -rln "high_dimensional_data"` -> conftest.py + test_wrappers.py only

---

### [gaps_selection_masking-01] No pure-null (zero-signal) FDR contract for RFECV, BorutaShap, ShapProxiedFS, HybridSelector, hetero_vote
Severity: P1
Kind: gap-mask
Files: tests/feature_selection/test_selectors_shared.py:347-359; tests/feature_selection/test_biz_value_mrmr_layer27.py:85-150 (MRMR-only); tests/feature_selection/test_biz_value_mrmr_quality_metrics.py:308-340 (MRMR-only); tests/feature_selection/test_biz_val_filters_boruta_shap.py:17 (has 2 informative cols, not null); tests/feature_selection/test_hetero_vote.py:21 (4 signal cols, not null)
Evidence: the only selector-agnostic null test is `test_all_noise_features` whose entire assertion is `assert hasattr(selector, "n_features_")` (test_selectors_shared.py:359) — a selector that selects ALL 5 noise features passes. Grep G5 shows every quantitative null/FPR assertion lives on the MRMR side. BorutaShap (shadow gate) and hetero_vote (shadow voting) exist PRECISELY for false-positive control, yet neither has a 0-informative-features contract.
Proposal: new `test_null_fdr_contract.py`, parametrized over `registry.available()` (MRMR, RFECV, BorutaShap, ShapProxiedFS) + hetero_vote + HybridSelector. Recipe: `rng=default_rng(seed)`, X = 15 iid N(0,1) cols, n=1000, y = rng.integers(0,2,n), seeds {0,1,2}. Assertions per family: BorutaShap(n_trials=30, gini): `len(selected_features_) <= 2`; hetero_vote(defaults): `len(accepted) <= 2`; ShapProxiedFS: `support_.sum() <= 3`; HybridSelector: `n_features_ <= 3`; MRMR(min_features_fallback=0): reuse the layer-27/QM ceiling; RFECV: pin `n_features_ <= p//2` AND a companion `select_features_noise_floor` run asserting `n_star <= 2` (the plateau rule is the documented RFECV null defence, test_noise_floor.py tests it only on synthetic CURVES, never end-to-end on a null dataset). Mark `@pytest.mark.slow` variants for multi-seed; fast mode = seed 0 only via `fast_subset`. Expected current behavior: unknown for all five — exactly why it masks; any failure is a real FP-control bug.
Effort: M

### [gaps_selection_masking-02] MRMR all-noise FP ceiling assertion is near-vacuous vs its own documented contract
Severity: P2
Kind: quality
Files: tests/feature_selection/test_biz_value_mrmr_quality_metrics.py:308-340
Evidence: docstring says "FP rate should be <= 30%" but the code asserts `n_selected < 10` on 10 noise features — a 90% FP rate passes. The inline comment concedes "~30-40% of pure-noise features can randomly survive" under `full_npermutations=3`, yet the assert does not encode even that 40%.
Proposal: tighten to a two-tier contract: (a) production defaults, 3 seeds, assert `median(n_selected) <= 4` (40% documented + headroom); (b) one seed with `full_npermutations=25` (power restored) assert `n_selected <= 2`. Keep `min_features_fallback=0` and the existing fallback_used_ guard. Verify tier (a) passes on current code before committing the threshold (calibrate per CLAUDE.md 5-15% headroom rule).
Effort: S

### [gaps_selection_masking-03] Interaction-only (XOR / zero-marginal-MI) signal never tested for BorutaShap, ShapProxiedFS, hetero_vote, HybridSelector
Severity: P1
Kind: gap-mask
Files: tests/feature_selection/test_wrappers_audit_p1_extra.py:176-205 (RFECV covered: tree finds XOR, linear documented-miss); tests/feature_selection/test_biz_value_mrmr_hard_cases.py:28 (MRMR covered); grep G1 -> EMPTY for the other four families; src/mlframe/feature_selection/hetero_vote.py:43-60 (panel = RF + LogReg + kNN, vote_threshold=0.5)
Evidence: hetero_vote's linear panel member has |coef| ~ 0 on a pure-interaction pair by construction, so the pair's best possible vote is 2/3; at any `vote_threshold > 2/3` a genuinely relevant XOR pair is structurally DROPPED, and even at the 0.5 default the contract is one kNN-permutation-importance flake away from failing. BorutaShap's gini importance on shallow trees can also under-rank XOR vs shadows. None of this is pinned.
Proposal: recipe: n=1500, x0,x1 ~ N(0,1), `y = ((x0>0) ^ (x1>0))` with 5% label flips, plus 10 N(0,1) noise cols, seed=0. Tests: (a) hetero_vote defaults: assert {x0,x1} subset of accepted AND `info['vote_fraction'][x0] >= 2/3` (pins which members carry it); (b) companion documented-limitation pin: `vote_threshold=0.9` -> x0,x1 NOT accepted (so the limitation is explicit, mirroring test_linear_estimator_misses_xor); (c) BorutaShap(n_trials=30): assert both accepted, noise <= 3; (d) ShapProxiedFS: assert both in support_; (e) HybridSelector: assert both survive vote2. Expected current behavior: (a)/(c)/(d) probably pass but are unpinned; any fail = real all-relevant-recall bug surfaced.
Effort: M

### [gaps_selection_masking-04] Extreme class imbalance (<=1% positives) tested only for MRMR
Severity: P1
Kind: gap-mask
Files: tests/feature_selection/test_biz_value_mrmr_layer13.py:48-161 (1%/0.5%/0.1% contracts, MRMR); tests/feature_selection/test_selectors_shared.py:589-599 (70/30 only, and on a y INDEPENDENT of X); grep G2 -> EMPTY for wrappers/boruta/shap_prox/hetero/hybrid
Evidence: layer13 proves the suite knows rare-class selection is fragile (per-class MI floors, noise leakage at 0.1%), yet RFECV's CV scorer (fold-wise log_loss/AUC at ~7 positives per fold), BorutaShap's shadow percentile (SHAP values concentrated on the majority class), and ShapProxiedFS's anchor subsampling are all untested below 30% prevalence.
Evidence 2 (memory rule rare_imbalance_needs_large_n): rare_1pct needs n>=5000 for a stable split — recipe below respects that.
Proposal: shared recipe mirroring layer13's `_build_imbalanced_data`: n=15000, p=10 (2 informative with strong effect on the rare class: `logit = -4.6 + 2.5*x0 + 2.0*x1`), p_pos ~= 1%, seeds {0,1,2}. Per family: RFECV(RandomForest, cv=3) assert {x0,x1} in support and `n_features_ <= 6`; BorutaShap assert both informative accepted, noise <= 3; ShapProxiedFS assert both in support; hetero_vote assert both accepted. Mark slow; fast mode = 1 seed, n=6000. Expected current behavior: unknown — any fold that draws 0 positives or a shadow gate that rejects the rare-class signal is a real bug this would surface.
Effort: M

### [gaps_selection_masking-05] sample_weight: not supported and not tested for BorutaShap / ShapProxiedFS / hetero_vote; no loud-failure pin when the training-suite weight marker is stamped on a weight-incapable selector
Severity: P2
Kind: param
Files: src/mlframe/feature_selection/_boruta_shap_fit_explain.py:276 (`def fit(self, X, y)` — no sample_weight); src/mlframe/feature_selection/shap_proxied_fs/_shap_proxied_fit.py:69 (`def fit(self, X, y)`); src/mlframe/feature_selection/hetero_vote.py:83-88 (no sample_weight kwarg); tests/feature_selection/test_sample_weights_fs.py:26-113 (marker plumbing tested with mocks only); tests/feature_selection/test_mrmr_sample_weight_unit.py:76 + test_rfecv_sample_weight_unit.py:63 ("runs without error" only — but behavioral flip tests DO exist at test_biz_val_filters_mrmr.py:1240 and test_biz_val_wrappers_rfecv.py:885, so MRMR/RFECV are covered)
Evidence: grep G3 — zero sample_weight references in boruta/shap_prox/hetero/hybrid tests or sources. A user who sets `use_sample_weights_in_fs=True` and wires BorutaShap/ShapProxiedFS via `custom_pre_pipelines` gets either a TypeError or (worse, if a **kwargs sink is ever added) a silently unweighted fit; neither outcome is pinned.
Proposal: (a) unit: `_passthrough_cols_fit_transform` with a selector whose fit signature lacks sample_weight but whose `_mlframe_use_sample_weights_in_fs_` marker is True -> assert a loud, actionable error (or a logged WARN + documented downgrade — pick one contract and pin it; per the enable-by-default memory rule, prefer the loud error). (b) feature-level: add sample_weight to BorutaShap (forward to the surrogate model fit + shadow fits — both LGBM/CatBoost surrogates accept it) with the recency-flip biz recipe already used at test_biz_val_filters_mrmr.py:1240 (two features whose informativeness swaps between early/late rows; weights upweight late rows; assert top feature flips).
Effort: M (a alone: S)

### [gaps_selection_masking-06] Heavy-tailed features + label noise untested for RFECV / BorutaShap / ShapProxiedFS / hetero_vote
Severity: P2
Kind: gap-mask
Files: grep G4 — all 20 heavy-tail files are MRMR/FE-side (e.g. test_biz_val_filters_screen_heavytail_floor.py, test_biz_value_mrmr_layer11.py:108-154 1000x outliers); none under test_wrappers*/test_boruta*/test_shap_prox*/test_hetero*
Evidence: MRMR is rank/bin-based (quantile discretization absorbs tails) and is the ONLY family with the contract. RFECV with permutation importance scored by log_loss, and SHAP-based gates, are the families actually exposed to Cauchy-scale leverage points — and they have zero coverage.
Proposal: recipe: n=2000, 3 informative `x_i ~ standard_t(df=2)`, `y = (0.8*x0 + 0.6*x1 + 0.4*x2 + t(2)-noise > 0)` with 10% label flips, 7 Cauchy noise cols (`rng.standard_cauchy`), seeds {0,1}. Assert per family: informative recall >= 2/3 and noise admitted <= 2; additionally for RFECV assert no fold crash with `importance_getter='permutation'`. Expected current: likely passes for tree estimators (masking confirmed absent) but pins the contract; LogReg-estimator RFECV may genuinely fail -> document with estimator-conditional assert like TestD21 does for XOR.
Effort: M

### [gaps_selection_masking-07] Monotone-warped duplicate: no test pins WHICH of {raw, warp(raw)} survives redundancy elimination
Severity: P2
Kind: gap-mask
Files: tests/feature_selection/test_edge_cases_robustness.py:196 (`test_mrmr_perfectly_correlated_pair_keeps_one` — asserts ONE survives, not which); tests/feature_selection/test_dcd_adversarial_mrmr.py:66 (perfect duplicates: no-crash only); grep G7 -> only engineered-side near-monotone-dup pruning comments (layer64:454)
Evidence: binned MI / SU are monotone-invariant, so for `g = exp(4*f)` the relevance and redundancy of f and g are identical up to binning jitter — DCD anchor choice between them is tie-noise (src: anchor = first greedy pick by marginal MI, swaps gated on conditional MI, both monotone-invariant). If g wins, a downstream LINEAR model loses the signal that f carried, and nothing in the suite would notice. This is the exact MI-monotone-invariance blind spot the FE side already corrects with linear-usability gates (memory: project_mlframe_fe_mi_vs_linear_usability) — the RAW-column redundancy path has no analogous contract.
Proposal: recipe: n=3000, f ~ N(0,1), `g = exp(4*f)` (strictly monotone, rank-identical), `y = (f + 0.3*N(0,1) > 0)`, 5 noise cols, seed=0. Test (a) MRMR(dcd_enable=True): assert exactly one of {f,g} selected AND the survivor is `f` — if current behavior keeps g (or flip-flops by seed), that surfaces the tie-break; if the team decides MI-equivalence makes either acceptable for tree downstream, then pin instead "survivor is DETERMINISTIC across 5 seeds and column orders" + a documented-limitation comment. Test (b) RFECV with `estimator=make_pipeline(StandardScaler(), LogisticRegression())`: f must outrank g (g's logistic fit is much worse) — assert f in support_ when n_features_selection_rule trims to 3.
Effort: M

### [gaps_selection_masking-08] Fit-time column-order permutation invariance of the SELECTED SET untested for every selector
Severity: P2
Kind: gap-mask
Files: grep G6 — tests/feature_selection/test_biz_value_mrmr_layer19.py:458-565 and test_shap_proxied_fs_contract.py:57-65 cover TRANSFORM-time reorder only; layer89/90/94 hits are one-hot reindex helpers
Evidence: greedy MRMR breaks score ties by column index; RFECV FI voting has a lexicographic tie-breaker (test_wrappers_rfecv_fi_semantics.py F7 — name-based, so order-safe IF names are used everywhere); DCD anchor formation iterates columns in order. No test fits the SAME data with permuted column order and compares selected sets, so an index-based tie-break regression (set ordering, argsort instability) would ship silently as selection nondeterminism across otherwise-identical pipelines.
Proposal: parametrize over MRMR / RFECV / BorutaShap: build the shared `small_clf_problem` (test_selectors_shared.py:73-81), fit selector A on X and selector B on `X[list(reversed(X.columns))]` (fresh instances, fixed seeds), assert `sorted(A.get_feature_names_out()) == sorted(B.get_feature_names_out())`. Add one adversarial tie case: two EXACTLY duplicated informative columns "a","z_dup" (forces a tie) — assert the kept name is the same under both orders OR document+pin the first-by-name rule. Cheap (reuses fixture), belongs in test_selectors_shared.py as Group S.
Effort: S

### [gaps_selection_masking-09] n<<p untested for BorutaShap, hetero_vote, HybridSelector
Severity: P2
Kind: gap-mask
Files: tests/feature_selection/conftest.py:220-239 (`high_dimensional_data`, n=50 p=103) — grep G10: consumed ONLY by test_wrappers.py; tests/feature_selection/test_biz_value_mrmr_hard_cases.py:279 (`test_high_dim_p_greater_than_n`, MRMR); tests/feature_selection/test_rfecv_wide_data_fi_guard.py (RFECV); tests/feature_selection/test_biz_val_shap_proxied_fs.py:79 (wide pipeline, ShapProxiedFS)
Evidence: 4 of 7 families have a p>n contract; BorutaShap (shadow doubling makes the effective width 2p — the most width-sensitive family), hetero_vote (kNN member degenerates in high-d), and HybridSelector have none.
Proposal: reuse the `high_dimensional_data` fixture (n=50, 3 informative, 100 noise): BorutaShap(n_trials=20): assert fit completes, informative recall >= 1/3, noise admitted <= 10; hetero_vote: assert completes and vote_fraction well-formed for all 103 cols, accepted noise <= 10; HybridSelector: completes, n_features_ <= 20. These are run-and-bounded-FP contracts, not recovery contracts (n=50 is honest about power).
Effort: S

### [gaps_selection_masking-10] RFECV adversarial feature scaling (x1e9) has no end-to-end selection contract; coef_scale_source is only unit-tested as "train differs from test"
Severity: P2
Kind: bizvalue
Files: tests/feature_selection/test_wrappers_rfecv_fi_semantics.py:131-149 (asserts only `fi_train != fi_test`); src/mlframe/feature_selection/wrappers/_rfecv.py:330-332 (coef_scale_source exists precisely to fix scale-dependent |coef| ranking); grep G8 -> no adversarial-scaling selection test anywhere
Evidence: with a linear estimator and `importance_getter='coef_'`, an informative feature scaled by 1e9 gets |coef| ~ 1e-9 and is eliminated FIRST unless the std-rescale works. The mechanism shipped (F4) but its biz value — "mis-scaled informative feature survives" — is unpinned; a regression that silently no-ops the rescale (e.g. stds computed on the wrong axis) would pass the current "differs" test.
Proposal: recipe: n=400, p=6, `X[:,0] *= 1e9` where x0 is the strongest informative (`y = (raw_x0 + 0.5*x1 > 0)`), Ridge/LogReg estimator, `importance_getter='coef_'`. Assert (a) `coef_scale_source='train'` (default): x0 in support_; (b) negative control `coef_scale_source='none'`: x0 NOT in support_ (pins that the test actually discriminates — per the regression-test memory rule, verify (b) fails the (a) assertion). Effort small; goes in test_wrappers_rfecv_fi_semantics.py.
Effort: S

### [gaps_selection_masking-11] Missingness-informative (MNAR) features untested for BorutaShap and RFECV
Severity: P2
Kind: gap-mask
Files: tests/feature_selection/test_biz_value_mrmr_layer7.py (MNAR, MRMR); test_biz_value_mrmr_hard_cases.py:399 (`test_nan_pattern_perfectly_predicts_y`, MRMR); src/mlframe/feature_selection/_boruta_shap_object_cats (NaN -> -1 ordinal sentinel, tests/feature_selection/test_boruta_shap_object_cats_i179.py:97 tests the ENCODER only)
Evidence: grep `missing_indicator|mnar|nan_informative` -> MRMR layers only (layer7/11/37/64). For BorutaShap the -1 sentinel makes the NaN pattern learnable by the surrogate — whether a feature that is informative ONLY through its missingness pattern is accepted (and whether its SHADOW preserves the NaN pattern under permutation — it does, permutation preserves marginals) is unpinned. For RFECV, NaN handling with CatBoost estimator vs the zero-variance filter is untested on MNAR data.
Proposal: recipe: n=1200, base ~ N(0,1) pure noise VALUES, mask: `x_mnar = base.where(y==0, NaN)` for 40% of y==1 rows (NaN pattern carries ~0.3 bits; values carry 0). Assert BorutaShap accepts x_mnar (the pattern is real signal in-distribution) and RFECV(CatBoost) keeps it; if a family structurally cannot see it (e.g. RFECV+LogReg imputes), pin the documented limitation like TestD21 does. Also assert the reverse control: MCAR mask (independent of y) -> x_mnar rejected.
Effort: M

### [gaps_selection_masking-12] Time-series look-ahead / near-leak: leakage gate only tested at corr~1; TSS auto-detect only smoke-tested
Severity: LOW
Kind: gap-mask
Files: tests/feature_selection/test_wrappers_audit_p0.py:83 (`X["leak"] = y + 0.01*noise` — corr ~ 1.0 direct leak); tests/feature_selection/test_wrappers_phase8.py:70-78 (TSS auto-detect asserted via caplog split-type only, regression target, no selection assert); tests/feature_selection/test_biz_value_mrmr_layer9.py + layer17 + layer92 (MRMR-side lag/leak coverage)
Evidence: nothing pins behavior for a PARTIAL leak (e.g. `lag_-1_target` with corr ~ 0.7 — below any sane leakage_corr_threshold, above every legit feature): RFECV will rank it #1 and ship it; MRMR's gain-ratio audit (layer17 TestGainRatioAuditSignal) is the documented detector but is tested only at corr~1.
Proposal: recipe: AR(1) series `s_t = 0.9 s_{t-1} + e_t`, y_t = s_t, features = {lag1..lag3 (legit), lead1 = s_{t+1} + 0.5*e (the look-ahead leak), 5 noise}; n=2000. Pin (a) MRMR: lead1 appears in support AND its gain-ratio audit value exceeds the flag threshold (extends layer17 to partial leaks); (b) RFECV with `leakage_corr_threshold=0.6`: lead1 excluded; with the default threshold: lead1 included — pin both sides so the default's blind spot is explicit and documented.
Effort: M

### [gaps_selection_masking-13] Graded redundancy chain a~b~c with complementary increments untested (star/duplicate clusters only)
Severity: P2
Kind: gap-mask
Files: tests/feature_selection/conftest.py:242-265 (`correlated_features_data`: two independent 2-clusters + their sum); tests/feature_selection/test_biz_value_mrmr_layer6.py (decoy = sum, star-shaped); tests/feature_selection/test_wrappers_audit_p1_stress.py:42-101 (EXACT duplicates only); grep G9 -> no chain dataset
Evidence: the classic greedy mRMR failure is a transitive chain: `a` (signal), `b = a + 0.6*e1` (bridge, no NEW signal), `c = b + 0.6*e2 + delta` where `delta` is an independent signal component. SU(a,b) and SU(b,c) are high but SU(a,c) is moderate; the right answer is {a, c}. A pairwise-tau redundancy drop keyed off the anchor `a` admits `b` (wasted slot) at intermediate tau, while a chain-merging cluster (if hierarchy post-hoc merges via single linkage, layer48) can swallow `c`'s independent delta. Neither failure direction is pinned for MRMR/DCD, and RFECV's dedup handles only bit-exact duplicates.
Proposal: recipe: n=3000, e1,e2 ~ N(0,1); a ~ N(0,1); b = a + 0.6*e1; c = b + 0.6*e2 + 1.0*d where d ~ N(0,1); `y = (a + d + 0.4*noise > 0)`; 5 noise cols; seeds {0,1,2}. Assert MRMR(dcd_enable=True, defaults): {a, c} subset of selected (c carries d, the only path to it), b NOT selected; same contract for HybridSelector (its corr-clusters are the same trap) and RFECV (tree estimator). Expected current behavior: genuinely unknown — chain merging at default `dcd_tau_cluster` may drop c; that would be a real selection-quality bug surfaced.
Effort: M

### [gaps_selection_masking-14] Shared-battery imbalance test target is independent of X and its assertion is vacuous
Severity: LOW
Kind: quality
Files: tests/feature_selection/test_selectors_shared.py:589-599
Evidence: `y = (rng.random(n) < 0.3)` is independent of X, and the assertion is `assert selector.n_features_ >= 0` — always true (n_features_ is a count). The test name promises "70/30 imbalance ... should work" but verifies nothing beyond no-crash.
Proposal: rebuild on an informative imbalanced recipe: `logit = -1.5 + 2.0*X[:,0]` (gives ~30% positives carrying signal in f0), assert `"f0" in get_feature_names_out()` for both registered factories. Keep the no-crash path as a separate `pytest.raises`-free smoke if desired, but the named contract must be behavioral.
Effort: S

### [gaps_selection_masking-15] Shared battery: broad skip-on-exception escape hatches let registered-selector regressions vanish as skips
Severity: P2
Kind: quality
Files: tests/feature_selection/test_selectors_shared.py:181-187 (`test_numpy_array_input` skips on ANY TypeError/ValueError/AttributeError), :294-303 (`test_works_in_sklearn_pipeline` wraps fit+predict in `except Exception: pytest.skip`), :547-554 (`test_unfitted_raises` converts an AssertionError into a skip), :103-106 + :219-227 + :316-318 (get_feature_names_out absence -> skip, for selectors that DO implement it)
Evidence: both registered factories (RFECV, MRMR) today support numpy input, sklearn Pipeline, and get_feature_names_out — so a regression that breaks any of these flips the test to SKIPPED, not FAILED. This is the documented-skip anti-pattern (memory: feedback_dont_accept_documented_skips) baked into the strongest cross-selector file.
Proposal: add a per-factory capability map next to `_SELECTOR_FACTORIES` (e.g. `{"RFECV": {"numpy": True, "pipeline": True, "names_out": True}, ...}`); where the capability is declared True, the skip branch becomes a hard `pytest.fail`. New selectors opt out explicitly instead of silently skipping. Zero new fixtures; ~20-line diff.
Effort: S

### [gaps_selection_masking-16] High-cardinality noise categorical / ID-like memorization FP control untested for BorutaShap
Severity: LOW
Kind: gap-mask
Files: tests/feature_selection/test_biz_val_shap_proxied_fs.py:394,426 (Zipf-cardinality control exists for ShapProxiedFS); tests/feature_selection/test_boruta_shap_object_cats_i179.py:49-115 (encoder mechanics only); tests/feature_selection/test_biz_value_mrmr_layer10.py (MRMR high-card)
Evidence: gini/split importance is biased toward high-cardinality columns; Boruta's defence is that shadows preserve cardinality — but that defence is unpinned. A 300-level random categorical (or a unique-int ID column) accepted by BorutaShap would be invisible to the current suite (the single biz test at test_biz_val_filters_boruta_shap.py:17 uses only continuous N(0,1) noise).
Proposal: recipe: n=1500, 2 informative numerics (as in the existing biz test), plus `cat_noise = rng.integers(0, 300, n)` (object dtype) and `id_col = rng.permutation(n)`; BorutaShap(importance_measure="gini", n_trials=30, seed=0). Assert both informative kept, and `cat_noise`/`id_col` both REJECTED. If gini fails, that is a real shadow-comparison bug (or motivates flipping the default to shap importance — accuracy-first default rule).
Effort: S

### [gaps_selection_masking-17] hetero_vote: regression-target path entirely untested
Severity: LOW
Kind: gap-mask
Files: tests/feature_selection/test_hetero_vote.py (3 tests, all `classification=True`); src/mlframe/feature_selection/hetero_vote.py:55-59 (regression panel: RF + Ridge + kNN regressors — live code, zero tests)
Evidence: `_default_panel(classification=False)` and the regression `_cv_skill` branch have no executing test; a typo in the regressor panel (wrong import, wrong skill metric sign) would ship undetected.
Proposal: mirror `test_hetero_vote_keeps_signal_drops_noise` with a continuous target: `y = z @ [1.5,-1.2,1.0,0.9] + 0.5*N(0,1)`, `classification=False`, assert all 4 signal cols accepted, noise <= 2, `info['n_models'] == 3`. One test, same `_data` helper with a `regression=True` flag.
Effort: S

---

## Summary

| Severity | Count |
|---|---|
| P0 | 0 |
| P1 | 3 (01, 03, 04) |
| P2 | 10 (02, 05, 06, 07, 08, 09, 10, 11, 13, 15) |
| LOW | 4 (12, 14, 16, 17) |
| Total | 17 |

Verdict: MRMR's adversarial coverage is exceptional (97 layer contracts spanning decoys, MNAR, imbalance, leakage, drift, outliers), but it acts as a spotlight that leaves the other selector families in shadow — BorutaShap, ShapProxiedFS, hetero_vote and HybridSelector have essentially ONE happy-path biz test each, with zero coverage for the null/FDR, interaction-only, extreme-imbalance, heavy-tail and n<<p attack surfaces that MRMR is tested against, and the shared battery's escape-hatch skips plus vacuous null/imbalance asserts mean a regression in those families degrades to a skip or a no-op assert rather than a failure. No P0 because no case was found where an existing test actively pins WRONG behavior; the risk profile is silent-masking, concentrated in the three P1s (null FDR, interaction-only recall, rare-class selection) where a real algorithmic failure would today be invisible.
