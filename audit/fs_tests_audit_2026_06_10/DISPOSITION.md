# FS/FE test-suite audit — disposition of all 146 findings (2026-06-10)

8 finder agents × 8 adversarial verifiers (0 REFUTED, ~50 CORRECTED-scope). Critic phase synthesized
manually (session-limit kill). Detailed evidence + proposals live in the per-key `*.verified.md` files in
this directory. Severity is post-verification (corrections that downgraded are reflected).

Disposition buckets: **RESOLVED** (fixed this program) · **FUTURE** (tracked, out of this batch) ·
**DOC** (documented, no code change) · **REJECTED** (anti-recommendation w/ reason).

## FINAL LANDED-STATUS RECONCILIATION (completeness check)

6 commits: 8c2c4038 (keystone+4 prod fixes), 5eba48e5 (FE-gate fail-open), 09f28a87 (artifacts/gitignore),
88f3023d (9 packs), 3533427c (null-FDR perf+value-proof). Every one of the 146 findings is in exactly one
bucket below — nothing silently dropped.

### LANDED + verified + committed (~40 findings)
- **Keystone contract infra** (new `_selector_factories.py` + `test_selector_contract_shared.py` + registry tripwire): shared_lift-02/04/07/08/11/12/13/15/16/22, param_axes-02, gaps_selection_masking-05/08/15, coverage_asymmetry-01/02, test_code_quality-15(new file uses capability flags). Plus shared_lift-01/03 (Boruta + GroupAware prod-wrap enrolled as specs).
- **9 biz-value/contract packs** (88f3023d): param_axes-04/06/07/17, coverage_asymmetry-08/09/10/11/12/14/16/17, gaps_selection_masking-01/02/03(boruta)/17, bizvalue-01/03/04/07.
- **B3 hygiene**: gaps_fe_masking-10 (golden NameError), training_integration-08 + param_axes-11 (cannot-fail tests), test_code_quality-06/17 (D:/Temp, .bak).
- **B4**: gaps_fe_masking-03 (FE-gate fail-open).

### PROD BUGS found by this work (beyond the 146) — 4 FIXED, 2 SENSORED
FIXED: RFECV.get_support added; RFECV global-RNG leak; BorutaShap/ShapProxiedFS/HybridSelector RNG leaks; MRMR caller-frame mutation; (B4) FE-gate fail-closed.
SENSORED (xfail regression sensors, fix tracked): PB-5 RFECV/ShapProxied/Hybrid pure-null FP-control; PB-6 MRMR empty support_ vs min_features_fallback.

### FUTURE — NOT yet implemented (~100 findings, each still has its row in the table below)
The large remaining tranche: the 97-layer consolidation (test_code_quality-01/02/16, param_axes-08, L-effort), RFECV 12-knob pack (coverage_asymmetry-03), ShapProxiedFS su_seeded knobs (coverage_asymmetry-15), pre_screen (coverage_asymmetry-18), cluster_aggregate extras, per-axis param families (param_axes-01/05/09/12/13/14/15/16, gaps_fe_masking-04..09/11..18), the remaining lifted-contract variants NOT in the new file (shared_lift-05 gfno-input_features, -06 set_output, -09 regression_df wiring, -10 determinism-unify, -14/17/18/19/20/21/23, -24 full), gaps_selection_masking-06/07/09/10/11/12/13/16, bizvalue-02/05/06/08(partial)/09/10/11/12/13/14/15, the full training_integration set incl. P0-01 save→load→predict parity + 02/03/04/05/06/07/09/10/11/12/13/14/15/16/17/18, test_code_quality-03(gpu markers)/04/05/07/08/09/10/11/13/14, the two PROD-BUG fixes (PB-5/PB-6), and the OLD-suite retrofit (merge test_fs_selector_contract.py / test_selectors_shared.py onto the single factory source — drift is currently killed via the new file + tripwire, not a full merge).

NOTE on honesty: the keystone landed a NEW comprehensive contract file rather than retrofitting the two pre-existing
parallel suites; the registry tripwire prevents future drift, but the OLD files still carry their own factory lists
(a tracked FUTURE merge). No finding's coverage is overstated above.

---

## Severity rollup (146 total)

| key | P0 | P1 | P2 | LOW | total |
|---|---|---|---|---|---|
| shared_lift | 0 | 4 | 10 | 10 | 24 |
| param_axes | 0 | 4 | 9 | 4 | 17 |
| gaps_selection_masking | 0 | 3 | 10 | 4 | 17 |
| gaps_fe_masking | 0 | 4 | 10 | 4 | 18 |
| bizvalue_value_proofs | 0 | 6 | 7 | 2 | 15 |
| test_code_quality | 0 | 4 | 9 | 4 | 17 |
| coverage_asymmetry_wrappers | 0 | 6 | 12 | 2 | 20 |
| training_integration | 1 | 5 | 8 | 4 | 18 |
| **TOTAL** | **1** | **36** | **75** | **34** | **146** |

## Cross-report dedup map (same underlying item, multiple angles)

- **DEDUP-A · shared contract suite covers only 2–3 of 4+ selectors**: shared_lift-01/02/03 ≡ param_axes-02 ≡ coverage_asymmetry-01/02 ≡ gaps_selection_masking-15/-22(escape hatches) ≡ shared_lift-22. *(keystone → KS-1)*
- **DEDUP-B · weak families (BorutaShap/ShapProxiedFS/HybridSelector/hetero_vote) ~1 happy-path test each**: gaps_selection_masking-01/03/04/05/06/09/16/17 ≡ bizvalue-07/08 ≡ coverage_asymmetry-04/05/06/07/08/09/10/11/12 ≡ param_axes-07.
- **DEDUP-C · null/FDR contract weak-or-absent**: gaps_selection_masking-01/02/14 ≡ gaps_fe_masking-01 ≡ bizvalue-03/04.
- **DEDUP-D · fe_max_steps=2 default unvalidated**: gaps_fe_masking-02 ≡ param_axes-05.
- **DEDUP-E · source-presence sensors / cannot-fail tests instead of behavioral**: shared_lift-23 ≡ test_code_quality-07 ≡ gaps_fe_masking-10(golden NameError) ≡ training_integration-02/08/09 ≡ param_axes-11.
- **DEDUP-F · column-order / pandas-index permutation invariance**: shared_lift-12/13 ≡ gaps_selection_masking-08.
- **DEDUP-G · 97-layer dedup + fast-mode + consolidation**: test_code_quality-01/02/16 ≡ param_axes-08/14 ≡ coverage_asymmetry-20.
- **DEDUP-H · external-baseline head-to-heads absent**: bizvalue-01/03/04/09.
- **DEDUP-I · sample_weight not lifted / suite-level untested**: shared_lift-11 ≡ param_axes-06 ≡ gaps_selection_masking-05 ≡ training_integration-04.
- **DEDUP-J · D:/Temp artifact paths + silent double-except**: gaps_fe_masking-17 ≡ test_code_quality-06.
- **DEDUP-K · np.random.seed global-state stragglers**: test_code_quality-13 ≡ training_integration-16.

## Missing angles (critic synthesis — not examined by any finder)

1. **Mutation testing** of the FS assertions — do they actually go red when src is broken? (the report repeatedly *infers* sensitivity; only a handful were empirically run). Next step: `cosmic-ray`/manual mutmut on `mrmr.py` greedy loop + `_rfecv` FI vote, confirm ≥1 layer fails per mutant.
2. **Property-based / hypothesis** testing — `_biz_val_synth.py` already ships dead `_property_*` shells (test_code_quality-09); none are live.
3. **GPU↔CPU selection parity** as a first-class contract (only kernel-level parity exists).
4. **Concurrency** — selectors fitted from threads/processes sharing the MRMR `_FIT_CACHE`; no multi-thread fit-race test.
5. **Memory-footprint / 100GB-frame** behavior of selectors (no `.nbytes`-gated path test).
6. **docs/README claims** — are the FS examples in README/docs executed? (`test_docs_examples_smoke.py` exists at repo root; FS coverage unverified).
7. **CI runtime budget** of the whole FS suite (6,619 tests, zero fast/slow tiering in the 48k-LOC layer slice).
8. **Golden-baseline staleness process** — no owner/cadence; machinery is currently dead (gaps_fe_masking-10).

---

## Disposition table (all 146)

Legend: R=RESOLVED, F=FUTURE, D=DOC, X=REJECTED. "batch" = which landing batch (KS=keystone, B#=batch).

### shared_lift (24)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| shared_lift-01 | P1 | R | KS-1 | BorutaShap+Hybrid into unified factory contract |
| shared_lift-02 | P1 | R | KS-1 | single `_selector_factories.py` + registry tripwire |
| shared_lift-03 | P1 | R | KS-1 | GroupAware(RFECV)/(BorutaShap) prod-wrap factories; lift missing invariants only |
| shared_lift-04 | P1 | R | B2 | pickle hard-fail via capability flag (no skip) |
| shared_lift-05 | P2 | R | B2 | gfno(input_features) protocol parametrized |
| shared_lift-06 | P2 | R | B2 | set_output(pandas) parametrized |
| shared_lift-07 | P2 | R | B2 | get_support mask/indices/half-fit parametrized |
| shared_lift-08 | P2 | R | B2 | transform n_features_in_ validation parametrized |
| shared_lift-09 | P2 | R | B2 | wire the dead regression_df fixture into the contract |
| shared_lift-10 | P2 | R | B2 | unify determinism floor via factory `deterministic` flag |
| shared_lift-11 | P2 | R | B2 | sample_weight semantics lifted (cap flag) [DEDUP-I] |
| shared_lift-12 | P2 | R | B2 | misaligned pandas index contract [DEDUP-F] |
| shared_lift-13 | P2 | R | B2 | fit-time column-order invariance [DEDUP-F] |
| shared_lift-14 | P2 | R | B2 | NaN/Inf policy table parametrized |
| shared_lift-15 | LOW | R | B2 | duplicate column-name rejection lifted |
| shared_lift-16 | LOW | R | B2 | fit-level global-RNG hygiene lifted (template: test_mrmr_fixes_p0_p1.py) |
| shared_lift-17 | LOW | F | — | n_jobs parity lift (StabilityMRMR/RFECV already partial) |
| shared_lift-18 | LOW | F | — | PYTHONHASHSEED subprocess harness (needs new fit-script) |
| shared_lift-19 | LOW | R | B2 | polars==pandas selection parity (cap flag) |
| shared_lift-20 | LOW | R | B2 | clone/get_params for GroupAware/Stability/Boruta |
| shared_lift-21 | LOW | F | — | Stability*/TreeRescued battery membership (partial cover exists) |
| shared_lift-22 | LOW | R | KS-1 | replace skip-hatches with capability flags [DEDUP-A] |
| shared_lift-23 | LOW | R | B3 | dtype-promotion e2e must call real `_append_engineered` [DEDUP-E] |
| shared_lift-24 | LOW | R | B2 | degenerate-y rejection unified |

### param_axes (17)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| param_axes-01 | P2 | F | — | non-orth target-aware FE families × regression/multiclass (orth already covered) |
| param_axes-02 | P1 | R | KS-1 | registry-derived factories incl. prod-wrapped [DEDUP-A] |
| param_axes-03 | P1 | R | B2 | polars container axis in shared contract [DEDUP-A] |
| param_axes-04 | P1 | R | B3 | score_pair_mi dispatcher direct contract (9 estimators) |
| param_axes-05 | P1 | R | B4 | all-FE-on kitchen-sink × prod-default redundancy [DEDUP-D] |
| param_axes-06 | P2 | F | — | sample_weight/groups × FE-mechanism axis |
| param_axes-07 | P2 | R | B5 | hetero_vote classification=False [DEDUP-B] |
| param_axes-08 | P2 | F | — | FE-lifecycle quartet → registry-parametrized (L) [DEDUP-G] |
| param_axes-09 | P2 | F | — | parametrized permuted-y leak gate over FE families |
| param_axes-10 | P2 | R | B4 | MRMR/FE n<p (wide-data) axis |
| param_axes-11 | P2 | R | B3 | quantization-method except-swallow → real failure [DEDUP-E] |
| param_axes-12 | P2 | F | — | layer83 per-cell parametrize (perf-only, low risk) |
| param_axes-13 | P2 | F | — | float32 fit+replay for modern FE recipes |
| param_axes-14 | LOW | F | — | layer35/104 within-file parametrize |
| param_axes-15 | LOW | F | — | miller_madow/unseen-cell parametrize |
| param_axes-16 | LOW | F | — | test_bases.py registry-iterate trio |
| param_axes-17 | LOW | R | B3 | cluster_aggregate methods → CLUSTER_AGGREGATE_METHODS parametrize |

### gaps_selection_masking (17)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| gaps_selection_masking-01 | P1 | R | B5 | null-FDR contract over all selectors [DEDUP-C] |
| gaps_selection_masking-02 | P2 | R | B5 | tighten MRMR all-noise ceiling (vacuous→FP-rate) [DEDUP-C] |
| gaps_selection_masking-03 | P1 | R | B5 | interaction-only recall for weak families [DEDUP-B] |
| gaps_selection_masking-04 | P1 | R | B5 | extreme imbalance for weak families [DEDUP-B] |
| gaps_selection_masking-05 | P2 | R | B2 | sample_weight loud-fail for weight-incapable selectors [DEDUP-I] |
| gaps_selection_masking-06 | P2 | F | — | heavy-tail+label-noise for RFECV/Boruta/Shap |
| gaps_selection_masking-07 | P2 | R | B5 | monotone-warped duplicate survivor pin |
| gaps_selection_masking-08 | P2 | R | B2 | fit-time column-order invariance [DEDUP-F] |
| gaps_selection_masking-09 | P2 | F | — | n<<p for Boruta/hetero/Hybrid |
| gaps_selection_masking-10 | P2 | R | B5 | RFECV adversarial scaling (coef_scale_source neg-control) |
| gaps_selection_masking-11 | P2 | F | — | MNAR for Boruta/RFECV |
| gaps_selection_masking-12 | LOW | F | — | partial time-series leak |
| gaps_selection_masking-13 | P2 | F | — | graded redundancy chain a~b~c |
| gaps_selection_masking-14 | LOW | R | B2 | vacuous shared imbalance assert → informative [DEDUP-A] |
| gaps_selection_masking-15 | P2 | R | KS-1 | skip-hatch → capability fail [DEDUP-A] |
| gaps_selection_masking-16 | LOW | F | — | high-card noise cat for BorutaShap |
| gaps_selection_masking-17 | LOW | R | B5 | hetero_vote regression path [DEDUP-B, ≡ param_axes-07] |

### gaps_fe_masking (18)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| gaps_fe_masking-01 | P1 | R | B4 | default-pipeline all-noise FWER sensor (strengthen quality_metrics) [DEDUP-C] |
| gaps_fe_masking-02 | P1 | R | B4 | fe_max_steps=2 keep/drop validation [DEDUP-D] |
| gaps_fe_masking-03 | P1 | R | B4 | _fe_accuracy_gate direct tests + fix fail-open/closed contradiction (PROD FIX) |
| gaps_fe_masking-04 | P1 | F | — | recipe-less drop transform branch (force via fe_max_steps=3) |
| gaps_fe_masking-05 | P2 | R | B3 | strict EXPECTED_XFAILS (XPASS visible) |
| gaps_fe_masking-06 | P2 | F | — | noise-only engineered-operand admission kind |
| gaps_fe_masking-07 | P2 | F | — | robust-prewarp recipe provenance replay |
| gaps_fe_masking-08 | P2 | F | — | prefer-engineered raw-wins large-n (needs ctor param thread) |
| gaps_fe_masking-09 | P2 | F | — | nullable dtype (Int64/pd.NA) FE coverage |
| gaps_fe_masking-10 | P2 | R | B3 | golden machinery: fix `ororjson` NameError, revive-or-delete [DEDUP-E] |
| gaps_fe_masking-11 | P2 | F | — | stability-vote single-survivor/kind-exemption |
| gaps_fe_masking-12 | P2 | R | B3 | rung-schedule needle operand assert + wall-clock robustness |
| gaps_fe_masking-13 | P2 | F | — | adaptive-FE anti-hijack e2e (monotone/Cauchy stays raw) |
| gaps_fe_masking-14 | P2 | R | B3 | exact replay parity (assert_allclose) where achievable |
| gaps_fe_masking-15 | LOW | R | B3 | vacuous recipe test → skip-or-strengthen |
| gaps_fe_masking-16 | LOW | F | — | dead GPU div-semantics parity (cheap-now PROD FIX) |
| gaps_fe_masking-17 | LOW | R | B3 | D:/Temp ledger → env-overridable in-repo [DEDUP-J] |
| gaps_fe_masking-18 | LOW | F | — | golden FE-drift capture (depends on revive 10) |

### bizvalue_value_proofs (15)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| bizvalue-01 | P1 | R | B5 | h2h vs SelectKBest-MI / RFE / random-K [DEDUP-H] |
| bizvalue-02 | P2 | F | — | sklearn-RFECV runtime-win at parity |
| bizvalue-03 | P1 | R | B5 | quality-vs-K frontier [DEDUP-H] |
| bizvalue-04 | P1 | R | B5 | bootstrap Nogueira stability vs baseline [DEDUP-C/H] |
| bizvalue-05 | P2 | F | — | suite FS wins on very-wide data + rename misleading test |
| bizvalue-06 | P1 | R | B6 | dead rfecv=True suite branch exercised |
| bizvalue-07 | P2 | R | B5 | BorutaShap downstream vs random-K/SHAP-top-k [DEDUP-B] |
| bizvalue-08 | P1 | R | B5 | Hybrid/hetero biz-value (quality≥best member) [DEDUP-B] |
| bizvalue-09 | P1 | F | — | real-data noise-injection value win (L) [DEDUP-H] |
| bizvalue-10 | P2 | F | — | label-noise degradation vs baseline |
| bizvalue-11 | P2 | F | — | runtime-budget wall-clock + scaling envelope |
| bizvalue-12 | P2 | F | — | FE value across model families (linear-lift/tree-no-harm) |
| bizvalue-13 | P2 | F | — | persisted value-leaderboard golden |
| bizvalue-14 | LOW | R | B3 | bench-adapter alive smoke (fs_selectors) |
| bizvalue-15 | LOW | F | — | layer83 fast-mode + LGBM downstream column |

### test_code_quality (17)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| test_code_quality-01 | P1 | F | — | ~3-4k LOC helper dedup → conftest/_biz_val_synth (L, AST-gated) [DEDUP-G] |
| test_code_quality-02 | P1 | F | — | fast/slow markers across 97 layers (L) [DEDUP-G] |
| test_code_quality-03 | P1 | R | B3 | gpu marker on 11+3 files (suite-crash workaround) |
| test_code_quality-04 | P2 | F | — | class-scoped fitted fixtures (hoist refits) |
| test_code_quality-05 | P2 | F | — | benchmark marker on 42 perf-assert files |
| test_code_quality-06 | P2 | R | B3 | D:/Temp artifact paths → helper [DEDUP-J] |
| test_code_quality-07 | P2 | R | B3 | read_text source sensor → behavioral [DEDUP-E] |
| test_code_quality-08 | P2 | R | B3 | dead conftest fixtures (known_mi_data delete etc.) |
| test_code_quality-09 | P2 | R | B3 | dead `_property_*` shells → real hypothesis tests |
| test_code_quality-10 | P2 | R | B3 | dual module-identity sys.path hack fix |
| test_code_quality-11 | P2 | F | — | seed-loop → parametrize (6 sites) |
| test_code_quality-12 | LOW | R | B3 | bare except in FS tests narrowed |
| test_code_quality-13 | LOW | F | — | np.random.seed → default_rng (8 sites) [DEDUP-K] |
| test_code_quality-14 | LOW | F | — | conftest fixture scope tuning |
| test_code_quality-15 | P2 | R | KS-1 | blanket Pipeline xfail → allowlist [DEDUP-E] |
| test_code_quality-16 | P1 | F | — | 97-layer → ~25-50 themed modules under AST inventory gate (L) [DEDUP-G] |
| test_code_quality-17 | LOW | R | B3 | .bak/.gitignore hygiene |

### coverage_asymmetry_wrappers (20)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| coverage_asymmetry-01 | P1 | R | KS-1 | Boruta+Hybrid into shared contract [DEDUP-A] |
| coverage_asymmetry-02 | P2 | R | KS-1 | registry ShapProxiedFS presence + instantiate-type tests [DEDUP-A] |
| coverage_asymmetry-03 | P1 | F | — | 12 RFECV unexercised knobs (L) |
| coverage_asymmetry-04 | P1 | R | B6 | enroll bench test_hybrid_tree_member.py into tests/ + 3 asserts [DEDUP-B] |
| coverage_asymmetry-05 | P1 | R | B6 | Hybrid degraded-member/vote-fallback [DEDUP-B] |
| coverage_asymmetry-06 | P2 | F | — | Hybrid boruta_driver/use_mrmr/prescreen paths |
| coverage_asymmetry-07 | P2 | R | B6 | Hybrid FE-mode pickle replay value-equality [DEDUP-B] |
| coverage_asymmetry-08 | P1 | R | B5 | hetero_vote regression path [DEDUP-B, ≡ ps-17/pa-07] |
| coverage_asymmetry-09 | P2 | R | B5 | hetero_vote knobs (models/percentile/ndarray/threshold-boundary) [DEDUP-B] |
| coverage_asymmetry-10 | P2 | R | B5 | hetero_vote skill-weighting discriminating test [DEDUP-B] |
| coverage_asymmetry-11 | P2 | R | B5 | hetero_vote _importance permutation-fallback [DEDUP-B] |
| coverage_asymmetry-12 | P1 | R | B5 | BorutaShap hard-case biz-value pack [DEDUP-B] |
| coverage_asymmetry-13 | P2 | R | B6 | hetero/Hybrid determinism + golden enroll |
| coverage_asymmetry-14 | P2 | F | — | cluster_aggregate 7 discovery knobs |
| coverage_asymmetry-15 | P2 | F | — | ShapProxiedFS su_seeded sub-knobs |
| coverage_asymmetry-16 | P2 | R | B6 | optbinning FS pipelines actually fitted + de-flake skip |
| coverage_asymmetry-17 | P2 | R | B6 | importance.py: sanitizer + explain_top_feature_importances |
| coverage_asymmetry-18 | P2 | F | — | pre_screen polars non-numeric/1-row/boundary |
| coverage_asymmetry-19 | LOW | R | B6 | Hybrid imports → production path not bench |
| coverage_asymmetry-20 | LOW | F | — | hybrid/hetero e2e fast mode |

### training_integration (18)
| id | sev | disp | batch | note |
|---|---|---|---|---|
| training_integration-01 | **P0** | R | B1 | save→load→predict FS round-trip parity (the keystone gap) |
| training_integration-02 | P1 | R | B1 | predict-recovery branch behavioral test (drop AST scan) [DEDUP-E] |
| training_integration-03 | P1 | R | B1 | engineered-FE through suite predict path |
| training_integration-04 | P1 | R | B6 | weight-aware FS suite-level (selection responds + reuse invariant) [DEDUP-I] |
| training_integration-05 | P1 | F | — | groups/TS-CV × FS through suite |
| training_integration-06 | P1 | R | B6 | FS × ensembling deterministic |
| training_integration-07 | P2 | R | B6 | empty-selection contract (fix vacuous assert; set min_relevance_gain_mode) |
| training_integration-08 | P2 | R | B6 | 3 cannot-fail tests in test_feature_selection.py [DEDUP-E] |
| training_integration-09 | P2 | R | B6 | report observability vs REAL selectors (drop skip-hatch) [DEDUP-E] |
| training_integration-10 | P2 | F | — | friend_graph/cluster_aggregate report fields |
| training_integration-11 | P2 | F | — | selected_features union semantics + per-model key |
| training_integration-12 | P2 | F | — | suite-level FS determinism (same-seed dual run) |
| training_integration-13 | P2 | F | — | categorical-survival-through-FS suite |
| training_integration-14 | P2 | F | — | BorutaShap deterministic suite + regression-target |
| training_integration-15 | LOW | F | — | early-stopping reaches RFECV inner fits |
| training_integration-16 | LOW | R | B3 | test_feature_selection.py module hygiene (8 seeds, dead docstring) [DEDUP-K] |
| training_integration-17 | LOW | R | B6 | bizvalue duplicate baseline run collapse |
| training_integration-18 | LOW | F | — | ShortlistTransformerAdapter needs_y=True suite path |

## Batch order (landing plan)

- **KS-1** — unified capability-flagged `_selector_factories.py` + registry tripwire + escape-hatch→capability conversion. Unlocks DEDUP-A.
- **B1** — P0 + predict-path family (training_integration-01/02/03).
- **B2** — lift shared sklearn-protocol + robustness contracts over all selectors (shared_lift-04..16/19/20/24, param_axes-03, gaps_selection_masking-05/08/14).
- **B3** — masking/hygiene bug-fixes (golden NameError, source sensors, dead fixtures/shells, D:/Temp, gpu marker, cannot-fail tests, exact replay, .bak). PROD FIX: none here except dead-GPU deferred.
- **B4** — FE correctness: _fe_accuracy_gate fail-closed PROD FIX + direct tests, fe_max_steps=2 validation, FWER null sensor, wide-data.
- **B5** — null-FDR + interaction-only + imbalance + adversarial-scaling + monotone-warp + external h2h + frontier + Nogueira stability + hetero_vote pack + BorutaShap biz pack.
- **B6** — weak-family + suite-integration (Hybrid tree/degrade/pickle/determinism, optbinning, importance, weight-aware suite, ensembling, empty-selection, observability-real, hygiene).
- **FUTURE** — the L-effort consolidation (97-layer merge, helper dedup, fast/slow markering, RFECV 12-knob pack) + the remaining single-angle param/gap tests, tracked here.

FUTURE items remain the explicit backlog; none is dropped — each has a row above with its disposition.

---

## Landing log (live)

### KS-1 — keystone (LANDED, 84 passed / 5 skip / 14 xfail in the new contract suite)
New files:
- `tests/feature_selection/_selector_factories.py` — single-source `SELECTOR_SPECS` with capability flags + robust `selected_mask`/`selected_names` (covers MRMR, RFECV, BorutaShap, ShapProxiedFS, HybridSelector, GroupAware(RFECV) — the production-default wrap).
- `tests/feature_selection/test_selector_contract_shared.py` — capability-gated cross-selector battery (universal API, pickle hard-fail, transform-width validation, column-order invariance, non-default index, sample_weight loud-fail, duplicate-name, global-RNG hygiene, NaN-policy, **fit-does-not-mutate-input-frame**).
- Registry tripwire + ShapProxiedFS presence + RFECV/ShapProxied instantiate-type sensors in `test_selector_registry.py`.

**4 REAL PROD BUGS surfaced + FIXED by the keystone (all beyond the original 146 — found by running the lifted contracts against real selectors):**
1. **RFECV had no `get_support()`** (claims TransformerMixin, breaks sklearn SelectorMixin tooling) → added `RFECV.get_support(indices=)` normalising bool/int `support_`. [src/mlframe/feature_selection/wrappers/_rfecv.py]
2. **RFECV clobbered the global numpy/random RNG every fit** (`set_random_seed` called inside fit — the exact thing its own docstring forbids) → `RFECV.fit` wrapped in `preserve_global_rng`. [new `preserve_global_rng` + `RFECV.fit` rebind]
3. **BorutaShap / ShapProxiedFS / HybridSelector all leaked the global RNG** → `@rng_hygienic_fit` on each fit. [misc.py + 3 selectors]
4. **MRMR (and HybridSelector via its MRMR member) MUTATED THE CALLER'S DataFrame IN PLACE** — `X[hinge_leg] = vals` at _mrmr_fit_impl.py:6398 leaked engineered columns into the caller's frame (10→15 cols), violating the CRITICAL CLAUDE.md 100GB-frame rule + silent schema corruption, uncaught by all 97 layer files → `@hygienic_fit` (mutate-and-restore) on `MRMR.fit`. [misc.py `hygienic_fit` + MRMR.fit]

These 4 map onto the shared_lift-16 (RNG hygiene) + the silent-correctness bug classes; they are P1-grade real bugs the audit's lifted contracts were designed to surface.

### B4 — FE accuracy gate fail-open [committed 5eba48e5]
`_fe_accuracy_gate.measure_feature_uplift` returned 0.0 on a probe exception/degenerate input → `0.0 >= threshold` False → engineered candidate silently DROPPED (fail-CLOSED), contradicting the documented fail-OPEN. Fixed (None sentinel on all can't-measure paths; both call sites keep on None) + new 9-test `test_fe_accuracy_gate.py`. gaps_fe_masking-03 RESOLVED.

### Additive packs workflow [committed 88f3023d] — 9 files, all independently re-verified
hetero_vote (14), mi_dispatch (9), importance (21), optbinning (17), cluster_aggregate (17), sample_weight×FE (10+6xf), BorutaShap hard-cases (16), null-FDR+Nogueira (4p/8slow-skip/1xf fast; slow tier trimmed B=10), external h2h (4p/1s/1xf). Closes: param_axes-04/06/07/17, coverage_asymmetry-08/09/10/11/12/14/16/17, gaps_selection_masking-01/02/03(boruta)/17, bizvalue-01/03/04/07.

### TWO MORE CONFIRMED PROD BUGS (independently reproduced; xfail-sensored, NOT yet fixed)
- **PB-5 (P1): RFECV(argmax, no plateau) selects ALL pure-noise features.** Reproduced n=1000/p=15 random-y: RFECV [15,15,15], ShapProxiedFS [8,4,9], HybridSelector [9,7,8], MRMR clean [2,2,2]. No false-positive control on a null dataset for those 3 families. Sensor: test_biz_val_null_fdr_and_stability.py xfails. FUTURE fix: default plateau/noise-floor rule for the argmax selection (a default-policy decision) + FP-control gate for ShapProxiedFS/Hybrid.
- **PB-6 (P1): MRMR returns EMPTY support_ (0-width transform) despite min_features_fallback>=1.** Was reproduced on make_signal_plus_noise (linear, no redundancy), FE on: seeds 2 & 4 of 6. **FIXED upstream by the parallel refactor's `f0fd18ad` ("raw-redundancy drop must not empty selection via un-replayable nested subsumer") -- verified 0/6 empties on the same fixture.** The h2h sensor's xfail was flipped to a passing within-epsilon value proof.

### FUTURE-tranche wave-1 (6 new files, integrated)
test_rfecv_unexercised_knobs (coverage_asymmetry-03), test_selector_contract_protocol_extra (shared_lift-05/06/09/19/20), test_biz_val_weak_family_adversarial (gaps_selection_masking-06/09/11/13/16), test_biz_val_hetero_hybrid_determinism (coverage_asymmetry-13), test_shap_proxied_knobs (coverage_asymmetry-15), test_pre_screen_edges (coverage_asymmetry-18). New PROD BUG sensored: RFECV estimators_save_path documented but inert (no dump call in the fit path).

### FUTURE-tranche wave-2 (5 new files, integrated)
test_hybrid_selector_deep (coverage_asymmetry-04/05/06/07: tree-member gate modes, degraded members, driver/use_mrmr/prescreen paths, FE-mode pickle replay), test_stability_selectors_contract (shared_lift-21: Stability*/MRMRTreeRescued API battery), test_biz_val_label_noise_and_scaling (bizvalue-10 + gaps_selection_masking-10), test_biz_val_runtime_budget (bizvalue-11: MRMR/RFECV honor max_runtime_mins + scaling envelope -- no bug, budget respected), test_biz_val_monotone_warp_and_ts_leak (gaps_selection_masking-07/12).
NEW prod bugs sensored (xfail strict=False): (1) MRMRTreeRescued.__init__ uses *args/**kwargs -> sklearn clone/get_params/set_params raise (validated: ctor at _mrmr_tree_rescue.py:47); (2) MRMR redundancy tie f vs strictly-monotone warp g=exp(4f) -- MI-monotone-invariance means the warp can win, costing a downstream linear model; (3) the layer-17 gain-ratio leak audit does not flag a PARTIAL time-series leak (corr~0.7, below corr~1 direct-leak detection).

### FUTURE-tranche wave-3 (5 new files, integrated)
test_selector_determinism_hygiene (shared_lift-16/17/18: fit-level global-RNG, n_jobs parity, PYTHONHASHSEED subprocess), test_biz_val_real_data_noise_injection (bizvalue-09: real datasets + noise injection, selector ADDS value vs unregularized all-features), test_fe_nullable_dtype (gaps_fe_masking-09), test_fe_replay_provenance_exactness (gaps_fe_masking-07/14: exact recipe replay parity + frozen-extra provenance), test_biz_val_rfecv_runtime_win (bizvalue-02).
NEW findings sensored (xfail strict=False): (1) MRMR(fe_max_steps>=1).fit CRASHES (TypeError) on a pandas nullable-dtype (Int64/Float64/pd.NA) frame -- nullable-backed .to_numpy() yields object+pd.NA the FE path can't consume; (2) mlframe's MBH RFECV is NOT faster than sklearn.feature_selection.RFECV at score parity on the measured fixture -- the runtime-win value hypothesis is refuted (honest, documented).

### FUTURE-tranche wave-4 (5 new files, integrated)
test_fe_mechanisms_task_axis (param_axes-01: non-orth FE families on regression+multiclass), test_fe_default_pipeline_null_fwer (gaps_fe_masking-01 + param_axes-05: shipped-default pipeline null FWER + all-FE-on x prod-default redundancy), test_fe_target_aware_leak_contract (param_axes-09: parametrized permuted-y OOF leak gate over 9 target-aware families), test_fe_float32_replay_parity (param_axes-13: modern-FE float32 fit+replay, f32==f64 selection + allclose recipe values), test_fe_value_across_model_families (bizvalue-12: FE lifts linear + no-harms GBDT, two-sided complementarity).

### FUTURE-tranche wave-5 (4 new files: training-integration, integrated)
test_predict_roundtrip_fs_parity (training_integration-01, THE P0: train(MRMR-FS) -> save -> load -> predict bit-equal + NO recovery fallback -- PASSES, fit-state survives the round trip), test_predict_fs_recovery_behavioral (training_integration-02: production predict-time FS recovery branch exercised behaviorally, no AST scan), test_weight_aware_fs_suite (training_integration-04), test_fs_empty_selection_and_observability (training_integration-07/09).
NEW prod bugs sensored (xfail strict=False): (1) use_sample_weights_in_fs=True does NOT propagate sample_weight to the MRMR selector (weights never reach fit -- weight-aware FS is inert); (2) _build_feature_selection_report reads a drifted RFECV importance attribute (the report observability surface diverged from the wrapper's real attrs).
