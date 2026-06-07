# train_mlframe_models_suite — consolidated critique & disposition (2026-06-04)

10 read-only critique agents, **157 findings**. Per-area detail in the sibling `A1..A10` files.
Disposition buckets: **RESOLVE-NOW** (fix this session, sequential) · **FUTURE** (real, out-of-scope this session, tracked) · **DOC** (docstring/doc caveat only) · **REJECTED** (anti-recommendation w/ reason) · **OK** (positive confirmation, no action).

## Headline

- **No P0 leakage of the honest holdout (test/OOS) into any decision** — confirmed across FS, FE, calibration, composite discovery, thresholding (A7-06..A7-11, A7-13). The suite is methodologically strong.
- **One genuine P0**: cross-target **stack weights are fit on the VAL/early-stopping split**, not honest OOF (A3-01) — biased, selection-inflating. → RESOLVE-NOW.
- **README + examples are the worst user-facing problem**: ~9 fictional/stale API snippets that `ImportError`/`TypeError` on copy-paste (A9-01..07, A10-09/10). → RESOLVE-NOW.
- **25 monoliths** have concrete split plans (A6) — planning was the ask ("продумать"); execution is a dedicated serial effort → FUTURE.

## Rollup
RESOLVE-NOW: 38 · FUTURE: 71 · DOC: 14 · OK: 19 · REJECTED: 1 · (memory-correction: 1, verify) 

---

## A1 — Feature selection (13)
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A1-01 | P1 | MRMR silently ignores `groups` | RESOLVE-NOW (surface to metadata + loud WARN; strict_groups when split uses groups) |
| A1-02 | Low | Suite default does NO supervised FS | DOC (document default explicitly) |
| A1-03 | P2 | MRMR `random_seed=None` not seeded from split | RESOLVE-NOW |
| A1-04 | Low | `skip_retraining_on_same_shape` misnomer | RESOLVE-NOW (rename + alias + test) |
| A1-05 | P2 | Pre-screen failure swallowed/latched | RESOLVE-NOW (make atomic/fatal) |
| A1-06 | P2 | Cross-target MRMR identity cache can skip FE | FUTURE (needs y-corr threshold bench) |
| A1-07 | P2 | FS re-fits per model-tier | FUTURE (cache support keyed on target+params+content) |
| A1-08 | Low | RFECV `auto→one_se_max` recall-oriented | DOC (surface resolved rule) |
| A1-09 | Low | Medoid reduction Pearson-only | FUTURE (offer SU/MI corr, bench first) |
| A1-10 | P2 | `screen_predictors` defaults diverge from MRMR.fit | RESOLVE-NOW (align or document override) |
| A1-11 | Low | Empty protected set only WARNs under group split | RESOLVE-NOW (skip/raise) |
| A1-12 | Low | FS retention logged only at verbose | RESOLVE-NOW (one INFO line + metadata) |
| A1-13 | Low | `min_features_fallback=1` silent | RESOLVE-NOW (WARN + metadata) |

## A2 — Feature engineering (14)
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A2-01 | P2 | Memory/docs claim 5 transformers wire into suite; agent found none do | VERIFY → then correct memory+docs (conflict w/ existing memory; do NOT overwrite until confirmed reachable-or-not via registry/config) |
| A2-02 | P1 | `apply_preprocessing_extensions` down-converts polars→pandas even with zero active stages | RESOLVE-NOW |
| A2-03 | P2 | RFF/Nystroem/dim-reducer `random_state` hardcoded 42 | RESOLVE-NOW |
| A2-04 | Low | `_filter_to_numeric` in-place bool→int8, per-split divergence risk | RESOLVE-NOW (compute kept-set on train, apply to val/test) |
| A2-05 | Low | TF-IDF dense object array via `.values` | FUTURE (measure first) |
| A2-06 | P2 | Standalone RFF fits on full X, no OOF/Mode-A | DOC (document train-only contract) |
| A2-07 | Low | local_lift PR-AUC mislabeled | DOC |
| A2-08 | Low | Composite discovery train-only — CLEAN | OK |
| A2-09 | Low | CompositeTargetEstimator no-down-conversion — CLEAN | OK |
| A2-10 | P2 | Predict-path extensions return RAW frame on transform failure (silent wrong preds) | RESOLVE-NOW (re-raise) |
| A2-11 | Low | Extensions fit once, no redundant recompute — CLEAN | OK |
| A2-12 | Low | Default target encoder leakage-safe but undocumented | DOC |
| A2-13 | Low | `copy.copy` config fallback in auto-tune | RESOLVE-NOW (deepcopy/model_copy) |
| A2-14 | Low | Mode-B class_distance self-label leak if query overlaps train | DOC (precondition) |

## A3 — Ensembling (14)
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A3-01 | **P0** | Cross-target stack weights derived on VAL/ES split, not honest OOF | RESOLVE-NOW (+ regression/biz_value test) |
| A3-02 | P1 | No `__getstate__`; stashed train-pred matrix pickles every save | RESOLVE-NOW |
| A3-03 | P1 | OOF refit `time_ordering`/`sample_weight`/`group_ids` always None (unbound ctx) | RESOLVE-NOW |
| A3-04 | P1 | Predict-time dropout refit on val-biased matrix, non-deterministic | RESOLVE-NOW (follows A3-01) |
| A3-05 | P1 | `_train_pred_cache` keyed on `id()`+shape | RESOLVE-NOW (content fingerprint) |
| A3-06 | P2 | Dead full-train predict at line 962 | RESOLVE-NOW (delete) |
| A3-07 | P2 | No diversity/decorrelation control | FUTURE (wire residual-corr drop; biz_value bench) |
| A3-08 | P2 | `oof_weighted` baseline = worst component | RESOLVE-NOW (pass dummy floor) |
| A3-09 | P2 | OOF gate renormalises nnls weights ≠ deployed predictor | RESOLVE-NOW |
| A3-10 | P2 | Pre-screen drops on leaky val_RMSE | RESOLVE-NOW (after A3-01) |
| A3-11 | Low | Module OOF LRU cache dead on suite path | FUTURE |
| A3-12 | Low | Auto time-split flips on any monotone base col | RESOLVE-NOW (opt-in / explicit time col) |
| A3-13 | Low | Composite-branch eval carve group-blind | RESOLVE-NOW (thread group_ids) |
| A3-14 | Low | MTR per-column NNLS fits on val | FUTURE (route honest OOF; equal_mean default until then) |
| — | — | `composite_cache.DiscoveryCache` well-built (content-hash, 1GiB ceiling, no live objects) | OK |

## A4 — Efficiency / caching / conversions (7) — suite already mature
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A4-01 | P2 | Pipeline cache key recomputes invariant digests per (pre_pipe×model) | FUTURE (cProfile first; likely <1% on narrow frames) |
| A4-02 | P2 | `_pre_pipeline_cache_key` computed twice/fit, relies on single-slot memo | RESOLVE-NOW (pass precomputed key — correctness hardening, bit-identical) |
| A4-03 | Low | target hash lacks memo X-side has | FUTURE |
| A4-04 | Low | `_canonical_dtype_pairs` re-stringifies invariant schema | FUTURE |
| A4-05 | Low | model-input fingerprint cache `id()`-keyed | FUTURE (content sig; consistency) |
| A4-06 | Low | verbose-only `get_strategy` re-walk | FUTURE |
| A4-07 | Low | mostly-inert pandas-view memo | REJECTED (leave as-is; agent recommends no change unless profiling shows 0 hits) |
> No hot-path `df.copy()`/`clone()`/reference-copy; polars→pandas deferred+size-gated+cached; pickle protected; caches content-keyed. All explicitly confirmed clean.

## A5 — Tests (14)
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A5-01 | P1 | `test_target_type_combinations.py` never calls the entrypoint | RESOLVE-NOW |
| A5-02 | P1 | Predict round-trip parity regression-only + self-skips on Windows | RESOLVE-NOW |
| A5-03 | P1 | No multiclass/multilabel predict parity test | RESOLVE-NOW |
| A5-04 | P2 | 435 source-text-read assertions (getsource ban renamed) | FUTURE (large triage) |
| A5-05 | P2 | Weak is-not-None FE coverage tests | FUTURE |
| A5-06 | P2 | Outlier/imbalance biz_value `xfail` on miss | FUTURE (re-tune to hard floor) |
| A5-07 | Low | is-not-None sole-assertion long tail | FUTURE |
| A5-08 | P2 | Thin `@pytest.mark.fast` adoption | FUTURE |
| A5-09 | P2 | Sleep-based mtime/LRU tests | FUTURE |
| A5-10 | P2 | Reload/sys.modules whole-file-guard pollution risk | FUTURE (harden meta-linter) |
| A5-11 | P2 | MTR lacks quantitative biz_value test | FUTURE |
| A5-12 | Low | ~95 `layerNN` mrmr files hurt discoverability | FUTURE (index, don't delete) |
| A5-13 | Low | Global `-x` in addopts hides full failure set | RESOLVE-NOW (drop -x) |
| A5-14 | Low | No multi-target shared fixture | FUTURE |

## A6 — Monolith split plans (25) → all FUTURE (plans delivered in A6_monolith_splits.md)
A6-01 _regression_extras (High/S) · A6-02 _classification_extras (High/S) · A6-03 discretization (High/M) · A6-04 engineered_recipes (High/L) · A6-05 _orthogonal_univariate_fe (High/M) · A6-06 _orthogonal_scorer_auto_fe (Med/S) · A6-07 wrappers/_helpers (Med/S) · A6-08 _shap_proxy_revalidate (High/M) · A6-09 boruta_shap (Low-Med/M) · A6-10 _rfecv (Med/M) · A6-11 _flat_torch_module (Med/M, pickle) · A6-12 _mrmr_fit_impl 5993 (High/L, highest care) · A6-13 mrmr.py (Med/M — `__init__` un-splittable, stays >1k; comment-hygiene only) · A6-14 _phase_train_one_target_body (Med-High/L) · A6-15 _mrmr_fe_step (Med/M-L) · A6-16 _feature_engineering_pairs (Med/M-L, shared lock) · A6-17 _composite_discovery_fit (Med/M) · A6-18 _screen_predictors (Med/M) · A6-19 _phase_composite_post_xt_ensemble (Med/M-L) · A6-20 shap_proxied_fs (Med/M) · A6-21 _dynamic_cluster_discovery (Med-High/M-L, _DCD_STATE) · A6-22 _confirm_predictor (Med/M) · A6-23 neural/base (Med/M-L, pickle) · A6-24 hermite_fe (Med/L, kernel-ladder ref) · A6-25 _param_oracle (Med/S-M).
> Recommended first execution targets (lowest risk, clean symbol carves): A6-01, A6-02, A6-25. Each split requires the AST-scope gate + a call-into-moved-body sensor (CLAUDE.md).

## A7 — ML best practices (14)
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A7-01 | P1 | `calib_size` declared/validated/documented but never carves a slice (inert) | FUTURE (wire end-to-end OR demote+fix docstring; design-heavy) |
| A7-02 | P1 | OOF uses shuffled KFold even on temporal splits; feeds ensemble selection | RESOLVE-NOW (+ test) |
| A7-03 | P2 | OOF seed hardcoded 42; CalibratedClassifierCV unseeded | RESOLVE-NOW |
| A7-04 | P2 | Threshold fixed 0.5, never tuned (not leaked) | DOC (or FUTURE: val/OOF tuning) |
| A7-05 | P2 | Tree "calibration" is eval-metric swap; post-hoc hook no-op | DOC (clarify docstring) |
| A7-06..11,13 | OK | 7 positive confirmations (disjoint-calib, OOF→val→test rank, train-only discovery, leak-careful split, nested RFECV, single test consumption, multilabel calib) | OK |
| A7-12 | Low | Drift report auto-wire claim is verbose-gated | RESOLVE-NOW (verify + ungate or soften doc) |
| A7-14 | OK/minor | Bootstrap CIs stratified+seeded; derive per-target seed from suite seed | RESOLVE-NOW (minor seed plumb) |

## A8 — Software standards (20)
| ID | Sev | Short | Disposition |
|----|-----|-------|-------------|
| A8-01 | OK | CompositeTargetEstimator sklearn conventions good | OK (add clone-raise test) |
| A8-02 | P2 | `check_estimator` never run | RESOLVE-NOW (add parametrize_with_checks test w/ expected failures) |
| A8-03 | P2 | Predict methods bound by runtime class-attr (invisible to tooling) | RESOLVE-NOW (delegating stubs) |
| A8-04 | P1 | `_ensure_config` dict path absorbs typo'd keys silently | RESOLVE-NOW |
| A8-05 | OK | 34-kwarg facade, 0 mutable defaults | OK |
| A8-06 | P1 | God-orchestrator: dual local/ctx state + `locals()` reflection copy | FUTURE (finish ctx migration; large) |
| A8-07 | P2 | Process-wide irreversible monkeypatching | DOC (document side effects) |
| A8-08 | Low | `print()` in library paths | RESOLVE-NOW (→ logger) |
| A8-09 | P2 | Private `_*` re-exported under public names | DOC |
| A8-10 | P2 | Bare `Tuple[Dict,Dict]` return contract | FUTURE (TypedDict) |
| A8-11 | Low | ctx config slots all `Any` | FUTURE (type w/ ctx migration) |
| A8-12 | P2 | Docstring documents ~13/34 params | RESOLVE-NOW |
| A8-13 | OK | Error-handling discipline strong | OK |
| A8-14 | P2 | 2 broad-except swallow sites in orchestrator | RESOLVE-NOW (narrow) |
| A8-15 | Low | `from typing import *` + dead import | RESOLVE-NOW |
| A8-16 | OK | Optional-dep guards textbook | OK |
| A8-17 | P2 | lgb_shim eval-set heuristic fragile | FUTURE (normalize at boundary) |
| A8-18 | OK | Live-cache `__getstate__` correct | OK |
| A8-19 | P2 | `_build_configs_from_params` drops unknown reporting kwargs | RESOLVE-NOW (rebuild from model_fields) |
| A8-20 | Low | `verbose` 0/1/2 contract not honored | DOC (or RESOLVE: map to level) |

## A9 — Docs (24): A9-01..07 README P0 stale API → RESOLVE-NOW; A9-08..11 calibration/honest-diagnostics module-path+phantom-knobs → RESOLVE-NOW; A9-12 FHC currency → RESOLVE-NOW; A9-13/14/15/18/24 research-doc-currency → RESOLVE-NOW (mark shipped / repoint); A9-16/19 default-drift → RESOLVE-NOW; A9-17 benchmark cmd → RESOLVE-NOW; A9-20/21 docs org+index → RESOLVE-NOW (add indexes; defer file moves); A9-22 leaked temp path → RESOLVE-NOW; A9-23 CONTRIBUTING py version → RESOLVE-NOW.

## A10 — Examples/notebooks (12): A10-01 notebook import → RESOLVE-NOW; A10-02 pip path → RESOLVE-NOW; A10-03 notebook fast-path/CI → FUTURE; A10-04/05 anaconda path in scripts → RESOLVE-NOW; A10-06/07/12 repros are dead weight (bugs fixed+covered) → FUTURE (delete pending user OK — files not authored this session); A10-08 default drift → RESOLVE-NOW; A10-09/10 README → RESOLVE-NOW (≡ A9-01..07); A10-11 scripts/README → RESOLVE-NOW.

---
### Notes
- A2-01 vs existing project memory `mlframe_fe_transformer_shortlist` conflict: do not overwrite memory until the registry/config reachability is re-confirmed.
- All edits this session are sequential, surgical (Edit, CRLF-preserving), no audit-IDs/dates in code comments (CLAUDE.md), no commits unless asked.
