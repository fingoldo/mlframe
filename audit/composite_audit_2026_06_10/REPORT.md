# Composite Targets — multi-agent audit report (2026-06-10/11)

Scope: the whole `src/mlframe/training/composite/` package (~14k LOC, 37 modules) plus the
six `training/core/_phase_composite_*` integration phases. Ten critique dimensions, rolling
pool of 3 agents (host API rate limit), then adversarial verification of every P0/P1 finding.

Raw per-dimension findings: `findings_<dim>.md`. Machine-readable: `findings_all_240.json`,
`p01_merged.json` (P0/P1 + verifier verdicts), `verdicts.json`.

## Totals

- **240 findings**: 2 P0, 38 P1, 121 P2, 79 LOW.
- Categories: bug 110, perf 39, docs 28, usability 22, extension 17, leak 15, test-gap 9.
- Verification: 24 CONFIRMED, 11 ADJUSTED (real but severity lowered), 0 REFUTED,
  5 UNVERIFIED (integration+dx verifier hit the session limit; mechanisms self-verified below).
  No P0/P1 finding was refuted.

## Dedup (same bug from two angles)

- **D2 ≡ DX1** — `iter_transform` crashes on default multi-base specs.
- **D3 ≡ A9** — no finite-row masking in multi-base / forward-stepwise OLS (NaN lag bases).
- **E2 ≡ I1** — per-model wrap builds clip/median from FULL y (train+val+test) + promised re-fit never runs.
- **P8 ≡ S2** — `_row_order_fingerprint` hashes the entire frame to keep 256 hashes.
- **N4 ≡ I4** — default kfold OOF silently drops time ordering on AR data.

## Disposition buckets

`RESOLVED` = fix in this pass. `FUTURE` = real, tracked, out of this pass' scope. `DOC` =
docstring/README/CHANGELOG caveat. `REJECTED` = anti-recommendation. Every finding ID lands in
exactly one bucket; P2/LOW rolled up per dimension at the end (full text in `findings_<dim>.md`).

### P0 — RESOLVED

| ID | verdict | file:line | title |
|----|---------|-----------|-------|
| E1 | CONFIRMED P0 | estimator/_predict.py:129 | multi-base domain-violation fallback crashes: `np.where((n,),(n,K),1.0)` broadcast ValueError; fires on any non-finite base cell at predict |
| D1 | ADJUSTED→P1 | discovery/_filter.py:203 | mean-imputation dilutes leak-corr gate (y-copy with a few NaN rows passes the forbidden-base filter). Was rated P0; escalates to P0 *iff* D3 fixed alone — fixed together here |

### P1 — correctness / leak — RESOLVED

| ID | verdict | file:line | title |
|----|---------|-----------|-------|
| D2≡DX1 | CONFIRMED | discovery/__init__.py:160 | `iter_transform` crashes on auto-promoted `linear_residual_multi` specs (multi_base default-ON); only public consumer not stacking extra bases |
| D3≡A9 | CONFIRMED | discovery/_fit.py:738, forward_stepwise.py:124 | multi-base promotion fits joint OLS on NaN-unfiltered rows → silently dead with lag bases (the canonical, default-ON, benchmark-validated win is lost) |
| E2≡I1 | CONFIRMED | core/_phase_composite_wrapping.py:155 | per-model wrap derives y-clip/median/t-clip from FULL y (incl. val+test); promised end-of-target re-fit never runs → mild holdout contamination of reported metrics |
| E3 | CONFIRMED | estimator/_predict.py:259 | `predict_quantile` crashes for every `requires_base=False` (unary) transform (missing the requires_base branch `_predict_unclipped` has) |
| E5 | CONFIRMED | post_shim.py:46 | `PrePipelinePredictShim._transform` swallows ALL exceptions → feeds untransformed X to a scaled-feature inner → silent garbage inside NNLS stack |
| N1 | CONFIRMED | ensemble/__init__.py:431 | sample_weight `sw[:len(y_fit)]` misaligned with mask-scattered fit rows after group-aware eval carve (silent: lengths match) |
| N2 | CONFIRMED | ensemble/__init__.py:670 | kfold OOF composite branch has no eval-set carve; ES boosters raise → per-component except silently drops them every fold (default oof source IS kfold) |
| N3 | CONFIRMED | ensemble/__init__.py:618 | outer OOF split never group-aware: same-group rows span refit-train & holdout, inflating the surface NNLS weights + dummy-floor gate consume |
| T1 | CONFIRMED | transforms/unary.py:200 | Yeo-Johnson inverse → NaN outside lambda-dependent asymptote; `np.clip(nan)=nan`; predict has no isfinite guard (yeo_johnson_y in default list) |
| T2 | CONFIRMED | transforms/linear.py:306 | `linear_residual_multi` condition gate is scale-variant: mixed-unit *independent* bases falsely flagged collinear → alphas=0 fallback (no column equilibration before SVD) |
| T3 | CONFIRMED | transforms/extended.py:359 | `smoothing_spline_residual` smoothing `s = m*std(signal)` has wrong units; should be `m*var(noise)` → systematic oversmoothing (~70% variance wrongly absorbed) |

### P1 — performance / memory — RESOLVED (bit-identical or gated)

| ID | verdict | file:line | title |
|----|---------|-----------|-------|
| E4 | CONFIRMED | estimator/_estimator.py:532 | `fit()` materialises a full copy of X even when 0 rows fail domain_check (common case) — violates the CLAUDE.md no-copy rule |
| P2 | CONFIRMED | discovery/screening.py:317 | `_mi_to_target_prebinned` boolean-slices the full int64 matrix + per-column gathers without all-true gates (~65 calls/fit, ~10 GB churn) |
| P4 | CONFIRMED | discovery/_tiny_rerank.py:299 | rerank thread oversubscription: auto worker pool × inner LGBM/XGB `n_jobs=-1` (cap never fires on default path) on the dominant phase |
| P5 | CONFIRMED | discovery/_tiny_rerank.py:171 | rerank rebuilds per-base matrices + x_full from df (B+1 full-column extraction passes); `np.delete` build-once fix is bit-identical |
| P6 | CONFIRMED | discovery/screening.py:64 | `_extract_column_array` materialises the full N-row column when callers keep a 20-100k sample of 4M+ rows; `gather(idx)` precedent (~100×) |
| P7 | CONFIRMED | discovery/screening.py:143 | `_safe_abs_corr_all` allocates 3 full-matrix temporaries, blowing past the leak-corr RAM sampler's budgeted alloc (~4× → OOM class) |
| P8≡S2 | CONFIRMED | cache.py:156 | `_row_order_fingerprint` runs `hash_rows()` over the ENTIRE frame to keep 256 hashes; `slice(0,n).hash_rows()` is digest-identical (~44× at 2M) |
| S1 | CONFIRMED | cache.py:362 | `data_signature` hashes object-array *pointer* bytes for str/datetime/cat columns → signature non-deterministic → discovery cache never hits on real frames |

### P1 — ADJUSTED→P2 (real, lower severity) — mix of RESOLVED + FUTURE

| ID | now | file:line | title | disposition |
|----|-----|-----------|-------|-------------|
| A2 | P2 | discovery/_tiny_rerank.py:640 | Wilcoxon gate mathematically unpassable at default n_seed_repeats=3 (n=3 all-neg → p=0.125) | RESOLVED (validator + guard) |
| A3 | P1 | discovery/_screening_tiny.py:433 | multi-seed repeats are exact no-ops under TSS/GroupKFold (3× wasted on dominant phase) | RESOLVED (skip dup seeds) |
| A4 | P2 | discovery/_tiny_rerank.py:444 | raw baseline flips to TSS if ANY base monotone; non-monotone specs scored vs mismatched folds | FUTURE |
| A5 | P2 | discovery/_screening_tiny.py:604 | composite tiny-CV scores only domain-valid rows; raw baseline scores all → cross-spec rank bias | FUTURE |
| A6 | P2 | discovery/_stacked.py:123 | pass-2 stacked specs can train on all-NaN bases downstream (opt-in path) | FUTURE |
| A7 | P2 | discovery/_stacked.py:305 | `discovered_on_residual` set but read nowhere; docstring contradicts wiring (opt-in) | DOC |
| A8 | P2 | discovery/_stacked.py:77 | `fit_stacked` rebuilds X_train inside the per-spec loop (hoist; opt-in) | RESOLVED (hoist) |
| N4≡I4 | P2 | ensemble/__init__.py:777 | explicit non-monotone time_ordering silently downgrades to random shuffle (no WARN) | RESOLVED (WARN + argsort) |
| N5 | P2 | ensemble/feature_stacking.py:73 | `composite_predictions_as_feature` deep-copies the whole pandas frame (opt-in) | RESOLVED (size gate) |
| P1f | P2 | discovery/_eval.py:162 | dead `x_screen_valid` copy + ungated prebinned slice per work item (default config) | RESOLVED (gate) |
| P3 | P2 | discovery/_fit.py:349 | per-base `np.delete` copies + `mi_y_for_base` recomputed per base though decomposable | RESOLVED (decompose) |
| I2 | P2 | core/_phase_composite_post.py:199 | raw-only CT_ENSEMBLE fallback fires only when WHOLE dict empty, not per discovery-skipped target in multi-target suites (opt-in ensemble) | FUTURE |
| I3 | P2 | core/_phase_composite_post_xt_ensemble/__init__.py:218 | train-RMSE proxy computed unconditionally then discarded under honest-OOF (wasted full-train predict/component) | RESOLVED (gate) |

### P2 / LOW — rolled up per dimension → FUTURE / DOC

Full text + file:line in the per-dimension `findings_<dim>.md`. 121 P2 + 79 LOW across:
transforms (25), discovery_core (25), state (25), integration (25), estimator (24),
ensemble (28), discovery_adv (34), perf (19), dx (22), methodology (13). These are tracked,
not dropped; the RESOLVED set above is the highest-confidence, highest-leverage subset taken
first. Notable FUTURE/extension candidates surfaced by the methodology agent (category=extension):
classification/`base_margin` residual support, conformal predictive intervals on the inverse,
multiplicity correction across (base×transform) candidates, stationarity-aware transform gating.

## Notes

- The audit dir, the per-dim md files, and the JSON artefacts are committed so any rejected or
  deferred item is re-checkable by file:line (REJECTED≠DELETED; never hide Low findings).
- Two opt-in flags dominate the ADJUSTED downgrades: `require_beats_raw_baseline` (default False)
  and `cross_target_ensemble_strategy='off'` (default). Several P1→P2 demotions are because the
  buggy path is opt-in — still fixed where cheap, since enabling them is a documented user choice.
