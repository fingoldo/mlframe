# gt_03: Banzhaf value feature ranking — noise-robust semivalue attribution

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

The Shapley value weights a feature's marginal contribution v(S∪{j})−v(S) by 1/(P·C(P−1,|S|)) —
coalition sizes are weighted unequally, and extreme sizes (tiny/near-full coalitions) get high
per-subset weight. The **Banzhaf value** weights ALL coalitions uniformly:
β_j = (1/2^{P−1}) · Σ_{S⊆N∖{j}} [v(S∪{j}) − v(S)].
Wang & Jia (AISTATS 2023, "Data Banzhaf: A Robust Data Valuation Framework for Machine Learning")
proved Banzhaf is the MOST noise-robust semivalue: when v(S) is evaluated with stochastic noise
(here: proxy-loss noise from OOF-SHAP fold variance, booster seed jitter, finite-sample loss),
Banzhaf's ranking is provably the most stable to perturbations of v among all semivalues — exactly
the property a feature RANKING consumed by a top-K prescreen cut needs. Their Maximum Sample Reuse
(MSR) estimator gets β for ALL features from ONE shared pool of sampled coalitions:
β̂_j = mean_{S∋j} v(S) − mean_{S∌j} v(S), over m coalitions sampled uniformly (each feature
included independently w.p. 1/2) — every sample informs every feature.

Fit with ShapProxiedFS: the prescreen ranks proxy columns by `mean|φ|` (Shapley-based, single
model's attribution). A Banzhaf ranking over the SAME proxy game (`v(S) = −subset_loss(S)`) costs
m × O(n) vector reductions for m coalitions — thousands are cheap — and is more stable across
seeds on noisy/low-SNR frames, which directly reduces the seed-to-seed churn of the selected set.

## 2. Integration verdict

**Phase 1 (this plan's mandatory scope): config extension of ShapProxiedFS** — new
`prescreen_ranking: str = "mean_abs_phi"` param with a `"banzhaf"` option, swapping the ranking
used in the prescreen keep-set. Cheap, contained, directly measurable.

**Phase 2 (optional, only if phase-1 bench shows large wins): full `BanzhafFS` selector** ranking
features purely by MSR-Banzhaf over a proxy game built from a single model fit — a lighter-weight
sibling of ShapProxiedFS without the search/trust/revalidate stack. Follow the 6-step selector
contract from README (registry.py, config flag, pre-pipelines branch, contract-test factory,
biz_val file). Do NOT start phase 2 before phase-1 results exist.

## 3. Existing machinery to reuse (verified paths)

- Proxy game oracle: `_Evaluator` (`src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_heuristics.py`)
  — memoised `loss(idx_tuple)`, contiguous-transpose fast path. m=4096 coalitions at n=3000,
  width≤112 is well under a second.
- Prescreen seam: `_shap_proxied_fit.py`, block `with _stage("prescreen"):` — currently
  `importance = np.abs(phi).mean(axis=0); top_keep = np.argsort(-importance)[:prescreen_top]`,
  then unions `noise_floor_rescue_keep_set(importance, top_keep)` and `_su_rescue_proxy_idx`.
  The swap point is the `importance` vector itself: with `"banzhaf"`, importance := β̂ (shifted
  nonnegative: β̂ − min(β̂), so the downstream noise-floor rescue math keeps working on a
  nonnegative vector — document this shift in a comment).
- Also swap-eligible: the knee ladder (`_resolve_knee_prescreen_cap` in
  `_shap_proxied_resolvers.py`) consumes the same importance vector — it inherits the swap for
  free since it receives `importance` as an argument.

## 4. Implementation steps (phase 1)

1. **New module** `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_banzhaf.py`:

   ```python
   def banzhaf_msr(
       phi: np.ndarray, base: np.ndarray, y: np.ndarray, *,
       classification: bool, metric: str | None,
       n_coalitions: int = 4096, rng: np.random.Generator,
       batch: int = 256,
   ) -> tuple[np.ndarray, dict]:
       """MSR Banzhaf estimate over the additive proxy game.

       Sample m boolean masks (n_features,) with p=0.5 per feature. v(S) = -loss(S) via a local
       _Evaluator. beta_j = mean(v | mask_j) - mean(v | ~mask_j). Guard: features never/always
       sampled (won't happen at m>=64 and P>=2, but guard anyway) get beta=0 + warning in info.
       Returns (beta (n_features,), info = dict(n_coalitions, v_mean, v_std,
       beta_stderr (n_features,) from the two-sample mean stderr)).
       Vectorization note: masks matrix (m, P) -> coalition margins can be computed as
       base + masks @ phi_T-like batched matmul: margins_all = base[None, :] + masks @ phi.T is an
       (m, n) matrix -- at n=3000, m=4096 that is ~98MB float64, hence the `batch` param: process
       masks in chunks of `batch`, computing chunk @ phi.T (batch, n) then per-row metric. This is
       MUCH faster than m sequential _Evaluator calls; implement the batched path directly (reuse
       score_margin_auto from _shap_proxy_objective for the per-row metric reduction)."""
   ```

2. **Constructor** (facade `__init__.py`): `prescreen_ranking: str = "mean_abs_phi"` (validate in
   `("mean_abs_phi", "banzhaf")`, store verbatim), `banzhaf_n_coalitions: int = 4096`.
3. **Prescreen swap** (`_shap_proxied_fit.py` prescreen block): resolve ranking mode; for
   `"banzhaf"` compute β̂, `importance = beta - beta.min()`; everything downstream (top-K cut,
   noise-floor rescue, su rescue union, knee ladder input) unchanged. Report:
   `report["prescreen"]["ranking"] = mode`, plus `banzhaf_stderr_max` when applicable.
4. **cProfile harness** `_benchmarks/profile_banzhaf_ranking.py`: batched-matmul path wall at
   (n, P, m) ∈ {(3000, 112, 4096), (10000, 112, 4096)}; assert prescreen stage wall increase ≤
   0.5s at the first point.

## 5. biz_val tests

File: `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_banzhaf_ranking.py`.

1. `test_biz_val_banzhaf_ranking_seed_stability_low_snr` — the headline claim. Bed: n=2000,
   p=500 post-prefilter-ish scale, 10 informative at LOW snr (make_regime_dataset with snr=1.5)
   or inline generator. Fit ShapProxiedFS 4 seeds × both rankings; metric = mean pairwise Jaccard
   of `selected_features_` across seeds. Threshold: Jaccard(banzhaf) ≥ Jaccard(mean_abs_phi) + 0.05
   (per Wang & Jia's robustness result; if measured delta is larger, floor 5-15% below measurement).
   `@pytest.mark.slow @pytest.mark.timeout(900)`.
2. `test_biz_val_banzhaf_ranking_no_regression_high_snr` — clean high-SNR bed: recall of
   informatives identical between rankings (both should be perfect), downstream-AUC within ±0.005.
3. `test_biz_val_banzhaf_estimator_matches_exact_small_p` — unit-ish biz test: P=10 features,
   compute EXACT Banzhaf by full 2^9 enumeration per feature over the proxy game; MSR at m=4096
   must correlate Spearman ≥ 0.95 with exact and top-5 sets must match ≥ 4/5.

Unit tests: dummy feature (φ column of zeros) gets β≈0; batched path vs sequential `_Evaluator`
path agree to 1e-10 on a small fixture (bit-identity of the vectorization); clone round-trip;
validator rejection.

## 6. Phase 2 sketch (BanzhafFS — implement ONLY on strong phase-1 results)

Selector: fit one booster on all columns (reuse ShapProxiedFS's prefilter for width), compute phi
once, rank by MSR-Banzhaf, select top-k with the noise-floor rescue, honest-validate the single
proposed set. Params: `top_k`, `n_coalitions`, `prefilter_top`. Value proposition vs ShapProxiedFS:
5-10× faster (no search/trust/revalidate/refine), for callers wanting a fast robust ranking rather
than a curated subset. All 6 contract steps from README §"Adding a full feature selector"; biz_val:
speed (≥3× faster than ShapProxiedFS same width) + stability (as above) + recall within 1
informative of ShapProxiedFS on the standard regimes.

## 7. Acceptance criteria
- Phase 1: params wired + clone-safe; batched estimator bit-identical to reference path;
  exact-vs-MSR agreement test green; both biz_val beds green locally
  (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`); cProfile committed.
- Default stays `"mean_abs_phi"` until a majority-win bench (stability wins alone do not flip a
  default that affects recall paths — bench recall regimes too, per README rules).
- Phase 2 gated on an explicit go-decision documented in the phase-1 bench file.

## 8. Known risks / rejected alternatives
- Semivalue over RETRAINED submodels (true Data-Banzhaf-style, v = retrained holdout score):
  rejected — thousands of retrains is the cost the proxy exists to avoid; the proxy game keeps it
  cheap and the noise-robustness argument applies to proxy noise identically.
- Beta on the SEARCH stage (replacing beam's loss ranking): out of scope — the search needs the
  actual coalition loss, not per-feature values; only the prescreen RANKING is swapped.
- Shifted-beta interaction with noise-floor rescue: the rescue's `median(bottom half)*4` floor
  assumes a noise-dominated tail; β̂'s tail for pure-noise features concentrates near 0 after the
  shift, same shape — but VERIFY on the low-SNR bed that `noise_floor_rescued` counts stay sane
  (add that assertion to biz_val test 1).
