# gt_09: Two-phase (residual) attribution — recovering weak-signal credit in ShapProxiedFS

Read `research/README.md` (shared conventions) first. This document is self-contained otherwise.

## 1. Problem & motivation

SHAP attribution on a FULL model divides credit among all features jointly. When a few strong
features explain most of the target, genuinely-predictive weak features receive small mean|φ| —
not because they carry no signal, but because the strong features absorb the shared credit. A model
RETRAINED without the strong features would extract more from the weak ones; the single-pass proxy
cannot see that (the documented "<50% coverage wall" in the ShapProxiedFS facade docstring).

Empirical trace (full details in
`src/mlframe/feature_selection/shap_proxied_fs/_benchmarks/PLAN_wide_dataframe_improvements.md`):
on a p=3000 fixture with 6 strong (weight 1.0) + 6 weak (weight 0.25) features, weak features
survive every pipeline cut (prefilter, clustering, prescreen, even the beam search's winning
candidate) yet end at mean|φ| ranks 6/7/20/45 of 112 — and downstream stages tuned for precision
(`parsimony_tol`) prune them. The weak features' phi is real but SMALL relative to strong.

Fix idea (user-proposed, this plan): a second attribution pass on the RESIDUALS of the first
model. In pass 2 the strong features' signal is already explained away, so weak features become
the dominant explanators of what remains and earn full credit. This is boosting's insight applied
to attribution rather than prediction.

## 2. Existing machinery (verified against current source)

`src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_explain.py:505`:

```python
def compute_shap_matrix(model_template, X: pd.DataFrame, y: np.ndarray, *,
    classification: bool, out_of_fold=True, n_splits=5, n_models=1,
    config_jitter=False, return_variance=False, rng=None, tqdm_desc=None,
    shap_backend="auto", n_jobs=1, n_estimators_cap=None, inner_n_jobs_cap=False,
    return_per_fold_phi_mean=False, cache_dir=None)
# -> (phi (n,f), base (n,), y_aligned)  [+ optional tails]
```

Facts that shape the design:
- **Custom `y` is already supported** — plain positional array. A residual pass is "call it again
  with `y = residuals`".
- **`classification=True` fits classifiers on discrete labels and attributes in margin/log-odds
  space.** Continuous residuals would break the classifier fit → the residual pass MUST run with
  `classification=False` (regression booster on continuous residuals), regardless of the outer
  task.
- **No `sample_weight` / `base_margin` / `init_score` passthrough exists** in `_fit_one` (`:394`)
  — a true boosting warm-start is NOT natively expressible. The residual pass is implemented at
  the caller: construct residual y, call again as regression. Do not add base_margin plumbing in
  this plan (larger blast radius; the caller-side residual is sufficient).
- `sparse_interaction_candidates` already demonstrates the exact call shape for an auxiliary pass:
  `out_of_fold=False, n_models=1` (`_shap_proxy_interactions.py:569`). For the residual pass use
  `out_of_fold=True` (same rigor as pass 1) but allow `n_splits` reuse.

Fit orchestration seams (`src/mlframe/feature_selection/shap_proxied_fs/_shap_proxied_fit.py`):
- Pass-1 OOF-SHAP call: the `with _stage("oof_shap"):` block (~`:477-498`).
- Prescreen keep-set construction: the `with _stage("prescreen"):` block — already unions THREE
  sources: top-K by mean|φ|, `noise_floor_rescue_keep_set(...)` rescued indices, and
  `_su_rescue_proxy_idx`. **The residual rescue is a FOURTH union member — the pattern is
  established, follow it.**
- Refine protection: `within_cluster_refine`
  (`_shap_proxy_revalidate/_shap_proxy_refine.py:535`) will re-prune residual-rescued weak
  features via `parsimony_tol` (this is exactly what happened in the empirical trace). See §3.4.

OOF predictions for residuals: pass 1's `compute_shap_matrix` does NOT return OOF predictions,
only (phi, base, y). Two options: (a) reconstruct margin predictions as
`margin_pred = base + phi.sum(axis=1)` — this is exactly the full-coalition proxy margin, zero
extra cost, consistent-by-construction with the attribution; (b) run a separate OOF predict. Use
**(a)**: it is free and the residual then measures precisely what the additive attribution failed
to explain, which is the quantity we want to re-attribute.

## 3. Design

### 3.1 New constructor params (facade `__init__.py`)
```python
residual_passes: int = 0,          # 0 = legacy single-pass (default until benched)
residual_merge: str = "rescue",    # "rescue" | "blend"
residual_lambda: float = 1.0,      # only used when residual_merge="blend"
residual_top_k: int | None = None, # rescue pool size per residual pass; None -> brute_force_max_features
residual_exclude_top: int = 0,     # "hard residual": drop this many pass-1 top features from pass-2 X
```
Store verbatim (sklearn clone rule); validate at fit time. `residual_passes` is capped at 2 in
validation (each pass costs one full OOF-SHAP; >2 has no plausible payoff — document, don't
silently clamp).

### 3.2 Residual construction
After pass-1 phi/base are available (post-prescreen is WRONG — run on the pre-prescreen phi so
rescue can save columns the prescreen would cut; hook immediately after the `oof_shap` stage and
BEFORE the knee/prescreen block):

- Regression (`classification=False` outer): `residual = y_phi - (base + phi.sum(axis=1))`.
- Classification (`classification=True` outer): work in margin space.
  `margin_pred = base + phi.sum(axis=1)` is already log-odds (TreeSHAP on binary xgboost
  attributes margin space; `base` is the margin base value). Target margin for a {0,1} label has
  no finite value, so use the standard gradient-of-logloss residual instead:
  `residual = y_phi - sigmoid(margin_pred)` (the pseudo-residual of logistic loss — bounded,
  continuous, exactly what a boosting step would fit). This avoids the log-odds(clip(p)) infinity
  handling entirely and is the mathematically standard choice.
- Pass 2: `phi2, base2, _ = compute_shap_matrix(model_template_regression, X_proxy_pass2,
  residual, classification=False, out_of_fold=True, n_splits=self.n_splits,
  n_estimators_cap=self.oof_shap_n_estimators, rng=..., n_jobs=self.n_jobs, cache_dir=...)`.
  `model_template_regression`: when the user passed `model=None`, build the default regressor
  template (the module already builds default boosters per task — reuse that path); when the user
  passed a custom classifier, derive an xgboost regressor with the same tree budget (document
  this in the param docstring; do NOT try to clone a classifier into a regressor generically).
- `X_proxy_pass2`: same `X_proxy` as pass 1, minus the top `residual_exclude_top` columns by
  pass-1 mean|φ| when that param > 0 ("hard residual" — forces pass 2 to explain without the
  strong features at all; benchmark both variants, see §5).

### 3.3 Merging credit — two modes
- `residual_merge="rescue"` (RECOMMENDED default): pass-1 coalition proxy stays untouched (search
  still optimizes the pass-1 additive proxy — mixing phi scales would corrupt the coalition
  value's meaning). The residual pass contributes ONLY a rescue set: top `residual_top_k` proxy
  columns by mean|φ₂| are unioned into the prescreen keep-set (fourth union member, §2) AND
  recorded as `residual_rescued` in `report["prescreen"]`.
- `residual_merge="blend"`: `phi_combined = phi1 + residual_lambda * phi2_aligned` (phi2 columns
  re-aligned to phi1's column order; excluded columns get 0) is used for the IMPORTANCE ranking
  only (prescreen keep order), while the search/coalition proxy still consumes raw phi1.
  Document clearly: blend changes ranking, never the proxy loss itself.

### 3.4 Protecting rescued features downstream (critical — the empirical trace shows prescreen
survival is NOT sufficient)
Rescued columns die again in `within_cluster_refine`'s greedy `parsimony_tol` drop. Add a
`protected_cols: set[int] | None = None` parameter to `within_cluster_refine`
(`_shap_proxy_refine.py:535`): protected members are excluded from drop trials (never proposed
for removal). The fit orchestration passes the residual-rescued ORIGINAL column indices when
`residual_passes > 0`. This is a small, backward-compatible extension (default None = no change);
it also creates the seam gt_02 (core-stability refine) plugs into. The honest REVALIDATION stage
still arbitrates freely — protection applies only to refine's greedy pruning, so a rescued
feature that genuinely hurts still loses in revalidate_top_n's candidate comparison.

### 3.5 Reporting
`report["residual_pass"] = dict(n_passes, merge, lambda, rescued=list_of_original_names,
excluded_top=list, pass2_top_importance=first_10_pairs, residual_std_before/after)` — the
before/after residual std doubles as a cheap convergence diagnostic (pass 2 explaining nothing →
std barely moves → telemetry for a future auto-gate).

## 4. biz_val tests

File: `tests/feature_selection/shap_proxied/test_biz_val_shap_proxied_residual_passes.py`.
Fixture generator: copy the session fixture (see
`test_biz_val_shap_proxied_parsimony_tol_recall.py::_make_mixed_strength_fixture` — n=3000,
p=3000, 6 strong w=1.0 at cols 0-5, 6 weak w=0.25 at cols 50-55, logit normalized to std*2).

1. `test_biz_val_residual_passes_recovers_weak_recall` — `residual_passes=1, residual_merge="rescue"`
   vs default 0. Measured baseline: weak recall 0/6 at defaults. Threshold: recall ≥ 3/6 with the
   residual pass, downstream-AUC (xgboost 300 trees on selected, 30% holdout) ≥ baseline − 0.005.
   Mark `@pytest.mark.slow @pytest.mark.timeout(900)` (two full fits).
2. `test_biz_val_residual_passes_no_noise_inflation` — pure-strong bed (6 strong + noise, NO weak):
   `residual_passes=1` must not add noise columns: n_selected(residual) ≤ n_selected(default) + 1
   and zero noise columns selected (precision guard — the residual of a well-explained target is
   noise; pass 2's top-k must fail the refine/revalidation arbitration).
3. `test_biz_val_residual_hard_vs_soft` — same fixture as (1), `residual_exclude_top=6` vs 0:
   report both weak recalls; assert hard ≥ soft (informational floor; if it fails consistently,
   the bench verdict is soft-wins — record numbers, adjust plan).

Unit tests: param validation (negative passes/λ rejected, passes>2 rejected), clone round-trip,
`protected_cols` honoured by `within_cluster_refine` (pure-function test: protected member never
dropped even at parsimony_tol=1.0), classification residual is `y - sigmoid(margin)` (numerical
check on a tiny fixture), report keys present.

## 5. Rollout & default decision
Ship as opt-in (`residual_passes=0` default). Then run the default-flip bench: fixtures = the
mixed-strength bed, a pure-strong bed, a pure-noise bed, and 2 `make_regime_dataset` regimes
(additive high-SNR, redundancy) × 3 seeds, arms = {0 passes, 1 pass rescue, 1 pass blend λ=1,
1 pass hard-residual}. Flip to `residual_passes=1` only on the standard majority-win rule (wins
weak-recall beds, ties pure beds, wall-clock overhead ≤ ~1.8× acceptable given a second OOF-SHAP —
report it honestly). Otherwise keep opt-in with the bench table committed (REJECTED ≠ DELETED).

## 6. Acceptance criteria
- All new params wired, validated, clone-safe; `report["residual_pass"]` populated.
- biz_val 1-3 green locally (`CUDA_VISIBLE_DEVICES="" ... --no-cov -p no:anyio`).
- `within_cluster_refine(protected_cols=...)` extension covered by a unit test and does not change
  behaviour when None (byte-identity spot check on one seeded fit).
- cProfile harness comparing 0-pass vs 1-pass wall at p={2000, 10000} committed to `_benchmarks/`.
- Default-flip bench committed with verdict recorded either way.

## 7. Known risks / notes for the implementer
- Do NOT compute residuals from post-prescreen phi — rescue must happen before the cut it rescues
  from (order: oof_shap → residual pass → knee/prescreen).
- phi2 column alignment: pass-2 X may be a column subset (hard residual); build an index map, never
  positional-assume.
- Memory: phi2 is another (n, f) float64 — at C4-scale widths that is significant; compute pass-2
  on the SAME already-narrowed X_proxy (post-prefilter), never the raw frame, and free phi2 after
  the rescue set / blend vector is extracted.
- Interaction with gt_08: if `proxy_mode="auto"` fires su_seeded rescue AND residual rescue in the
  same fit, both union into the keep-set independently — no coupling needed, but the report must
  disambiguate the source of each rescued column.
