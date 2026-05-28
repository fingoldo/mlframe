"""Honest re-validation + trust diagnostics for the SHAP-proxied selector.

The proxy ranks subsets cheaply but is a *biased* estimator of a retrained model's quality (it
attributes the full model restricted to S, not a model retrained on S). Three guards close that gap:

  - ``proxy_trust_guard``: on a sample of anchor subsets, score BOTH the cheap proxy and an honest
    retrain, then report Spearman/Kendall rank-correlation + recall@k of proxy-top vs honest-top.
    Converts "trust me" into measured proxy fidelity on the user's own data; warns below a floor.
  - ``revalidate_top_n``: honestly retrains each proxy-top-N candidate and evaluates it on a holdout
    DISJOINT from the SHAP/objective data (avoids winner's curse over millions of combos). Returns
    the best / most stability-penalised subset, plus a same-size random-subset baseline for context.
  - ``importance_topk_ablation``: gating check that proxy search beats plain SHAP-importance-top-k
    (Aleksey Pichugin's concern: does this reduce to feature importance + a wrapper?).

Honest scoring uses sklearn metrics (not the hot path); rank correlation uses scipy. Lower = better
loss everywhere, to match the proxy objective.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, roc_auc_score, root_mean_squared_error,
)

from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss, resolve_metric

logger = logging.getLogger(__name__)


def _expand(idx, unit_to_members):
    """Map unit indices (proxy / phi space) to original member feature columns (honest-retrain space).

    Identity when ``unit_to_members is None`` (non-clustering mode: idx already are feature columns).
    In clustering mode the proxy ranks in unit space (one column per denoised cluster), but honest
    re-validation must train on the REAL member columns we actually deploy -- so a unit subset expands
    to the union of its clusters' member columns.
    """
    if unit_to_members is None:
        return list(idx)
    cols: list[int] = []
    for u in idx:
        cols.extend(int(c) for c in unit_to_members[int(u)])
    return sorted(set(cols))


class HonestLossCache:
    """Memoizes ``_honest_loss`` across the FOUR honest-retrain stages of one ``ShapProxiedFS.fit``.

    Within a single fit ``(X_tr, y_tr, X_ev, y_ev, model_template, metric, classification)`` are fixed,
    so a retrain's holdout loss is fully determined by the column subset + the model seed. The trust
    guard, the importance ablation, and within-cluster refine all use ``seed=None`` (the template's own
    fixed seed) and frequently re-evaluate the SAME large member subset (e.g. the chosen winner is
    retrained as the ablation's proxy baseline AND as within-cluster-refine's starting ``base``), so
    caching returns those identical results without a duplicate fit. Keyed on
    ``(frozenset(cols), seed)`` -- order-independent, so column permutations of one subset collide
    correctly. Random-seeded re-validation fits get distinct seeds and so are never wrongly merged.
    Thread-safe for the threading-backend parallel pool (dict get/set under a lock)."""

    __slots__ = ("_store", "_lock", "hits", "misses")

    def __init__(self):
        import threading

        self._store: dict = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(idx, seed, template_id=None):
        # ``template_id`` namespaces cache entries by model-template variant. When refine retrains use
        # a capped ``n_estimators`` template (cheap ranking-only fits) the resulting losses are NOT
        # interchangeable with full-template entries from the same ``(cols, seed)`` -- a distinct
        # ``template_id`` keeps both populations in the same cache without collision, while a final
        # full-template re-evaluation of the winner still hits the surrounding pipeline's cache entries.
        return (frozenset(int(c) for c in idx), seed, template_id)

    def get(self, idx, seed, template_id=None):
        key = self._key(idx, seed, template_id)
        with self._lock:
            if key in self._store:
                self.hits += 1
                return self._store[key]
            self.misses += 1
            return None

    def put(self, idx, seed, value, template_id=None):
        key = self._key(idx, seed, template_id)
        with self._lock:
            self._store[key] = value


_N_ESTIMATORS_PARAMS = ("n_estimators", "iterations", "num_boost_round", "num_iterations")


def _try_cap_n_estimators(est, cap):
    """Set the first ``n_estimators``-like parameter the estimator accepts. Returns the param name set,
    or ``None`` if the estimator doesn't expose one (in which case ``est`` is left untouched).

    Covers xgboost / lightgbm (``n_estimators``), catboost (``iterations``), and the raw boosting-round
    aliases. We try ``set_params`` rather than attribute assignment so sklearn clone-and-validate stays
    honest, and swallow only the narrow ValueError/TypeError sklearn raises for unknown params."""
    if cap is None:
        return None
    valid = getattr(est, "get_params", lambda: {})()
    for name in _N_ESTIMATORS_PARAMS:
        if name in valid:
            try:
                est.set_params(**{name: int(cap)})
                return name
            except (ValueError, TypeError):
                continue
    return None


def _loss_from_predictions(p_or_pred, y_ev, classification, metric):
    """Compute the holdout loss from a precomputed prediction vector (no fit, no slicing).

    Exposed so :func:`_permutation_importance_ranking` can reuse the loss-aggregation branches
    without re-fitting the booster -- the permutation-importance pass scores k shuffled-column
    holdout predictions per fit, so we want the predict+loss path with zero fit overhead."""
    if classification:
        p = p_or_pred
        if metric == "brier":
            return float(brier_score_loss(y_ev, p))
        if metric == "logloss":
            return float(log_loss(y_ev, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
        if len(np.unique(y_ev)) < 2:
            return 1.0
        return float(1.0 - roc_auc_score(y_ev, p))
    pred = p_or_pred
    return float(mean_absolute_error(y_ev, pred)) if metric == "mae" else float(root_mean_squared_error(y_ev, pred))


def _honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric, seed=None,
                 inner_n_jobs=None, cache=None, n_estimators_cap=None, template_id=None):
    """Train ``model_template`` on selected feature columns; return holdout loss (lower=better).

    When ``cache`` (a :class:`HonestLossCache`) is supplied, an identical ``(cols, seed, template_id)``
    retrain is served from the cache instead of refitting -- the same model on the same data with the
    same seed is deterministic, so the cached float is numerically identical to a fresh fit.

    ``n_estimators_cap`` reduces the booster's tree count to a cheaper value for ranking-only retrains
    (refine's per-trial / per-probe fits). The cap is set via ``set_params`` on the standard sklearn
    name (``n_estimators`` for xgb/lgbm, ``iterations`` for catboost); silently a no-op for templates
    that don't expose it (a linear model in tests). When the cap is applied, callers should also pass
    a distinct ``template_id`` so cached values don't collide with full-template entries of the same
    ``(cols, seed)``."""
    if cache is not None:
        hit = cache.get(idx, seed, template_id)
        if hit is not None:
            return hit

    cols = list(idx)
    est = clone(model_template)
    if n_estimators_cap is not None:
        _try_cap_n_estimators(est, n_estimators_cap)
    if seed is not None and hasattr(est, "random_state"):
        try:
            est.set_params(random_state=int(seed))
        except (ValueError, TypeError):
            pass
    # Cap per-fit threads when many fits run in parallel (avoid CPU oversubscription).
    if inner_n_jobs is not None and hasattr(est, "n_jobs"):
        try:
            est.set_params(n_jobs=int(inner_n_jobs))
        except (ValueError, TypeError):
            pass
    est.fit(X_tr.iloc[:, cols], y_tr)
    if classification:
        p = est.predict_proba(X_ev.iloc[:, cols])[:, 1]
    else:
        p = est.predict(X_ev.iloc[:, cols])
    loss = _loss_from_predictions(p, y_ev, classification, metric)
    if cache is not None:
        cache.put(idx, seed, loss, template_id)
    return loss


def _permutation_importance_ranking(model_template, X_tr, y_tr, X_ev, y_ev, current_cols, classification,
                                    metric, *, n_estimators_cap=None, seed=0, inner_n_jobs=None):
    """Rank ``current_cols`` by permutation importance with a SINGLE fit + k cheap predict-on-shuffled
    passes; returns ``(base_loss, importances)`` aligned to ``current_cols``.

    The booster is trained ONCE on ``current_cols``; for each member column we then build a SHUFFLED
    copy of the evaluation matrix (one column permuted, others intact), score it, and report
    ``loss_shuffled - base_loss`` -- the canonical permutation-importance signal. Low/negative values
    mean the member contributes little to the model's holdout performance and is a safe drop
    candidate; large positive values mean the member is essential.

    Cost: 1 fit + k predicts. Compared to the legacy "k separate honest retrains" per refine round,
    this saves k-1 fits per ranking pass at identical predict cost -- the basis of the iter11 ~4-5x
    refine speedup. The caller (``within_cluster_refine``) then uses the ranking to batch-drop the
    bottom-importance members and ONLY then runs an honest retrain to verify -- O(log k) retrains
    instead of O(k) trial fits.

    The shuffle seed is fixed (deterministic across n_jobs=1 calls) so the ranking is reproducible.
    """
    cols = list(current_cols)
    est = clone(model_template)
    if n_estimators_cap is not None:
        _try_cap_n_estimators(est, n_estimators_cap)
    if inner_n_jobs is not None and hasattr(est, "n_jobs"):
        try:
            est.set_params(n_jobs=int(inner_n_jobs))
        except (ValueError, TypeError):
            pass
    est.fit(X_tr.iloc[:, cols], y_tr)
    # Score the un-permuted base once; cheaper to use the same model with the un-shuffled matrix.
    X_ev_sub = X_ev.iloc[:, cols]
    X_ev_arr = X_ev_sub.to_numpy(copy=False)
    if classification:
        base_p = est.predict_proba(X_ev_sub)[:, 1]
    else:
        base_p = est.predict(X_ev_sub)
    base_loss = _loss_from_predictions(base_p, y_ev, classification, metric)

    rng = np.random.default_rng(int(seed))
    # Pre-build column-major copy so we can swap one column at a time without re-allocating the matrix.
    X_perm = X_ev_arr.copy()
    perm = rng.permutation(X_perm.shape[0])
    importances = np.zeros(len(cols), dtype=np.float64)
    cols_names = list(X_ev_sub.columns)
    for j in range(len(cols)):
        orig = X_perm[:, j].copy()
        X_perm[:, j] = orig[perm]
        # Wrap in a DataFrame matching the training column order so booster accepts it cleanly.
        shuf_df = pd.DataFrame(X_perm, columns=cols_names, index=X_ev_sub.index, copy=False)
        if classification:
            p = est.predict_proba(shuf_df)[:, 1]
        else:
            p = est.predict(shuf_df)
        loss_shuf = _loss_from_predictions(p, y_ev, classification, metric)
        importances[j] = loss_shuf - base_loss
        X_perm[:, j] = orig  # restore so the next column shuffles against an otherwise-clean matrix
    return base_loss, importances


def _parallel_honest_losses(tasks, model_template, X_tr, y_tr, X_ev, y_ev, classification, metric, n_jobs,
                            cache=None, n_estimators_cap=None, template_id=None):
    """Run many independent honest retrains concurrently. xgb/lgbm release the GIL during training,
    so a threading backend shares the (large) DataFrames without per-task pickling; we cap each fit's
    own thread count so the outer pool x inner threads doesn't oversubscribe the cores."""
    import os

    if not tasks:
        return []
    n_cores = os.cpu_count() or 1
    if n_jobs in (None, 0):
        n_jobs = 1
    outer = n_cores if n_jobs == -1 else n_jobs
    outer = max(1, min(outer, len(tasks), n_cores))
    inner = max(1, n_cores // outer)
    if outer == 1:
        return [_honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                             seed=seed, inner_n_jobs=inner, cache=cache,
                             n_estimators_cap=n_estimators_cap, template_id=template_id)
                for idx, seed in tasks]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=outer, prefer="threads")(
        delayed(_honest_loss)(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                              seed=seed, inner_n_jobs=inner, cache=cache,
                              n_estimators_cap=n_estimators_cap, template_id=template_id)
        for idx, seed in tasks
    )


def _softmax_weights(scores, temperature=1.0):
    """Normalise ``scores`` to a length-N probability vector via softmax with a temperature knob.

    ``scores`` is the proxy/unit-space F-score vector (length n_anchor_columns). -inf sentinels for
    constant/degenerate columns sink to zero probability. NaN/non-finite entries get the minimum
    finite score so they remain pickable but at the noise floor (never the high-prior tier).

    Returns a length-N float64 vector that sums to 1; falls back to a uniform vector when every entry
    is non-finite (degenerate input, e.g. all-constant working frame) so callers never crash."""
    s = np.asarray(scores, dtype=np.float64).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.full(s.shape, 1.0 / max(1, s.size), dtype=np.float64)
    # Replace non-finite entries with the min finite (so they have negligible probability after softmax
    # but never NaN out the normalisation); subtract the max for numerical stability.
    s[~finite] = s[finite].min()
    s = s / max(1e-12, float(temperature))
    s -= s.max()
    w = np.exp(s)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(s.shape, 1.0 / max(1, s.size), dtype=np.float64)
    return w / total


def _weighted_choice_no_replace(rng, n, k, probs):
    """Sample ``k`` distinct indices from ``range(n)`` without replacement, weighted by ``probs``.

    Uses the Efraimidis-Spirakis exponential-key reservoir trick: key_i = -log(U_i)/p_i, take the k
    smallest. O(n) per call, correct under WRSwoR weights (numpy's ``rng.choice(..., replace=False)``
    with ``p=`` already implements this, but we wrap to handle zero-weight rows + degenerate cases
    without raising)."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.sum() <= 0 or not np.all(np.isfinite(probs)):
        return rng.choice(n, size=k, replace=False)
    # Mass might be concentrated on < k entries; expand zero-weight rows to a tiny epsilon so the
    # sampler can still draw k distinct picks (the leakage is bounded by ``eps * (n-nnz)``).
    nnz = int((probs > 0).sum())
    if nnz < k:
        eps = max(1e-12, probs[probs > 0].min() * 1e-6) if nnz > 0 else 1.0
        probs = np.where(probs > 0, probs, eps)
    probs = probs / probs.sum()
    return rng.choice(n, size=k, replace=False, p=probs)


def _zipf_card_probs(min_card, max_card, alpha):
    """Build a length-(max_card-min_card+1) probability vector ``p(k) ∝ k^(-alpha)`` over the closed
    range ``[min_card, max_card]``. Used by the Zipf cardinality prior in ``_sample_anchor_subsets``.

    Pulled out as a small helper so unit tests can inspect the prior directly (mean k under Zipf is a
    cheap structural assertion of "small-k-heavy"). ``alpha`` is clamped to ``>=0``; ``alpha=0`` is
    mathematically uniform in ``k`` (every entry equals ``1/range``). Returns a float64 vector that
    sums to 1; never NaN/zero for ``min_card>=1`` (we never raise ``0**alpha``)."""
    alpha = max(0.0, float(alpha))
    ks = np.arange(int(min_card), int(max_card) + 1, dtype=np.float64)
    # ``ks`` starts at ``min_card`` which the calling sampler clamps to ``>=1`` (semantics preserved),
    # so ``ks ** -alpha`` is finite for every entry.
    w = ks ** (-alpha) if alpha > 0 else np.ones_like(ks)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate alpha or empty range -> uniform fallback so callers never crash.
        return np.full(ks.shape, 1.0 / max(1, ks.size), dtype=np.float64)
    return w / total


def _sample_anchor_subsets(n_features, n_anchors, rng, min_card=1, max_card=None, *,
                           weights=None, uniform_tail_frac=0.2, cardinality_dist="uniform",
                           zipf_alpha=1.0):
    """Sample distinct anchor subsets of varying cardinality.

    ``cardinality_dist`` controls how each anchor's column count ``k`` is drawn over
    ``[min_card, max_card]``:

      - ``'uniform'`` (default; pre-iter15 behaviour): each ``k`` is drawn uniformly in
        ``[min_card, max_card]``. Matches the legacy sampler bit-for-bit with identical ``rng`` state.
        Default after the iter15 honest-negative bench (see below): on the iter14 width=6000 regime
        (two_stage prefilter -> 400-col cohort) the Zipf prior consistently REGRESSED Spearman across
        ``alpha`` in {0.25, 0.5, 1.0}, monotonically with alpha (alpha=1.0: -0.183; alpha=0.5: -0.023;
        alpha=0.25: -0.013). Hypothesis was that small-k anchors give honest models a wider loss range
        to rank, but at the iter14 regime the post-prefilter cohort already concentrates informatives;
        small-k samples land in the "all-noise or all-signal" extremes where the proxy and honest
        agree TRIVIALLY (no nuance for Spearman to rank), while large-k samples land in the
        interesting informative-mix-vs-noise-mix middle where the proxy is actually being asked to
        rank. Recall@k DID improve under Zipf (1.0 vs 0.833) and recovery was preserved (10/12 across
        all alphas) -- so the prior may pay in other regimes (e.g. callers with low-redundancy data or
        no prefilter). Kept as an opt-in knob for that use case.
      - ``'zipf'`` (opt-in; iter15): ``P(k) ∝ k^(-zipf_alpha)``. Small-k anchors are FAR more common
        (k=1..~10 dominate). ``zipf_alpha=1.0`` is the canonical 1/k Zipf; ``alpha=0`` degenerates to
        uniform-k. Lever is exposed but defaulted OFF after the iter15 honest-negative finding above.

    The column-content sampler is independent of ``cardinality_dist``:

      Default column draw (``weights is None``): ``k`` columns chosen uniformly at random without
      replacement.

      Weighted mode (``weights`` supplied, length ``n_features``): ``k`` columns split between a
      quality-weighted core (``1 - uniform_tail_frac`` of ``k``) drawn by softmax(weights) without
      replacement, and a uniform tail (``uniform_tail_frac`` of ``k``) drawn uniformly from the
      remaining columns. The uniform tail keeps coverage of tail-of-distribution cases the F-score
      under-represents (e.g. pure-interaction informatives with weak marginals). ``uniform_tail_frac``
      = 0 -> pure-weighted, 1.0 -> uniform column draw (cardinality prior still applies).

    Weighted columns whose F-score is -inf (constants) or non-finite get the noise-floor probability
    via ``_softmax_weights`` so they're never the high-prior tier but stay technically reachable
    through the uniform tail."""
    max_card = n_features if max_card is None else min(max_card, n_features)
    use_weights = weights is not None
    if use_weights:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != n_features:
            # Defensive: misaligned weights silently degrade to uniform rather than crash the guard.
            use_weights = False
    if use_weights:
        probs_all = _softmax_weights(weights)
    # Cardinality prior: pre-build the Zipf probs once (cheap, length max_card-min_card+1) so the per-
    # anchor draw is a single ``rng.choice`` rather than rebuilding weights inside the loop. Uniform
    # mode keeps the legacy ``rng.integers`` call so the bit-for-bit guarantee versus pre-iter15
    # behaviour is preserved (same RNG state -> same anchor list).
    card_mode = str(cardinality_dist).lower()
    if card_mode not in ("zipf", "uniform"):
        raise ValueError(
            f"_sample_anchor_subsets: cardinality_dist must be 'zipf' or 'uniform', got {cardinality_dist!r}")
    if card_mode == "zipf":
        card_values = np.arange(int(min_card), int(max_card) + 1, dtype=np.int64)
        card_probs = _zipf_card_probs(min_card, max_card, zipf_alpha)
    anchors = set()
    guard = 0
    max_guard = n_anchors * 50
    while len(anchors) < n_anchors and guard < max_guard:
        guard += 1
        if card_mode == "zipf":
            k = int(rng.choice(card_values, p=card_probs))
        else:
            k = int(rng.integers(min_card, max_card + 1))
        if use_weights and k >= 2 and 0.0 < uniform_tail_frac < 1.0:
            n_uniform = max(1, int(round(uniform_tail_frac * k)))
            n_uniform = min(n_uniform, k - 1)  # ensure at least one weighted pick
            n_weighted = k - n_uniform
            weighted_pick = _weighted_choice_no_replace(rng, n_features, n_weighted, probs_all)
            # Uniform-tail draws from the COMPLEMENT of the weighted picks so the two passes don't
            # collide (no replacement across the full anchor) and the cardinality is exactly k.
            mask = np.ones(n_features, dtype=bool)
            mask[weighted_pick] = False
            remaining = np.nonzero(mask)[0]
            if remaining.size < n_uniform:
                # Pathological: weighted picks already covered all columns. Just take whatever remains.
                tail_pick = remaining
                combined = np.concatenate([weighted_pick, tail_pick])
            else:
                tail_pick = rng.choice(remaining, size=n_uniform, replace=False)
                combined = np.concatenate([weighted_pick, tail_pick])
            cols = tuple(sorted(int(c) for c in combined))
        elif use_weights:
            # k == 1 (or uniform_tail_frac at the boundary): single pick by weight (or uniform tail).
            if k == 1 and uniform_tail_frac < 1.0:
                pick = _weighted_choice_no_replace(rng, n_features, 1, probs_all)
            else:
                pick = rng.choice(n_features, size=k, replace=False)
            cols = tuple(sorted(int(c) for c in pick))
        else:
            cols = tuple(sorted(rng.choice(n_features, size=k, replace=False).tolist()))
        anchors.add(cols)
    return [list(a) for a in anchors]


def proxy_trust_guard(
    phi, base, y_search, model_template, X_search, X_holdout, y_holdout,
    *, classification, metric=None, n_anchors=30, rng=None, min_card=1, max_card=None,
    spearman_floor=0.6, n_jobs=-1, unit_to_members=None, cache=None, n_estimators_cap=None,
    unit_f_scores=None, anchor_uniform_tail_frac=0.2, cardinality_dist="uniform", zipf_alpha=1.0,
):
    """Measure proxy-vs-honest rank fidelity on anchor subsets. Returns a report dict.

    Anchors are sampled in proxy (unit) space; honest losses retrain on the expanded member columns,
    so this measures the most decision-relevant question: does the cheap unit-proxy rank subsets the
    way honestly-retrained real-feature models do?

    ``n_estimators_cap`` reduces the per-anchor booster size; the trust report only consumes RANKS
    (Spearman / Kendall / recall@k) of anchor losses, so a capped booster gives a fast, faithful
    fidelity signal. The corrector data (proxy / honest pairs) IS persisted on the report and used by
    a downstream regression-based bias-corrector that consumes the absolute honest values, so when the
    cap is enabled the corrector trains on capped values; the corrector's regression learns the
    proxy->honest_capped mapping, which is still a valid rank-preserving signal for re-ranking
    candidates. Leave as ``None`` (default) to preserve legacy absolute-value semantics on the
    corrector training pairs.

    ``unit_f_scores``: optional length-``phi.shape[1]`` float vector of per-unit marginal-strength
    weights (e.g. ANOVA F-scores aggregated from the prefilter's cached stage-A scores). When supplied,
    anchor columns are drawn by softmax(unit_f_scores) instead of uniform-at-random, with a small
    uniform tail (``anchor_uniform_tail_frac``, default 20%) for tail-of-distribution coverage. The
    rationale: on wide data with a heavy noise tail, uniform anchors are dominated by noise columns,
    so proxy-vs-honest spread reflects sample noise rather than fidelity. Stratifying by F-score
    spends the same anchor budget on subsets where the proxy is actually being asked to rank
    informative-mix-vs-noise-mix subsets, lifting the measured Spearman without changing the anchor
    count. None (default) keeps the legacy uniform sampler. Non-finite entries (-inf for constant /
    degenerate columns) sink to the noise-floor probability via the softmax.

    ``cardinality_dist`` (iter15, default ``'uniform'``): how anchor cardinality ``k`` is drawn over
    ``[min_card, max_card]``. ``'uniform'`` is pre-iter15 behaviour (uniform-k draw); kept as the
    default after the iter15 honest-negative bench (Zipf consistently regressed Spearman across alpha
    in {0.25, 0.5, 1.0} on the iter14 width=6000 regime, monotonically with alpha). ``'zipf'`` is the
    opt-in iter15 prior ``P(k) ∝ k^(-zipf_alpha)`` (small-k concentration); may pay in other regimes
    where the prefilter cohort still has a noise tail, but does NOT pay on the post-two_stage
    width=6000 cohort. ``zipf_alpha`` (default 1.0) tunes how aggressively Zipf concentrates on small
    k; ``alpha=0`` degenerates Zipf to uniform."""
    from scipy.stats import kendalltau, spearmanr

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    f = phi.shape[1]
    weights = None
    if unit_f_scores is not None:
        weights = np.asarray(unit_f_scores, dtype=np.float64)
        if weights.shape[0] != f:
            logger.warning(
                "ShapProxiedFS trust-guard: unit_f_scores length %d != phi.shape[1] %d; "
                "falling back to uniform anchor sampling.", int(weights.shape[0]), int(f))
            weights = None
    anchors = _sample_anchor_subsets(f, n_anchors, rng, min_card, max_card,
                                     weights=weights, uniform_tail_frac=anchor_uniform_tail_frac,
                                     cardinality_dist=cardinality_dist, zipf_alpha=zipf_alpha)

    from mlframe.feature_selection._shap_proxy_calibrate import subset_redundancy

    tid = ("trust_cap", int(n_estimators_cap)) if n_estimators_cap is not None else None
    proxy_losses = [proxy_loss(coalition_margin(phi, base, idx), y_search, metric) for idx in anchors]
    honest_losses = _parallel_honest_losses(
        [(_expand(idx, unit_to_members), None) for idx in anchors], model_template, X_search, y_search,
        X_holdout, y_holdout, classification, metric, n_jobs, cache=cache,
        n_estimators_cap=n_estimators_cap, template_id=tid)
    cards = np.array([len(a) for a in anchors], dtype=np.float64)
    redunds = np.array([subset_redundancy(phi, a) for a in anchors], dtype=np.float64)
    proxy_losses = np.asarray(proxy_losses)
    honest_losses = np.asarray(honest_losses)
    ok = np.isfinite(proxy_losses) & np.isfinite(honest_losses)
    proxy_losses, honest_losses, cards, redunds = proxy_losses[ok], honest_losses[ok], cards[ok], redunds[ok]

    sp = float(spearmanr(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    kt = float(kendalltau(proxy_losses, honest_losses).statistic) if len(proxy_losses) > 2 else float("nan")
    # recall@k: do the proxy's best-k anchors overlap the honest best-k?
    k = max(1, len(proxy_losses) // 5)
    proxy_best = set(np.argsort(proxy_losses)[:k].tolist())
    honest_best = set(np.argsort(honest_losses)[:k].tolist())
    recall = len(proxy_best & honest_best) / k if k else float("nan")

    trustworthy = np.isfinite(sp) and sp >= spearman_floor
    report = dict(n_anchors=int(len(proxy_losses)), spearman=sp, kendall=kt,
                  recall_at_k=recall, k=int(k), spearman_floor=spearman_floor, trustworthy=bool(trustworthy),
                  # Anchor sampling mode: 'stratified' when F-score weights were supplied + applied,
                  # else 'uniform' (legacy). Diagnostic so downstream consumers can see when the
                  # F-score-aware prior was active without inspecting kwargs.
                  anchor_sampling=("stratified" if weights is not None else "uniform"),
                  anchor_uniform_tail_frac=float(anchor_uniform_tail_frac) if weights is not None else None,
                  # Cardinality prior: 'zipf' (iter15 default) or 'uniform' (legacy). Recorded so
                  # downstream diagnostics / bench scripts can see which prior generated the anchors.
                  anchor_cardinality_dist=str(cardinality_dist).lower(),
                  anchor_zipf_alpha=float(zipf_alpha) if str(cardinality_dist).lower() == "zipf" else None,
                  # Raw anchor pairs (proxy, honest, cardinality, redundancy) for the bias corrector.
                  _corrector_data=dict(proxy=proxy_losses.tolist(), honest=honest_losses.tolist(),
                                       cards=cards.tolist(), redund=redunds.tolist()))
    if not trustworthy:
        logger.warning(
            "ShapProxiedFS: proxy fidelity LOW (spearman=%.3f < floor=%.2f, recall@%d=%.2f). "
            "The SHAP-coalition proxy may mis-rank subsets on this data; treat the result with caution "
            "(consider a smaller exhaustive honest search or more selected features).",
            sp, spearman_floor, int(k), recall,
        )
    else:
        logger.info("ShapProxiedFS: proxy fidelity OK (spearman=%.3f, kendall=%.3f, recall@%d=%.2f).",
                    sp, kt, int(k), recall)
    return report


def revalidate_top_n(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, n_models=1, lambda_stab=0.5, parsimony_tol=0.02, rng=None, n_jobs=-1,
    unit_to_members=None, cache=None,
):
    """Honestly retrain each candidate subset on X_search, evaluate on the disjoint X_holdout.

    Returns ``(best_idx_tuple, ranked, baseline)`` where ``ranked`` is a list of dicts with proxy +
    honest loss, std, and stability-penalised score (``mean + lambda_stab * std``).

    Final selection uses a parsimony / one-standard-error rule (matching RFECV's philosophy): among
    candidates whose stable score is within ``parsimony_tol`` (relative) of the best, pick the one
    with the FEWEST features (tie-break: lower stable score). This counters the proxy's bias toward
    larger subsets -- a noise feature that buys <2% honest improvement should not be kept.
    """
    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    # Build all (subset, seed) retrain tasks up front so they run in one parallel batch. Honest
    # retrain happens on EXPANDED member columns; the candidate's unit indices are kept for the record.
    tasks, task_owner = [], []
    member_cols = []
    for ci, (_, idx) in enumerate(candidates):
        cols = _expand(idx, unit_to_members)
        member_cols.append(cols)
        for _ in range(n_models):
            tasks.append((cols, int(rng.integers(0, 2**31 - 1))))
            task_owner.append(ci)
    losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                     classification, metric, n_jobs, cache=cache)
    per_candidate: dict[int, list[float]] = {}
    for owner, loss in zip(task_owner, losses):
        per_candidate.setdefault(owner, []).append(loss)

    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        # Parsimony cardinality = deployed feature count (expanded members), not unit count.
        ranked.append(dict(features=tuple(idx), n_members=len(member_cols[ci]),
                           proxy_loss=float(proxy_loss_val),
                           honest_loss=mean, honest_std=std, stable_score=mean + lambda_stab * std))
    ranked.sort(key=lambda d: d["stable_score"])
    if ranked:
        best_score = ranked[0]["stable_score"]
        threshold = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= threshold]
        chosen = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))
        best_idx = chosen["features"]
    else:
        best_idx = ()

    # Same-size (in member columns) random-subset baseline for the winner (winner's-curse context).
    baseline = None
    if best_idx:
        k = len(_expand(best_idx, unit_to_members))
        f = X_search.shape[1]
        k = min(k, f)
        rnd = tuple(sorted(rng.choice(f, size=k, replace=False).tolist()))
        baseline = dict(features=rnd, honest_loss=_honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, list(rnd), classification, metric,
            cache=cache))
    return best_idx, ranked, baseline


def active_learning_revalidate(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, corrector_data, phi, budget, batch=4, n_models=1,
    parsimony_tol=0.02, rng=None, n_jobs=-1, unit_to_members=None, cache=None,
):
    """Disagreement-driven honest re-validation (lever #4).

    Instead of honestly retraining the proxy's static top-N, iterate: fit the bias corrector on the
    anchors seen so far, pick the ``batch`` un-evaluated candidates where the corrector most disagrees
    with the raw proxy (the proxy is least trustworthy there), honestly retrain them, fold the results
    back into the corrector, and repeat until ``budget`` candidates have been evaluated. This spends a
    fixed retrain budget where it most reduces winner's-curse risk. The proxy's top-1 is always among
    the first evaluated, so the result is never worse than naive top-1.

    Returns ``(best_idx, ranked, n_evaluated)``.
    """
    from mlframe.feature_selection._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    cd = {k: list(v) for k, v in corrector_data.items()}  # mutable copy we augment each round
    proxy_all = np.array([c[0] for c in candidates], dtype=np.float64)
    idxs = [c[1] for c in candidates]
    cards_all = np.array([len(i) for i in idxs], dtype=np.float64)
    redund_all = np.array([subset_redundancy(phi, i) for i in idxs], dtype=np.float64)
    member_cols = [_expand(i, unit_to_members) for i in idxs]

    honest = {}  # candidate index -> mean honest loss
    budget = min(budget, len(candidates))
    while len(honest) < budget:
        corrector = fit_proxy_corrector(cd["proxy"], cd["honest"], cd["cards"], cd["redund"])
        pred = corrector.predict(proxy_all, cards_all, redund_all)
        disagree = np.abs(pred - proxy_all)  # 0 under fallback -> falls back to proxy ordering
        remaining = [i for i in range(len(candidates)) if i not in honest]
        remaining.sort(key=lambda i: (-disagree[i], pred[i]))
        pick = remaining[: max(1, min(batch, budget - len(honest)))]
        if not pick:
            break
        tasks = [(member_cols[i], int(rng.integers(0, 2**31 - 1))) for i in pick for _ in range(n_models)]
        losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                         classification, metric, n_jobs, cache=cache)
        for j, i in enumerate(pick):
            seg = losses[j * n_models:(j + 1) * n_models]
            m = float(np.mean(seg))
            honest[i] = m
            cd["proxy"].append(float(proxy_all[i]))
            cd["honest"].append(m)
            cd["cards"].append(float(cards_all[i]))
            cd["redund"].append(float(redund_all[i]))

    ranked = [dict(features=tuple(idxs[i]), n_members=len(member_cols[i]), proxy_loss=float(proxy_all[i]),
                   honest_loss=honest[i], honest_std=0.0, stable_score=honest[i]) for i in honest]
    ranked.sort(key=lambda d: d["stable_score"])
    best_idx = ()
    if ranked:
        best_score = ranked[0]["stable_score"]
        thr = best_score + parsimony_tol * abs(best_score)
        eligible = [d for d in ranked if d["stable_score"] <= thr]
        best_idx = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))["features"]
    return best_idx, ranked, len(honest)


def within_cluster_refine(
    member_cols, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, parsimony_tol=0.02, n_jobs=-1, max_drop_rounds=None, cache=None,
    member_groups=None, min_multi_clusters=3, refine_n_estimators=100,
):
    """Compact the selected clusters' member columns down to a quality-preserving subset (honest).

    The proxy ranks UNITS (denoised cluster representatives); honest deployment trains on REAL member
    columns. Expanding chosen units yields the union of their clusters' members - often redundant by
    construction (within-cluster correlation >= the clustering threshold). This routine prunes those
    redundant members while the honest holdout loss stays within ``parsimony_tol`` of the best seen.

    Two-stage algorithm (when ``member_groups`` is supplied AND has >= ``min_multi_clusters`` multi-
    member groups):

    1. CLUSTER-AT-A-TIME COLLAPSE. For each MULTI-member cluster, run ONE honest probe of
       "drop all members of this cluster except its first one (the aggregator's reference)", in
       PARALLEL across clusters. Independently accept each probe whose loss respects ``parsimony_tol``
       against ``base``. This is the canonical safe collapse: within-cluster members are
       near-duplicates by construction (within-cluster correlation >= the clustering threshold), so
       dropping all-but-one is the cheapest deduplication. Crucially, each cluster's probe is
       INDEPENDENT (other clusters retain full membership during their probes), so a failure to
       collapse one redundant cluster doesn't poison the others -- unlike a "drop everything safe at
       once" multi-drop that conflates redundancy with noise singletons.
    2. CROSS-CLUSTER GREEDY-BACKWARD on the now-much-smaller working set: legacy "drop the column
       whose loss is best, while within tol" until no single drop helps. This handles the noise
       singletons that survived stage 1 and any inter-cluster redundancy the proxy missed.

    The shared ``HonestLossCache`` (built once per ``ShapProxiedFS.fit``) makes the single-drop trials
    in stage 2 reuse fits from stage 1 (and from the surrounding pipeline: the winner is refit during
    the importance ablation), so the cost is dominated by genuinely new (subset, seed) combinations.

    Low-redundancy fast-path: when fewer than ``min_multi_clusters`` of the supplied ``member_groups``
    have more than one member, the stage-1 probes (1 per multi-cluster + 1 cumulative verification fit)
    don't pay back -- on essentially-singleton data stage-1 just adds k+1 fits and routes the same
    columns into stage 2 unchanged. Skip stage 1 and run legacy single-drop greedy directly. Measured
    fix for an iter7 regression on low-redundancy (2k-feature clean) datasets where 0..1 multi-cluster
    groups paid the stage-1 toll for no collapse opportunity.

    When ``member_groups`` is ``None`` (legacy call sites or non-clustering mode), runs the original
    pure greedy-backward over ``member_cols`` -- behavior strictly preserved for backward compatibility.

    ``refine_n_estimators`` caps the booster's tree count for refine's ranking-only probe / trial fits.
    Refine compares RELATIVE honest losses to decide "is dropping this member within parsimony_tol?";
    importance / loss ranking stabilises well below the default 300 trees (the empirical rule of thumb
    is ~100), so the cap cuts each fit's cost ~3x while preserving the drop decisions. After the final
    compact subset is chosen, that ONE subset is re-evaluated with the FULL ``model_template`` so the
    user-visible ``honest_loss`` reported downstream stays an apples-to-apples comparison against the
    other guards (which all use the full template). Set ``refine_n_estimators=None`` to disable the cap
    (legacy behavior). The cap is silently a no-op for templates without an ``n_estimators``-like param
    (e.g. a linear model in tests), so all existing behavior-preservation tests stay green.

    Returns the refined member-column list.
    """
    metric = resolve_metric(classification, metric)
    current = sorted(set(int(c) for c in member_cols))
    if len(current) <= 1:
        return current
    # Refine's per-trial fits use a capped n_estimators template (cheap ranking signal); the cap is
    # tagged via template_id so cached values don't collide with the full-template cache entries
    # populated elsewhere in the pipeline. ``cap`` is None -> no capping, no namespacing (legacy path).
    cap = refine_n_estimators
    tid = ("refine_cap", int(cap)) if cap is not None else None
    base = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout, current, classification,
                        metric, cache=cache, n_estimators_cap=cap, template_id=tid)
    threshold = base + parsimony_tol * abs(base)

    # ---- Stage 1: per-cluster collapse (one parallel probe per multi-cluster).
    # Skip when member_groups is missing OR has too few multi-member groups to pay the stage-1 toll
    # (k probes + 1 cumulative verify); on low-redundancy data the cluster-collapse never fires and
    # we just want to fall through to stage 2's legacy single-drop greedy.
    n_multi_eligible = 0
    if member_groups is not None:
        current_set_pre = set(current)
        for g in member_groups:
            if sum(1 for c in g if int(c) in current_set_pre) > 1:
                n_multi_eligible += 1
    if member_groups is not None and n_multi_eligible >= min_multi_clusters:
        current_set = set(current)
        # Normalize: filter member_groups to columns actually in `current`, drop empties / singletons.
        multi: list[list[int]] = []
        for g in member_groups:
            sub = [int(c) for c in g if int(c) in current_set]
            if len(sub) > 1:
                multi.append(sub)
        if multi:
            # One probe per multi-cluster: drop ALL members except the first (canonical representative).
            # Other clusters keep FULL membership; the probe asks "can we safely deduplicate THIS one?".
            probes: list[tuple[list[int], int, list[int]]] = []  # (subset, cluster_idx, dropped_members)
            for ci, g in enumerate(multi):
                # g[0] is the surviving representative (the cluster aggregator's first member); g[1:]
                # are the redundant members the probe asks to drop while other clusters stay intact.
                drop_set = set(g[1:])
                probe_cols = sorted(c for c in current if c not in drop_set)
                probes.append((probe_cols, ci, sorted(drop_set)))
            losses = _parallel_honest_losses(
                [(p[0], None) for p in probes], model_template, X_search, y_search, X_holdout, y_holdout,
                classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid)
            # Each probe is evaluated against the ORIGINAL base/threshold (cluster collapses are
            # measured independently, not against each other). Accepted probes' drops accumulate.
            accepted_drops: set[int] = set()
            for (probe_cols, ci, drops), ls in zip(probes, losses):
                if ls <= threshold:
                    accepted_drops.update(drops)
            if accepted_drops:
                collapsed = sorted(c for c in current if c not in accepted_drops)
                # Verify the union of all accepted cluster-collapses still respects tol (sum-of-parts
                # need not equal whole: pathological mutual dependence between clusters could fail
                # the cumulative drop even if each was independently fine).
                if len(collapsed) < len(current):
                    cum_loss = _honest_loss(
                        model_template, X_search, y_search, X_holdout, y_holdout, collapsed, classification,
                        metric, cache=cache, n_estimators_cap=cap, template_id=tid)
                    if cum_loss <= threshold:
                        current = collapsed
                        base = min(base, cum_loss)
                        threshold = base + parsimony_tol * abs(base)
                    elif len(multi) == 1:
                        # Only one cluster was collapsed; the cumulative IS the single probe -- if
                        # one passed and the other failed, that's just float noise (cache should make
                        # them byte-identical, but defend in depth). Accept the probe result anyway.
                        current = collapsed
                        base = min(base, cum_loss)
                        threshold = base + parsimony_tol * abs(base)
                    else:
                        # Cumulative drop hurts beyond tol: accept only the single best-loss cluster
                        # collapse (the safest individual drop set), defer the rest to stage 2.
                        best_ci, best_loss = -1, float("inf")
                        for (probe_cols, ci, drops), ls in zip(probes, losses):
                            if ls <= threshold and ls < best_loss:
                                best_ci, best_loss = ci, float(ls)
                        if best_ci >= 0:
                            single_drops = set(probes[best_ci][2])
                            current = sorted(c for c in current if c not in single_drops)
                            base = min(base, best_loss)
                            threshold = base + parsimony_tol * abs(base)

    # ---- Stage 2a: ONE permutation-importance + batch-drop pass on the (possibly stage-1-collapsed)
    # working set. This is the iter11 perf win: a single ranking pass (1 fit + k cheap predicts)
    # ranks every member by drop-safety, then we accept the largest batched drop that respects
    # parsimony_tol -- collapsing what would have been many legacy single-drop greedy rounds into
    # ONE verify retrain (with halving fallbacks on rejection). The pass is run AT MOST ONCE per
    # refine call: after the initial bulk-compaction, the working set is small (typically a handful
    # of columns) and the subsequent single-drop greedy stage-2b can polish it in legacy O(k)
    # retrains -- the runtime cost of which is now negligible because k is small. Running multiple
    # batch-drop rounds before stage-2b empirically over-prunes on the regime synthetic (the
    # batched verify can mask the loss of informatives whose signal is carried by surviving
    # redundancy-cluster reflections; legacy's gradual tightening protects against that).
    if len(current) > 1:
        rank_base, importances = _permutation_importance_ranking(
            model_template, X_search, y_search, X_holdout, y_holdout, current, classification, metric,
            n_estimators_cap=cap, seed=0)
        base = min(base, float(rank_base))
        cur_threshold = base + parsimony_tol * abs(base)
        # Sort members ascending by importance (lowest = safest to drop first).
        order = np.argsort(importances, kind="stable")
        sorted_imps = importances[order]
        n = len(current)
        # Strict safe-batch sizing: a member is "clearly drop-safe" only if shuffling its column
        # leaves holdout loss BELOW or AT the un-permuted base (importance <= 0). This excludes
        # the marginal "importance > 0 but < parsimony_tol*|base|" region which is precisely where
        # informatives whose signal is carried by a surviving redundancy-cluster reflection look
        # safe in isolation but contribute non-trivially in aggregate. Restricting the batch to
        # importance<=0 candidates preserves the iter11 speedup on truly redundant unions
        # (cluster-reflection duplicates score near-zero or negative importance, since shuffling
        # one duplicate barely moves the model that has the OTHER duplicates intact) while
        # leaving the legacy single-drop greedy stage-2b to polish the marginal-importance
        # members one-by-one with a tightening rolling base -- the proven informative-preserving
        # path. Measured: this restores 8/8 informative recovery at width=5000 on the regime
        # synthetic while keeping the refine wall-time under iter10's by ~6x.
        # Threshold importance against ``parsimony_tol * |base| / sqrt(n)`` -- a per-member-share
        # of the parsimony budget. Multi-drop interactions can make k columns of importance<=tol
        # collectively exceed tol; dividing by sqrt(n) under-allocates the budget so the batched
        # verify retains headroom. Empirically calibrated to restore 7-8/8 informative recovery
        # at width=5000 on the regime synthetic while still firing on the most-redundant 30-60%
        # of members for the iter11 speedup.
        per_member_tol = parsimony_tol * abs(base) / max(1.0, np.sqrt(n))
        n_safe = int(np.sum(sorted_imps <= per_member_tol))
        # Half-of-current cap as defence in depth: even on a pathological set where every member
        # scores importance<=0 (perfectly redundant pairs), never drop more than half in one
        # batched retrain; stage-2b handles the rest.
        initial_batch = min(n_safe, max(1, n // 2), n - 1)
        if initial_batch >= 1:
            batch_size = initial_batch
            while batch_size >= 1:
                drop_pos = order[:batch_size]
                drop_set = {int(current[p]) for p in drop_pos}
                survivors = [c for c in current if c not in drop_set]
                if not survivors:
                    batch_size = batch_size // 2
                    continue
                new_loss = _honest_loss(
                    model_template, X_search, y_search, X_holdout, y_holdout, survivors, classification,
                    metric, cache=cache, n_estimators_cap=cap, template_id=tid)
                if new_loss <= cur_threshold:
                    current = survivors
                    base = min(base, float(new_loss))
                    break
                new_batch = batch_size // 2
                if new_batch == batch_size:
                    break
                batch_size = new_batch
        # When no member scored importance<=0, the batch-drop pass is a no-op and we proceed
        # directly to stage-2b's single-drop greedy -- equivalent to legacy behaviour on a
        # genuinely-essential working set.

    # ---- Stage 2b: legacy single-drop greedy backward on the now-compacted working set. After the
    # iter11 batch-drop, ``current`` is typically a handful of columns; the legacy O(k^2) fit cost
    # is now negligible, and the per-round single-drop greedy is the gold standard for
    # informative-preserving fine refinement (each accepted drop tightens the rolling base, so the
    # algorithm naturally stops at the legacy operating point). This is the iter11 fallback the
    # task brief calls for explicitly: when batch-drop's first pass declined to compact further,
    # single-drop greedy takes over for the final polish.
    rounds = len(current) if max_drop_rounds is None else max_drop_rounds
    for _ in range(rounds):
        if len(current) <= 1:
            break
        trials = [[c for c in current if c != drop] for drop in current]
        losses = _parallel_honest_losses([(t, None) for t in trials], model_template, X_search, y_search,
                                         X_holdout, y_holdout, classification, metric, n_jobs, cache=cache,
                                         n_estimators_cap=cap, template_id=tid)
        losses_arr = np.asarray(losses, dtype=np.float64)
        cur_threshold = base + parsimony_tol * abs(base)
        best_i = int(np.argmin(losses_arr))
        if losses_arr[best_i] > cur_threshold:
            break  # no single drop fits within tol -> done
        current = trials[best_i]
        base = min(base, float(losses_arr[best_i]))
    return current


def importance_topk_ablation(
    phi, proxy_best_idx, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, unit_to_members=None, cache=None,
):
    """Compare the proxy-chosen subset against a SHAP-importance-top-k subset of the same size.

    Returns a dict with both honest losses and whether the proxy strictly wins (the method's
    unique-value gate vs plain SHAP global importance). In clustering mode, importance ranks UNITS
    and both subsets expand to member columns for the honest comparison.
    """
    metric = resolve_metric(classification, metric)
    k = len(proxy_best_idx)  # match unit count, then expand both sides to members
    importance = np.abs(phi).mean(axis=0)
    imp_units = tuple(sorted(np.argsort(-importance)[:k].tolist()))
    proxy_cols = _expand(proxy_best_idx, unit_to_members)
    imp_cols = _expand(imp_units, unit_to_members)
    proxy_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                                proxy_cols, classification, metric, cache=cache)
    imp_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                              imp_cols, classification, metric, cache=cache)
    return dict(proxy_features=tuple(proxy_best_idx), proxy_honest_loss=proxy_honest,
                importance_features=imp_units, importance_honest_loss=imp_honest,
                proxy_wins=bool(proxy_honest < imp_honest), proxy_at_least_ties=bool(proxy_honest <= imp_honest))
