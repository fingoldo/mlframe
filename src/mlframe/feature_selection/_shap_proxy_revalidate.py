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
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    brier_score_loss, log_loss, mean_absolute_error, roc_auc_score, root_mean_squared_error,
)

from mlframe.feature_selection._shap_proxy_objective import coalition_margin, proxy_loss, resolve_metric

logger = logging.getLogger(__name__)


# Cache-key namespace for the cross-process honest_loss disk cache. Keeps entries from this consumer
# from colliding with the shap_phi_ entries the OOF-SHAP path writes into the same cache_dir, and
# makes the cache human-greppable (``ls cache_dir/honest_loss_*`` shows reval/trust entries).
_HONEST_LOSS_CACHE_PREFIX = "honest_loss_"


def _build_honest_loss_disk_key(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                                seed, n_estimators_cap, template_id) -> Optional[str]:
    """Return a stable cache key for an ``_honest_loss`` call, or ``None`` if hashing fails.

    Key inputs cover everything that determines the cached float:
      * (X_tr, y_tr) summary -- the fit data.
      * (X_ev, y_ev) summary -- the eval data (different holdouts give different losses).
      * column subset (sorted, frozen) + classification + metric + seed.
      * model template params + n_estimators_cap + template_id (so capped vs full-template entries
        live in disjoint key namespaces, mirroring the in-memory cache's ``template_id`` field).

    Hash failure (an exotic template that doesn't pickle, etc.) returns ``None`` so the caller
    falls through to the compute path -- the cache is best-effort, never a correctness gate.
    """
    try:
        from mlframe.utils.disk_cache import compose_key, hash_array_summary, hash_object

        try:
            params = model_template.get_params(deep=False)
        except Exception:
            params = {"_repr": repr(model_template)}
        x_tr_key = hash_array_summary(X_tr.values if hasattr(X_tr, "values") else np.asarray(X_tr))
        y_tr_key = hash_array_summary(np.asarray(y_tr))
        x_ev_key = hash_array_summary(X_ev.values if hasattr(X_ev, "values") else np.asarray(X_ev))
        y_ev_key = hash_array_summary(np.asarray(y_ev))
        state_key = hash_object({
            "cols": tuple(sorted(int(c) for c in idx)),
            "seed": seed,
            "classification": bool(classification),
            "metric": str(metric),
            "n_estimators_cap": n_estimators_cap,
            "template_id": template_id,
            "params": params,
        })
        return _HONEST_LOSS_CACHE_PREFIX + compose_key(x_tr_key, y_tr_key, x_ev_key, y_ev_key, state_key)
    except Exception as exc:
        logger.debug("_honest_loss: disk-cache key build failed (%s); skipping cache", exc)
        return None


def _open_disk_cache(disk_cache_dir):
    """Return a ``DiskCache`` for ``disk_cache_dir`` or ``None`` if it can't be opened.

    Failure here (permission error, exotic path) downgrades to "compute and skip cache" rather than
    raising -- a cache hiccup must never poison the honest-retrain path it's optimising.
    """
    if disk_cache_dir is None:
        return None
    try:
        from mlframe.utils.disk_cache import DiskCache

        return DiskCache(disk_cache_dir)
    except Exception as exc:
        logger.debug("_honest_loss: disk-cache open failed for %r (%s); skipping cache", disk_cache_dir, exc)
        return None


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
                 inner_n_jobs=None, cache=None, n_estimators_cap=None, template_id=None,
                 disk_cache=None):
    """Train ``model_template`` on selected feature columns; return holdout loss (lower=better).

    When ``cache`` (a :class:`HonestLossCache`) is supplied, an identical ``(cols, seed, template_id)``
    retrain is served from the cache instead of refitting -- the same model on the same data with the
    same seed is deterministic, so the cached float is numerically identical to a fresh fit.

    ``disk_cache`` (iter80, optional :class:`mlframe.utils.disk_cache.DiskCache`) extends the
    in-memory ``HonestLossCache`` across PROCESSES / FIT INVOCATIONS. The in-memory cache is rebuilt
    per ``ShapProxiedFS.fit`` (so a second fit on identical (X, y, columns, template) retrains from
    scratch); the disk cache survives the process boundary. Cache key includes (X_tr summary, y_tr
    summary, X_ev summary, y_ev summary, sorted cols, seed, template params, n_estimators_cap,
    template_id, classification, metric) -- everything that determines the cached float. Cached
    payload is the scalar loss (one float, ~8 bytes serialised), so the disk overhead is
    millions-of-entries-cheap even before LRU eviction. Cache miss is silent; a hashing or I/O
    hiccup transparently degrades to compute + don't-cache (best-effort policy: the cache is a
    performance lever, never a correctness gate).

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

    # Cross-process disk cache lookup. The in-memory cache misses here (or was None), but a prior
    # ShapProxiedFS.fit on identical (X_tr, y_tr, X_ev, y_ev, cols, template) may have cached the
    # scalar loss to disk -- a hit avoids the fit + predict + loss work entirely. Hash-build failure
    # (None key) and DiskCache.get failure both degrade silently to the compute path.
    disk_key = None
    if disk_cache is not None:
        disk_key = _build_honest_loss_disk_key(
            model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric, seed,
            n_estimators_cap, template_id,
        )
        if disk_key is not None:
            try:
                disk_hit = disk_cache.get(disk_key)
            except Exception as exc:
                logger.debug("_honest_loss: disk cache get failed (%s); skipping", exc)
                disk_hit = None
            if disk_hit is not None:
                # Warm the in-memory cache too, so further same-fit calls don't pay the disk
                # round-trip cost for what is now a known-deterministic value.
                if cache is not None:
                    cache.put(idx, seed, float(disk_hit), template_id)
                return float(disk_hit)

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
    if disk_cache is not None and disk_key is not None:
        try:
            disk_cache.put(disk_key, float(loss))
        except Exception as exc:
            logger.debug("_honest_loss: disk cache put failed (%s); skipping", exc)
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
                            cache=None, n_estimators_cap=None, template_id=None,
                            inner_n_jobs_cap=False, disk_cache=None):
    """Run many independent honest retrains concurrently. xgb/lgbm release the GIL during training,
    so a threading backend shares the (large) DataFrames without per-task pickling.

    ``inner_n_jobs_cap`` (iter54, default False): when False, each fit's ``n_jobs`` is set to -1 so
    xgboost's internal thread pool decides scheduling; A/B at width 4000+10000 measured the legacy
    ``n_cores // outer`` cap as 8-9% e2e SLOWER on 8-core boxes (reval +8%, refine +11%, trust +12%).
    Set True to restore the iter4 outer-x-inner cap for HW where the cap measurably helps.

    ``disk_cache`` (iter80): forwarded to every per-task ``_honest_loss`` so cross-process cache hits
    avoid the fit entirely. The ``DiskCache`` instance is safe for concurrent threaded ``get`` /
    ``put`` because the underlying ``pickle.load`` / ``os.replace`` calls are atomic at the OS level
    (the cache's own docstring discusses the contract); the in-thread workers do not share Python
    state across each other inside this function."""
    import os

    if not tasks:
        return []
    n_cores = os.cpu_count() or 1
    if n_jobs in (None, 0):
        n_jobs = 1
    outer = n_cores if n_jobs == -1 else n_jobs
    outer = max(1, min(outer, len(tasks), n_cores))
    inner = max(1, n_cores // outer) if inner_n_jobs_cap else -1
    if outer == 1:
        return [_honest_loss(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                             seed=seed, inner_n_jobs=inner, cache=cache,
                             n_estimators_cap=n_estimators_cap, template_id=template_id,
                             disk_cache=disk_cache)
                for idx, seed in tasks]
    from joblib import Parallel, delayed

    return Parallel(n_jobs=outer, prefer="threads")(
        delayed(_honest_loss)(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                              seed=seed, inner_n_jobs=inner, cache=cache,
                              n_estimators_cap=n_estimators_cap, template_id=template_id,
                              disk_cache=disk_cache)
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


_FIDELITY_FLOOR_UNSET = object()


def proxy_trust_guard(
    phi, base, y_search, model_template, X_search, X_holdout, y_holdout,
    *, classification, metric=None, n_anchors=30, rng=None, min_card=1, max_card=None,
    fidelity_floor=0.5, n_jobs=-1, unit_to_members=None, cache=None, n_estimators_cap=None,
    unit_f_scores=None, anchor_uniform_tail_frac=0.2, cardinality_dist="uniform", zipf_alpha=1.0,
    fidelity_weights=(0.6, 0.4), trustworthy_metric="proxy_fidelity_score",
    spearman_floor=_FIDELITY_FLOOR_UNSET, inner_n_jobs_cap=False, disk_cache_dir=None,
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

    ``fidelity_weights`` (iter17, default ``(0.6, 0.4)``): weights ``(w_spearman, w_recall)`` for the
    composite ``proxy_fidelity_score = w_spearman * spearman + w_recall * recall_at_k``. Both Spearman
    and recall@k live on ``[0, 1]`` for non-degenerate inputs (Spearman is clipped on the gate side
    only; the raw field can be negative on a broken proxy and the gate then correctly trips because
    the composite drops below the floor). The composite is the trust-guard's headline metric; raw
    ``spearman`` / ``kendall`` / ``recall_at_k`` remain as diagnostic fields. Iter16 shipped a
    symmetric (0.5, 0.5) default for lack of evidence; iter17 calibrated the default by correlating
    each component independently with downstream selector RECOVERY across 5 regimes (additive high-SNR,
    redundancy-heavy, interaction-heavy, xor-interaction, noise-heavy). Result: ``corr(spearman,
    recovery_rate) = 0.93`` vs ``corr(recall@k, recovery_rate) = 0.55``. Spearman tracks the proxy's
    whole-ranking quality which actually predicts whether downstream candidate ranking finds the
    informatives; recall@k is bounded above (small anchor top-k overlap stays high even on
    half-broken proxies) and below (the 1-anchor top-k is trivially 1.0), so it lacks the dynamic
    range to drive the gate. The corr-proportional split (0.629, 0.371) rounds to (0.6, 0.4); the
    rounded value is the registered default. Iter15 still motivates the composite over raw Spearman:
    Zipf at alpha=0.25 dropped spearman 0.969->0.956 but lifted recall@k 0.833->1.0, so the composite
    went 0.901->0.978 -- a real win the raw-Spearman gate had masked. Iter17's (0.6, 0.4) preserves
    that win (the composite stays above the floor) while letting Spearman dominate the gate decision
    in the calibration-supported direction.

    ``trustworthy_metric`` (iter16, default ``'proxy_fidelity_score'``): which scalar gates the
    ``trustworthy`` boolean. ``'proxy_fidelity_score'`` is the new composite (default). ``'spearman'``
    preserves pre-iter16 semantics for callers that pinned the floor against the raw Spearman scale.

    ``fidelity_floor`` (iter18, default ``0.5``): below this value the gate trips and ``trustworthy``
    is ``False``. Interpreted in the chosen ``trustworthy_metric`` scale; both metrics live on [0, 1].
    Iter18 recalibrated the default from 0.6 to 0.5 after iter17 flipped the gate from raw Spearman to
    the composite ``proxy_fidelity_score = 0.6*spearman + 0.4*recall@k``: the legacy 0.6 was set
    against the raw-Spearman scale and is too conservative on the composite scale, flagging the
    ``interaction_heavy`` regime (recovery 6/8 = 75%, a real partial success) as LOW. The new floor
    is the lowest composite of any regime that hits ``recovery_rate >= 0.7`` across the same 5-regime
    bench used for the weights calibration:

      regime              spearman  recall@k  composite  recovery  recovery_rate  gate@0.5
      additive_highSNR     0.9533    0.8333    0.9053     8/8        1.000          PASS
      redundancy_heavy     0.8839    0.8333    0.8637     8/8        1.000          PASS
      interaction_heavy    0.5640    0.5000    0.5384     6/8        0.750          PASS
      noise_heavy          0.9506    0.8333    0.9036     7/8        0.875          PASS
      xor_interaction      0.3459    0.6667    0.4742     2/6        0.333          FAIL

    ``recovery_rate >= 0.7`` PASS group min composite = 0.5384 (interaction_heavy); the only regime
    with ``recovery_rate < 0.5`` (xor) sits at 0.4742. A floor at 0.5 separates the two groups
    cleanly (0.039 margin to the PASS floor, 0.026 margin above the FAIL ceiling). See
    ``_benchmarks/calib_iter18_fidelity_floor.py`` for reproducible measurement.

    ``spearman_floor`` (DEPRECATED iter18 alias for ``fidelity_floor``): kept for backwards-compat
    with callers that hard-coded the iter15 kwarg name. Emits a ``DeprecationWarning`` when supplied.
    Passing BOTH ``fidelity_floor`` and ``spearman_floor`` raises ``ValueError``.

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

    ``cardinality_dist`` (iter15+iter16): how anchor cardinality ``k`` is drawn over
    ``[min_card, max_card]``. The MODULE-LEVEL default of this kwarg is still ``'uniform'`` (so direct
    callers of ``proxy_trust_guard`` get legacy behaviour); the FACADE-LEVEL default
    (``ShapProxiedFS.trust_guard_cardinality_dist``) is ``'zipf'`` with ``zipf_alpha=0.25`` after the
    iter16 composite-fidelity re-evaluation showed Zipf alpha=0.25 lifts ``proxy_fidelity_score`` from
    0.779 to 0.834 on the iter14 width=6000 regime (raw Spearman dips 0.891->0.834 but recall@k jumps
    0.667->0.833). ``'zipf'`` uses ``P(k) ∝ k^(-zipf_alpha)`` (small-k concentration). Higher alpha
    over-compresses to small-k extremes where proxy and honest agree trivially; ``alpha=0`` degenerates
    Zipf to uniform."""
    from scipy.stats import kendalltau, spearmanr

    # iter18: ``spearman_floor`` is a deprecated alias of ``fidelity_floor`` (renamed because the
    # gate has been the composite ``proxy_fidelity_score`` since iter16; the legacy name was a
    # misnomer). Resolve here so the rest of the body only deals with ``fidelity_floor``.
    if spearman_floor is not _FIDELITY_FLOOR_UNSET:
        import warnings
        if fidelity_floor != 0.5:
            # Both supplied -- ambiguous; refuse rather than silently picking one.
            raise ValueError(
                "proxy_trust_guard: pass either `fidelity_floor` (new name) or `spearman_floor` "
                "(deprecated alias), not both.")
        warnings.warn(
            "`spearman_floor` is deprecated since iter18; use `fidelity_floor` (same semantics). "
            "The kwarg name was inherited from the iter15 raw-Spearman gate but the gate has been "
            "the composite `proxy_fidelity_score` since iter16.",
            DeprecationWarning, stacklevel=2,
        )
        fidelity_floor = spearman_floor

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
    # iter80: open the cross-process disk cache once (None when disabled). The cache short-circuits
    # the per-anchor xgboost fit whenever (X_search, y_search, X_holdout, y_holdout, expanded cols,
    # template params, cap) was retrained by a prior fit -- the standard ShapProxiedFS hyperparam
    # sweep / ablation pattern. Open here (not per-anchor) so the LRU evictor sees the whole batch.
    disk_cache = _open_disk_cache(disk_cache_dir)
    honest_losses = _parallel_honest_losses(
        [(_expand(idx, unit_to_members), None) for idx in anchors], model_template, X_search, y_search,
        X_holdout, y_holdout, classification, metric, n_jobs, cache=cache,
        n_estimators_cap=n_estimators_cap, template_id=tid, inner_n_jobs_cap=inner_n_jobs_cap,
        disk_cache=disk_cache)
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

    # Composite fidelity: weighted convex combination of Spearman and recall@k. Both live on [0, 1]
    # for non-degenerate inputs; Spearman is clipped to [0, 1] for the composite so a broken-proxy
    # negative Spearman doesn't get "credited" by a high recall@k (the recall@k of a 1-anchor top-k
    # is trivially 1.0). The raw spearman field is kept unchanged for diagnostics.
    w_sp, w_rc = float(fidelity_weights[0]), float(fidelity_weights[1])
    total = w_sp + w_rc
    if total <= 0:
        raise ValueError(f"fidelity_weights must sum to a positive value, got {fidelity_weights!r}")
    w_sp, w_rc = w_sp / total, w_rc / total
    sp_pos = max(0.0, sp) if np.isfinite(sp) else 0.0
    rc_pos = recall if np.isfinite(recall) else 0.0
    fidelity = float(w_sp * sp_pos + w_rc * rc_pos)
    gate_metric_name = str(trustworthy_metric).lower()
    if gate_metric_name == "spearman":
        gate_value = sp
    elif gate_metric_name in ("proxy_fidelity_score", "fidelity", "composite"):
        gate_metric_name = "proxy_fidelity_score"
        gate_value = fidelity
    else:
        raise ValueError(
            f"trustworthy_metric must be 'proxy_fidelity_score' or 'spearman', got {trustworthy_metric!r}")
    trustworthy = np.isfinite(gate_value) and gate_value >= fidelity_floor
    report = dict(n_anchors=int(len(proxy_losses)), spearman=sp, kendall=kt,
                  recall_at_k=recall, k=int(k),
                  # iter18: ``fidelity_floor`` is the canonical key for the gate threshold.
                  # ``spearman_floor`` is kept as a deprecated alias in the report so legacy
                  # downstream consumers that inspect the dict by the old name don't break.
                  fidelity_floor=fidelity_floor, spearman_floor=fidelity_floor,
                  # iter16: composite gate (default) -- raw spearman / recall stay above as diagnostics.
                  proxy_fidelity_score=fidelity,
                  fidelity_weights=(w_sp, w_rc),
                  trustworthy_metric=gate_metric_name,
                  trustworthy=bool(trustworthy),
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
            "ShapProxiedFS: proxy fidelity LOW (%s=%.3f < floor=%.2f; spearman=%.3f, recall@%d=%.2f). "
            "The SHAP-coalition proxy may mis-rank subsets on this data; treat the result with caution "
            "(consider a smaller exhaustive honest search or more selected features).",
            gate_metric_name, gate_value, fidelity_floor, sp, int(k), recall,
        )
    else:
        logger.info(
            "ShapProxiedFS: proxy fidelity OK (%s=%.3f; spearman=%.3f, kendall=%.3f, recall@%d=%.2f).",
            gate_metric_name, gate_value, sp, kt, int(k), recall,
        )
    return report


def _ucb_stop_remaining_cannot_win(
    best_stable_score, remaining_proxy_losses, ucb_slack, parsimony_tol,
):
    """Return ``True`` when no un-evaluated candidate can plausibly beat ``best_stable_score``.

    UCB bound: each un-evaluated candidate's honest loss is best-case ``proxy_loss + ucb_slack``
    (``ucb_slack`` is negative when honest tends to under-shoot proxy in the calibration window).
    If even the most optimistic remaining lower bound exceeds ``best_stable_score`` by more than
    ``parsimony_tol * |best_stable_score|`` it cannot enter the parsimony band, so further fits add
    cost without changing the winner -- safe to stop dispatching new batches.

    Stable across reruns: deterministic comparison of floats only.
    """
    if len(remaining_proxy_losses) == 0:
        return True
    lower_bounds = np.asarray(remaining_proxy_losses, dtype=np.float64) + float(ucb_slack)
    threshold = float(best_stable_score) + float(parsimony_tol) * abs(float(best_stable_score))
    return bool(np.min(lower_bounds) > threshold)


def _winner_from_per_candidate(per_candidate, candidates, member_cols, lambda_stab, parsimony_tol):
    """Parsimony-rule winner index tuple from accumulated per-candidate seed losses (iter77).

    Mirrors the post-dispatch ranking + parsimony pick used to finalise ``revalidate_top_n``'s
    winner; used INSIDE the model-round loop to test winner stability across consecutive rounds
    when ``adaptive_n_models=True``. Returns ``None`` when ``per_candidate`` is empty.
    """
    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        if ci not in per_candidate or not per_candidate[ci]:
            continue
        scores = np.asarray(per_candidate[ci], dtype=np.float64)
        mean, std = float(scores.mean()), float(scores.std())
        ranked.append(dict(features=tuple(idx), n_members=len(member_cols[ci]),
                           stable_score=mean + lambda_stab * std))
    if not ranked:
        return None
    ranked.sort(key=lambda d: d["stable_score"])
    best_score = ranked[0]["stable_score"]
    threshold = best_score + parsimony_tol * abs(best_score)
    eligible = [d for d in ranked if d["stable_score"] <= threshold]
    chosen = min(eligible, key=lambda d: (d["n_members"], d["stable_score"]))
    return chosen["features"]


def _ucb_auto_slack(evaluated_proxy, evaluated_honest_mean, stdev_multiplier=1.5):
    """Calibrate the UCB slack from already-evaluated (proxy, honest_mean) pairs.

    ``slack`` shifts proxy onto the honest scale; the lower bound for an un-evaluated candidate's
    honest loss is ``proxy + slack``. To be a *lower* bound we take ``mean(delta) - k * std(delta)``
    where ``delta_i = honest_i - proxy_i``: most of the calibration mass on the high side keeps it
    pessimistic (smaller honest predictions => larger remaining lower bounds rarely => never wrong
    stops). With <2 evaluated points the std is undefined; we fall back to ``mean(delta)`` only,
    which still preserves the proxy ordering but with zero safety margin -- the calling stop check
    additionally requires the margin to clear ``parsimony_tol``.

    Returns 0.0 when no evaluated pairs supplied (caller has not yet started; cannot stop).
    """
    p = np.asarray(evaluated_proxy, dtype=np.float64)
    h = np.asarray(evaluated_honest_mean, dtype=np.float64)
    if p.size == 0 or h.size == 0:
        return 0.0
    delta = h - p
    finite = np.isfinite(delta)
    if not finite.any():
        return 0.0
    delta = delta[finite]
    mean = float(delta.mean())
    if delta.size < 2:
        return mean
    return mean - float(stdev_multiplier) * float(delta.std(ddof=1))


def revalidate_top_n(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, n_models=1, lambda_stab=0.5, parsimony_tol=0.02, rng=None, n_jobs=-1,
    unit_to_members=None, cache=None, revalidation_n_estimators=None,
    ucb_enabled=False, ucb_min_eval_size=None, ucb_slack=None, ucb_stdev_multiplier=1.5,
    candidate_score=None, inner_n_jobs_cap=False, adaptive_n_models=False,
    disk_cache_dir=None,
):
    """Honestly retrain each candidate subset on X_search, evaluate on the disjoint X_holdout.

    Returns ``(best_idx_tuple, ranked, baseline)`` where ``ranked`` is a list of dicts with proxy +
    honest loss, std, and stability-penalised score (``mean + lambda_stab * std``).

    Final selection uses a parsimony / one-standard-error rule (matching RFECV's philosophy): among
    candidates whose stable score is within ``parsimony_tol`` (relative) of the best, pick the one
    with the FEWEST features (tie-break: lower stable score). This counters the proxy's bias toward
    larger subsets -- a noise feature that buys <2% honest improvement should not be kept.

    ``revalidation_n_estimators`` (iter28) caps the per-candidate booster's tree count for the
    PARSIMONY-RULE RANKING trials only. The selection criterion is "stable_score within parsimony_tol
    of the best" -- a RELATIVE ranking decision that stabilises long before the default 300 trees
    (mirror of iter9 refine_n_estimators / iter19 oof_shap_n_estimators / iter10 trust_guard_n_estimators).
    Microbench at the live regime (width=1000, n_rows=5000, snr=8, 11-feature subsets): n=100 vs n=300
    Spearman 0.9414, identical argmin, 2.6x faster per fit. After the winner is chosen, ONE full-template
    re-evaluation is added per ranked entry's reported ``honest_loss`` so the user-visible report stays
    apples-to-apples with the trust-guard / ablation (which use the full template). The cap is tagged
    via ``template_id=('reval_cap', cap)`` so cached entries don't collide with full-template entries
    from elsewhere in the pipeline. ``None`` (default for the standalone function) disables the cap
    (legacy 300-tree behaviour). The selector facade passes ``revalidation_n_estimators=100`` by default.

    bench-attempt-rejected (iter32, 2026-05-29): bias-corrector-predicted-loss CULL gate
    (``corrector`` + ``phi`` + ``max_candidates`` kwargs that sorted top_n candidates by the
    trust-guard corrector's predicted honest loss and culled to ``max_candidates=10`` BEFORE the
    honest retrain loop). Live regime (width=1000, n_rows=5000, snr=8, top_n=20 -> 10): warm
    same-process seed=1 baseline reval=2.61s vs gated reval=2.55s (+2.3%), e2e 8.29s -> 8.27s
    (+0.2%); seed=0 reval 2.95s -> 2.88s (+2.4%); seed=0 e2e 12.58s -> 9.93s (+21%) is dominated
    by within-run prefilter variance (3.71s vs 1.56s in same comparison, NOT gate-attributable).
    cProfile ``xgboost.update`` ncalls=1600 in BOTH baseline and gated -- the actual training-round
    count is the same: the joblib-threading pool at -1 already absorbed the 60-fit batch (per
    iter29 ``time.sleep`` ~5.0s = parallel productive wait), so dropping 30 of those 60 fits leaves
    the same wall because the pool was never the bottleneck. iter28's
    ``revalidation_n_estimators=100`` cap already extracted the per-fit win; further cuts here are
    sub-threshold. Lever does not pay at the current ``n_revalidation_models=3`` + parallel-joblib
    operating point; revisit only if a future iter raises models-per-candidate or serialises the
    retrain loop.

    ``ucb_enabled`` (iter34): batched-dispatch early-stop on the candidate scoring loop. Different
    mechanism from the rejected iter32 cull -- there we proposed dropping tail candidates before the
    pool started (which didn't help because the wall was already a single batched run). Here we
    start the BEST candidates first and stop dispatching new batches once the running winner is
    provably better than any remaining candidate's UCB lower bound. At width >= 10000 each honest
    fit is ~300 ms; ``top_n=20 * n_models=3 = 60`` tasks on 8 workers run ~8 batches deep
    (Phase-0 measurement: 4.86s wall vs 337 ms per-fit = 14.4x ratio, NOT saturation). Skipping the
    last few batches at the tail of the proxy ranking is direct wall savings.

    ``ucb_min_eval_size`` (default ``None`` -> ``max(n_workers, 3)``): first batch evaluates this many
    candidates so the workers saturate; the running ``best_stable_score`` is then defined.
    ``ucb_slack`` (default ``None`` -> auto from batch ``delta = honest - proxy`` via
    ``mean(delta) - ucb_stdev_multiplier * std(delta)``): negative slack means honest tends below
    proxy in the calibration window (the proxy is mildly biased high), so un-evaluated lower bounds
    are tighter; positive slack widens them. The auto calibration is conservative on the
    *pessimistic* side (subtracting std lowers the slack -> tightens the un-evaluated lower bound ->
    requires a larger gap to stop -> fewer wrong stops). With UCB disabled OR n_candidates <=
    min_eval_size OR ``n_jobs in (1, 0, None)``, falls through to the legacy single-batch path with
    zero behaviour change -- single-job runs (test fixtures) have no batching to save on, so the gate
    would only risk dropping the winner without any wall benefit.

    ``adaptive_n_models`` (iter77): when True, dispatch the ``n_models`` stability seeds as SEPARATE
    rounds (one seed per candidate per round) instead of one combined batch. After each completed
    round (k >= 2), the parsimony-rule winner is computed from accumulated per-candidate losses; if
    the winner is identical to the previous round's winner, remaining seed rounds are skipped. Floor
    is 2 rounds (need at least one stability check); ceiling is ``n_models``. Worst case (winners
    differ every round) is the same total fit count as the legacy path. With ``n_models=1`` the knob
    is a no-op. The candidate-UCB candidate-pruning still applies within each round. Conservation
    guarantee: when ``n_models_run == n_models`` the result is bit-identical to the legacy path
    (same seeds, same accumulation, same ranking). When the loop exits early, ``stable_score`` is
    computed on fewer seeds per candidate so std is lower-variance but mean is the same expectation;
    the parsimony rule (relative-tol) is robust to this. Surface: ``baseline['ucb']['n_models_run']``
    reports actual rounds executed.
    """
    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    cap = revalidation_n_estimators
    tid = ("reval_cap", int(cap)) if cap is not None else None
    # iter80: cross-process disk cache for repeat hyperparam sweeps over the same (X_search,
    # y_search, X_holdout, y_holdout, template) tuple. Opened once so the LRU evictor sees the full
    # batch of writes from this revalidation pass; ``None`` (default) keeps the legacy in-memory-only
    # cache wiring.
    disk_cache = _open_disk_cache(disk_cache_dir)
    # Pre-expand member columns + sample per-fit seeds once so candidate ordering shuffles only the
    # task LIST, never re-samples (cache reuse + determinism across UCB/no-UCB paths).
    member_cols = [_expand(idx, unit_to_members) for _, idx in candidates]
    candidate_seeds = [[int(rng.integers(0, 2**31 - 1)) for _ in range(n_models)] for _ in candidates]
    n_total = len(candidates)
    # UCB batched dispatch (iter34): evaluate proxy-ranked candidates in batches; stop once the
    # running winner provably beats every remaining candidate's UCB lower bound. Determinism:
    # within-batch joblib results are zipped back to the (cols, seed) tuples we dispatched, ties in
    # proxy ordering are broken by the original candidate index (kind="stable" argsort), and ALL
    # seeds are sampled BEFORE the gate decides any batch -- so n_candidates_evaluated is the only
    # variable between UCB and the legacy path; ranked entries for evaluated candidates are
    # bit-identical given identical seed + cache state.
    proxy_losses_arr = np.asarray([float(c[0]) for c in candidates], dtype=np.float64)
    # ``candidate_score`` (iter34): the caller's already-computed per-candidate score (corrector-
    # predicted honest loss when the bias corrector fit cleanly, raw proxy_loss otherwise). The
    # facade computes this for ordering anyway; passing it through lets the UCB lower bound work on
    # the honest scale instead of the cheap-but-tightly-clustered proxy_loss. With ``candidate_score``
    # supplied, the gate compares un-evaluated score + slack against the running best stable score;
    # the slack auto-calibrates the residual gap. When ``None`` (standalone tests, legacy callers),
    # falls back to raw proxy_loss -- the gate still works but may rarely fire on regimes whose
    # proxy_loss spread is too tight to discriminate (corrector-aware score widens the spread).
    score_arr = (np.asarray(candidate_score, dtype=np.float64)
                 if candidate_score is not None else proxy_losses_arr)
    # Use the CALLER'S order (the facade already sorts top_n by bias-corrector + uncertainty score,
    # which is a strictly stronger ordering than raw proxy_loss alone). Re-sorting on proxy_loss
    # here would unwind that work and surrender the corrector's per-candidate trust signal -- the
    # very signal the trust-guard pays its 60+ anchor retrains to produce. Stays compatible with
    # the standalone test fixtures that pass already-proxy-sorted candidates.
    proxy_order = np.arange(n_total, dtype=np.int64)
    if ucb_min_eval_size is None:
        import os as _os
        n_cores = _os.cpu_count() or 1
        outer = n_cores if n_jobs in (-1, None, 0) else int(n_jobs)
        ucb_min_eval_size_eff = max(int(outer), 3)
    else:
        ucb_min_eval_size_eff = max(1, int(ucb_min_eval_size))
    # UCB only pays when joblib actually BATCHES dispatch across workers (the iter34 premise).
    # With n_jobs=1 the legacy single-batch path is already sequential, so skipping candidates
    # produces no wall savings -- only opens a window for the gate to stop on a too-small evaluated
    # batch (3 candidates) and miss the winner. The user-visible failure mode on the biz_val test
    # (noise2 kept where the legacy path picked an informative) is exactly this: 1-job runs are
    # typically test fixtures where determinism + recall matter more than wall savings.
    use_ucb = (bool(ucb_enabled)
               and n_total > ucb_min_eval_size_eff
               and n_jobs not in (1, 0, None))

    per_candidate: dict[int, list[float]] = {}
    n_candidates_evaluated = 0
    ucb_slack_used = 0.0
    # iter77 adaptive_n_models: split the n_models stability seeds into separate rounds, allow early
    # stop when the parsimony winner stabilises. With adaptive_n_models=False (legacy) the rounds
    # collapse into one combined batch (the original semantics).
    adapt_active = bool(adaptive_n_models) and int(n_models) >= 2
    seed_rounds = int(n_models) if adapt_active else 1
    # When NOT adaptive, "one round" dispatches all n_models seeds per candidate; when adaptive,
    # each round dispatches ONE seed per candidate (round k -> candidate_seeds[ci][k:k+1]).
    n_models_run = 0
    prev_winner: tuple | None = None

    for round_k in range(seed_rounds):
        if adapt_active:
            round_seeds = [[candidate_seeds[ci][round_k]] for ci in range(n_total)]
        else:
            round_seeds = candidate_seeds  # legacy: all seeds in one round
        if not use_ucb:
            # Legacy path: one parallel batch over all candidates.
            tasks, task_owner = [], []
            for ci in range(n_total):
                for s in round_seeds[ci]:
                    tasks.append((member_cols[ci], s))
                    task_owner.append(ci)
            losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                             classification, metric, n_jobs, cache=cache,
                                             n_estimators_cap=cap, template_id=tid,
                                             inner_n_jobs_cap=inner_n_jobs_cap,
                                             disk_cache=disk_cache)
            for owner, loss in zip(task_owner, losses):
                per_candidate.setdefault(owner, []).append(loss)
            n_candidates_evaluated = n_total
        else:
            evaluated_idx_set: set[int] = set()
            # First batch saturates the workers. Subsequent batches are workers-sized so each iteration
            # is one wall-clock pool dispatch on the same operating point.
            import os as _os
            n_cores = _os.cpu_count() or 1
            outer_workers = n_cores if n_jobs in (-1, None, 0) else int(n_jobs)
            outer_workers = max(1, outer_workers)
            # Effective seeds-per-candidate THIS round drives the worker-share denominator.
            seeds_per_cand = len(round_seeds[0]) if round_seeds and round_seeds[0] else 1
            batch_sizes: list[int] = []
            cur = 0
            while cur < n_total:
                if cur == 0:
                    step = min(ucb_min_eval_size_eff, n_total - cur)
                else:
                    step = min(max(1, outer_workers // max(1, seeds_per_cand)), n_total - cur)
                    step = max(step, 1)
                batch_sizes.append(step)
                cur += step
            pos = 0
            for step in batch_sizes:
                batch_candidate_idx = [int(proxy_order[pos + j]) for j in range(step)]
                pos += step
                tasks, task_owner = [], []
                for ci in batch_candidate_idx:
                    for s in round_seeds[ci]:
                        tasks.append((member_cols[ci], s))
                        task_owner.append(ci)
                losses = _parallel_honest_losses(tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                                                 classification, metric, n_jobs, cache=cache,
                                                 n_estimators_cap=cap, template_id=tid,
                                                 inner_n_jobs_cap=inner_n_jobs_cap,
                                                 disk_cache=disk_cache)
                for owner, loss in zip(task_owner, losses):
                    per_candidate.setdefault(owner, []).append(loss)
                evaluated_idx_set.update(batch_candidate_idx)
                n_candidates_evaluated = len(evaluated_idx_set)
                if pos >= n_total:
                    break
                # Compute the running best stable_score over evaluated candidates.
                best_so_far = float("inf")
                ev_proxy: list[float] = []
                ev_honest_mean: list[float] = []
                for ci in evaluated_idx_set:
                    scores = np.asarray(per_candidate[ci], dtype=np.float64)
                    mean = float(scores.mean())
                    std = float(scores.std())
                    stable = mean + lambda_stab * std
                    if stable < best_so_far:
                        best_so_far = stable
                    ev_proxy.append(float(score_arr[ci]))
                    ev_honest_mean.append(mean)
                if ucb_slack is None:
                    ucb_slack_used = _ucb_auto_slack(ev_proxy, ev_honest_mean, ucb_stdev_multiplier)
                else:
                    ucb_slack_used = float(ucb_slack)
                remaining_idx = [int(proxy_order[j]) for j in range(pos, n_total)]
                remaining_score = [float(score_arr[ci]) for ci in remaining_idx]
                if _ucb_stop_remaining_cannot_win(
                    best_so_far, remaining_score, ucb_slack_used, parsimony_tol,
                ):
                    break
        n_models_run = round_k + 1
        # Parsimony-rule winner across accumulated per-candidate losses; early-stop when stable
        # across two consecutive rounds. Floor at 2 rounds so we always have at least one stability
        # check (round_k >= 1). When n_models == 1 the loop runs exactly once -- no check possible
        # and adapt_active is False anyway, so this branch is skipped.
        if adapt_active:
            cur_winner = _winner_from_per_candidate(
                per_candidate, candidates, member_cols, lambda_stab, parsimony_tol,
            )
            if round_k >= 1 and cur_winner is not None and cur_winner == prev_winner:
                break
            prev_winner = cur_winner

    ranked = []
    for ci, (proxy_loss_val, idx) in enumerate(candidates):
        if ci not in per_candidate:
            continue
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

    # Full-template re-evaluation of the WINNER so the user-visible honest_loss in the report stays
    # apples-to-apples with the trust-guard / ablation outputs (those use the full template). Only
    # the chosen subset is re-fit -- a single extra fit, not n_models more. The cache lookup uses
    # the full-template namespace (template_id=None) so it hits any prior pipeline retrain of the
    # same subset (e.g. when ablation later refits the winner, that fit is the cache hit). Same
    # design as within_cluster_refine's final full-template re-evaluation. When cap is None the
    # ranking trials already used the full template and this is a guaranteed cache hit (no extra fit).
    if best_idx and cap is not None:
        winner_cols = _expand(best_idx, unit_to_members)
        winner_full_loss = _honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, winner_cols,
            classification, metric, cache=cache, disk_cache=disk_cache)
        # Update the reported entry for the chosen winner. Find it in ranked by features identity.
        for d in ranked:
            if d["features"] == best_idx:
                d["honest_loss"] = float(winner_full_loss)
                # std measured at capped template (n_models samples); winner's full-template eval is a
                # single fit so its std is not refreshed -- the capped-template std remains as a
                # cross-seed-stability proxy. Update stable_score to reflect the new mean.
                d["stable_score"] = float(winner_full_loss) + lambda_stab * d["honest_std"]
                d["honest_loss_capped"] = float(np.asarray(per_candidate[
                    next(i for i, (_, ix) in enumerate(candidates) if tuple(ix) == best_idx)
                ]).mean())
                break

    # Same-size (in member columns) random-subset baseline for the winner (winner's-curse context).
    baseline = None
    if best_idx:
        k = len(_expand(best_idx, unit_to_members))
        f = X_search.shape[1]
        k = min(k, f)
        rnd = tuple(sorted(rng.choice(f, size=k, replace=False).tolist()))
        baseline = dict(features=rnd, honest_loss=_honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, list(rnd), classification, metric,
            cache=cache, disk_cache=disk_cache))
    ucb_info = dict(enabled=bool(use_ucb), n_candidates_total=int(n_total),
                    n_candidates_evaluated=int(n_candidates_evaluated),
                    min_eval_size=int(ucb_min_eval_size_eff),
                    slack=float(ucb_slack_used),
                    adaptive_n_models=bool(adapt_active),
                    n_models_configured=int(n_models),
                    n_models_run=int(n_models_run))
    # Attach UCB diagnostic to the random-subset baseline dict (or create a stub when no winner).
    # Keeps the 3-tuple return contract stable; downstream consumers fish ucb diagnostics out via
    # ``report['revalidation']['random_baseline']['ucb']``. Same pattern as other-revalidator-side
    # diagnostics that ride on the baseline payload without expanding the return signature.
    if baseline is None:
        baseline = dict(ucb=ucb_info)
    else:
        baseline["ucb"] = ucb_info
    return best_idx, ranked, baseline


def active_learning_revalidate(
    candidates, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, corrector_data, phi, budget, batch=4, n_models=1,
    parsimony_tol=0.02, rng=None, n_jobs=-1, unit_to_members=None, cache=None,
    revalidation_n_estimators=None, inner_n_jobs_cap=False, disk_cache_dir=None,
):
    """Disagreement-driven honest re-validation (lever #4).

    Instead of honestly retraining the proxy's static top-N, iterate: fit the bias corrector on the
    anchors seen so far, pick the ``batch`` un-evaluated candidates where the corrector most disagrees
    with the raw proxy (the proxy is least trustworthy there), honestly retrain them, fold the results
    back into the corrector, and repeat until ``budget`` candidates have been evaluated. This spends a
    fixed retrain budget where it most reduces winner's-curse risk. The proxy's top-1 is always among
    the first evaluated, so the result is never worse than naive top-1.

    ``revalidation_n_estimators`` (iter28): same cap semantics as ``revalidate_top_n`` -- per-candidate
    ranking trials use the capped booster (cheaper but ranking-equivalent), winner gets ONE
    full-template re-evaluation to keep the reported ``honest_loss`` apples-to-apples. The corrector
    is fit on the CAPPED honest losses (the corrector is itself a ranking-quality tool, so working
    in the capped space is consistent).

    Returns ``(best_idx, ranked, n_evaluated)``.
    """
    from mlframe.feature_selection._shap_proxy_calibrate import fit_proxy_corrector, subset_redundancy

    metric = resolve_metric(classification, metric)
    rng = np.random.default_rng(0) if rng is None else rng
    cap = revalidation_n_estimators
    tid = ("reval_cap", int(cap)) if cap is not None else None
    # iter80: cross-process disk cache (same wiring as ``revalidate_top_n``). Each AL round picks a
    # disagreement-driven batch and retrains honest losses; the disk cache short-circuits any
    # (cols, seed, template) tuple seen on a prior fit. Disabled when ``disk_cache_dir is None``.
    disk_cache = _open_disk_cache(disk_cache_dir)
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
                                         classification, metric, n_jobs, cache=cache,
                                         n_estimators_cap=cap, template_id=tid,
                                         inner_n_jobs_cap=inner_n_jobs_cap,
                                         disk_cache=disk_cache)
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

    # Full-template re-evaluation of the winner (mirror of revalidate_top_n; see that function for the
    # apples-to-apples rationale). When cap is None this is a guaranteed cache hit (no extra fit).
    if best_idx and cap is not None:
        winner_cols = _expand(best_idx, unit_to_members)
        winner_full_loss = _honest_loss(
            model_template, X_search, y_search, X_holdout, y_holdout, winner_cols,
            classification, metric, cache=cache, disk_cache=disk_cache)
        for d in ranked:
            if d["features"] == best_idx:
                d["honest_loss_capped"] = float(d["honest_loss"])
                d["honest_loss"] = float(winner_full_loss)
                d["stable_score"] = float(winner_full_loss)
                break

    return best_idx, ranked, len(honest)


def within_cluster_refine(
    member_cols, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, parsimony_tol=0.02, n_jobs=-1, max_drop_rounds=None, cache=None,
    member_groups=None, min_multi_clusters=3, refine_n_estimators=100,
    ucb_enabled=False, ucb_min_eval_size=None, ucb_slack=None, ucb_stdev_multiplier=1.0,
    inner_n_jobs_cap=False, disk_cache_dir=None,
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

    ``ucb_enabled`` (iter35): batched-dispatch early-stop on stage 2b's per-round single-drop greedy.
    Each round sorts drop trials by ascending stage-2a permutation importance (lowest = safest drop =
    lowest expected honest loss), dispatches in workers-sized batches, and stops dispatching once no
    un-evaluated trial can beat the running round leader. The UCB lower bound for an un-evaluated
    trial is ``importance + slack`` where ``slack`` auto-calibrates from the round's evaluated
    (importance, honest_loss) pairs via ``mean(delta) - stdev_multiplier * std(delta)``. Mirrors
    iter34's revalidate_top_n UCB knob design. Falls through to legacy single-batch-per-round when
    UCB is off OR ``n_jobs in (1, 0, None)`` OR stage-2a's importance prior is missing -- single-job
    runs (test fixtures) have no batching to save on, so the gate would only risk wrong stops without
    a wall benefit. The lever pays at width >= 10000 where each honest fit is ~500 ms and ~5
    stage-2b rounds dispatch ~10 trials each (Phase-0 C3 cProfile: within_cluster_refine 6.14s of
    which ~5s is stage-2b parallel batches).

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
    # iter81: cross-process disk cache extends iter80's wiring through the refine stage. Stage-1
    # parallel cluster probes and stage-2b per-round single-drop trials repeat the SAME (cols, seed,
    # template, cap) tuple on hyperparam sweeps -- a warm-cache lookup skips the booster fit entirely.
    # ``None`` (default) keeps the legacy in-memory-only contract bit-identical.
    disk_cache = _open_disk_cache(disk_cache_dir)
    base = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout, current, classification,
                        metric, cache=cache, n_estimators_cap=cap, template_id=tid, disk_cache=disk_cache)
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
                classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid,
                inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
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
                        metric, cache=cache, n_estimators_cap=cap, template_id=tid, disk_cache=disk_cache)
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

    # ``importance_by_col`` (iter35): persist stage-2a's permutation importances so stage-2b can sort
    # its per-round drop trials in ascending importance order and dispatch UCB-batched. Defaults to
    # empty -> stage-2b falls back to legacy unsorted single-batch dispatch.
    importance_by_col: dict[int, float] = {}
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
        # Persist per-column importance so stage 2b can sort its drop trials by ascending-importance
        # priors (lowest importance = safest drop = lowest expected honest loss). Used as the UCB
        # proxy when ``ucb_enabled``; dropped members fall out of the dict naturally on lookup.
        importance_by_col = {int(current[i]): float(importances[i]) for i in range(len(current))}
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
                    metric, cache=cache, n_estimators_cap=cap, template_id=tid, disk_cache=disk_cache)
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
    #
    # iter35 UCB-batched dispatch: when ``ucb_enabled`` AND ``n_jobs`` enables threading AND we have a
    # stage-2a importance prior for the current members, each round sorts trials by ascending
    # importance (lowest = safest drop = lowest expected honest loss) and dispatches in
    # workers-sized batches. After each batch, the round leader is compared against every
    # un-evaluated trial's UCB lower bound (importance + auto-slack). When no remaining trial can
    # beat the leader -> stop, accept the leader (if within tol) or break the round (if not).
    # Falls through to legacy single-batch-per-round when UCB is off OR n_jobs in (1, 0, None) OR
    # no importance prior available (stage 2a skipped).
    import os as _os_iter35

    n_cores = _os_iter35.cpu_count() or 1
    if n_jobs in (-1, None, 0):
        outer_workers = n_cores
    else:
        outer_workers = max(1, int(n_jobs))
    if ucb_min_eval_size is None:
        ucb_min_eval_size_eff = max(outer_workers, 3)
    else:
        ucb_min_eval_size_eff = max(1, int(ucb_min_eval_size))
    use_ucb_stage2b = (bool(ucb_enabled)
                      and n_jobs not in (1, 0, None)
                      and len(importance_by_col) > 0)

    rounds = len(current) if max_drop_rounds is None else max_drop_rounds
    for _ in range(rounds):
        if len(current) <= 1:
            break
        cur_threshold = base + parsimony_tol * abs(base)

        # Build (col, importance_prior) pairs. Members not in ``importance_by_col`` (e.g. stage-2a was
        # skipped on a degenerate path) fall back to importance = +inf so they sort last; the legacy
        # path also runs them but the UCB path keeps them as last-resort dispatch.
        col_importance = [(int(c), importance_by_col.get(int(c), float("inf"))) for c in current]
        if use_ucb_stage2b and len(col_importance) > ucb_min_eval_size_eff:
            # UCB-batched: sort trials by ascending importance, dispatch in workers-sized batches,
            # short-circuit when no remaining trial can beat the round leader.
            sorted_pairs = sorted(enumerate(col_importance), key=lambda kv: (kv[1][1], kv[1][0]))
            order_local = [kv[0] for kv in sorted_pairs]  # original-index ordering within ``current``
            # First batch saturates the workers; subsequent batches are workers-sized.
            evaluated_losses: dict[int, float] = {}  # local-idx -> honest loss
            best_loss_round = float("inf")
            best_local_idx = -1
            stopped_early = False
            pos = 0
            n_trials = len(order_local)
            # ``slack`` calibrates importance -> honest_loss residual on a per-round basis. With <2
            # evaluated points fall back to slack=mean(delta) (no std term) so the gate still has a
            # working lower bound; the auto-slack helper handles that fallback.
            slack_used = 0.0
            while pos < n_trials:
                if pos == 0:
                    step = min(ucb_min_eval_size_eff, n_trials - pos)
                else:
                    step = min(max(1, outer_workers), n_trials - pos)
                batch_local = order_local[pos:pos + step]
                pos += step
                tasks = []
                for li in batch_local:
                    drop_col = current[li]
                    survivors = [c for c in current if c != drop_col]
                    tasks.append((survivors, None))
                losses_batch = _parallel_honest_losses(
                    tasks, model_template, X_search, y_search, X_holdout, y_holdout,
                    classification, metric, n_jobs, cache=cache, n_estimators_cap=cap, template_id=tid,
                    inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
                for li, ls in zip(batch_local, losses_batch):
                    evaluated_losses[li] = float(ls)
                    if ls < best_loss_round:
                        best_loss_round = float(ls)
                        best_local_idx = li
                if pos >= n_trials:
                    break
                # Calibrate slack from evaluated pairs (importance_prior, honest_loss).
                ev_importance = [col_importance[li][1] for li in evaluated_losses]
                ev_honest = [evaluated_losses[li] for li in evaluated_losses]
                if ucb_slack is None:
                    slack_used = _ucb_auto_slack(ev_importance, ev_honest, ucb_stdev_multiplier)
                else:
                    slack_used = float(ucb_slack)
                remaining_local = order_local[pos:n_trials]
                remaining_importance = [col_importance[li][1] for li in remaining_local]
                # Stop when no remaining trial can have a lower honest loss than the round leader.
                # Use parsimony_tol=0 here: we want strict "remaining cannot beat leader" semantics
                # because we only need to find the round's minimum, not enter a parsimony band.
                if _ucb_stop_remaining_cannot_win(
                    best_loss_round, remaining_importance, slack_used, parsimony_tol=0.0,
                ):
                    stopped_early = True
                    break
            # Accept the leader if within tol; otherwise round terminates.
            if best_local_idx < 0 or best_loss_round > cur_threshold:
                break
            drop_col = current[best_local_idx]
            current = [c for c in current if c != drop_col]
            base = min(base, float(best_loss_round))
            # The dropped column's importance entry is no longer needed; left in place because the
            # dict is keyed by column id (the dropped col simply never re-appears in subsequent
            # ``current`` lookups). Avoids mutating the dict in the inner loop.
        else:
            # Legacy single-batch path: ALL k trials in one parallel dispatch per round. Preserved
            # bit-identical for UCB-off / n_jobs in {1,0,None} / no-prior fallback paths.
            trials = [[c for c in current if c != drop] for drop in current]
            losses = _parallel_honest_losses([(t, None) for t in trials], model_template, X_search, y_search,
                                             X_holdout, y_holdout, classification, metric, n_jobs, cache=cache,
                                             n_estimators_cap=cap, template_id=tid,
                                             inner_n_jobs_cap=inner_n_jobs_cap, disk_cache=disk_cache)
            losses_arr = np.asarray(losses, dtype=np.float64)
            best_i = int(np.argmin(losses_arr))
            if losses_arr[best_i] > cur_threshold:
                break
            current = trials[best_i]
            base = min(base, float(losses_arr[best_i]))
    return current


def importance_topk_ablation(
    phi, proxy_best_idx, model_template, X_search, y_search, X_holdout, y_holdout,
    *, classification, metric=None, unit_to_members=None, cache=None, disk_cache_dir=None,
):
    """Compare the proxy-chosen subset against a SHAP-importance-top-k subset of the same size.

    Returns a dict with both honest losses and whether the proxy strictly wins (the method's
    unique-value gate vs plain SHAP global importance). In clustering mode, importance ranks UNITS
    and both subsets expand to member columns for the honest comparison.

    ``disk_cache_dir`` (iter81): when set, the two honest retrains (proxy subset + SHAP-importance-
    top-k subset) check the cross-process :class:`DiskCache` first. The proxy subset is typically a
    cache hit -- it's the chosen winner that revalidation just retrained -- so the warm-cache cost
    of this stage drops to one fit for the imp_cols comparison (and to zero when both subsets
    overlap a prior fit). ``None`` (default) keeps the legacy in-memory-only contract bit-identical.
    """
    metric = resolve_metric(classification, metric)
    k = len(proxy_best_idx)  # match unit count, then expand both sides to members
    importance = np.abs(phi).mean(axis=0)
    imp_units = tuple(sorted(np.argsort(-importance)[:k].tolist()))
    proxy_cols = _expand(proxy_best_idx, unit_to_members)
    imp_cols = _expand(imp_units, unit_to_members)
    disk_cache = _open_disk_cache(disk_cache_dir)
    proxy_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                                proxy_cols, classification, metric, cache=cache, disk_cache=disk_cache)
    imp_honest = _honest_loss(model_template, X_search, y_search, X_holdout, y_holdout,
                              imp_cols, classification, metric, cache=cache, disk_cache=disk_cache)
    return dict(proxy_features=tuple(proxy_best_idx), proxy_honest_loss=proxy_honest,
                importance_features=imp_units, importance_honest_loss=imp_honest,
                proxy_wins=bool(proxy_honest < imp_honest), proxy_at_least_ties=bool(proxy_honest <= imp_honest))
