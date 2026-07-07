"""Honest-loss retraining + caching primitives for the SHAP-proxied selector.

Leaf module: disk-cache key builders, the per-fit :class:`HonestLossCache`, the single honest
retrain (:func:`_honest_loss`), the permutation-importance ranking pass, and the parallel
honest-loss pool. No back-import to the revalidate parent so the refine sibling can depend on it.
Lower = better loss everywhere, matching the proxy objective.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    log_loss, roc_auc_score, root_mean_squared_error,
)
from mlframe.metrics.core import fast_brier_score_loss, fast_log_loss, fast_roc_auc, fast_mean_absolute_error

logger = logging.getLogger(__name__)


def _slice_cols_to_numpy(X, cols):
    """Gather ``cols`` from ``X`` (DataFrame or ndarray) as a plain float numpy array.

    Speed lever (2026-06-08): the honest-retrain hot loop (revalidation / within-cluster refine /
    trust guard / ablation) trains + predicts the booster hundreds of times per fit on a column
    slice of the SAME ``X_search`` / ``X_hold`` frames. Passing the slice as a *named* pandas
    DataFrame forces xgboost to (a) extract C-string feature names via ``from_cstr_to_pystr`` on
    every fit AND every predict, and (b) run ``_validate_features`` name-matching -- cProfile @
    width=4000 attributed ~1.0s (8% of a 12.6s fit) to ``from_cstr_to_pystr`` / ``_get_feature_info``
    / ``_validate_features``, all GIL-holding Python work that also serialises the threaded honest
    pool. Tree splits depend only on column VALUES + POSITIONS, never names; a microbench confirmed
    fit-on-numpy then predict-on-numpy is BIT-IDENTICAL to the named-DataFrame path (max abs diff
    0.0, ``np.array_equal`` True) at ~1.13x per fit. So we strip the names: slice columns to a
    contiguous numpy array once and hand xgboost positional columns. Identity is preserved because
    fit and predict use the SAME column order (``cols``) so positional alignment == name alignment.
    Accepts ndarray input unchanged (already nameless) for callers that pre-converted upstream."""
    if hasattr(X, "iloc"):
        return X.iloc[:, cols].to_numpy(copy=False)
    return np.asarray(X)[:, cols]


# Cache-key namespace for the cross-process honest_loss disk cache. Keeps entries from this consumer
# from colliding with the shap_phi_ entries the OOF-SHAP path writes into the same cache_dir, and
# makes the cache human-greppable (``ls cache_dir/honest_loss_*`` shows reval/trust entries).
_HONEST_LOSS_CACHE_PREFIX = "honest_loss_"

# Cache-key namespace for the cross-process permutation-importance FITTED BOOSTER cache (iter82).
# Stage 2a of within_cluster_refine fits a single booster on ``current_cols`` then issues k cheap
# predict-on-shuffled passes; the fit dominates the runtime. Caching the fitted booster pickle (not
# the scalar loss, unlike honest_loss_) lets a warm re-run of the same (X_tr, y_tr, cols, template)
# skip the fit entirely while keeping per-feature shuffle predicts deterministic. Booster pickles
# are typically 10-100 KB at our regimes.
_PERM_IMP_FIT_CACHE_PREFIX = "perm_imp_fit_"


def _build_honest_loss_disk_key(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric, seed, n_estimators_cap, template_id) -> Optional[str]:
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


def _build_perm_fit_disk_key(model_template, X_tr, y_tr, idx, n_estimators_cap, template_id) -> Optional[str]:
    """Stable cache key for the FITTED booster used by ``_permutation_importance_ranking`` (iter82).

    Permutation importance fits a single booster on (X_tr[:, idx], y_tr); subsequent per-column
    shuffle predictions are evaluated against that fitted booster. The cached payload here is the
    pickled fitted booster itself -- different from ``_HONEST_LOSS_CACHE_PREFIX`` entries (scalar
    losses). Key inputs therefore depend ONLY on the fit determinants (training data + column
    subset + template params + n_estimators_cap + template_id); the permutation seed is intentionally
    NOT included because the seed affects only the post-fit shuffle predicts, never the fit itself.

    The X_ev / y_ev evaluation data is likewise NOT in the key: the cached booster is independent of
    where it gets scored. Callers that change X_ev re-use the cached fit and only re-do the cheap
    predict pass.
    """
    try:
        from mlframe.utils.disk_cache import compose_key, hash_array_summary, hash_object

        try:
            params = model_template.get_params(deep=False)
        except Exception:
            params = {"_repr": repr(model_template)}
        x_tr_key = hash_array_summary(X_tr.values if hasattr(X_tr, "values") else np.asarray(X_tr))
        y_tr_key = hash_array_summary(np.asarray(y_tr))
        state_key = hash_object({
            "cols": tuple(sorted(int(c) for c in idx)),
            "n_estimators_cap": n_estimators_cap,
            "template_id": template_id,
            "params": params,
        })
        return _PERM_IMP_FIT_CACHE_PREFIX + compose_key(x_tr_key, y_tr_key, state_key)
    except Exception as exc:
        logger.debug("_permutation_importance_ranking: disk-cache key build failed (%s); skipping cache", exc)
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
    valid: dict = getattr(est, "get_params", lambda: {})()
    for name in _N_ESTIMATORS_PARAMS:
        if name in valid:
            try:
                est.set_params(**{name: int(cap)})
                return name
            except (ValueError, TypeError):
                continue
    return None


def _classification_proba(est, X_ev_arr):
    """Probability prediction for the holdout, multiclass-aware.

    Returns the positive-class column ``(n,)`` for a binary fit (booster classes ``{0,1}``) and the full
    probability matrix ``(n, C)`` for a >2-class fit. A single-class anchor returns a 1-column proba
    (sklearn convention); a hard ``[:, 1]`` would raise IndexError on such rare-class anchors, so we fall
    back to the lone column (all-1.0 for the sole class) and let the loss layer's NaN/AUC guards handle the
    degenerate holdout. The downstream :func:`_loss_from_predictions` dispatches on the shape: a 1-D vector
    is the binary positive-class probability, a 2-D matrix is the multiclass probability matrix."""
    proba = est.predict_proba(X_ev_arr)
    if proba.shape[1] <= 2:
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    return proba


# Back-compat alias: the prior name leaked into a regression test (L2). It now also covers the
# multiclass matrix path, so the name is generalised but the symbol is preserved.
_positive_class_proba = _classification_proba


def _loss_from_predictions(p_or_pred, y_ev, classification, metric):
    """Compute the holdout loss from a precomputed prediction vector (no fit, no slicing).

    Exposed so :func:`_permutation_importance_ranking` can reuse the loss-aggregation branches
    without re-fitting the booster -- the permutation-importance pass scores k shuffled-column
    holdout predictions per fit, so we want the predict+loss path with zero fit overhead.

    Multiclass (L3): when the classification probability is a 2-D ``(n, C>2)`` matrix the binary
    closed forms (``brier_score_loss`` / ``log_loss(labels=[0,1])`` / scalar ``roc_auc_score``) are
    wrong, so we route to multiclass log-loss (over the observed class set) for the pointwise metrics
    and one-vs-rest macro AUC for the ranking metric. The proxy search itself stays binary (a single
    SHAP margin -> sigmoid), but the honest-loss layer must score a multiclass target correctly rather
    than silently mis-scoring or crashing."""
    if classification:
        p = p_or_pred
        if np.ndim(p) > 1:  # multiclass probability matrix (n, C)
            return _multiclass_loss_from_proba(p, y_ev, metric)
        if metric == "brier":
            return float(fast_brier_score_loss(y_ev, p))
        if metric == "logloss":
            # eps=1e-7 reproduces the previous np.clip(p, 1e-7, 1 - 1e-7) floor bit-for-bit (kernel clips internally).
            return float(fast_log_loss(y_ev, p, eps=1e-7))
        # A single-class holdout has no defined AUC; return NaN (dropped by every downstream finite-mask) rather than a magic
        # 1.0 that masquerades as a measured loss and biases the corrector / stable-score ranking on rare-class anchors.
        if len(np.unique(y_ev)) < 2:
            return float("nan")
        return float(1.0 - fast_roc_auc(y_ev, p))
    pred = p_or_pred
    return float(fast_mean_absolute_error(y_ev, pred)) if metric == "mae" else float(root_mean_squared_error(y_ev, pred))


def _multiclass_loss_from_proba(proba, y_ev, metric):
    """Honest loss for a >2-class holdout from the full ``(n, C)`` probability matrix (lower=better).

    ``brier``/``logloss`` map to multiclass log-loss (a proper scoring rule that subsumes binary
    log-loss); ``auc`` maps to one-vs-rest macro AUC turned into a loss (``1 - auc``). The holdout may
    not contain every training class, so we pass the booster's class count as ``labels`` to keep the
    probability columns aligned. A holdout with a single realised class has no defined AUC -> NaN
    (dropped by the finite-mask, mirroring the binary single-class guard)."""
    n_classes = proba.shape[1]
    all_labels = list(range(n_classes))
    if metric == "auc":
        if len(np.unique(y_ev)) < 2:
            return float("nan")
        auc = roc_auc_score(y_ev, proba, multi_class="ovr", average="macro", labels=all_labels)
        return float(1.0 - auc)
    # brier + logloss both fold into multiclass log-loss (the proper multiclass scoring rule).
    return float(log_loss(y_ev, np.clip(proba, 1e-7, 1.0), labels=all_labels))


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
    # Numpy-slice the fit/predict columns (strip pandas feature names -> xgboost skips the
    # per-call ``from_cstr_to_pystr`` + ``_validate_features`` marshalling; bit-identical, see
    # ``_slice_cols_to_numpy``). Same ``cols`` order on both sides keeps positional == named.
    est.fit(_slice_cols_to_numpy(X_tr, cols), y_tr)
    X_ev_arr = _slice_cols_to_numpy(X_ev, cols)
    if classification:
        p = _positive_class_proba(est, X_ev_arr)
    else:
        p = est.predict(X_ev_arr)
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
                                    metric, *, n_estimators_cap=None, seed=0, inner_n_jobs=None,
                                    disk_cache=None, template_id=None):
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

    ``disk_cache`` (iter82): when supplied, the fitted booster pickle is cached under a key derived
    from (X_tr summary, y_tr summary, sorted cols, template params, n_estimators_cap, template_id).
    A warm second call with identical fit determinants reloads the booster from disk and skips the
    fit entirely -- the perm-importance shuffle predicts (cheap) still run on the cached estimator.
    Cache hit / miss / pickle errors degrade silently to the compute path (best-effort policy).
    """
    cols = list(current_cols)
    # Cross-process fit cache (iter82): try to deserialize a previously-trained booster on the same
    # (X_tr, y_tr, cols, template). Hash-build failure (None key) and DiskCache.get failure both
    # fall through to the standard fit path; the cache is never a correctness gate. Note: we cache
    # the FITTED estimator, not the loss / importance vector, so the per-feature shuffle predicts
    # still execute against a numerically-identical model.
    est = None
    fit_disk_key = None
    if disk_cache is not None:
        fit_disk_key = _build_perm_fit_disk_key(model_template, X_tr, y_tr, cols, n_estimators_cap, template_id)
        if fit_disk_key is not None:
            try:
                cached_est = disk_cache.get(fit_disk_key)
            except Exception as exc:
                logger.debug("_permutation_importance_ranking: disk cache get failed (%s); skipping", exc)
                cached_est = None
            if cached_est is not None:
                est = cached_est
    if est is None:
        est = clone(model_template)
        if n_estimators_cap is not None:
            _try_cap_n_estimators(est, n_estimators_cap)
        if inner_n_jobs is not None and hasattr(est, "n_jobs"):
            try:
                est.set_params(n_jobs=int(inner_n_jobs))
            except (ValueError, TypeError):
                pass
        # Fit on a nameless numpy slice (strip pandas feature names -> xgboost skips the
        # ``from_cstr_to_pystr`` + ``_validate_features`` marshalling; bit-identical to the named
        # path, see ``_slice_cols_to_numpy``). The shuffle-predict loop below feeds numpy too.
        est.fit(_slice_cols_to_numpy(X_tr, cols), y_tr)
        if disk_cache is not None and fit_disk_key is not None:
            try:
                disk_cache.put(fit_disk_key, est)
            except Exception as exc:
                logger.debug("_permutation_importance_ranking: disk cache put failed (%s); skipping", exc)
    # Score the un-permuted base once; cheaper to use the same model with the un-shuffled matrix.
    X_ev_arr = _slice_cols_to_numpy(X_ev, cols)
    if classification:
        base_p = _positive_class_proba(est, X_ev_arr)
    else:
        base_p = est.predict(X_ev_arr)
    base_loss = _loss_from_predictions(base_p, y_ev, classification, metric)

    rng = np.random.default_rng(int(seed))
    # Pre-build column-major copy so we can swap one column at a time without re-allocating the matrix.
    # Predict directly on the numpy buffer: the booster was fit nameless above, so it accepts
    # positional numpy columns -- this drops the per-shuffle DataFrame rewrap (k extra allocations +
    # name marshalling per ranking pass), bit-identical because the column order never changes.
    # Own a contiguous, writable copy (X_ev_arr may be a view into X_ev) so the in-place column
    # swaps below never touch the caller's buffer; ``predict`` needs C-contiguous input anyway.
    X_perm = np.ascontiguousarray(X_ev_arr).copy()
    perm = rng.permutation(X_perm.shape[0])
    importances = np.zeros(len(cols), dtype=np.float64)
    for j in range(len(cols)):
        orig = X_perm[:, j].copy()
        X_perm[:, j] = orig[perm]
        if classification:
            p = _positive_class_proba(est, X_perm)
        else:
            p = est.predict(X_perm)
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
    # bench-attempt-rejected (iter103/iter104): swapping joblib.Parallel(prefer="threads") for
    # concurrent.futures.ThreadPoolExecutor was attempted on the strength of cProfile attributing
    # ~13.8s tottime to time.sleep inside joblib's _retrieve polling loop on a 22.4s fit. Microbench
    # at 60 short xgb fits on 8 cores: trial 1 = +3.2% (futures faster), trial 2 = -7.7% (joblib
    # faster). Full-fit re-profile after the swap measured +14% wall (regression), but reproducibility
    # was poor at 1 trial. Both backends are within noise on this workload; the polling overhead is
    # real in the profile but doesn't translate to measurable end-to-end wall. Keeping joblib.
    from joblib import Parallel, delayed

    return Parallel(n_jobs=outer, prefer="threads")(
        delayed(_honest_loss)(model_template, X_tr, y_tr, X_ev, y_ev, idx, classification, metric,
                              seed=seed, inner_n_jobs=inner, cache=cache,
                              n_estimators_cap=n_estimators_cap, template_id=template_id,
                              disk_cache=disk_cache)
        for idx, seed in tasks
    )
