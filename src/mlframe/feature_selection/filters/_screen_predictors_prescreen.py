"""Self-contained pre-screen helpers for ``screen_predictors``.

Two pure passes lifted out of the screening spine so the parent stays under the
1k-LOC ceiling. Neither snapshots / restores the global RNG -- that stays on the
``screen_predictors`` try/finally spine; ``compute_fdr_gain_floor`` only forwards
the already-seeded ``random_seed`` into the permutation-null kernel.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cardinality_prescreen(factors_data, factors_nbins, factors_names, x, y, verbose):
    """Drop high-cardinality columns before candidate enumeration.

    The Miller-Madow bias on plug-in MI is ~(|X|-1)*(|Y|-1)/(2n) nats; columns with
    ``nbins_x > 2*sqrt(n)`` are too cardinality-biased to score honestly (user_id-style
    high-level categoricals), so they are removed from the active factor index set ``x``.
    Returns ``(x, refused_set)`` where ``x`` matches the caller's container type (set or list)
    minus the refused indices.

    Caller gates on ``cardinality_bias_correction`` and ``factors_data.shape[1] > 0`` before
    invoking; this helper assumes both hold.
    """
    _n_for_screen = int(factors_data.shape[0])
    _y_idx_for_screen = int(y[0]) if hasattr(y, "__len__") else int(y)
    # Refuse columns with nbins_x > 2*sqrt(n) (cat-FE convention). At n=2500 this is 100 - catches user_id (1200 levels) cleanly; at n=309 it is 35, passing 5-bin numerics.
    _nbins_x_ceiling = 2.0 * float(np.sqrt(_n_for_screen))
    _refused = []
    _refused_set = set()
    for _col_idx in range(factors_data.shape[1]):
        if _col_idx == _y_idx_for_screen:
            continue
        _nbins_x = int(factors_nbins[_col_idx])
        if _nbins_x <= 1:
            continue
        if _nbins_x > _nbins_x_ceiling:
            _refused.append(_col_idx)
            _refused_set.add(_col_idx)
    if _refused and verbose >= 1:
        _names = [factors_names[i] if factors_names is not None and i < len(factors_names) else f"col_{i}" for i in _refused]
        logger.info(
            "screen_predictors: pre-screening dropped %d high-cardinality column(s) "
            "(nbins_x > 2*sqrt(n)=%.0f at n=%d): %s. Bin or target-encode before "
            "fitting if they carry real signal. Disable via "
            "cardinality_bias_correction=False.",
            len(_refused), _nbins_x_ceiling, _n_for_screen,
            _names[:10] + (["..."] if len(_names) > 10 else []),
        )
    # Remove refused columns from the active factor index set so they're never enumerated as candidates at any interactions_order.
    if _refused_set:
        if isinstance(x, set):
            x = x - _refused_set
        else:
            x = [i for i in x if i not in _refused_set]
    return x, _refused_set


def compute_fdr_gain_floor(
    factors_data,
    factors_nbins,
    x,
    y,
    *,
    screen_fdr_null_permutations,
    screen_fdr_null_quantile,
    screen_fdr_min_features,
    screen_fdr_target_oversplit_ratio,
    screen_fdr_min_rows_per_joint_cell,
    cardinality_bias_correction,
    random_seed,
    verbose,
    # 2026-07-09 fix: optional cross-round cache, keyed on the candidate-pool identity + all inputs
    # that affect the result. ``pooled_permutation_null_gain_floor`` recomputes a full (n_permutations
    # x n_candidates x n_rows) histogram+MI pass on EVERY call; when the SAME raw-column pool recurs
    # across screen_predictors rounds (the common case -- data only grows via appended engineered
    # columns, existing column content never changes), the earlier round's floor is mathematically
    # identical, not approximate, so a hit skips that whole pass. Threaded the same way as
    # screen_predictors' ``seed_caches``: the CALLER owns the dict's lifetime (must be scoped to one
    # fit, never shared across separate ``.fit()`` calls or concurrent fits). ``None`` (default)
    # disables caching entirely -- legacy behaviour, always recompute.
    maxt_floor_cache: "dict | None" = None,
):
    """Westfall-Young maxT permutation-null gain floor over the finalised order-1 pool.

    Fires on a WIDE pool (>= ``screen_fdr_min_features``, best-of-p selection bias) OR a NARROW pool
    meeting the target-over-split gate (MDLP-over-split heavy-tailed regression target whose plug-in
    MI bias lifts pure-noise columns past the gain floors). Returns the gain floor (0.0 = no-op).

    Does NOT touch the global RNG: ``random_seed`` is forwarded to the permutation-null kernel, which
    runs after the screening spine already seeded + snapshotted the process RNG.
    """
    if not (screen_fdr_null_permutations and int(screen_fdr_null_permutations) > 0):
        return 0.0
    _fdr_pool = np.asarray(sorted(x) if isinstance(x, set) else list(x))
    from ._permutation_null import (
        pooled_permutation_null_gain_floor,
        target_oversplit_floor_applies,
    )

    _y_idx_fdr = int(y[0]) if hasattr(y, "__len__") else int(y)
    _wide_pool = len(_fdr_pool) >= int(screen_fdr_min_features)
    _narrow_oversplit = (
        not _wide_pool
        and len(_fdr_pool) >= 2
        and target_oversplit_floor_applies(
            factors_nbins,
            _fdr_pool,
            _y_idx_fdr,
            int(factors_data.shape[0]),
            oversplit_ratio=float(screen_fdr_target_oversplit_ratio),
            min_rows_per_joint_cell=float(screen_fdr_min_rows_per_joint_cell),
        )
    )
    if not (_wide_pool or _narrow_oversplit):
        return 0.0

    _cache_key = None
    if maxt_floor_cache is not None:
        _cache_key = (
            tuple(int(v) for v in _fdr_pool.tolist()),
            _y_idx_fdr,
            int(screen_fdr_null_permutations),
            float(screen_fdr_null_quantile),
            bool(cardinality_bias_correction),
            None if random_seed is None else int(random_seed),
            int(factors_data.shape[0]),
        )
        _cached = maxt_floor_cache.get(_cache_key)
        if _cached is not None:
            if _cached > 0.0 and verbose >= 1:
                logger.info(
                    "screen_predictors: maxT permutation-null gain floor=%.5f over p=%d "
                    "candidates (q=%.2f, K=%d) - rejects chance-max noise at scale [cache hit].",
                    _cached, len(_fdr_pool), float(screen_fdr_null_quantile),
                    int(screen_fdr_null_permutations),
                )
            return _cached

    _fdr_gain_floor = pooled_permutation_null_gain_floor(
        factors_data,
        factors_nbins,
        _fdr_pool,
        _y_idx_fdr,
        n_permutations=int(screen_fdr_null_permutations),
        quantile=float(screen_fdr_null_quantile),
        cardinality_bias_correction=cardinality_bias_correction,
        random_seed=random_seed,
    )
    if _cache_key is not None and maxt_floor_cache is not None:
        maxt_floor_cache[_cache_key] = _fdr_gain_floor
    if _fdr_gain_floor > 0.0 and verbose >= 1:
        logger.info(
            "screen_predictors: maxT permutation-null gain floor=%.5f over p=%d "
            "candidates (q=%.2f, K=%d) - rejects chance-max noise at scale.",
            _fdr_gain_floor, len(_fdr_pool), float(screen_fdr_null_quantile),
            int(screen_fdr_null_permutations),
        )
    return _fdr_gain_floor
