"""Self-contained pre-screen helpers for ``screen_predictors``.

Two pure passes lifted out of the screening spine so the parent stays under the
1k-LOC ceiling. Neither snapshots / restores the global RNG - that stays on the
``screen_predictors`` try/finally spine; ``compute_fdr_gain_floor`` only forwards
the already-seeded ``random_seed`` into the permutation-null kernel.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from ._screen_predictors_gate import _single_int

logger = logging.getLogger(__name__)


def cardinality_prescreen(
    factors_data: np.ndarray,
    factors_nbins: Sequence[int] | np.ndarray,
    factors_names: Sequence[str] | None,
    x: set[int] | list[int],
    y: Sequence[int] | int,
    verbose: int,
    raw_cardinality_cols: set[str] | None = None,
) -> tuple[set[int] | list[int], set[int]]:  # unused-ok: verbose: positional stage signature; refusals are logged at warning/info regardless
    """Drop high-cardinality columns before candidate enumeration.

    The Miller-Madow bias on plug-in MI is ~(|X|-1)*(|Y|-1)/(2n) nats; columns with
    ``nbins_x > 2*sqrt(n)`` are too cardinality-biased to score honestly (user_id-style
    high-level categoricals), so they are removed from the active factor index set ``x``.
    Returns ``(x, refused_set)`` where ``x`` matches the caller's container type (set or list)
    minus the refused indices.

    ``raw_cardinality_cols`` is the set of column NAMES whose bin count really is a raw level count. Only those
    are eligible for the ceiling. A numeric column discretised by a SUPERVISED strategy (MDLP, the MRMR
    default) earns more bins the better it explains the target, so judging it by bin count inverts this
    guard: measured on a 500-row fixture whose feature A drives the target on 80% of rows, MDLP gave A 50
    bins against 2-5 for the pure-noise columns, A tripped the 2*sqrt(500)=44.7 ceiling, and MRMR returned a
    noise column with ``fallback_used_=False`` -- a confident wrong answer. The plug-in bias those columns do
    carry is already handled downstream by the permutation-null debiasing and its significance gate, and by
    MDLP's own MDL stopping rule. ``None`` -- or any column this set cannot name -- keeps the legacy behaviour of
    judging by bin count. ``verbose`` no longer gates the refusal warning: dropping a column from selection is
    reported unconditionally.

    Caller gates on ``cardinality_bias_correction`` and ``factors_data.shape[1] > 0`` before
    invoking; this helper assumes both hold.
    """
    _n_for_screen = int(factors_data.shape[0])
    _y_idx_for_screen = _single_int(y)
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
        # Only skip a column when it is POSITIVELY known to be supervised-binned. Without names there is nothing
        # to classify by, so the legacy bin-count judgement stands rather than the guard quietly going inert.
        if raw_cardinality_cols is not None and factors_names is not None and _col_idx < len(factors_names):
            if factors_names[_col_idx] not in raw_cardinality_cols:
                continue
        if _nbins_x > _nbins_x_ceiling:
            _refused.append(_col_idx)
            _refused_set.add(_col_idx)
    # Unconditional: this silently removed the single most informative column from a production fit, and the
    # only trace was an INFO line behind verbose>=1 that the suite never enables.
    if _refused:
        _names = [factors_names[i] if factors_names is not None and i < len(factors_names) else f"col_{i}" for i in _refused]
        logger.warning(
            "screen_predictors: pre-screening dropped %d high-cardinality column(s) "
            "(nbins_x > 2*sqrt(n)=%.0f at n=%d): %s. Bin or target-encode before "
            "fitting if they carry real signal. Disable via "
            "cardinality_bias_correction=False.",
            len(_refused),
            _nbins_x_ceiling,
            _n_for_screen,
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
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    x: set[int] | list[int],
    y: Sequence[int] | int,
    *,
    screen_fdr_null_permutations: int | None,
    screen_fdr_null_quantile: float,
    screen_fdr_min_features: int,
    screen_fdr_target_oversplit_ratio: float,
    screen_fdr_min_rows_per_joint_cell: float,
    cardinality_bias_correction: bool,
    random_seed: int | None,
    verbose: int,
    # fix: optional cross-round cache, keyed on the candidate-pool identity + all inputs
    # that affect the result. ``pooled_permutation_null_gain_floor`` recomputes a full (n_permutations
    # x n_candidates x n_rows) histogram+MI pass on EVERY call; when the SAME raw-column pool recurs
    # across screen_predictors rounds (the common case - data only grows via appended engineered
    # columns, existing column content never changes), the earlier round's floor is mathematically
    # identical, not approximate, so a hit skips that whole pass. Threaded the same way as
    # screen_predictors' ``seed_caches``: the CALLER owns the dict's lifetime (must be scoped to one
    # fit, never shared across separate ``.fit()`` calls or concurrent fits). ``None`` (default)
    # disables caching entirely - legacy behaviour, always recompute.
    maxt_floor_cache: "dict | None" = None,
    honest_null_codes_fn=None,
    honest_full_factors=None,
    honest_eval_rows=None,
) -> float:
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
    from ._permutation_null import pooled_permutation_null_gain_floor

    _y_idx_fdr = _single_int(y)
    _wide_pool = len(_fdr_pool) >= int(screen_fdr_min_features)
    # Family-wise over every narrow pool of two or more candidates: a narrow pool with a binary or low-cardinality target lets the best
    # of its columns clear the per-column gain floors by chance (pure-noise suite).
    _narrow_oversplit = not _wide_pool and len(_fdr_pool) >= 2
    if not (_wide_pool or _narrow_oversplit):
        return 0.0
    if honest_null_codes_fn is not None:
        # Supervised binning: the null re-bins on every permuted target (see _honest_mdlp_null.honest_fdr_floor); the cached/pooled
        # permutation floor below would keep the in-sample cuts and cannot see their selection bias.
        from ._honest_mdlp_null import honest_fdr_floor

        # The honest null refits the binning on the whole permuted target; when the screen scores a row subsample the MI is evaluated on
        # those rows, matching how the observed codes were fitted (whole target) and scored (subsample).
        _hf = honest_full_factors if honest_full_factors is not None else factors_data
        _he = honest_eval_rows if honest_full_factors is not None else None
        return honest_fdr_floor(
            honest_null_codes_fn,
            _hf,
            factors_nbins,
            _y_idx_fdr,
            _fdr_pool,
            int(screen_fdr_null_permutations),
            float(screen_fdr_null_quantile),
            random_seed,
            _he,
        )

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
        _cached: float | None = maxt_floor_cache.get(_cache_key)
        if _cached is not None:
            if _cached > 0.0 and verbose >= 1:
                logger.info(
                    "screen_predictors: maxT permutation-null gain floor=%.5f over p=%d "
                    "candidates (q=%.2f, K=%d) - rejects chance-max noise at scale [cache hit].",
                    _cached,
                    len(_fdr_pool),
                    float(screen_fdr_null_quantile),
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
            "screen_predictors: maxT permutation-null gain floor=%.5f over p=%d " "candidates (q=%.2f, K=%d) - rejects chance-max noise at scale.",
            _fdr_gain_floor,
            len(_fdr_pool),
            float(screen_fdr_null_quantile),
            int(screen_fdr_null_permutations),
        )
    return _fdr_gain_floor
