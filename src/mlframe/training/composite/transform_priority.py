"""``recommend_transform_candidates``: prune the additive-residual transform for positive-scale pairs.

Source: 1st_optiver-realized-volatility-prediction.md -- "I trained my LightGBM model on a multiplicative
residual (target / book.log_return.realized_volatility). It gave a slight but real improvement over target
directly"; the main post notes plain additive residual on ``target - realized_volatility`` did NOT work at
all for this strictly-positive, scale-like target.

``CompositeTargetDiscovery``'s transform search is exhaustive and order-independent (every configured
transform in ``config.transforms`` is CV-scored and the best wins by MI-gain, regardless of trial order --
confirmed in ``discovery/_fit.py``), so "trying ratio before diff" changes nothing about which transform
ultimately wins. What DOES matter for a strictly-positive, scale-like target/base pair is COMPUTE: the
additive ``diff``/``linear_residual`` transforms are a wasted CV-scoring pass when the underlying quantity is
multiplicative by construction (volatility, volume, counts, prices) -- Optiver's own writeup found the
additive residual simply doesn't help there. This helper is a cheap, standalone pre-filter: given sample
target/base arrays, it recommends the subset of ``candidate_transforms`` worth evaluating, dropping the
additive-residual family when both are strictly positive.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_ADDITIVE_RESIDUAL_TRANSFORMS = frozenset({"diff", "linear_residual"})
_MULTIPLICATIVE_TRANSFORMS = frozenset({"ratio", "logratio"})

# Default CV(ratio) < threshold * CV(target) bar for the auto-detection probe below. Chosen loosely (ratio
# needs to at least halve the relative dispersion) rather than tightly -- a false positive here only costs an
# extra CV-scoring pass on "ratio" (cheap), while a false negative silently keeps the caller stuck on
# additive-only search for a genuinely multiplicative pair (the actual failure mode this closes).
_AUTO_DETECT_CV_REDUCTION_THRESHOLD: float = 0.5


def _is_multiplicative_regime(
    target_arr: np.ndarray, base_arr: np.ndarray, cv_reduction_threshold: float,
) -> bool:
    """Probe whether ``target``/``base`` look ratio-stationary rather than additively related.

    Strict positivity alone (as used by the pruning branch below) says the additive residual is *unlikely*
    to be the right shape, but says nothing about whether ``ratio`` actually helps -- a positive-but-additive
    pair (e.g. ``target = base + noise`` with both shifted positive) would pass the sign check yet gain
    nothing from dividing. This probe checks the actual effect of dividing: a multiplicative pair like
    Optiver's volatility ratio collapses the relative dispersion (CV = std/mean) far more by dividing by
    ``base`` than a merely-shifted-positive additive pair does, since ``base`` explains the *scale* of
    ``target`` rather than just an offset.
    """
    if not (np.all(target_arr > 0) and np.all(base_arr > 0)):
        return False
    mean_target = float(np.mean(target_arr))
    if mean_target <= 0:
        return False
    cv_target = float(np.std(target_arr)) / mean_target
    if cv_target <= 0:
        return False
    ratio = target_arr / base_arr
    mean_ratio = float(np.mean(ratio))
    if mean_ratio <= 0:
        return False
    cv_ratio = float(np.std(ratio)) / mean_ratio
    return bool(cv_ratio < cv_reduction_threshold * cv_target)


def recommend_transform_candidates(
    target: np.ndarray,
    base: np.ndarray,
    candidate_transforms: Sequence[str],
    auto_detect: bool = False,
    cv_reduction_threshold: float = _AUTO_DETECT_CV_REDUCTION_THRESHOLD,
) -> list:
    """Return the subset of ``candidate_transforms`` worth evaluating for this ``(target, base)`` pair.

    Parameters
    ----------
    target, base
        Sample arrays for the target and the candidate base column (a representative subsample is fine --
        this only checks sign and dispersion, not exact magnitude).
    candidate_transforms
        The transform names under consideration (e.g. ``config.transforms`` before it's handed to
        ``CompositeTargetDiscovery``).
    auto_detect
        Opt-in (default ``False``, behavior below is unchanged when omitted). When ``True``, runs a
        ratio-stationarity probe (:func:`_is_multiplicative_regime`) directly on ``(target, base)`` instead
        of requiring the caller to have already included ``"ratio"``/``"logratio"`` in
        ``candidate_transforms`` for the pruning to take effect. If the probe fires, ``"ratio"`` is ADDED to
        the recommendation (when not already present) before the additive-residual pruning is applied, so a
        genuinely multiplicative pair gets the multiplicative candidate even when the caller never asserted
        that regime.
    cv_reduction_threshold
        Only used when ``auto_detect=True``. The pair is judged multiplicative when
        ``CV(target / base) < cv_reduction_threshold * CV(target)``.

    Returns
    -------
    list
        ``candidate_transforms`` with additive-residual transforms (``"diff"``, ``"linear_residual"``)
        DROPPED when both ``target`` and ``base`` are strictly positive AND at least one multiplicative
        transform (``"ratio"``, ``"logratio"``) is present in the (possibly auto-detect-augmented) candidate
        set (never drops the only candidates available, and never touches any transform not in the
        additive-residual family).
    """
    target_arr = np.asarray(target)
    base_arr = np.asarray(base)
    working_candidates = list(candidate_transforms)

    if auto_detect and not _MULTIPLICATIVE_TRANSFORMS.intersection(working_candidates):
        if _is_multiplicative_regime(target_arr, base_arr, cv_reduction_threshold):
            working_candidates.append("ratio")

    both_strictly_positive = bool(np.all(target_arr > 0) and np.all(base_arr > 0))
    has_multiplicative_alternative = bool(_MULTIPLICATIVE_TRANSFORMS.intersection(working_candidates))

    if both_strictly_positive and has_multiplicative_alternative:
        return [t for t in working_candidates if t not in _ADDITIVE_RESIDUAL_TRANSFORMS]
    return working_candidates


__all__ = ["recommend_transform_candidates"]
