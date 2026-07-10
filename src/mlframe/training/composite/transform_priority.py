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


def recommend_transform_candidates(
    target: np.ndarray,
    base: np.ndarray,
    candidate_transforms: Sequence[str],
) -> list:
    """Return the subset of ``candidate_transforms`` worth evaluating for this ``(target, base)`` pair.

    Parameters
    ----------
    target, base
        Sample arrays for the target and the candidate base column (a representative subsample is fine --
        this only checks sign, not magnitude).
    candidate_transforms
        The transform names under consideration (e.g. ``config.transforms`` before it's handed to
        ``CompositeTargetDiscovery``).

    Returns
    -------
    list
        ``candidate_transforms`` with additive-residual transforms (``"diff"``, ``"linear_residual"``)
        DROPPED when both ``target`` and ``base`` are strictly positive AND at least one multiplicative
        transform (``"ratio"``, ``"logratio"``) is present in ``candidate_transforms`` (never drops the only
        candidates available, and never touches any transform not in the additive-residual family).
    """
    target_arr = np.asarray(target)
    base_arr = np.asarray(base)
    both_strictly_positive = bool(np.all(target_arr > 0) and np.all(base_arr > 0))
    has_multiplicative_alternative = bool(_MULTIPLICATIVE_TRANSFORMS.intersection(candidate_transforms))

    if both_strictly_positive and has_multiplicative_alternative:
        return [t for t in candidate_transforms if t not in _ADDITIVE_RESIDUAL_TRANSFORMS]
    return list(candidate_transforms)


__all__ = ["recommend_transform_candidates"]
