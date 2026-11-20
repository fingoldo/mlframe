"""Plan of the integer storage of the quantised codes: the narrowest dtype that holds every code the chosen binning can produce.

``quantization_dtype`` is a hint, not an order. What the codes need follows from the other settings: the bin count of the quantile / uniform binner, the interval ceiling of the supervised and adaptive strategies,
the cardinality cap of the categorical columns and the NaN code one past the last real one. The plan takes the largest of those bounds and returns the narrowest signed dtype that holds it for the discretiser to allocate
(a user dtype that is wider is narrowed, one that is too narrow is widened instead of failing), and the dtype the finished matrix is stored in (``MRMR.fit`` narrows it again by the range actually seen: unsigned byte for codes 0..255).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._code_dtype import narrowest_code_dtype

_MDLP_STRATEGIES = ("mdlp", "fayyad_irani", "mdlp_validated", "fayyad_irani_validated")
_SIGNED = (np.int8, np.int16, np.int32, np.int64)
DEFAULT_MDLP_MAX_DEPTH = 8
DEFAULT_MAX_ADAPTIVE_NBINS = 256


@dataclass(frozen=True)
class CodeStoragePlan:
    """The resolved code storage: ``alloc_dtype`` for the discretiser, ``storage_dtype`` for the finished matrix, ``bound`` the largest code any column can reach, and the reasons that fixed the bound."""

    alloc_dtype: type
    storage_dtype: type
    bound: int
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        """JSON-friendly view for the fit report."""
        return {
            "alloc_dtype": np.dtype(self.alloc_dtype).name,
            "storage_dtype": np.dtype(self.storage_dtype).name,
            "max_code": int(self.bound),
            "reasons": list(self.reasons),
        }


def _numeric_bound(method: str, n_bins: int, strategy: "str | None", strategy_kwargs: "Mapping[str, Any] | None", max_adaptive_nbins: int) -> tuple[int, str]:
    """The largest code a numeric column can take (the NaN bin is one past the last interval) and why."""
    kwargs = dict(strategy_kwargs or {})
    if strategy is None:
        return int(n_bins), f"{method} binner with {int(n_bins)} bins (+ NaN bin)"
    name = str(strategy).lower()
    if name in _MDLP_STRATEGIES:
        depth = int(kwargs.get("max_depth", DEFAULT_MDLP_MAX_DEPTH))
        return 2 ** max(1, depth), f"{name} depth {depth}: at most {2 ** max(1, depth)} intervals (+ NaN bin)"
    ceiling = int(kwargs.get("max_nbins", max_adaptive_nbins))
    return ceiling, f"{name}: at most {ceiling} bins (+ NaN bin)"


def plan_code_storage(
    *,
    n_bins: int,
    method: str = "quantile",
    nbins_strategy: "str | None" = None,
    nbins_strategy_kwargs: "Mapping[str, Any] | None" = None,
    max_adaptive_nbins: int = DEFAULT_MAX_ADAPTIVE_NBINS,
    max_categorical_cardinality: "int | None" = None,
    categorical_cardinalities: "list[int] | None" = None,
    user_dtype: "type | None" = None,
) -> CodeStoragePlan:
    """Resolve the code dtype from the binning settings; ``user_dtype`` is only widened when the bound needs it, or kept as the allocation dtype with ``MLFRAME_MRMR_COMPACT_CODES=0``.

    ``categorical_cardinalities`` are the distinct counts of the categorical columns when known; without them a cap alone bounds the categorical codes and no cap leaves them unbounded (the discretiser widens on the observed range)."""
    bound, why = _numeric_bound(method, int(n_bins), nbins_strategy, nbins_strategy_kwargs, int(max_adaptive_nbins))
    reasons = [why]
    cap = int(max_categorical_cardinality) if max_categorical_cardinality else 0
    if categorical_cardinalities:
        card = max(int(c) for c in categorical_cardinalities)
        cat_bound = min(card, cap) if cap else card  # the +1 NaN shift makes the top code equal to the number of distinct values
        reasons.append(f"categorical: {card} levels, cap {cap or 'none'} -> codes up to {cat_bound}")
        bound = max(bound, cat_bound)
    elif cap:
        reasons.append(f"categorical cap {cap} bounds the categorical codes")
        bound = max(bound, cap)
    storage = narrowest_code_dtype(0, bound) or np.int64
    signed = next((d for d in _SIGNED if bound <= np.iinfo(d).max), np.int64)
    if os.environ.get("MLFRAME_MRMR_COMPACT_CODES", "1").strip().lower() in ("0", "false", "off", "no"):
        keep = user_dtype if user_dtype is not None else signed
        if np.dtype(keep).kind in "iu" and bound > np.iinfo(keep).max:
            keep = signed
            reasons.append("compact storage off: the requested dtype is widened to hold the codes")
        return CodeStoragePlan(keep, keep, int(bound), tuple(reasons))
    if user_dtype is not None and np.dtype(user_dtype).kind in "iu":
        if np.iinfo(user_dtype).max < bound:
            reasons.append(f"requested {np.dtype(user_dtype).name} holds up to {np.iinfo(user_dtype).max}: widened to {np.dtype(signed).name}")
        elif np.dtype(user_dtype).itemsize > np.dtype(signed).itemsize:
            reasons.append(f"requested {np.dtype(user_dtype).name} is wider than the codes need: narrowed to {np.dtype(signed).name}")
    return CodeStoragePlan(signed, storage, int(bound), tuple(reasons))


def code_alloc_dtype(estimator: Any) -> Any:
    """The dtype the codes of ``estimator``'s current fit are allocated in: the plan's when the fit has one (``_code_alloc_dtype``), else the requested ``quantization_dtype``."""
    return getattr(estimator, "_code_alloc_dtype", None) or estimator.quantization_dtype
