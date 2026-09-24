"""Unit- and provenance-tagged spec scores, and the one ranking helper every spec sort goes through.

Several ranking defects came from ordering numbers that were not comparable: RMSE fractions against MI nats in the
cross-target budget, honest-holdout RMSE against in-sample CV RMSE in the rerank, a multi-base spec carrying its seed's MI.
A ``Score`` names its unit, the rows it was measured on, the estimator and the transform it was measured for, and
``rank_specs`` refuses a ranking whose scores disagree on any of them.
"""

from __future__ import annotations

import math
from typing import Any, Callable, NamedTuple, Optional, Sequence


class Score(NamedTuple):
    """One spec's ranking score and what it is a score of."""

    value: float
    unit: str  # e.g. "y_rmse", "rmse_frac", "mi_nats", "waic_nats", "frequency"
    split: str  # the rows it was measured on, e.g. "screen", "cv", "honest_holdout"
    estimator: str  # what produced it, e.g. "bin_mi", "tiny_cv", "waic"
    measured_for: Optional[str] = None  # the transform it was measured for; None when the item carries no transform


class MixedScoresError(ValueError):
    """A ranking was asked to order scores of different units, splits or estimators, or of another transform."""


def _transform_of(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        return item.get("transform_name")
    return getattr(item, "transform_name", None)


def _name_of(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("name", ""))
    return str(getattr(item, "name", ""))


def rank_specs(
    items: Sequence[Any],
    key: Callable[[Any], Score],
    *,
    descending: bool = False,
    tiebreak: Optional[str] = "name",
    name: Optional[Callable[[Any], str]] = None,
    transform: Optional[Callable[[Any], Optional[str]]] = None,
) -> list:
    """``items`` ordered by ``key(item).value`` (ascending unless ``descending``); non-finite scores go last.

    Ties are broken by name (``tiebreak="name"``) or keep their input order (``tiebreak=None``). Raises
    ``MixedScoresError`` when the scores differ in unit, split or estimator, or when a score's ``measured_for`` is not the
    item's transform. ``name`` / ``transform`` read an item's name and transform when it is not a spec, a spec dict or a
    name string (e.g. an index into a spec list).
    """
    items = list(items)
    scores = [key(it) for it in items]
    kinds = {(s.unit, s.split, s.estimator) for s in scores}
    if len(kinds) > 1:
        raise MixedScoresError(f"cannot rank scores of different kinds together: {sorted(kinds)}")
    get_tf = transform or _transform_of
    for it, s in zip(items, scores):
        tf = get_tf(it)
        if s.measured_for is not None and tf is not None and s.measured_for != tf:
            raise MixedScoresError(f"{_name_of(it) or it!r}: score measured for {s.measured_for!r}, not its transform {tf!r}")
    get_name = name or _name_of
    sign = -1.0 if descending else 1.0

    def order(pair):
        it, s = pair
        v = float(s.value)
        finite = math.isfinite(v)
        k = (not finite, sign * v if finite else 0.0)
        return k + ((get_name(it),) if tiebreak == "name" else ())

    return [it for it, _ in sorted(zip(items, scores), key=order)]


__all__ = ["MixedScoresError", "Score", "rank_specs"]
