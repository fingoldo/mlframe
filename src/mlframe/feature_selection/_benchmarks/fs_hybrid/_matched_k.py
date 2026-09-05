"""Matched-`K` evaluation: turn an arm into a feature *ranking*, then cut it at pre-declared cardinalities.

Comparing arms at whatever cardinality each one chose measures the stopping rule, not the ranking, and
mixes two effects the pre-registration wants separated. The primary outcome is therefore quality at
matched `K`, evaluated at one, two and five times the target-set size, with the arm's own self-chosen `K`
reported as its own separate row.

This also dissolves most of the `score_kind` incomparability problem: a `continuous` arm and a
`selection_order` arm both yield a prefix of length `K`, and prefixes are directly comparable even when
the objects that produced them are not.

The ranking source per `score_kind` (fields as declared by the shared `ArmResult` dataclass):

* `continuous` / `ordinal` -- `score`, descending, ties broken by original column order.
* `selection_order` -- `ranked_prefix`, which is already the greedy pick order; features outside it have
  no score at all, so a cut beyond its length is truncated and flagged via `coverage`.
* `none` -- no ranking exists. Matched-`K` rows are not produced; only the self-chosen-`K` row is, and
  the aggregate must report the arm as absent from matched-`K` tables rather than inventing a score
  (synthesising a 1/0 pseudo-score and ranking on it silently computes a different statistic per arm).
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "K_MULTIPLIERS",
    "SELF_CHOSEN_K",
    "Ranking",
    "arm_result_type",
    "ranking_from_arm_result",
    "matched_k_grid",
    "cut_at_k",
]

# One, two and five times the target-set size, exactly as pre-registered.
K_MULTIPLIERS: Tuple[int, ...] = (1, 2, 5)
# Sentinel label for the arm's own cardinality; reported separately, never mixed into the matched-K rows.
SELF_CHOSEN_K = "self"


def arm_result_type() -> Optional[type]:
    """Return the shared `ArmResult` dataclass, or `None` while its module is not yet importable.

    Imported dynamically rather than with a `from . import` statement so this module stays importable
    (and testable) independently of the sibling that owns `_arm_result.py`.
    """
    try:
        module = importlib.import_module(f"{__package__}._arm_result")
    except ImportError:
        return None
    return getattr(module, "ArmResult", None)


@dataclass(frozen=True)
class Ranking:
    """An arm's features in preference order, plus what produced it."""

    order: Tuple[str, ...]
    score_kind: str
    selected: Tuple[str, ...]
    # Fraction of the feature space the ranking actually covers; below 1.0 a deep cut is truncated.
    coverage: float

    def is_rankable(self) -> bool:
        """True when the arm supplies an order that can be cut at an arbitrary `K`."""
        return self.score_kind != "none" and len(self.order) > 0


def _names_from_support(support: Any, feature_names: Sequence[str]) -> List[str]:
    """Return the feature names flagged by a boolean support mask (or an index array)."""
    arr = np.asarray(support)
    if arr.dtype == bool:
        if arr.shape[0] != len(feature_names):
            raise ValueError(f"support length {arr.shape[0]} != n_features {len(feature_names)}")
        return [feature_names[i] for i in np.flatnonzero(arr)]
    return [feature_names[int(i)] for i in arr]


def ranking_from_arm_result(result: Any, feature_names: Sequence[str]) -> Ranking:
    """Build a `Ranking` from an `ArmResult` (or any object exposing the same attribute contract).

    Accepts a legacy fs_hybrid adapter too (one exposing only `raw_selected_`), which is treated as
    `score_kind="none"`: a selected set with no internal order.
    """
    names = list(feature_names)
    support = getattr(result, "support", None)
    if support is None:
        selected = [c for c in getattr(result, "raw_selected_", []) if c in names]
        return Ranking(order=(), score_kind="none", selected=tuple(selected), coverage=0.0)

    selected = _names_from_support(support, names)
    kind = str(getattr(result, "score_kind", "none"))
    order: List[str] = []

    if kind in ("continuous", "ordinal"):
        score = getattr(result, "score", None)
        if score is None:
            # A `continuous` arm without a score is fatal by pre-registration: it means the arm's own
            # score extraction silently failed, and ranking it on anything else compares two statistics.
            raise ValueError(f"score_kind={kind!r} but score is None -- refusing to invent a ranking")
        vals = np.asarray(score, dtype=float)
        if vals.shape[0] != len(names):
            raise ValueError(f"score length {vals.shape[0]} != n_features {len(names)}")
        finite = np.where(np.isfinite(vals), vals, -np.inf)
        # Stable sort on the negated score keeps original column order as the tie-break.
        order = [names[i] for i in np.argsort(-finite, kind="stable")]
    elif kind == "selection_order":
        prefix = getattr(result, "ranked_prefix", None)
        if prefix is None:
            raise ValueError("score_kind='selection_order' but ranked_prefix is None")
        order = [names[int(i)] for i in prefix]

    coverage = len(order) / len(names) if names else 0.0
    return Ranking(order=tuple(order), score_kind=kind, selected=tuple(selected), coverage=coverage)


# Absolute K grid for beds with no declared target set, fixed by docs/BENCHMARK_PREREGISTRATION.md section 3a.
# Deliberately a module constant and NOT a caller-tunable parameter: a tunable invites exactly the silent drift
# the pre-registration exists to prevent. Changing these values once results exist is a POST-HOC deviation and
# ships labelled as one.
ABSOLUTE_K_GRID = (5, 10, 20, 50, 100, 200)


def matched_k_grid(target_size: int, n_features: int) -> Dict[str, int]:
    """Return `{label: K}` for the pre-registered `1x/2x/5x` grid, capped at `n_features`."""
    if target_size <= 0:
        return {}
    grid: Dict[str, int] = {}
    for mult in K_MULTIPLIERS:
        k = min(int(target_size) * mult, int(n_features))
        grid[f"{mult}k"] = k
    return grid


def absolute_k_grid(n_features: int) -> Dict[str, int]:
    """Return `{label: K}` for the pre-registered absolute grid, dropping entries at or above `n_features`.

    A real dataset declares no target set, so the multiplier grid has no denominator there and the runner must
    not invent one. Entries are dropped rather than capped: two labels collapsing onto the same K would double
    count one comparison, and a K equal to `n_features` is the null hypothesis itself, not a selection.
    """
    grid: Dict[str, int] = {}
    for k in ABSOLUTE_K_GRID:
        if k < int(n_features):
            grid[f"k{k}"] = k
    return grid


def k_grid_for_bed(target_size: Optional[int], n_features: int) -> Tuple[Dict[str, int], str]:
    """Return `(grid, mode)` for one bed: the multiplier grid when a target set is declared, else the absolute one.

    `mode` is recorded on every cell so the synthetic and real legs can never be pooled by accident -- their K
    labels mean different things, and an aggregate mixing them would be averaging over two different questions.
    """
    if target_size is not None and int(target_size) > 0:
        return matched_k_grid(int(target_size), n_features), "multiplier"
    return absolute_k_grid(n_features), "absolute"


def cut_at_k(ranking: Ranking, k: int) -> Optional[List[str]]:
    """Return the top-`k` prefix of `ranking`, or `None` when the arm supplies no usable order.

    A prefix shorter than `k` (a `selection_order` arm that stopped early) is returned as-is; the caller
    records the realised length so the table never claims a cardinality the arm did not deliver.
    """
    if not ranking.is_rankable() or k <= 0:
        return None
    return list(ranking.order[:k])
