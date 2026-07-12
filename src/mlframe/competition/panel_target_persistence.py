"""Panel-data lag/lead target imputation for weak-identifier grouping.

COMPETITION/EXPLORATORY USE ONLY -- NOT FOR PRODUCTION.

Kaggle competitions occasionally anonymize a true panel structure (e.g. repeat
customers/policies) behind noisy pseudo-identifier columns. Grouping rows by a
candidate weak identifier and sorting within-group by a plausible ordering key
(row index, a timestamp-like feature, etc.) can reveal that the target is highly
persistent/autocorrelated across the ordered sequence within each group -- a strong
signal the grouping key really does recover panel structure.

``check_target_persistence`` is a legitimate diagnostic: it only measures whether a
candidate grouping+ordering reconstructs a persistent panel, which is informative
about data structure in general.

``lag_target_within_group``/``lead_target_within_group``, by contrast, build
lag(target)/lead(target)-within-group FEATURES straight from the observed target.
These are explicitly leak-prone: ``lead_target_within_group`` uses future rows'
labels directly (look-ahead leakage) and even ``lag_target_within_group`` leaks the
current row's label into neighboring rows' features whenever the grouping key is
only a proxy identifier shared with the evaluation set. They must only ever be used
inside a proper out-of-fold (OOF) scheme (compute lag/lead target purely from other
folds' rows) and never wired into a production feature pipeline -- see
``mlframe.competition`` package docstring and ``MLFRAME_IDEAS_competitions.md``
("Panel-data lag/lead target imputation for weak-identifier grouping").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "TargetPersistenceResult",
    "check_target_persistence",
    "lag_target_within_group",
    "lead_target_within_group",
]


@dataclass
class TargetPersistenceResult:
    """Per-group and aggregate target-persistence diagnostics.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring.

    Attributes:
        flip_rate: fraction of within-group consecutive pairs (ordered by ``order``)
            where the target changes value (only meaningful for a binary/discrete
            target). Lower means more persistent.
        lag1_autocorrelation: Pearson correlation between ``y[i]`` and ``y[i-1]``
            across all within-group consecutive pairs, pooled over groups. Higher
            (closer to 1) means more persistent.
        n_pairs: number of within-group consecutive (i-1, i) pairs used.
        n_groups_with_pairs: number of groups that contributed at least one pair
            (groups of size 1 contribute none).
        is_persistent: convenience verdict, ``True`` when persistence looks strong
            enough that leak-prone lag/lead features are likely to carry real signal
            (``lag1_autocorrelation >= persistence_threshold`` and
            ``flip_rate <= flip_rate_threshold``).
    """

    flip_rate: float
    lag1_autocorrelation: float
    n_pairs: int
    n_groups_with_pairs: int
    is_persistent: bool
    per_group_flip_rate: dict = field(default_factory=dict)


def _group_sort_permutation(group_ids: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Return a permutation that sorts rows by group (ascending) then ``order`` within group.

    A single ``np.lexsort`` (O(n log n), stable w.r.t. original row order on ties) replaces
    looping ``np.where(group_ids == gid)`` per group, which is O(n_groups * n) and was the
    dominant cost at realistic panel sizes (cProfile-confirmed hotspot).
    """
    return np.asarray(np.lexsort((order, group_ids)))


def check_target_persistence(
    group_ids: np.ndarray,
    order: np.ndarray,
    y: np.ndarray,
    *,
    persistence_threshold: float = 0.5,
    flip_rate_threshold: float = 0.3,
) -> TargetPersistenceResult:
    """Diagnose whether a candidate grouping+ordering reconstructs a persistent panel.

    COMPETITION/EXPLORATORY USE ONLY -- see module docstring. This function itself
    only reads the target to MEASURE persistence; it does not leak the target into
    any feature. It is safe to run before deciding whether to build (leak-prone)
    lag/lead features at all.

    Args:
        group_ids: 1-D array of candidate group/panel identifiers, one per row.
        order: 1-D array of a within-group ordering key (e.g. row index, timestamp),
            one per row. Ties are broken by original array order (stable sort).
        y: 1-D array of the target, one per row (binary or continuous).
        persistence_threshold: minimum lag-1 autocorrelation to call the panel
            "persistent".
        flip_rate_threshold: maximum flip rate to call the panel "persistent"
            (only informative for near-binary targets; ignored for continuous
            targets beyond being reported).

    Returns:
        ``TargetPersistenceResult``. If fewer than 1 within-group consecutive pair
        exists at all (e.g. every group has size 1), ``flip_rate`` and
        ``lag1_autocorrelation`` are ``nan`` and ``is_persistent`` is ``False``.
    """
    group_ids = np.asarray(group_ids)
    order = np.asarray(order)
    y = np.asarray(y, dtype=float)
    n = group_ids.shape[0]
    if not (order.shape[0] == n and y.shape[0] == n):
        raise ValueError("group_ids, order, and y must all have the same length")

    if n < 2:
        return TargetPersistenceResult(
            flip_rate=float("nan"),
            lag1_autocorrelation=float("nan"),
            n_pairs=0,
            n_groups_with_pairs=0,
            is_persistent=False,
            per_group_flip_rate={},
        )

    perm = _group_sort_permutation(group_ids, order)
    sorted_group = group_ids[perm]
    sorted_y = y[perm]

    same_group = sorted_group[1:] == sorted_group[:-1]
    prev_all = sorted_y[:-1][same_group]
    cur_all = sorted_y[1:][same_group]

    if prev_all.size == 0:
        return TargetPersistenceResult(
            flip_rate=float("nan"),
            lag1_autocorrelation=float("nan"),
            n_pairs=0,
            n_groups_with_pairs=0,
            is_persistent=False,
            per_group_flip_rate={},
        )

    # Per-group flip-rate breakdown, computed off the same sorted array via group boundaries
    # rather than another O(n_groups * n) scan.
    uniq_groups, group_start = np.unique(sorted_group, return_index=True)
    group_end = np.concatenate((group_start[1:], [n]))
    per_group_flip_rate: dict = {}
    n_groups_with_pairs = 0
    for gid, start, end in zip(uniq_groups, group_start, group_end):
        if end - start < 2:
            continue
        seq = sorted_y[start:end]
        per_group_flip_rate[gid] = float(np.mean(seq[:-1] != seq[1:]))
        n_groups_with_pairs += 1

    n_pairs = int(prev_all.size)
    flip_rate = float(np.mean(prev_all != cur_all))

    if np.std(prev_all) == 0 or np.std(cur_all) == 0:
        # Degenerate: no variance in one of the two sequences means correlation is
        # undefined; a target that never changes at all is maximally persistent.
        lag1_autocorr = 1.0 if flip_rate == 0.0 else 0.0
    else:
        lag1_autocorr = float(np.corrcoef(prev_all, cur_all)[0, 1])

    is_persistent = bool(lag1_autocorr >= persistence_threshold and flip_rate <= flip_rate_threshold)

    return TargetPersistenceResult(
        flip_rate=flip_rate,
        lag1_autocorrelation=lag1_autocorr,
        n_pairs=n_pairs,
        n_groups_with_pairs=n_groups_with_pairs,
        is_persistent=is_persistent,
        per_group_flip_rate=per_group_flip_rate,
    )


def lag_target_within_group(group_ids: np.ndarray, order: np.ndarray, y: np.ndarray, *, periods: int = 1) -> np.ndarray:
    """Compute a leak-prone lag(target)-within-group feature.

    COMPETITION/EXPLORATORY USE ONLY -- NOT FOR PRODUCTION, OOF-ONLY.

    For each row, returns the target value of the row ``periods`` steps EARLIER in
    the same group's ``order``-sorted sequence (``nan`` where no such row exists).
    This directly leaks other rows' observed labels into a feature: only ever
    compute this from a fold's OWN held-out target column when the labels of other
    folds are legitimately known (proper OOF construction), never on the full
    train+test target column at once, and never in a production pipeline.

    Args:
        group_ids: 1-D array of group identifiers, one per row.
        order: 1-D array of within-group ordering key, one per row.
        y: 1-D array of the (leak-source) target, one per row.
        periods: how many ordered steps back to look (must be >= 1).

    Returns:
        1-D float array, same length and original row order as the inputs, with
        ``nan`` for rows that have no ``periods``-th predecessor in their group.
    """
    return _shift_target_within_group(group_ids, order, y, periods=periods)


def lead_target_within_group(group_ids: np.ndarray, order: np.ndarray, y: np.ndarray, *, periods: int = 1) -> np.ndarray:
    """Compute a leak-prone lead(target)-within-group feature.

    COMPETITION/EXPLORATORY USE ONLY -- NOT FOR PRODUCTION, OOF-ONLY.

    For each row, returns the target value of the row ``periods`` steps LATER in
    the same group's ``order``-sorted sequence (``nan`` where no such row exists).
    This is look-ahead leakage by construction (it literally reads a future row's
    label): it must never be used for any real inference, only inside an OOF
    diagnostic/feature-importance exploration where the leak is explicitly
    acknowledged and the resulting model is never deployed.

    Args:
        group_ids: 1-D array of group identifiers, one per row.
        order: 1-D array of within-group ordering key, one per row.
        y: 1-D array of the (leak-source) target, one per row.
        periods: how many ordered steps ahead to look (must be >= 1).

    Returns:
        1-D float array, same length and original row order as the inputs, with
        ``nan`` for rows that have no ``periods``-th successor in their group.
    """
    return _shift_target_within_group(group_ids, order, y, periods=-periods)


def _shift_target_within_group(group_ids: np.ndarray, order: np.ndarray, y: np.ndarray, *, periods: int) -> np.ndarray:
    if periods == 0:
        raise ValueError("periods must be non-zero")
    group_ids = np.asarray(group_ids)
    order = np.asarray(order)
    y = np.asarray(y, dtype=float)
    n = group_ids.shape[0]
    if not (order.shape[0] == n and y.shape[0] == n):
        raise ValueError("group_ids, order, and y must all have the same length")

    if n == 0:
        return np.full(0, np.nan, dtype=float)

    perm = _group_sort_permutation(group_ids, order)
    sorted_group = group_ids[perm]
    sorted_y = y[perm]

    k = abs(periods)
    shifted_sorted = np.full(n, np.nan, dtype=float)
    if k < n:
        if periods > 0:
            candidate_pos = np.arange(k, n)
            same_group = sorted_group[candidate_pos] == sorted_group[candidate_pos - k]
            valid_pos = candidate_pos[same_group]
            shifted_sorted[valid_pos] = sorted_y[valid_pos - k]
        else:
            candidate_pos = np.arange(0, n - k)
            same_group = sorted_group[candidate_pos] == sorted_group[candidate_pos + k]
            valid_pos = candidate_pos[same_group]
            shifted_sorted[valid_pos] = sorted_y[valid_pos + k]

    out = np.empty(n, dtype=float)
    out[perm] = shifted_sorted
    return out
