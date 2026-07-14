"""``concat_categorical_group``: build a composite categorical column from multiple raw categorical columns.

Source: 2nd_porto-seguro-safe-driver-prediction.md -- building ``new_ind``/``new_reg``/``new_car`` by
string-concatenating groups of raw categorical columns, then computing frequency/count encodings of both raw
and composite categoricals. mlframe already has full frequency/count encoding coverage
(``feature_selection.filters._count_freq_interaction_fe.frequency_encode_fit``/``count_encode_fit``) -- the
genuinely missing piece is the CONCATENATOR itself, a standalone precursor step (feed its output into the
existing frequency encoder) rather than the vectorized ``.str.cat`` pattern currently buried inline inside
``two_step_target_encode.py``.

Extension: ``discover_categorical_groups``/``auto_concat_categorical_groups`` -- the base concatenator (and
the sibling ``categorical_powerset_concat``, which enumerates every subset of a caller-chosen key set) both
require the caller to already know which columns belong together. Neither proposes the GROUPING itself. This
extension greedily partitions a whole column pool into groups by growing each group while the composite's
mutual information with the target keeps improving over the best MI seen so far for that group (seeded at the
lone column's own marginal MI) -- recovering the same "which specific columns must be combined" decision the
porto-seguro entry made by hand, for pools too large to hand-pick.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def concat_categorical_group(df: pd.DataFrame, columns: Sequence[str], separator: str = "_", feature_name: str = "concat_group") -> pd.DataFrame:
    """Append a composite categorical column built by string-concatenating ``columns``.

    Parameters
    ----------
    df
        Source frame.
    columns
        Categorical columns to concatenate, in order.
    separator
        Joiner between each column's value.
    feature_name
        Name for the appended composite column.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) plus one new ``object`` (string) column: each row's values from ``columns``
        joined by ``separator``. Uses vectorized ``Series.str.cat`` (measured ~13x faster than
        ``df[columns].agg(separator.join, axis=1)`` at scale, per the existing pattern this reuses from
        ``two_step_target_encode.py``).
    """
    if len(columns) < 2:
        raise ValueError("concat_categorical_group: need at least 2 columns to concatenate")

    first, *rest = columns
    composite = df[first].astype(str).str.cat([df[c].astype(str) for c in rest], sep=separator)

    out = df.copy(deep=False)
    out[feature_name] = composite
    return out


def _composite_mi_with_target(values: pd.Series, y: np.ndarray, random_state: int) -> float:
    """Mutual information between a (possibly composite) categorical column and a discrete target.

    Uses integer codes rather than raw strings -- ``mutual_info_classif`` needs a numeric feature matrix, and
    ``pd.factorize`` codes preserve the exact partition of rows into levels that MI actually depends on (the
    numeric code values themselves carry no order the estimator relies on, since ``discrete_features=True``).
    """
    codes, _ = pd.factorize(values)
    mi = mutual_info_classif(codes.reshape(-1, 1), y, discrete_features=True, random_state=random_state)
    return float(mi[0])


def discover_categorical_groups(
    df: pd.DataFrame,
    columns: Sequence[str],
    y: np.ndarray,
    separator: str = "_",
    min_mi_gain: float = 0.0,
    max_group_size: Optional[int] = None,
    random_state: int = 0,
) -> List[List[str]]:
    """Greedily partition ``columns`` into groups by concatenated-composite MI with ``y``.

    For each still-unassigned column (processed as the group seed, in ``columns`` order), repeatedly adds
    whichever remaining candidate column maximizes the growing group's concatenated MI with ``y``, stopping a
    group as soon as no candidate improves on the best MI seen so far for that group by more than
    ``min_mi_gain`` (the seed's own marginal MI is the group's starting best-MI, so a column that gains
    nothing from any partner is emitted as its own singleton group). This mirrors, automatically, the manual
    "which raw columns combine into one meaningful composite" judgment call
    ``2nd_porto-seguro-safe-driver-prediction.md`` made by hand for a handful of columns.

    Parameters
    ----------
    df
        Source frame.
    columns
        Pool of categorical columns to partition into groups. Every column ends up in exactly one group.
    y
        Discrete target (e.g. binary/multiclass labels) driving the MI search.
    separator
        Joiner used when building trial composites, forwarded to :func:`concat_categorical_group`.
    min_mi_gain
        Minimum MI improvement over a group's current best MI required to accept another column into that
        group. ``0.0`` (default) accepts any strictly-positive improvement; raise it to require a more
        decisive joint-signal gain before growing a group (guards against accepting noise-level MI wobble).
    max_group_size
        Cap on how many columns a single group may absorb. ``None`` (default) allows a group to absorb the
        whole remaining pool. Bounds the O(k^2) worst-case number of trial-composite MI evaluations for a
        pool of size k.
    random_state
        Forwarded to ``mutual_info_classif`` (its k-NN based estimator is stochastic).

    Returns
    -------
    list of list of str
        One entry per discovered group, each a list of original column names (order of insertion: seed
        first, then accepted partners). Singleton groups (no informative partner found) are included too.
    """
    if len(columns) < 1:
        raise ValueError("discover_categorical_groups: need at least 1 column to partition")

    y_arr = np.asarray(y)
    cap = max_group_size if max_group_size is not None else len(columns)
    if cap < 1:
        raise ValueError("discover_categorical_groups: max_group_size must be >= 1")

    remaining = list(columns)
    groups: List[List[str]] = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        best_mi = _composite_mi_with_target(df[seed], y_arr, random_state)
        candidates = list(remaining)

        while candidates and len(group) < cap:
            trial_mis = {}
            for cand in candidates:
                trial_cols = [*group, cand]
                composite = df[trial_cols[0]].astype(str).str.cat([df[c].astype(str) for c in trial_cols[1:]], sep=separator)
                trial_mis[cand] = _composite_mi_with_target(composite, y_arr, random_state)
            best_cand = max(trial_mis, key=lambda c: trial_mis[c])
            if trial_mis[best_cand] > best_mi + min_mi_gain:
                group.append(best_cand)
                best_mi = trial_mis[best_cand]
                candidates.remove(best_cand)
                remaining.remove(best_cand)
            else:
                break

        groups.append(group)

    return groups


def auto_concat_categorical_groups(
    df: pd.DataFrame,
    columns: Sequence[str],
    y: np.ndarray,
    separator: str = "_",
    min_mi_gain: float = 0.0,
    max_group_size: Optional[int] = None,
    feature_prefix: str = "concat_group",
    random_state: int = 0,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """Discover column groupings via :func:`discover_categorical_groups`, then materialize their composites.

    Opt-in convenience wrapper: appends one new composite column per discovered group of size >= 2 (singleton
    groups already exist as their own raw column, so nothing is appended for them), reusing
    :func:`concat_categorical_group` per group rather than reimplementing the string-join.

    Parameters
    ----------
    df, columns, y, separator, min_mi_gain, max_group_size, random_state
        Forwarded to :func:`discover_categorical_groups`.
    feature_prefix
        Prefix for each appended composite column's name (full name is ``f"{feature_prefix}__{'_'.join(group)}"``).

    Returns
    -------
    tuple of (pd.DataFrame, list of list of str)
        ``df`` (shallow copy) plus one new composite column per discovered group of size >= 2, and the
        discovered groups themselves (including singletons) for caller inspection/logging.
    """
    groups = discover_categorical_groups(df, columns, y, separator=separator, min_mi_gain=min_mi_gain, max_group_size=max_group_size, random_state=random_state)

    out = df.copy(deep=False)
    for group in groups:
        if len(group) >= 2:
            feature_name = f"{feature_prefix}__{separator.join(group)}"
            out = concat_categorical_group(out, columns=group, separator=separator, feature_name=feature_name)

    return out, groups


__all__ = ["concat_categorical_group", "discover_categorical_groups", "auto_concat_categorical_groups"]
