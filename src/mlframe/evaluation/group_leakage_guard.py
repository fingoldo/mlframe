"""Runtime assertion that a CV split never places the same group's rows on both sides of a fold.

A classic silent leakage bug: a plain (non-grouped) KFold over a child/nested table where multiple rows share
a parent entity key lets the model learn per-entity artifacts (a Home-Credit writeup's example: repeated
``AMT_INSTALMENT`` values within a loan let the model memorize "this exact value -> this target" across the
fold boundary) -- CV looks great, LB doesn't move, because the "signal" never generalizes past the leaked
entity identity. This is a mandatory, cheap runtime check for any nested/child-table OOF featurizer: assert
that no group id appears in both the train and test index sets of any fold.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def assert_no_group_leakage(cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]], groups: np.ndarray) -> None:
    """Raise ``ValueError`` if any fold places the same group's rows in both train and test.

    Parameters
    ----------
    cv_splits
        Iterable of ``(train_idx, test_idx)`` index-array pairs (e.g. from a fitted splitter's ``.split()``).
        Consumed eagerly (materialized to a list) so a generator can still be checked without exhausting it
        for the caller.
    groups
        ``(n_samples,)`` group/entity id per row, aligned to the indices used in ``cv_splits``.

    Raises
    ------
    ValueError
        On the first fold (by iteration order) where a group id appears in both ``train_idx`` and
        ``test_idx``, naming the fold index, the offending group id(s) (up to 5), and the leaked row count.
    """
    groups = np.asarray(groups)
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        overlap = train_groups & test_groups
        if overlap:
            overlap_list = sorted(overlap, key=str)[:5]
            leaked_rows = int(np.isin(groups[test_idx], list(overlap)).sum())
            raise ValueError(
                f"assert_no_group_leakage: fold {fold_idx} places {len(overlap)} group id(s) on both sides of the "
                f"train/test split (e.g. {overlap_list}), leaking {leaked_rows} test row(s) from an entity already "
                f"seen in training -- use a group-aware splitter (GroupKFold / GroupTimeSeriesSplit) instead of a "
                f"plain (non-grouped) split."
            )


__all__ = ["assert_no_group_leakage"]
