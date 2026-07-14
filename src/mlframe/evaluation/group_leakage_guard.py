"""Runtime assertion that a CV split never places the same group's rows on both sides of a fold.

A classic silent leakage bug: a plain (non-grouped) KFold over a child/nested table where multiple rows share
a parent entity key lets the model learn per-entity artifacts (a Home-Credit writeup's example: repeated
``AMT_INSTALMENT`` values within a loan let the model memorize "this exact value -> this target" across the
fold boundary) -- CV looks great, LB doesn't move, because the "signal" never generalizes past the leaked
entity identity. This is a mandatory, cheap runtime check for any nested/child-table OOF featurizer: assert
that no group id appears in both the train and test index sets of any fold.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def assert_no_group_leakage(
    cv_splits: Iterable[Tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
    *,
    near_duplicate_features: Optional[np.ndarray] = None,
    near_duplicate_metric: str = "euclidean",
    near_duplicate_threshold: Optional[float] = None,
    near_duplicate_max_neighbor_distance: Optional[float] = None,
) -> None:
    """Raise ``ValueError`` if any fold places the same group's rows in both train and test.

    Parameters
    ----------
    cv_splits
        Iterable of ``(train_idx, test_idx)`` index-array pairs (e.g. from a fitted splitter's ``.split()``).
        Consumed eagerly (materialized to a list) so a generator can still be checked without exhausting it
        for the caller.
    groups
        ``(n_samples,)`` group/entity id per row, aligned to the indices used in ``cv_splits``.
    near_duplicate_features
        Opt-in extension. ``(n_samples, n_features)`` numeric feature matrix, aligned to ``groups``/the indices
        used in ``cv_splits``. When given, an additional check flags folds where a test row has a near-identical
        neighbor (by ``near_duplicate_metric``) in the train split -- the case where no explicit group id column
        exists but rows still share a parent entity implicitly (e.g. duplicated/near-duplicated records, or
        multiple snapshots of the same underlying entity re-scaled/re-encoded differently). Left ``None``
        (default), this whole code path is skipped and behavior is bit-identical to the pre-extension function.
    near_duplicate_metric
        Distance metric passed to ``sklearn.neighbors.NearestNeighbors`` -- ``"euclidean"`` (default) or
        ``"cosine"``. Only used when ``near_duplicate_features`` is given.
    near_duplicate_threshold
        Cosine-similarity threshold above which a test row is flagged (only used when
        ``near_duplicate_metric="cosine"``). Required (no default) when using the cosine metric, so callers must
        make a deliberate choice rather than inherit a silently-wrong cutoff.
    near_duplicate_max_neighbor_distance
        Euclidean-distance threshold below which a test row is flagged (only used when
        ``near_duplicate_metric="euclidean"``). Required (no default) for the same reason as above.

    Raises
    ------
    ValueError
        On the first fold (by iteration order) where a group id appears in both ``train_idx`` and
        ``test_idx``, naming the fold index, the offending group id(s) (up to 5), and the leaked row count.
        If ``near_duplicate_features`` is given and no exact group overlap is found, raised instead on the
        first fold where a test row has a near-duplicate neighbor in the train split, naming the fold index,
        the count of flagged test rows, and the closest observed distance/similarity.
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

        if near_duplicate_features is not None:
            _assert_no_near_duplicate_leakage(
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                features=near_duplicate_features,
                metric=near_duplicate_metric,
                cosine_similarity_threshold=near_duplicate_threshold,
                euclidean_max_distance=near_duplicate_max_neighbor_distance,
            )


def _assert_no_near_duplicate_leakage(
    *,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: np.ndarray,
    metric: str,
    cosine_similarity_threshold: Optional[float],
    euclidean_max_distance: Optional[float],
) -> None:
    """Flag test rows whose nearest train-split neighbor is a near-duplicate (implicit, un-labeled group leak).

    Uses ``sklearn.neighbors.NearestNeighbors`` (a ball-tree/kd-tree under the hood for euclidean, brute-force
    for cosine) rather than a naive O(n_train * n_test) pairwise matrix -- keeps the check tractable at the row
    counts a real nested-table featurizer runs over.
    """
    from sklearn.neighbors import NearestNeighbors

    features = np.asarray(features)
    train_x = features[train_idx]
    test_x = features[test_idx]
    if len(train_x) == 0 or len(test_x) == 0:
        return

    if metric == "cosine":
        if cosine_similarity_threshold is None:
            raise ValueError("assert_no_group_leakage: near_duplicate_threshold is required when near_duplicate_metric='cosine'.")
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(train_x)
        distances, _neighbor_pos = nn.kneighbors(test_x)
        similarities = 1.0 - distances[:, 0]
        flagged = similarities >= cosine_similarity_threshold
        best_score = float(similarities.max()) if len(similarities) else float("-inf")
        score_name, score_op = "cosine similarity", ">="
    elif metric == "euclidean":
        if euclidean_max_distance is None:
            raise ValueError("assert_no_group_leakage: near_duplicate_max_neighbor_distance is required when " "near_duplicate_metric='euclidean'.")
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(train_x)
        distances, _neighbor_pos = nn.kneighbors(test_x)
        flagged = distances[:, 0] <= euclidean_max_distance
        best_score = float(distances[:, 0].min()) if len(distances) else float("inf")
        score_name, score_op = "euclidean distance", "<="
    else:
        raise ValueError(f"assert_no_group_leakage: unsupported near_duplicate_metric {metric!r} (use 'euclidean' or 'cosine').")

    if flagged.any():
        n_flagged = int(flagged.sum())
        raise ValueError(
            f"assert_no_group_leakage: fold {fold_idx} places {n_flagged} test row(s) within a near-duplicate "
            f"feature neighborhood of a train row (best observed {score_name} {score_op} threshold: {best_score:.6g}) "
            f"-- rows likely share an implicit parent entity with no explicit group id column; deduplicate or add "
            f"an explicit group key and use a group-aware splitter instead of a plain (non-grouped) split."
        )


__all__ = ["assert_no_group_leakage"]
