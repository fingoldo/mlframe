"""Index-level split helpers for make_train_test_split.

Shape-aware stratified splitting and leakage-free calibration carving, lifted
from splitting.py so the public splitter stays under the LOC ceiling.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _stratified_split(
    indices: np.ndarray,
    test_size: float,
    stratify_y: np.ndarray,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shape-aware stratified split for the no-timestamps row-based path.

    1-D ``stratify_y``: sklearn ``StratifiedShuffleSplit``.
    2-D ``stratify_y`` (multilabel): ``iterstrat.MultilabelStratifiedShuffleSplit``
    (lazy-imported, raises with helpful message on missing optional dep).

    Parameters
    ----------
    indices : np.ndarray (M,)
        The row indices to split (e.g. arange(N) for the first call,
        train_idx slice for the second).
    test_size : float
        Fraction of ``indices`` for the right-hand split (val or test).
    stratify_y : np.ndarray (M,) or (M, K)
        Labels for stratification, ALIGNED to ``indices`` (caller must
        slice if indices is not the full range).
    random_state : int | None

    Returns
    -------
    (left_idx, right_idx) -- both are slices of ``indices``, not 0..M-1.
    """
    if stratify_y.ndim == 1:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        )
    elif stratify_y.ndim == 2:
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        except ImportError as e:
            raise ImportError(
                "Multilabel stratification requires the optional dependency "
                "'iterative-stratification'. Install via: "
                "pip install iterative-stratification\n"
                f"Original error: {e}"
            ) from e
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state,
        )
    else:
        raise ValueError(
            f"stratify_y must be 1-D or 2-D, got shape {stratify_y.shape}"
        )

    left_pos, right_pos = next(splitter.split(np.zeros(len(indices)), stratify_y))
    return indices[left_pos], indices[right_pos]


def _use_multilabel_3way(groups, stratify_y, test_size: float, val_size: float) -> bool:
    """True when the single-pass multilabel 3-way carve applies (2-D strat, no groups, both carves)."""
    return (
        groups is None
        and stratify_y is not None
        and getattr(stratify_y, "ndim", 0) == 2
        and test_size > 0
        and val_size > 0
    )


def _stratified_split_3way(
    indices: np.ndarray,
    test_size: float,
    val_size: float,
    stratify_y: np.ndarray,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass 3-way stratified partition (train / val / test).

    Replaces the two-call ``MultilabelStratifiedShuffleSplit`` pattern (test carve
    over all N, then val carve over the remainder) -- each of which re-ran the full
    O(n*K*iters) greedy from scratch -- with ONE greedy pass over all N into three
    folds. Uses ``iterstrat.IterativeStratification`` directly (the same kernel the
    shuffle-splitter wraps), so the split stays a valid, leakage-free, label-balanced
    multilabel split and is deterministic for a fixed seed.

    Fractions follow the suite convention: ``test_size`` and ``val_size`` are both
    fractions of the WHOLE ``indices`` (NOT of the post-test remainder), matching the
    cumulative semantics the two-call path produced (test over N, then val over the
    train remainder == val_size*N rows). Train gets ``1 - test_size - val_size``.

    1-D ``stratify_y`` falls back to the exact two-call ``_stratified_split`` path
    (single-label sklearn carve is already cheap; no greedy re-run to fold).

    Returns ``(train_idx, val_idx, test_idx)`` -- all slices of ``indices``.
    """
    if stratify_y.ndim == 1:
        train_idx, test_idx = _stratified_split(
            indices, test_size=test_size, stratify_y=stratify_y, random_state=random_state,
        )
        # val_size is a fraction of the whole; rescale to the train remainder.
        rem = max(1.0 - test_size, 1e-12)
        strat_train = stratify_y[_positions(indices, train_idx)]
        train_idx, val_idx = _stratified_split(
            train_idx, test_size=min(val_size / rem, 1.0 - 1e-9),
            stratify_y=strat_train, random_state=random_state,
        )
        return train_idx, val_idx, test_idx

    if stratify_y.ndim != 2:
        raise ValueError(f"stratify_y must be 1-D or 2-D, got shape {stratify_y.shape}")

    try:
        from iterstrat.ml_stratifiers import IterativeStratification
    except ImportError as e:
        raise ImportError(
            "Multilabel stratification requires the optional dependency "
            "'iterative-stratification'. Install via: "
            "pip install iterative-stratification\n"
            f"Original error: {e}"
        ) from e
    from sklearn.utils import check_random_state

    train_frac = max(1.0 - test_size - val_size, 0.0)
    # Fold order [train, val, test]; r normalised so the greedy desired-counts match.
    r = np.array([train_frac, val_size, test_size], dtype=float)
    r = r / r.sum()

    rng = check_random_state(random_state)
    y = np.asarray(stratify_y, dtype=bool)
    n = y.shape[0]
    perm = np.arange(n)
    rng.shuffle(perm)
    folds = IterativeStratification(labels=y[perm], r=r, random_state=rng)
    # Map fold labels back to original positions in ``indices``.
    fold_of_pos = folds[np.argsort(perm)]
    train_idx = indices[fold_of_pos == 0]
    val_idx = indices[fold_of_pos == 1]
    test_idx = indices[fold_of_pos == 2]
    return train_idx, val_idx, test_idx


def _positions(indices: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Positions of ``subset`` values within ``indices`` (both are index arrays)."""
    order = np.argsort(indices, kind="stable")
    sorted_idx = indices[order]
    pos_in_sorted = np.searchsorted(sorted_idx, subset)
    return order[pos_in_sorted]


def _carve_calib_from_train(
    train_idx: np.ndarray,
    calib_size: float,
    *,
    n_total: int,
    timestamps: Optional[pd.Series],
    groups: Optional[np.ndarray],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a disjoint calibration slice from ``train_idx`` ONLY (never val/test).

    ``calib_size`` is a fraction of the WHOLE dataset (same convention as
    ``test_size`` / ``val_size``); the slice is taken from the train portion so
    the base model -- fit on the returned ``train_idx`` -- never sees calib rows.

    Group-aware: when ``groups`` is supplied, whole groups move to calib (no
    group spans the calib/train boundary). Time-ordered: with ``timestamps`` the
    calib slice is the OLDEST train rows so the remaining (newer) train rows stay
    adjacent to val/test on the timeline.

    Returns ``(new_train_idx, calib_idx)`` (both unsorted; caller sorts).
    """
    n_calib_target = int(round(n_total * calib_size))
    if n_calib_target <= 0 or len(train_idx) == 0:
        return train_idx, np.array([], dtype=train_idx.dtype)
    n_calib_target = min(n_calib_target, len(train_idx))

    if groups is not None:
        # Move whole groups (oldest-first under timestamps, else random) until the
        # row budget is met. A group never straddles the calib/train boundary.
        _groups_arr = np.asarray(groups)
        train_groups = _groups_arr[train_idx]
        if timestamps is not None:
            _ts_vals = pd.Series(timestamps).values
            _order = np.argsort(_ts_vals[train_idx], kind="stable")
        else:
            _order = rng.permutation(len(train_idx))
        seen: dict = {}
        for _pos in _order:
            seen.setdefault(train_groups[_pos], []).append(_pos)
        chosen_local: list = []
        for _g, _positions in seen.items():
            if len(chosen_local) >= n_calib_target:
                break
            chosen_local.extend(_positions)
        chosen_local_arr = np.array(sorted(chosen_local), dtype=np.intp)
        calib_idx = train_idx[chosen_local_arr]
        _mask = np.ones(len(train_idx), dtype=bool)
        _mask[chosen_local_arr] = False
        return train_idx[_mask], calib_idx

    if timestamps is not None:
        # Oldest train rows form the calib slice; newer train rows stay adjacent
        # to val/test (temporal honesty preserved).
        _ts_vals = pd.Series(timestamps).values
        _order = np.argsort(_ts_vals[train_idx], kind="stable")
        calib_local = _order[:n_calib_target]
        train_local = _order[n_calib_target:]
        return train_idx[train_local], train_idx[calib_local]

    perm = rng.permutation(len(train_idx))
    calib_local = perm[:n_calib_target]
    train_local = perm[n_calib_target:]
    return train_idx[train_local], train_idx[calib_local]
