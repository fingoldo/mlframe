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
