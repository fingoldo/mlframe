"""
Train/validation/test splitting utilities for mlframe.

Provides flexible data splitting with support for:
- Sequential and shuffled splits
- Date-based (whole-day) splitting
- Training set aging limits
"""

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    shuffle_val: bool = False,
    shuffle_test: bool = False,
    val_sequential_fraction: Optional[float] = None,
    test_sequential_fraction: Optional[float] = None,
    trainset_aging_limit: Optional[float] = None,
    timestamps: Optional[pd.Series] = None,
    wholeday_splitting: bool = True,
    random_seed: Optional[int] = None,
    val_placement: Literal["forward", "backward"] = "forward",
    stratify_y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """
    Split data into train, validation, and test sets with flexible sequential/shuffled control.

    Supports three modes:
    1. Date-based splitting (wholeday_splitting=True, timestamps provided)
    2. Row-based splitting with timestamps (wholeday_splitting=False, timestamps provided)
    3. Simple row-based splitting (no timestamps, uses sklearn)

    Args:
        df: Input DataFrame to split.
        test_size: Fraction of data for test set (0.0-1.0).
        val_size: Fraction of remaining data for validation set (0.0-1.0).
        shuffle_val: If True and val_sequential_fraction is None, fully shuffle validation set.
        shuffle_test: If True and test_sequential_fraction is None, fully shuffle test set.
        val_sequential_fraction: Fraction of validation set that should be sequential (0.0-1.0).
            If None and shuffle_val=True: fully shuffled.
            If None and shuffle_val=False: fully sequential.
        test_sequential_fraction: Fraction of test set that should be sequential (0.0-1.0).
            If None and shuffle_test=True: fully shuffled.
            If None and shuffle_test=False: fully sequential.
        trainset_aging_limit: If set, keep only the most recent fraction of training data.
            Must be between 0 and 1.
        timestamps: Series of timestamps for time-based splitting.
        wholeday_splitting: If True and timestamps provided, split by whole days.
        random_seed: Random seed for reproducibility.
        val_placement: Temporal placement of val relative to train/test.
            - "forward" (default): ``[train] [val] [test]`` -- val immediately
              precedes test on the timeline (conventional time-series split).
            - "backward": ``[val] [train] [test]`` -- val precedes train
              ("First test then train", Mazzanti 2024). Chosen when you want
              val-metric to approximate deployment-time performance under
              drift: forward's val->train gap is ~0 while train->prod gap is
              large, so val overstates prod; backward mirrors the gaps so
              val-metric is sampled from the same drift-distance regime as
              deployment. Only meaningful when ``timestamps`` is provided
              -- without time axis, placement is ill-defined and this
              argument is ignored (caller gets a plain sklearn shuffle
              split). Also a no-op when ``val_size`` is 0.

    Returns:
        Tuple of (train_idx, val_idx, test_idx, train_details, val_details, test_details)
        where *_idx are sorted numpy arrays of indices and *_details are description strings.
    """
    # Local RNG -- never mutate global numpy random state (policy).
    rng = np.random.default_rng(random_seed)
    # Derive a 32-bit int seed for sklearn splitters that need an integer seed.
    sklearn_seed = int(rng.integers(0, 2**32 - 1)) if random_seed is not None else None

    # Input validation (2026-04-19 proactive probe pass).
    # Previously negative ``val_size``/``test_size`` silently produced
    # empty splits; ``trainset_aging_limit=0`` silently no-op'd despite
    # the non-zero-only validator below contradicting that behaviour.
    if not (0.0 <= test_size <= 1.0):
        raise ValueError(f"test_size must be in [0, 1], got {test_size}")
    if not (0.0 <= val_size <= 1.0):
        raise ValueError(f"val_size must be in [0, 1], got {val_size}")
    # trainset_aging_limit: None = no-aging; otherwise must be strictly in (0, 1).
    # 0 is rejected explicitly because it would mean "keep zero of training
    # data", which is unusable and used to silently fall through the
    # ``if trainset_aging_limit:`` falsy short-circuit.
    if trainset_aging_limit is not None and not (0 < trainset_aging_limit < 1.0):
        raise ValueError(f"trainset_aging_limit must be in (0, 1), got {trainset_aging_limit}")
    if val_placement not in ("forward", "backward"):
        raise ValueError(
            f"val_placement must be 'forward' or 'backward', got {val_placement!r}"
        )

    # 2026-04-27 Session 7 batch 7: normalise ``timestamps`` to pd.Series
    # at function entry. Several downstream branches use ``timestamps.iloc``
    # / ``.dt`` accessors that only work on pandas Series. Callers may
    # pass either a Series (FTE.transform on pandas df) or a numpy
    # datetime64 ndarray (FTE.transform on polars df -> ``.to_numpy()``)
    # or a plain Python list. Coerce once so the rest of the function
    # has a stable contract.
    if timestamps is not None and not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)

    # Backward placement is time-axis-specific. Without timestamps there is
    # no "before/after" to place val relative to train, so we silently fall
    # back to forward -- the sklearn-shuffle path below doesn't order rows
    # by time anyway. ``val_size=0`` makes placement moot too.
    _effective_val_placement = val_placement
    if timestamps is None or val_size == 0:
        _effective_val_placement = "forward"

    # Diagnostic: surface the effective placement at INFO so a user who
    # passed ``val_placement="backward"`` but sees a forward-style split
    # in the log can immediately confirm whether (a) the value reached
    # this function at all (config wiring), (b) it was downgraded to
    # forward by the no-timestamps / val_size=0 fallback above, or
    # (c) the placement is honoured but the layout still looks
    # forward-ish for some other reason. Caller-attributed log so the
    # line shows up next to the existing "{N} train rows ... val rows
    # ..." summary at the bottom.
    if val_placement != _effective_val_placement:
        logger.info(
            "val_placement=%r requested but downgraded to %r "
            "(timestamps=%s, val_size=%s)",
            val_placement, _effective_val_placement,
            "present" if timestamps is not None else "None",
            val_size,
        )
    elif val_placement != "forward":
        logger.info("val_placement=%r (Mazzanti backward layout)", val_placement)

    if _effective_val_placement == "backward" and trainset_aging_limit is not None:
        # Aging trims the OLDEST train rows -- which in backward layout are
        # the ones closest to the (earlier) val block. Trimming them
        # widens the val->train gap silently, defeating the "gap mirror"
        # whole point of the Mazzanti split. Refuse explicitly rather
        # than produce a subtly wrong split.
        raise ValueError(
            "val_placement='backward' is incompatible with "
            "trainset_aging_limit -- aging removes the oldest train rows, "
            "which are the ones adjacent to the backward-placed val. "
            "Drop one of the two."
        )

    def _calculate_split_sizes(total_size, target_size, shuffle, sequential_fraction):
        """Calculate sequential and shuffled portions for a split."""
        n_total = int(total_size * target_size)

        if sequential_fraction is not None:
            if not (0.0 <= sequential_fraction <= 1.0):
                raise ValueError(f"sequential_fraction must be between 0.0 and 1.0, got {sequential_fraction}")
            n_sequential = int(n_total * sequential_fraction)
            n_shuffled = n_total - n_sequential
        else:
            # Legacy behavior: fully shuffled or fully sequential
            n_shuffled = n_total if shuffle else 0
            n_sequential = 0 if shuffle else n_total

        return n_sequential, n_shuffled

    def _perform_split(sorted_items, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng=rng):
        """Perform the actual splitting on sorted items (dates or indices).

        With ``_effective_val_placement == "backward"`` the sequential val
        slice is taken from the OLDEST end of ``remaining`` after the test
        slice is removed, producing the Mazzanti layout
        ``[val(oldest)] [train(middle)] [test(newest)]``. Forward (default)
        takes it from the newest end of the same remainder, producing
        ``[train(oldest)] [val(middle)] [test(newest)]``. The shuffled
        portions are drawn from the remainder after BOTH sequential slices
        are removed, so the shuffled-val / shuffled-test samples don't
        overlap with the sequential blocks regardless of placement.
        """
        remaining = sorted_items.copy()
        test_seq = val_seq = None
        test_list = []
        val_list = []

        # Sequential test (most recent) -- same for both placements.
        if n_test_seq > 0:
            test_seq = remaining[-n_test_seq:]
            test_list.append(test_seq)
            remaining = remaining[:-n_test_seq]

        # Sequential val: newest end (forward) or oldest end (backward).
        if n_val_seq > 0:
            if _effective_val_placement == "backward":
                val_seq = remaining[:n_val_seq]
                val_list.append(val_seq)
                remaining = remaining[n_val_seq:]
            else:
                val_seq = remaining[-n_val_seq:]
                val_list.append(val_seq)
                remaining = remaining[:-n_val_seq]

        # Shuffled test from remaining
        if n_test_shuf > 0:
            test_shuf_idx = rng.choice(len(remaining), n_test_shuf, replace=False)
            test_list.append(remaining[test_shuf_idx])
            remaining = np.delete(remaining, test_shuf_idx)

        # Shuffled val from remaining
        if n_val_shuf > 0:
            val_shuf_idx = rng.choice(len(remaining), n_val_shuf, replace=False)
            val_list.append(remaining[val_shuf_idx])
            remaining = np.delete(remaining, val_shuf_idx)

        test_items = np.concatenate(test_list) if test_list else np.array([], dtype=remaining.dtype)
        val_items = np.concatenate(val_list) if val_list else np.array([], dtype=remaining.dtype)
        train_items = remaining

        return train_items, val_items, test_items, val_seq, test_seq

    def _build_details(timestamps, idx, sequential_idx, n_shuffled, unit) -> str:
        """Build a detail string for a split set.

        Format: ``{first_date}/{last_date} +{n_shuffled}{unit}``

        The ``+N<unit>`` suffix means "N extra items, sampled at random from
        *outside* the sequential date window, mixed into this split." It lets
        the user know at a glance that the split is not purely contiguous.
        ``unit`` is a single letter:
          * ``R`` -- ``N`` additional **rows** (row-based splitting)
          * ``D`` -- ``N`` additional **days** (whole-day splitting)

        Example: ``90_000 val rows 2014-01-20/2014-04-05 +45000R`` =
        45k val rows from outside the Jan-Apr 2014 window were shuffled in
        on top of the sequential 45k that fell inside the window.
        """
        if sequential_idx is not None and len(sequential_idx) > 0:
            details = f"{timestamps.iloc[sequential_idx].min():%Y-%m-%d}/{timestamps.iloc[sequential_idx].max():%Y-%m-%d}"
            if n_shuffled > 0:
                details += f" +{n_shuffled}{unit}"
        else:
            if len(idx) > 0:
                details = f"{timestamps.iloc[idx].min():%Y-%m-%d}/{timestamps.iloc[idx].max():%Y-%m-%d}"
            else:
                details = ""
        return details

    # Calculate split sizes
    if wholeday_splitting and timestamps is not None:
        # `.dt.floor('D')` is vectorized over datetime64 and stays in datetime dtype
        # (unlike `.dt.date` which yields a Python-object Series -- much slower for isin).
        # 2026-04-27: ``pd.to_datetime`` on a numpy datetime64 ndarray returns a
        # ``DatetimeIndex`` (no ``.dt`` accessor); on a pandas Series it returns
        # a Series (has ``.dt``). FTE.transform may pass either shape -- wrap
        # in pd.Series first so ``.dt.floor`` works uniformly.
        dates = pd.to_datetime(pd.Series(timestamps)).dt.floor("D")
        unique_dates = dates.unique()
        n_total = len(unique_dates)
    else:
        n_total = len(df)

    n_test_seq, n_test_shuf = _calculate_split_sizes(n_total, test_size, shuffle_test, test_sequential_fraction)
    n_val_seq, n_val_shuf = _calculate_split_sizes(n_total - (n_test_seq + n_test_shuf), val_size, shuffle_val, val_sequential_fraction)

    # Perform splitting
    if wholeday_splitting and timestamps is not None:
        # Date-based splitting
        sorted_dates = np.sort(unique_dates)
        train_dates, val_dates, test_dates, val_dates_seq, test_dates_seq = _perform_split(
            sorted_dates, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf
        )

        # Apply aging limit BEFORE computing train_idx / train_details,
        # so the printed train date range reflects the actually-used rows
        # (consistent with the row-timestamp branch below).
        if trainset_aging_limit:
            n_dates_to_keep = int(len(train_dates) * trainset_aging_limit)
            if n_dates_to_keep > 0:
                train_dates = np.sort(train_dates)[-n_dates_to_keep:]

        # Map dates -> split label once, then derive all index arrays from the cached
        # label array. Each `dates.isin(...)` call would re-hash the full Series; doing
        # it once via a unique-date categorical keeps the work O(n) instead of O(n*k).
        # Label convention: 0=train, 1=val, 2=test, -1=dropped by aging.
        uniq = pd.unique(dates)
        date_to_label = {d: -1 for d in uniq}
        for d in train_dates:
            date_to_label[d] = 0
        for d in val_dates:
            date_to_label[d] = 1
        for d in test_dates:
            date_to_label[d] = 2
        labels = dates.map(date_to_label).to_numpy()
        train_idx = np.flatnonzero(labels == 0)
        val_idx = np.flatnonzero(labels == 1)
        test_idx = np.flatnonzero(labels == 2)

        def _dates_to_idx(dates_subset):
            if dates_subset is None:
                return None
            subset_map = {d: True for d in dates_subset}
            return np.flatnonzero(dates.map(lambda d: subset_map.get(d, False)).to_numpy())

        val_seq_idx = _dates_to_idx(val_dates_seq)
        test_seq_idx = _dates_to_idx(test_dates_seq)

        # Build detail strings. ``.min()`` on empty index yields NaT, which
        # then crashes the ``:%Y-%m-%d`` formatter. Guard for empty train.
        if len(train_idx) > 0:
            train_details = f"{timestamps.iloc[train_idx].min():%Y-%m-%d}/{timestamps.iloc[train_idx].max():%Y-%m-%d}"
        else:
            train_details = "(empty)"

        val_details = _build_details(timestamps, val_idx, val_seq_idx, n_val_shuf, "D")
        test_details = _build_details(timestamps, test_idx, test_seq_idx, n_test_shuf, "D")

    elif timestamps is not None:
        # Row-based splitting with timestamps
        sorted_idx = np.argsort(timestamps.values)
        train_idx, val_idx, test_idx, val_idx_seq, test_idx_seq = _perform_split(
            sorted_idx, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf
        )

        # Apply aging limit
        if trainset_aging_limit:
            train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)):]

        # Build detail strings (same NaT-on-empty guard as above).
        if len(train_idx) > 0:
            train_details = f"{timestamps.iloc[train_idx].min():%Y-%m-%d}/{timestamps.iloc[train_idx].max():%Y-%m-%d}"
        else:
            train_details = "(empty)"
        val_details = _build_details(timestamps, val_idx, val_idx_seq, n_val_shuf, "R")
        test_details = _build_details(timestamps, test_idx, test_idx_seq, n_test_shuf, "R")

    else:
        # Row-based splitting without timestamps (fallback to sklearn).
        # 2026-04-24: stratify_y support -- when provided, route through
        # sklearn StratifiedShuffleSplit (1-D y) or
        # iterstrat.MultilabelStratifiedShuffleSplit (2-D y, multilabel).
        # Both REQUIRE shuffle (cannot stratify a sequential split). When
        # stratify_y is provided AND shuffle_test/shuffle_val is False,
        # we emit a WARNING and stratify anyway (correctness > sequential
        # nicety; user explicitly asked for class balance).
        # 2026-05-04: ``groups`` support -- when provided (typically for
        # learning-to-rank), route through sklearn ``GroupShuffleSplit`` so
        # all rows belonging to one query land in the same split. Mutually
        # exclusive with stratify_y (sklearn doesn't ship a stratified
        # group-split out of the box; if a user really needs both, they
        # can pre-bucket groups by stratum first).
        all_idx = np.arange(len(df))

        if groups is not None and stratify_y is not None:
            raise ValueError(
                "make_train_test_split: groups and stratify_y are mutually "
                "exclusive in the row-based fallback (no group-stratified "
                "splitter is wired). Pick one: groups for LTR / "
                "leave-one-group-out semantics, stratify_y for class-balance."
            )

        if stratify_y is not None and timestamps is not None:
            logger.warning(
                "stratify_y provided but timestamps active -- stratification "
                "ignored (stratification is ill-defined for time-based splits)."
            )
            _stratify_active = None
        elif stratify_y is not None:
            _stratify_active = np.asarray(stratify_y)
            if _stratify_active.shape[0] != len(df):
                raise ValueError(
                    f"stratify_y length {_stratify_active.shape[0]} does not "
                    f"match df length {len(df)}"
                )
            if _stratify_active.ndim not in (1, 2):
                raise ValueError(
                    f"stratify_y must be 1-D (single-label) or 2-D (multilabel), "
                    f"got shape {_stratify_active.shape}"
                )
        else:
            _stratify_active = None

        # Group-aware splitter: GroupShuffleSplit on the row-based path.
        # Validate length once; the shape contract is per-row (len == len(df)).
        _groups_arr = None
        if groups is not None:
            _groups_arr = np.asarray(groups)
            if _groups_arr.shape[0] != len(df):
                raise ValueError(
                    f"groups length {_groups_arr.shape[0]} does not match "
                    f"df length {len(df)}"
                )
            if _groups_arr.ndim != 1:
                raise ValueError(
                    f"groups must be 1-D (one query-id per row), got shape "
                    f"{_groups_arr.shape}"
                )

        if test_size > 0:
            if _groups_arr is not None:
                from sklearn.model_selection import GroupShuffleSplit
                gss_test = GroupShuffleSplit(
                    n_splits=1, test_size=test_size, random_state=sklearn_seed,
                )
                train_idx, test_idx = next(gss_test.split(all_idx, groups=_groups_arr))
                # GroupShuffleSplit returns positions into all_idx, but here
                # all_idx == np.arange(len(df)) so they coincide -- normalise
                # to int arrays for consistency with downstream sort.
                train_idx = np.asarray(train_idx, dtype=np.intp)
                test_idx = np.asarray(test_idx, dtype=np.intp)
            elif _stratify_active is not None:
                train_idx, test_idx = _stratified_split(
                    all_idx, test_size=test_size,
                    stratify_y=_stratify_active, random_state=sklearn_seed,
                )
            else:
                train_idx, test_idx = train_test_split(
                    all_idx, test_size=test_size, shuffle=shuffle_test,
                    random_state=sklearn_seed if shuffle_test else None,
                )
        else:
            train_idx, test_idx = all_idx, np.array([], dtype=np.intp)

        if val_size > 0:
            if _groups_arr is not None:
                from sklearn.model_selection import GroupShuffleSplit
                gss_val = GroupShuffleSplit(
                    n_splits=1, test_size=val_size, random_state=sklearn_seed,
                )
                _train_groups = _groups_arr[train_idx]
                _train_local_train, _train_local_val = next(
                    gss_val.split(train_idx, groups=_train_groups)
                )
                # gss returns positions into train_idx, not into all_idx.
                val_idx = train_idx[_train_local_val]
                train_idx = train_idx[_train_local_train]
            elif _stratify_active is not None:
                # Stratify val from the remaining train indices.
                strat_train = _stratify_active[train_idx]
                train_idx, val_idx = _stratified_split(
                    train_idx, test_size=val_size,
                    stratify_y=strat_train, random_state=sklearn_seed,
                )
            else:
                train_idx, val_idx = train_test_split(
                    train_idx, test_size=val_size, shuffle=shuffle_val,
                    random_state=sklearn_seed if shuffle_val else None,
                )
        else:
            val_idx = np.array([], dtype=np.intp)

        if trainset_aging_limit:
            train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)):]

        train_details, val_details, test_details = "", "", ""

    # 2026-05-04: Group-spans-cutoff resolution for time-based splits.
    # When ``groups`` is supplied alongside ``timestamps`` (typical LTR
    # scenario where queries are time-stamped), a query may straddle a
    # train->val or val->test cutoff. Default behaviour (per user 2026-05-04):
    # assign the WHOLE spanning group to the LATER side -- preserves
    # temporal ordering, so train never sees rows from a query that
    # leaked into val/test. Emit a single INFO summarising how many
    # groups were reassigned. The row-based path with ``groups`` already
    # uses GroupShuffleSplit so groups stay together by construction --
    # this block only matters when timestamps drive the split AND groups
    # are also supplied.
    if groups is not None and timestamps is not None:
        _groups_arr_post = np.asarray(groups)
        train_g = set(_groups_arr_post[train_idx].tolist()) if len(train_idx) else set()
        val_g = set(_groups_arr_post[val_idx].tolist()) if len(val_idx) else set()
        test_g = set(_groups_arr_post[test_idx].tolist()) if len(test_idx) else set()

        train_val_overlap = train_g & val_g
        train_test_overlap = train_g & test_g
        val_test_overlap = val_g & test_g

        n_reassigned = 0
        if train_val_overlap or train_test_overlap or val_test_overlap:
            # Build the reassignment plan. A group spanning train+test goes
            # entirely to test (the latest split wins -- matches user
            # preference of "assign to later side"). A group spanning
            # train+val goes to val. A group spanning val+test goes to
            # test. Combined train+val+test -> test.
            promote_to_val = train_val_overlap - train_test_overlap
            promote_to_test = train_test_overlap | val_test_overlap

            if promote_to_val or promote_to_test:
                row_group = _groups_arr_post
                # Mask of rows currently in train that need to leave train.
                _to_val_mask = np.isin(row_group, list(promote_to_val)) if promote_to_val else np.zeros(len(df), dtype=bool)
                _to_test_mask = np.isin(row_group, list(promote_to_test)) if promote_to_test else np.zeros(len(df), dtype=bool)

                # Drop reassigned-row positions from train_idx and val_idx,
                # then add them to the destination split.
                _new_train_mask = np.zeros(len(df), dtype=bool)
                _new_train_mask[train_idx] = True
                _new_train_mask &= ~(_to_val_mask | _to_test_mask)

                _new_val_mask = np.zeros(len(df), dtype=bool)
                _new_val_mask[val_idx] = True
                _new_val_mask &= ~_to_test_mask
                _new_val_mask |= _to_val_mask

                _new_test_mask = np.zeros(len(df), dtype=bool)
                _new_test_mask[test_idx] = True
                _new_test_mask |= _to_test_mask

                # Verify partition invariants before commit.
                assert not (_new_train_mask & _new_val_mask).any()
                assert not (_new_train_mask & _new_test_mask).any()
                assert not (_new_val_mask & _new_test_mask).any()

                n_reassigned = int(_to_val_mask.sum() + _to_test_mask.sum())
                train_idx = np.flatnonzero(_new_train_mask).astype(np.intp)
                val_idx = np.flatnonzero(_new_val_mask).astype(np.intp)
                test_idx = np.flatnonzero(_new_test_mask).astype(np.intp)

                logger.warning(
                    "Group-spanning cutoff resolution: %d row(s) from %d "
                    "spanning group(s) reassigned to the later split "
                    "(train->val: %d groups, train+val->test: %d groups). "
                    "This preserves group integrity for LTR / per-group "
                    "metrics. To eliminate spanning, widen aging_limit so "
                    "the cutoff falls outside any group's timespan, or "
                    "drop spanning groups before calling the splitter.",
                    n_reassigned,
                    len(promote_to_val) + len(promote_to_test),
                    len(promote_to_val), len(promote_to_test),
                )

    # Silent-empty-split warning: user requested a non-zero val/test fraction
    # but wound up with 0 rows. Happens in practice when wholeday_splitting=True
    # and all rows share a single date (n_unique_days=1, int(1*0.1)=0), or
    # when val_size*test_size was small enough to round to 0 at low n.
    # Emit a WARNING so the user notices instead of silently losing splits.
    if val_size > 0 and len(val_idx) == 0:
        logger.warning(
            "val_size=%s requested but val split is empty (possibly because "
            "wholeday_splitting collapsed to a single date, or n_rows*val_size<1).",
            val_size,
        )
    if test_size > 0 and len(test_idx) == 0:
        logger.warning(
            "test_size=%s requested but test split is empty (possibly because "
            "wholeday_splitting collapsed to a single date, or n_rows*test_size<1).",
            test_size,
        )

    logger.info(
        f"{len(train_idx):_} train rows {train_details}, "
        f"{len(val_idx):_} val rows {val_details}, "
        f"{len(test_idx):_} test rows {test_details}."
    )

    return (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
        train_details,
        val_details,
        test_details,
    )


__all__ = ["make_train_test_split"]
