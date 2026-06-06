"""
Train/validation/test splitting utilities for mlframe.

Provides flexible data splitting with support for:
- Sequential and shuffled splits
- Date-based (whole-day) splitting
- Training set aging limits
"""

from __future__ import annotations


import logging
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


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
    calib_size: Optional[float] = None,
    return_calib: bool = False,
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

        calib_size: Fraction of the WHOLE dataset to carve as a disjoint calibration slice
            from the TRAIN portion only (never val/test). The base model is fit on the
            returned (shrunk) train_idx, so it never sees calib rows -- the calibrator can
            then be fit on calib leakage-free. Group-aware (whole groups move to calib) and
            time-ordered (oldest train rows) when applicable. None/0 -> no carve.
        return_calib: When True, the return tuple is extended with (calib_idx, calib_details).

    Returns:
        Tuple of (train_idx, val_idx, test_idx, train_details, val_details, test_details)
        where *_idx are sorted numpy arrays of indices and *_details are description strings.
        When ``return_calib=True``, two extra elements are appended: (calib_idx, calib_details).
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

    # 2026-05-18 explicit log: surface group-aware splitting status so the
    # operator sees at a glance whether GroupShuffleSplit is in effect.
    # Pre-fix this was implicit ("if groups is not None: ..."). Users with
    # ``SimpleFeaturesAndTargetsExtractor(group_field=...)`` could not tell
    # from the log whether the group_field propagated through to the
    # splitter.
    if groups is not None:
        try:
            _n_groups = int(np.unique(np.asarray(groups)).shape[0])
        except Exception:
            _n_groups = -1
        logger.info(
            "Group-aware splitting: ENABLED (n_groups=%d). "
            "Each group stays within ONE split (no per-row leakage).",
            _n_groups,
        )
        logger.warning(
            "Group-aware split: downstream models with unbounded output "
            "ranges (Identity-MLP, LinearRegression, plain MLP without "
            "output clipping) can extrapolate catastrophically on test "
            "groups whose feature distribution differs from train. "
            "Observed in prod: Identity-MLP collapsed R^2=-326 "
            "on unseen-group test split while Ridge nailed R^2=1.00 on "
            "identical data. Mitigations: (a) prefer Ridge over plain "
            "LinearRegression, (b) use a real nonlinearity (nn.ReLU / "
            "nn.GELU) instead of nn.Identity in MLP configs, (c) let "
            "composite-target discovery propose a residualised target "
            "(``y - alpha*top_AR_feature``) that bounds the residual "
            "variance so prediction extrapolation can't blow up.",
        )
    else:
        logger.info(
            "Group-aware splitting: disabled (groups=None; rows split "
            "independently). To enable, supply group_field= in your "
            "extractor and keep TrainingSplitConfig.use_groups=True (default)."
        )

    # Backward placement is time-axis-specific. Without timestamps there is
    # no "before/after" to place val relative to train, so we silently fall
    # back to forward -- the sklearn-shuffle path below doesn't order rows
    # by time anyway. ``val_size=0`` makes placement moot too.
    _effective_val_placement = val_placement
    if timestamps is None or val_size == 0:
        _effective_val_placement = "forward"

    # Implied-temporal-layout INFO: when caller supplied timestamps AND default val_placement="forward" (a quiet default that assumes iid-friendly target), surface the resulting val/train/test ranges + the val->train gap + estimated train->prod gap so the time-series user can see at a glance how their split lays out. Cheap one-time log; never auto-flips the default.
    if (
        timestamps is not None
        and val_placement == "forward"
        and _effective_val_placement == "forward"
        and val_size > 0
    ):
        try:
            _ts = pd.to_datetime(timestamps, errors="coerce")
            _ts_sorted = _ts.sort_values().reset_index(drop=True)
            _n = len(_ts_sorted)
            if _n >= 3:
                _n_test = max(1, int(round(_n * test_size))) if test_size > 0 else 0
                _n_val = max(1, int(round(_n * val_size)))
                _train_end_pos = _n - _n_test - _n_val
                _val_end_pos = _n - _n_test
                _t_train_max = _ts_sorted.iloc[max(0, _train_end_pos - 1)]
                _t_val_min = _ts_sorted.iloc[_train_end_pos] if _train_end_pos < _n else _ts_sorted.iloc[-1]
                _t_val_max = _ts_sorted.iloc[max(0, _val_end_pos - 1)]
                _t_test_min = _ts_sorted.iloc[_val_end_pos] if _val_end_pos < _n else _ts_sorted.iloc[-1]
                _t_test_max = _ts_sorted.iloc[-1]
                _gap_val_train_days = float((_t_val_min - _t_train_max).total_seconds() / 86400.0) if pd.notna(_t_val_min) and pd.notna(_t_train_max) else float("nan")
                _gap_train_prod_days = float((_t_test_max - _t_train_max).total_seconds() / 86400.0) if pd.notna(_t_test_max) and pd.notna(_t_train_max) else float("nan")
                logger.info(
                    "Temporal layout (val_placement='forward', default): train_max=%s, val=[%s..%s], test=[%s..%s], val->train_gap=%.2fd, train->prod_estimated_gap=%.2fd. Mazzanti backward layout typically gives a better prod-error proxy under drift; set val_placement='backward' to switch.",
                    _t_train_max, _t_val_min, _t_val_max, _t_test_min, _t_test_max,
                    _gap_val_train_days, _gap_train_prod_days,
                )
        except Exception as _layout_err:
            logger.debug("Temporal-layout INFO compute failed: %s", _layout_err)

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
        # Caller asked for temporal honesty (newest data -> val) but no timestamps were
        # supplied or val_size=0 forced fallback. INFO-level: downgrade is the explicit
        # fallback for the no-timestamp case, not a configuration error. If group_field
        # was supplied, GroupShuffleSplit still keeps each group entirely on ONE side
        # (no per-row group leakage); only the temporal ORDERING of which groups go
        # where is lost.
        if groups is not None:
            try:
                _ng = int(np.unique(np.asarray(groups)).shape[0])
            except Exception:
                _ng = -1
            _group_clause = f"; groups (n_groups={_ng}) still kept whole per split"
        else:
            _group_clause = ""
        _reason = (
            "no timestamps_column"
            if timestamps is None
            else f"val_size={val_size}"
        )
        if val_placement == "backward":
            logger.warning(
                "val_placement=%r requested but downgraded to %r (%s)%s. "
                "Temporal honesty lost: val will be drawn at random instead of newest-data. "
                "Supply timestamps_column to honor backward placement.",
                val_placement, _effective_val_placement, _reason, _group_clause,
            )
        else:
            logger.info(
                "val_placement=%r downgraded to %r (%s)%s.",
                val_placement, _effective_val_placement, _reason, _group_clause,
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

        # Shuffled test from remaining. rng.choice(N, k, replace=False) raises
        # when k > N; that surfaces as a ValueError far from the cause (e.g.
        # when the sequential test block already consumed most of the input,
        # or test_size + val_size jointly exceed n_total). Clamp with a WARN
        # so the caller sees the issue and the split still produces a result.
        if n_test_shuf > 0:
            _eff_test_shuf = min(n_test_shuf, len(remaining))
            if _eff_test_shuf < n_test_shuf:
                logger.warning(
                    "Shuffled test size %d exceeds remaining pool %d after "
                    "sequential allocation; clamping to %d.",
                    n_test_shuf, len(remaining), _eff_test_shuf,
                )
            if _eff_test_shuf > 0:
                test_shuf_idx = rng.choice(len(remaining), _eff_test_shuf, replace=False)
                test_list.append(remaining[test_shuf_idx])
                remaining = np.delete(remaining, test_shuf_idx)

        # Shuffled val from remaining (same clamp rationale).
        if n_val_shuf > 0:
            _eff_val_shuf = min(n_val_shuf, len(remaining))
            if _eff_val_shuf < n_val_shuf:
                logger.warning(
                    "Shuffled val size %d exceeds remaining pool %d after "
                    "sequential / test allocation; clamping to %d.",
                    n_val_shuf, len(remaining), _eff_val_shuf,
                )
            if _eff_val_shuf > 0:
                val_shuf_idx = rng.choice(len(remaining), _eff_val_shuf, replace=False)
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
        def _fmt_ts(value):
            """Format a single timestamp for log lines. Datetime-typed
            values use ``%Y-%m-%d``; numeric (int/float epoch-seconds or
            generic numeric proxy) values fall back to ``repr(value)``
            because the ``%Y-%m-%d`` format-spec raises
            ``ValueError: Invalid format specifier '%Y-%m-%d' for object
            of type 'int'`` on non-datetime inputs (the FTE's ``ts_field``
            plumbing accepts any monotone numeric column).
            """
            try:
                return format(value, "%Y-%m-%d")
            except (ValueError, TypeError):
                return str(value)

        if sequential_idx is not None and len(sequential_idx) > 0:
            details = f"{_fmt_ts(timestamps.iloc[sequential_idx].min())}/{_fmt_ts(timestamps.iloc[sequential_idx].max())}"
            if n_shuffled > 0:
                details += f" +{n_shuffled}{unit}"
        else:
            if len(idx) > 0:
                details = f"{_fmt_ts(timestamps.iloc[idx].min())}/{_fmt_ts(timestamps.iloc[idx].max())}"
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
        # Numeric ts (epoch-seconds int64) can't be ``pd.to_datetime``'d
        # without a unit hint and isn't a meaningful date axis for the
        # wholeday-splitter; route those to the row-based path. Surfaced by
        # iter-48 300k seed=7 cb-regression where a fuzz-axis combo passed
        # int64 ts AND wholeday_splitting=True; pd.to_datetime then
        # collapsed every row into the 1970-01-01 epoch single date and
        # the splitter produced empty val/test, leading to
        # ``CatBoostError: Input data must have at least one feature``.
        _ts_series = pd.Series(timestamps)
        _kind = getattr(_ts_series.dtype, "kind", None)
        if _kind in {"i", "u", "f"}:
            logger.warning(
                "wholeday_splitting=True requested but timestamps dtype "
                "is numeric (%s); falling back to row-based split. "
                "Numeric ts has no calendar-day semantics for the "
                "wholeday-splitter.",
                _ts_series.dtype,
            )
            wholeday_splitting = False
            n_total = len(df)
        else:
            dates = pd.to_datetime(_ts_series).dt.floor("D")
            unique_dates = dates.unique()
            n_total_days = len(unique_dates)
            # Iter-48 guard: predict the actual val/test day counts
            # ``_calculate_split_sizes`` would produce (floor on int(n *
            # frac)) and fall back to row-based if any REQUESTED non-zero
            # split would come out as 0 days. Examples that trigger:
            #   - n_days=1, any val_size>0 or test_size>0 (collapses)
            #   - n_days=3, val_size=0.34, test_size=0.34
            #     -> int(3*0.34)=1 test, then int(2*0.34)=0 val (empty)
            # Empty val/test then crash downstream model.fit
            # (CatBoost: "Input data must have at least one feature").
            _predicted_n_test = int(n_total_days * test_size)
            _predicted_n_val = int(
                (n_total_days - _predicted_n_test) * val_size,
            )
            _val_empty_wrongly = val_size > 0 and _predicted_n_val == 0
            _test_empty_wrongly = test_size > 0 and _predicted_n_test == 0
            if _val_empty_wrongly or _test_empty_wrongly:
                logger.warning(
                    "wholeday_splitting=True requested but only %d "
                    "unique day(s) found, which yields "
                    "predicted_n_val=%d, predicted_n_test=%d at "
                    "val_size=%s, test_size=%s; falling back to "
                    "row-based split so val/test stay non-empty.",
                    n_total_days, _predicted_n_val, _predicted_n_test,
                    val_size, test_size,
                )
                wholeday_splitting = False
                n_total = len(df)
            else:
                n_total = n_total_days
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
        # label array. ``dates.map(dict)`` triggers a per-row Python apply path
        # on pandas Series of datetime values when the lookup dict is small,
        # measured ~6x slower than factorize+gather on a 1M-row daily span.
        # We factorize once to get integer codes per unique date, then build
        # a (n_unique,)-sized label LUT and gather labels[codes] in pure NumPy.
        # Label convention: 0=train, 1=val, 2=test, -1=dropped by aging.
        codes, uniq_index = pd.factorize(dates, sort=False)
        # uniq_index is a pandas Index of unique dates in first-occurrence
        # order; build a O(1) lookup from datetime -> position via a dict.
        # 2026-05-21: normalise keys + lookups to int64 nanoseconds so the
        # ``dict.get(d)`` calls below work regardless of whether ``train_dates``
        # is a numpy datetime64 array (default on most pandas versions) or a
        # pandas DatetimeIndex of Timestamps (pandas 3.0 path). The previous
        # ``{d: i for i, d in enumerate(uniq_index)}`` keyed by Timestamps
        # silently lost lookups when ``train_dates`` was np.datetime64 because
        # Timestamp.__hash__ != np.datetime64.__hash__ on pandas 3.0; every
        # label_lut entry stayed -1 and every split came out empty.
        def _as_ns_int(_d):
            try:
                return int(pd.Timestamp(_d).value)
            except (ValueError, TypeError, OverflowError):
                return _d

        def _ns_keys(_arr):
            # Vectorised datetime -> int64-ns keys. ``pd.DatetimeIndex(...).asi8``
            # is the batch equivalent of the per-element ``int(pd.Timestamp(_d).value)``
            # (verified identical), ~10x faster on a 50k-date span by avoiding one
            # Timestamp construction per date across uniq_index + train/val/test. The
            # per-element ``_as_ns_int`` path is kept as the fallback for arrays that
            # are not datetime-convertible: DatetimeIndex raises on any unconvertible
            # element, so a successful conversion implies every element matches
            # pd.Timestamp, and the fallback then reproduces the old value-as-is keys.
            try:
                return pd.DatetimeIndex(np.asarray(_arr)).asi8.tolist()
            except (ValueError, TypeError, OverflowError):
                return [_as_ns_int(_d) for _d in _arr]

        date_to_code = {k: i for i, k in enumerate(_ns_keys(uniq_index))}
        label_lut = np.full(len(uniq_index), -1, dtype=np.int8)
        for _lbl, _split_dates in ((0, train_dates), (1, val_dates), (2, test_dates)):
            for k in _ns_keys(_split_dates):
                _i = date_to_code.get(k)
                if _i is not None:
                    label_lut[_i] = _lbl
        # `codes` already aligned to `dates` order; gather is a single C-loop.
        # Rows whose date was outside any of train/val/test dates retain -1
        # (codes == -1 from factorize for unknown values; clamp via where).
        labels = np.where(codes >= 0, label_lut[codes], -1)
        train_idx = np.flatnonzero(labels == 0)
        val_idx = np.flatnonzero(labels == 1)
        test_idx = np.flatnonzero(labels == 2)

        def _dates_to_idx(dates_subset):
            if dates_subset is None:
                return None
            subset_mask = np.zeros(len(uniq_index), dtype=bool)
            for d in dates_subset:
                # Use the same int64-ns normalisation as the label-LUT loops
                # so dict lookups work regardless of whether `dates_subset`
                # is a numpy datetime64 array or a pandas DatetimeIndex.
                _i = date_to_code.get(_as_ns_int(d))
                if _i is not None:
                    subset_mask[_i] = True
            return np.flatnonzero(np.where(codes >= 0, subset_mask[codes], False))

        val_seq_idx = _dates_to_idx(val_dates_seq)
        test_seq_idx = _dates_to_idx(test_dates_seq)

        # Build detail strings. ``.min()`` on empty index yields NaT, which
        # then crashes the ``:%Y-%m-%d`` formatter. Guard for empty train.
        # Numeric ts also crashes ``:%Y-%m-%d`` formatter -- reuse the
        # ``_fmt_ts`` helper defined inside ``_build_details``. (Promoted
        # to module-private if more sites grow this need; for now inline.)
        def _fmt_ts(value):
            try:
                return format(value, "%Y-%m-%d")
            except (ValueError, TypeError):
                return str(value)

        if len(train_idx) > 0:
            train_details = f"{_fmt_ts(timestamps.iloc[train_idx].min())}/{_fmt_ts(timestamps.iloc[train_idx].max())}"
        else:
            train_details = "(empty)"

        val_details = _build_details(timestamps, val_idx, val_seq_idx, n_val_shuf, "D")
        test_details = _build_details(timestamps, test_idx, test_seq_idx, n_test_shuf, "D")

    elif timestamps is not None:
        # Row-based splitting with timestamps. Use ``kind="stable"`` so that
        # ties keep their original positional order; default quicksort
        # reshuffles ties run-to-run which makes seeded splits non-reproducible.
        sorted_idx = np.argsort(timestamps.values, kind="stable")
        train_idx, val_idx, test_idx, val_idx_seq, test_idx_seq = _perform_split(
            sorted_idx, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf
        )

        # Apply aging limit
        if trainset_aging_limit:
            train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)):]

        # Build detail strings (same NaT-on-empty guard as above; also
        # numeric-ts safe via the inline ``_fmt_ts`` fallback).
        def _fmt_ts(value):
            try:
                return format(value, "%Y-%m-%d")
            except (ValueError, TypeError):
                return str(value)

        if len(train_idx) > 0:
            train_details = f"{_fmt_ts(timestamps.iloc[train_idx].min())}/{_fmt_ts(timestamps.iloc[train_idx].max())}"
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

        # ``StratifiedGroupKFold`` (sklearn >=1.0) preserves the class /
        # bucket distribution of ``stratify_y`` ACROSS splits while keeping
        # whole groups together. Multilabel + group is only supported via
        # the optional ``iterative-stratification`` package
        # (``MultilabelStratifiedGroupKFold``); when the user passes a 2-D
        # stratify_y + groups without that package we fall back to
        # ``GroupShuffleSplit`` (groups precedence) and warn.
        _strat_groups_active = (groups is not None) and (stratify_y is not None)
        if _strat_groups_active:
            _strat_arr = np.asarray(stratify_y)
            if _strat_arr.ndim == 2:
                try:
                    from iterstrat.ml_stratifiers import (  # noqa: F401
                        MultilabelStratifiedGroupKFold,
                    )
                    _multilabel_group_strat = True
                except ImportError:
                    _multilabel_group_strat = False
                if not _multilabel_group_strat:
                    logger.info(
                        "make_train_test_split: multilabel stratify_y + groups "
                        "supplied but iterative-stratification not installed. "
                        "Falling back to GroupShuffleSplit (groups precedence). "
                        "pip install iterative-stratification to enable "
                        "MultilabelStratifiedGroupKFold."
                    )
                    _strat_groups_active = False
                    stratify_y = None

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
            if _groups_arr is not None and _strat_groups_active and _stratify_active is not None and _stratify_active.ndim == 1:
                # Both constraints simultaneously: stratify by target
                # bucket / class AND keep whole groups together. sklearn
                # ``StratifiedGroupKFold`` requires n_splits >= 2; we
                # derive n_splits from the requested test_size and take
                # the first fold as test (groups + class proportions
                # preserved by construction).
                from sklearn.model_selection import StratifiedGroupKFold
                _n_splits_test = max(2, int(round(1.0 / max(test_size, 1e-9))))
                sgkf = StratifiedGroupKFold(
                    n_splits=_n_splits_test, shuffle=True, random_state=sklearn_seed,
                )
                train_idx, test_idx = next(
                    sgkf.split(all_idx, _stratify_active, groups=_groups_arr),
                )
                train_idx = np.asarray(train_idx, dtype=np.intp)
                test_idx = np.asarray(test_idx, dtype=np.intp)
            elif _groups_arr is not None:
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
            if _groups_arr is not None and _strat_groups_active and _stratify_active is not None and _stratify_active.ndim == 1:
                # Same StratifiedGroupKFold strategy as the test split,
                # restricted to the post-test train rows. val_size is
                # interpreted as a fraction of the REMAINING train pool
                # (matching the existing GroupShuffleSplit semantics).
                from sklearn.model_selection import StratifiedGroupKFold
                _n_splits_val = max(2, int(round(1.0 / max(val_size, 1e-9))))
                _train_groups = _groups_arr[train_idx]
                _train_strat = _stratify_active[train_idx]
                sgkf_val = StratifiedGroupKFold(
                    n_splits=_n_splits_val, shuffle=True, random_state=sklearn_seed,
                )
                _train_local_train, _train_local_val = next(
                    sgkf_val.split(train_idx, _train_strat, groups=_train_groups),
                )
                val_idx = train_idx[_train_local_val]
                train_idx = train_idx[_train_local_train]
            elif _groups_arr is not None:
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

                # Verify partition invariants before commit. Using raise (not
                # assert) so the check survives `python -O` / `PYTHONOPTIMIZE=1`
                # which strips bare asserts - a partition bug would then
                # silently corrupt the split rather than fail loud.
                if (_new_train_mask & _new_val_mask).any():
                    raise RuntimeError(
                        "Group-spanning cutoff resolution: train and val masks overlap after promote"
                    )
                if (_new_train_mask & _new_test_mask).any():
                    raise RuntimeError(
                        "Group-spanning cutoff resolution: train and test masks overlap after promote"
                    )
                if (_new_val_mask & _new_test_mask).any():
                    raise RuntimeError(
                        "Group-spanning cutoff resolution: val and test masks overlap after promote"
                    )

                n_reassigned = int(_to_val_mask.sum() + _to_test_mask.sum())
                train_idx = np.flatnonzero(_new_train_mask).astype(np.intp)
                val_idx = np.flatnonzero(_new_val_mask).astype(np.intp)
                test_idx = np.flatnonzero(_new_test_mask).astype(np.intp)

                # Bucket the promote_to_test set by origin for the operator
                # message: groups that previously spanned train+test go from
                # the train side, while groups spanning val+test only leave
                # val. ``train_test_overlap`` and ``val_test_overlap`` can
                # intersect (3-way span) - count those as ``train+val->test``.
                _three_way = train_test_overlap & val_test_overlap
                _train_only_to_test = train_test_overlap - val_test_overlap
                _val_only_to_test = val_test_overlap - train_test_overlap
                logger.warning(
                    "Group-spanning cutoff resolution: %d row(s) from %d "
                    "spanning group(s) reassigned to the later split "
                    "(train->val: %d groups, train->test: %d groups, "
                    "val->test: %d groups, train+val->test: %d groups). "
                    "This preserves group integrity for LTR / per-group "
                    "metrics. To eliminate spanning, widen aging_limit so "
                    "the cutoff falls outside any group's timespan, or "
                    "drop spanning groups before calling the splitter.",
                    n_reassigned,
                    len(promote_to_val) + len(promote_to_test),
                    len(promote_to_val),
                    len(_train_only_to_test),
                    len(_val_only_to_test),
                    len(_three_way),
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

    # Calibration carve: take a disjoint slice from train ONLY, after all
    # train/val/test + group-spanning resolution, so it inherits the same
    # group-integrity / temporal-ordering guarantees. Base model is fit on the
    # shrunk train_idx -> calib rows are leakage-free for the calibrator.
    calib_idx = np.array([], dtype=train_idx.dtype)
    calib_details = ""
    _calib = calib_size if calib_size is not None else 0.0
    if _calib > 0:
        if not (0.0 < _calib < 1.0):
            raise ValueError(f"calib_size must be in (0, 1), got {calib_size}")
        train_idx, calib_idx = _carve_calib_from_train(
            train_idx, _calib, n_total=len(df), timestamps=timestamps, groups=groups, rng=rng,
        )
        # Hard leakage asserts (raise, not warn): calib must be disjoint from val AND test.
        if len(calib_idx) > 0:
            if np.intersect1d(calib_idx, test_idx, assume_unique=False).size > 0:
                raise RuntimeError("calib_size carve produced calib rows overlapping test_idx")
            if np.intersect1d(calib_idx, val_idx, assume_unique=False).size > 0:
                raise RuntimeError("calib_size carve produced calib rows overlapping val_idx")
            if np.intersect1d(calib_idx, train_idx, assume_unique=False).size > 0:
                raise RuntimeError("calib_size carve left calib rows still inside train_idx")
        if len(calib_idx) == 0:
            logger.warning("calib_size=%s requested but calib slice came out empty (n*calib_size<1 or train too small).", calib_size)
        elif timestamps is not None:
            calib_details = f"{len(calib_idx)} calib rows (oldest-train)"
        else:
            calib_details = f"{len(calib_idx)} calib rows"

    logger.info(
        "%d train rows %s, %d val rows %s, %d test rows %s%s.",
        len(train_idx), train_details,
        len(val_idx), val_details,
        len(test_idx), test_details,
        f", {len(calib_idx)} calib rows" if len(calib_idx) > 0 else "",
    )

    _base = (
        np.sort(train_idx),
        np.sort(val_idx),
        np.sort(test_idx),
        train_details,
        val_details,
        test_details,
    )
    if return_calib:
        return (*_base, np.sort(calib_idx), calib_details)
    return _base


__all__ = ["make_train_test_split", "_carve_calib_from_train"]

from ._split_helpers import _carve_calib_from_train, _stratified_split  # noqa: F401,E402
