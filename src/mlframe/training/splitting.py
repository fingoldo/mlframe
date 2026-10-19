"""
Train/validation/test splitting utilities for mlframe.

Provides flexible data splitting with support for:
- Sequential and shuffled splits
- Date-based (whole-day) splitting
- Training set aging limits
"""

from __future__ import annotations


import logging
from typing import Literal, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

from mlframe.training._splitting_helpers import _calculate_split_sizes

# Derived from the timestamps and the requested sizes, so it describes the layout BEFORE a random-row val top-up moves
# the boundaries; the realised split is reported separately.
_IMPLIED_FORWARD_LAYOUT_MSG = (
    "Implied temporal layout BEFORE any val augmentation (val_placement='forward', default; derived "
    "from the timestamps and the requested sizes, so a later random-row top-up moves these "
    "boundaries -- the realised split is reported separately below): train_max=%s, val=[%s..%s], "
    "test=[%s..%s], val->train_gap=%.2fd, train->prod_estimated_gap=%.2fd. Mazzanti backward layout "
    "typically gives a better prod-error proxy under drift; set val_placement='backward' to switch."
)


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
) -> Tuple:
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

        stratify_y: Optional class labels to stratify the split on (forwarded to sklearn's
            ``train_test_split`` on the no-timestamps path; ignored on the timestamp-based paths).
        groups: Optional group labels so a group never straddles the train/val/test boundary
            (whole groups move together, same as ``calib_size``'s group-aware carve).
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

    # Input validation (proactive probe pass).
    # Previously negative ``val_size``/``test_size`` silently produced
    # empty splits; ``trainset_aging_limit=0`` silently no-op'd despite
    # the non-zero-only validator below contradicting that behaviour.
    from mlframe.training._splitting_validation import _validate_split_arguments
    _validate_split_arguments(test_size, val_size, trainset_aging_limit, val_placement)

    # Session 7 batch 7: normalise ``timestamps`` to pd.Series
    # at function entry. Several downstream branches use ``timestamps.iloc``
    # / ``.dt`` accessors that only work on pandas Series. Callers may
    # pass either a Series (FTE.transform on pandas df) or a numpy
    # datetime64 ndarray (FTE.transform on polars df -> ``.to_numpy()``)
    # or a plain Python list. Coerce once so the rest of the function
    # has a stable contract.
    if timestamps is not None and not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)

    # explicit log: surface group-aware splitting status so the
    # operator sees at a glance whether GroupShuffleSplit is in effect.
    # Pre-fix this was implicit ("if groups is not None: ..."). Users with
    # ``SimpleFeaturesAndTargetsExtractor(group_field=...)`` could not tell
    # from the log whether the group_field propagated through to the
    # splitter.
    _make_train_test_sp_splitter(groups)

    # Backward placement is time-axis-specific. Without timestamps there is
    # no "before/after" to place val relative to train, so we silently fall
    # back to forward -- the sklearn-shuffle path below doesn't order rows
    # by time anyway. ``val_size=0`` makes placement moot too.
    _effective_val_placement = val_placement
    if timestamps is None or val_size == 0:
        _effective_val_placement = "forward"

    # Implied-temporal-layout INFO: when caller supplied timestamps AND default val_placement="forward" (a quiet default that assumes iid-friendly target), surface the resulting val/train/test ranges + the val->train gap + estimated train->prod gap so the time-series user can see at a glance how their split lays out. Cheap one-time log; never auto-flips the default.
    _make_train_test_sp_implied_temporal_layout_info(timestamps, val_placement, _effective_val_placement, val_size, test_size)

    # Diagnostic: surface the effective placement at INFO so a user who
    # passed ``val_placement="backward"`` but sees a forward-style split
    # in the log can immediately confirm whether (a) the value reached
    # this function at all (config wiring), (b) it was downgraded to
    # forward by the no-timestamps / val_size=0 fallback above, or
    # (c) the placement is honoured but the layout still looks
    # forward-ish for some other reason. Caller-attributed log so the
    # line shows up next to the existing "{N} train rows ... val rows
    # ..." summary at the bottom.
    _make_train_test_sp_summary_bottom(val_placement, _effective_val_placement, groups, timestamps, val_size)

    if _effective_val_placement == "backward" and trainset_aging_limit is not None:
        # Aging trims the OLDEST train rows -- which in backward layout are
        # the ones closest to the (earlier) val block. Trimming them
        # widens the val->train gap silently, defeating the "gap mirror"
        # whole point of the Mazzanti split. Refuse explicitly rather
        # than produce a subtly wrong split.
        #
        # This raise only guards genuine backward placement; it relies on the downgrade-to-forward fallback a few
        # lines above (the "no timestamps / val_size=0" branch that rewrites _effective_val_placement to "forward")
        # having already run. Removing either guard alone reopens a silent temporal-honesty-loss path: without the
        # downgrade, a no-timestamp caller could reach "backward" semantics that were never actually honoured;
        # without this raise, a genuine backward split combined with aging would silently widen the val-to-train
        # gap. Keep both checks together if this function is ever refactored.
        raise ValueError(
            "val_placement='backward' is incompatible with "
            "trainset_aging_limit -- aging removes the oldest train rows, "
            "which are the ones adjacent to the backward-placed val. "
            "Drop one of the two."
        )

    _base, calib_details, calib_idx = _make_train_test_sp_wholeday_splitting_timestamps_none(wholeday_splitting, timestamps, df, test_size, val_size, shuffle_test, test_sequential_fraction, shuffle_val, val_sequential_fraction, rng, _effective_val_placement, trainset_aging_limit, groups, stratify_y, sklearn_seed, calib_size)
    if return_calib:
        return (*_base, np.sort(calib_idx), calib_details)
    return tuple(_base)


def _make_train_test_sp_wholeday_splitting_timestamps_none(wholeday_splitting, timestamps, df, test_size, val_size, shuffle_test, test_sequential_fraction, shuffle_val, val_sequential_fraction, rng, _effective_val_placement, trainset_aging_limit, groups, stratify_y, sklearn_seed, calib_size):
    """Block of make_train_test_split starting at ``if wholeday_splitting and timestamps is not None:``."""
    val_idx: Any = None
    train_idx: Any = None
    test_idx: Any = None
    if wholeday_splitting and timestamps is not None:
        # `.dt.floor('D')` is vectorized over datetime64 and stays in datetime dtype
        # (unlike `.dt.date` which yields a Python-object Series -- much slower for isin).
        # ``pd.to_datetime`` on a numpy datetime64 ndarray returns a
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
        from mlframe.training._splitting_wholeday import _split_by_wholeday
        test_details, test_idx, train_details, train_idx, val_details, val_idx = _split_by_wholeday(unique_dates, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, _effective_val_placement, trainset_aging_limit, dates, timestamps, test_idx, train_idx, val_idx)

    elif timestamps is not None:
        from mlframe.training._splitting_rowbased import _split_row_based_with_timestamps
        test_details, test_idx, train_details, train_idx, val_details, val_idx = _split_row_based_with_timestamps(timestamps, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, _effective_val_placement, trainset_aging_limit, test_idx, train_idx, val_idx)

    else:
        from mlframe.training._splitting_rowbased import _split_row_based_without_timestamps
        test_details, test_idx, train_details, train_idx, val_details, val_idx = _split_row_based_without_timestamps(df, groups, stratify_y, timestamps, test_size, val_size, sklearn_seed, shuffle_test, val_idx, shuffle_val, trainset_aging_limit)

    # Group-spans-cutoff resolution for time-based splits.
    # When ``groups`` is supplied alongside ``timestamps`` (typical LTR
    # scenario where queries are time-stamped), a query may straddle a
    # train->val or val->test cutoff. Default behaviour:
    # assign the WHOLE spanning group to the LATER side -- preserves
    # temporal ordering, so train never sees rows from a query that
    # leaked into val/test. Emit a single INFO summarising how many
    # groups were reassigned. The row-based path with ``groups`` already
    # uses GroupShuffleSplit so groups stay together by construction --
    # this block only matters when timestamps drive the split AND groups
    # are also supplied.
    test_idx, train_idx, val_idx = _make_train_test_sp_also_supplied(groups, timestamps, train_idx, val_idx, test_idx, df)

    from mlframe.training._splitting_validation import _assert_nonempty_splits
    _assert_nonempty_splits(test_size, train_idx, df, wholeday_splitting, timestamps, val_size, trainset_aging_limit, val_idx, val_sequential_fraction, test_idx, test_sequential_fraction)

    # Calibration carve: take a disjoint slice from train ONLY, after all
    # train/val/test + group-spanning resolution, so it inherits the same
    # group-integrity / temporal-ordering guarantees. Base model is fit on the
    # shrunk train_idx -> calib rows are leakage-free for the calibrator.
    calib_idx = np.array([], dtype=train_idx.dtype)
    calib_details = ""
    _calib = calib_size if calib_size is not None else 0.0
    calib_details, calib_idx, train_idx = _make_train_test_sp_calib(_calib, calib_size, train_idx, df, timestamps, groups, rng, test_idx, val_idx, calib_details, calib_idx)

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
    return _base, calib_details, calib_idx


def _make_train_test_sp_splitter(groups):
    """Block of make_train_test_split starting at ``if groups is not None:``."""
    if groups is not None:
        try:
            _n_groups = int(np.unique(np.asarray(groups)).shape[0])
        except Exception as e:
            logger.debug("group-count computation failed: %s", e)
            _n_groups = -1
        logger.info(
            "Group-aware splitting: ENABLED (n_groups=%d). " "Each group stays within ONE split (no per-row leakage).",
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


def _make_train_test_sp_implied_temporal_layout_info(timestamps, val_placement, _effective_val_placement, val_size, test_size):
    """Block of make_train_test_split starting at ``if timestamps is not None and val_placement == "forward" and _effectiv``."""
    if timestamps is not None and val_placement == "forward" and _effective_val_placement == "forward" and val_size > 0:
        try:
            _ts = pd.to_datetime(timestamps, errors="coerce")
            _n_nat = int(pd.isna(_ts).sum())
            if _n_nat > 0:
                logger.debug("splitting: temporal-layout diagnostic coerced %d unparseable timestamp(s) to NaT.", _n_nat)
            _ts_sorted = _ts.sort_values().reset_index(drop=True)
            _n = len(_ts_sorted)
            if _n >= 3:
                _n_test = max(1, round(_n * test_size)) if test_size > 0 else 0
                _n_val = max(1, round(_n * val_size))
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
                    _IMPLIED_FORWARD_LAYOUT_MSG,
                    _t_train_max, _t_val_min, _t_val_max, _t_test_min, _t_test_max,
                    _gap_val_train_days, _gap_train_prod_days,
                )
        except Exception as _layout_err:
            logger.debug("Temporal-layout INFO compute failed: %s", _layout_err)


def _make_train_test_sp_summary_bottom(val_placement, _effective_val_placement, groups, timestamps, val_size):
    """Block of make_train_test_split starting at ``if val_placement != _effective_val_placement:``."""
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
            except Exception as e:
                logger.debug("group-count computation failed: %s", e)
                _ng = -1
            _group_clause = f"; groups (n_groups={_ng}) still kept whole per split"
        else:
            _group_clause = ""
        _reason = "no timestamps_column" if timestamps is None else f"val_size={val_size}"
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


def _make_train_test_sp_consistent_row_timestamp_branch(trainset_aging_limit, train_dates):
    """Block of make_train_test_split starting at ``if trainset_aging_limit is not None:``."""
    if trainset_aging_limit is not None:
        n_dates_to_keep = int(len(train_dates) * trainset_aging_limit)
        if n_dates_to_keep > 0:
            train_dates = np.sort(train_dates)[-n_dates_to_keep:]
    return train_dates


def _make_train_test_sp_lbl_split_dates_train(train_dates, val_dates, test_dates, _ns_keys, date_to_code, label_lut):
    """Block of make_train_test_split starting at ``for _lbl, _split_dates in ((0, train_dates), (1, val_dates), (2, test_``."""
    for _lbl, _split_dates in ((0, train_dates), (1, val_dates), (2, test_dates)):
        for k in _ns_keys(_split_dates):
            _i = date_to_code.get(k)
            if _i is not None:
                label_lut[_i] = _lbl


def _make_train_test_sp_though_shuffling_was_originally(_fully_sequential, train_idx, timestamps, _effective_val_placement, val_idx, test_idx):
    """Block of make_train_test_split starting at ``if _fully_sequential and len(train_idx):``."""
    if _fully_sequential and len(train_idx):
        _ts_vals = timestamps.values
        if _effective_val_placement == "backward":
            # Layout [val(oldest)] [train] [test]: de-leak val|train then train|test.
            val_idx, train_idx, test_idx = _deleak_tied_boundaries(
                _ts_vals, val_idx, train_idx, test_idx,
            )
        else:
            train_idx, val_idx, test_idx = _deleak_tied_boundaries(
                _ts_vals, train_idx, val_idx, test_idx,
            )
    return test_idx, train_idx, val_idx


def _make_train_test_sp_groupshufflesplit_groups_precedence_warn(_strat_groups_active, stratify_y, groups):
    """Block of make_train_test_split starting at ``if _strat_groups_active:``."""
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
                # iterative-stratification absent: rather than drop ALL
                # stratification (the pre-fix behaviour silently degraded to
                # GroupShuffleSplit, losing every label's balance), derive a
                # single 1-D composite class id from the K label columns and
                # route through sklearn StratifiedGroupKFold. This preserves
                # the JOINT label-combination balance across splits while
                # keeping whole groups together. Gated on the derived
                # cardinality being small enough for StratifiedGroupKFold
                # (>=2 distinct ids, each appearing on >=2 groups); otherwise
                # fall back to plain GroupShuffleSplit and WARN that multilabel
                # proportions are not enforced.
                _derived_ok = False
                try:
                    _, _derived_ids = np.unique(_strat_arr, axis=0, return_inverse=True)
                    _derived_ids = np.asarray(_derived_ids).ravel()
                    _du = np.unique(_derived_ids)
                    # Per-class group counts: StratifiedGroupKFold needs each
                    # class spread over enough groups to place it in every fold.
                    _grp_arr = np.asarray(groups)
                    _min_groups_per_class = min(int(np.unique(_grp_arr[_derived_ids == _c]).shape[0]) for _c in _du) if len(_du) else 0
                    if 2 <= len(_du) <= 200 and _min_groups_per_class >= 2:
                        stratify_y = _derived_ids
                        _derived_ok = True
                        logger.warning(
                            "make_train_test_split: multilabel stratify_y + groups but "
                            "iterative-stratification not installed; stratifying on a derived "
                            "1-D composite label-combination id (%d classes) via "
                            "StratifiedGroupKFold instead of dropping stratification. Per-label "
                            "marginals are NOT individually enforced (only the joint combination "
                            "is); pip install iterative-stratification for exact multilabel "
                            "MultilabelStratifiedGroupKFold.",
                            len(_du),
                        )
                except Exception as _derive_err:
                    logger.warning(
                        "make_train_test_split: derived-label fallback for multilabel+groups " "failed (%s: %s); dropping stratification.",
                        type(_derive_err).__name__,
                        _derive_err,
                    )
                if not _derived_ok:
                    logger.warning(
                        "make_train_test_split: multilabel stratify_y + groups supplied but "
                        "iterative-stratification not installed AND a derived composite label "
                        "is unusable (too many combinations or too few groups per class). "
                        "Falling back to GroupShuffleSplit -- MULTILABEL CLASS PROPORTIONS ARE "
                        "NOT PRESERVED across train/val/test; rare labels may be absent from a "
                        "split. pip install iterative-stratification to enable "
                        "MultilabelStratifiedGroupKFold.",
                    )
                    _strat_groups_active = False
                    stratify_y = None
    return _strat_groups_active, stratify_y


def _make_train_test_sp_stratify_none_timestamps_none(stratify_y, timestamps, df):
    """Block of make_train_test_split starting at ``if stratify_y is not None and timestamps is not None:``."""
    if stratify_y is not None and timestamps is not None:
        logger.warning("stratify_y provided but timestamps active -- stratification " "ignored (stratification is ill-defined for time-based splits).")
        _stratify_active = None
    elif stratify_y is not None:
        _stratify_active = np.asarray(stratify_y)
        if _stratify_active.shape[0] != len(df):
            raise ValueError(f"stratify_y length {_stratify_active.shape[0]} does not " f"match df length {len(df)}")
        if _stratify_active.ndim not in (1, 2):
            raise ValueError(f"stratify_y must be 1-D (single-label) or 2-D (multilabel), " f"got shape {_stratify_active.shape}")
    else:
        _stratify_active = None
    return _stratify_active


def _make_train_test_sp_validate_length_once_shape(groups, df, _groups_arr):
    """Block of make_train_test_split starting at ``if groups is not None:``."""
    if groups is not None:
        _groups_arr = np.asarray(groups)
        if _groups_arr.shape[0] != len(df):
            raise ValueError(f"groups length {_groups_arr.shape[0]} does not match " f"df length {len(df)}")
        if _groups_arr.ndim != 1:
            raise ValueError(f"groups must be 1-D (one query-id per row), got shape " f"{_groups_arr.shape}")
    return _groups_arr


def _make_train_test_sp_multilabel_single_greedy_way(_ml_3way_done, _stratify_active, all_idx, test_size, val_size, sklearn_seed, _groups_arr, _strat_groups_active, shuffle_test, val_idx):
    """Block of make_train_test_split starting at ``if _ml_3way_done:``."""
    if _ml_3way_done:
        assert _stratify_active is not None  # guaranteed by _use_multilabel_3way's own not-None check
        train_idx, val_idx, test_idx = _stratified_split_3way(
            all_idx, test_size=test_size, val_size=val_size,
            stratify_y=_stratify_active, random_state=sklearn_seed,
        )
    elif test_size > 0:
        if _groups_arr is not None and _strat_groups_active and _stratify_active is not None and _stratify_active.ndim == 1:
            # Both constraints simultaneously: stratify by target
            # bucket / class AND keep whole groups together. sklearn
            # ``StratifiedGroupKFold`` requires n_splits >= 2; we
            # derive n_splits from the requested test_size and take
            # the first fold as test (groups + class proportions
            # preserved by construction).
            from sklearn.model_selection import StratifiedGroupKFold
            _n_splits_test = max(2, round(1.0 / max(test_size, 1e-9)))
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
    return test_idx, train_idx, val_idx


def _make_train_test_sp_ml_way_done(_ml_3way_done, val_size, _groups_arr, _strat_groups_active, _stratify_active, train_idx, sklearn_seed, shuffle_val, val_idx):
    """Block of make_train_test_split starting at ``if _ml_3way_done:``."""
    if _ml_3way_done:
        pass
    elif val_size > 0:
        if _groups_arr is not None and _strat_groups_active and _stratify_active is not None and _stratify_active.ndim == 1:
            # Same StratifiedGroupKFold strategy as the test split,
            # restricted to the post-test train rows. val_size is
            # interpreted as a fraction of the REMAINING train pool
            # (matching the existing GroupShuffleSplit semantics).
            from sklearn.model_selection import StratifiedGroupKFold
            _n_splits_val = max(2, round(1.0 / max(val_size, 1e-9)))
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
            _train_local_train, _train_local_val = next(gss_val.split(train_idx, groups=_train_groups))
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
    return train_idx, val_idx


def _make_train_test_sp_also_supplied(groups, timestamps, train_idx, val_idx, test_idx, df):
    """Block of make_train_test_split starting at ``if groups is not None and timestamps is not None:``."""
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
                    raise RuntimeError("Group-spanning cutoff resolution: train and val masks overlap after promote")
                if (_new_train_mask & _new_test_mask).any():
                    raise RuntimeError("Group-spanning cutoff resolution: train and test masks overlap after promote")
                if (_new_val_mask & _new_test_mask).any():
                    raise RuntimeError("Group-spanning cutoff resolution: val and test masks overlap after promote")

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
    return test_idx, train_idx, val_idx


def _make_train_test_sp_calib(_calib, calib_size, train_idx, df, timestamps, groups, rng, test_idx, val_idx, calib_details, calib_idx):
    """Block of make_train_test_split starting at ``if _calib > 0:``."""
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
    return calib_details, calib_idx, train_idx


__all__ = ["make_train_test_split", "_carve_calib_from_train"]

from ._split_helpers import (
    _carve_calib_from_train,
    _deleak_tied_boundaries,
    _stratified_split,
    _stratified_split_3way,
)
