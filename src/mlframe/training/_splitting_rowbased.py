"""Stages of ``_make_train_test_sp_wholeday_splitting_timestamps_none``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.training._split_helpers import _use_multilabel_3way
from mlframe.training._splitting_helpers import _build_details
from mlframe.training._splitting_helpers import _perform_split
from mlframe.training._splitting_helpers import fmt_ts
from mlframe.training.splitting import _make_train_test_sp_groupshufflesplit_groups_precedence_warn
from mlframe.training.splitting import _make_train_test_sp_ml_way_done
from mlframe.training.splitting import _make_train_test_sp_multilabel_single_greedy_way
from mlframe.training.splitting import _make_train_test_sp_stratify_none_timestamps_none
from mlframe.training.splitting import _make_train_test_sp_though_shuffling_was_originally
from mlframe.training.splitting import _make_train_test_sp_validate_length_once_shape
# --- end imports ---


def _split_row_based_without_timestamps(df, groups, stratify_y, timestamps, test_size, val_size, sklearn_seed, shuffle_test, val_idx, shuffle_val, trainset_aging_limit):
    """Stage of ``_make_train_test_sp_wholeday_splitting_timestamps_none``."""
    # Row-based splitting without timestamps (fallback to sklearn).
    # stratify_y support -- when provided, route through
    # sklearn StratifiedShuffleSplit (1-D y) or
    # iterstrat.MultilabelStratifiedShuffleSplit (2-D y, multilabel).
    # Both REQUIRE shuffle (cannot stratify a sequential split). When
    # stratify_y is provided AND shuffle_test/shuffle_val is False,
    # we emit a WARNING and stratify anyway (correctness > sequential
    # nicety; user explicitly asked for class balance).
    # ``groups`` support -- when provided (typically for
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
    _strat_groups_active, stratify_y = _make_train_test_sp_groupshufflesplit_groups_precedence_warn(_strat_groups_active, stratify_y, groups)

    _stratify_active = _make_train_test_sp_stratify_none_timestamps_none(stratify_y, timestamps, df)

    # Group-aware splitter: GroupShuffleSplit on the row-based path.
    # Validate length once; the shape contract is per-row (len == len(df)).
    _groups_arr = None
    _groups_arr = _make_train_test_sp_validate_length_once_shape(groups, df, _groups_arr)

    # Multilabel single greedy 3-way pass replaces two full O(n*K*iters) carves (no groups).
    _ml_3way_done = _use_multilabel_3way(_groups_arr, _stratify_active, test_size, val_size)
    test_idx, train_idx, val_idx = _make_train_test_sp_multilabel_single_greedy_way(_ml_3way_done, _stratify_active, all_idx, test_size, val_size, sklearn_seed, _groups_arr, _strat_groups_active, shuffle_test, val_idx)

    train_idx, val_idx = _make_train_test_sp_ml_way_done(_ml_3way_done, val_size, _groups_arr, _strat_groups_active, _stratify_active, train_idx, sklearn_seed, shuffle_val, val_idx)

    if trainset_aging_limit is not None:
        train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)) :]

    train_details, val_details, test_details = "", "", ""
    return test_details, test_idx, train_details, train_idx, val_details, val_idx


def _split_row_based_with_timestamps(timestamps, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, _effective_val_placement, trainset_aging_limit, test_idx, train_idx, val_idx):
    """Stage of ``_make_train_test_sp_wholeday_splitting_timestamps_none``."""
    # Row-based splitting with timestamps. Use ``kind="stable"`` so that
    # ties keep their original positional order; default quicksort
    # reshuffles ties run-to-run which makes seeded splits non-reproducible.
    sorted_idx = np.argsort(timestamps.values, kind="stable")
    train_idx, val_idx, test_idx, val_idx_seq, test_idx_seq, eff_test_shuf, eff_val_shuf = _perform_split(
        sorted_idx, n_test_seq, n_test_shuf, n_val_seq, n_val_shuf, rng, _effective_val_placement
    )

    # Boundary de-leak: positional slicing cuts THROUGH a block of rows
    # sharing one timestamp (common in multi-entity panels where many
    # entities share each timestamp), leaving the same timestamp value in
    # BOTH train and the later split -- a time leak (model validated/tested
    # on a timestamp it trained on). Snap any boundary-tied rows to the
    # LATER split so max(train_ts) < min(val_ts) <= min(test_ts) strictly.
    # Only for the fully-sequential path: shuffled / partial-sequential
    # portions intentionally interleave timestamps, so a tie split there is
    # by request, not a bug. Uses the EFFECTIVE (post-clamp) counts: if the
    # shuffled pool was exhausted and _perform_split clamped both to 0, the
    # actual output IS fully sequential and needs the de-leak fix even
    # though shuffling was originally requested.
    _fully_sequential = eff_test_shuf == 0 and eff_val_shuf == 0
    test_idx, train_idx, val_idx = _make_train_test_sp_though_shuffling_was_originally(_fully_sequential, train_idx, timestamps, _effective_val_placement, val_idx, test_idx)

    # Apply aging limit
    if trainset_aging_limit is not None:
        train_idx = train_idx[int(len(train_idx) * (1 - trainset_aging_limit)) :]

    # Build detail strings (same NaT-on-empty guard as above; also
    # numeric-ts safe via the ``fmt_ts`` fallback).
    if len(train_idx) > 0:
        train_details = f"{fmt_ts(timestamps.iloc[train_idx].min())}/{fmt_ts(timestamps.iloc[train_idx].max())}"
    else:
        train_details = "(empty)"
    val_details = _build_details(timestamps, val_idx, val_idx_seq, eff_val_shuf, "Rnd")
    test_details = _build_details(timestamps, test_idx, test_idx_seq, eff_test_shuf, "Rnd")
    return test_details, test_idx, train_details, train_idx, val_details, val_idx
