"""Public-API consistency / validation regression tests for mlframe.metrics.

Each test pins a confirmed public-surface bug fix:
  * API-P0-1 -- fast_* wrappers raise ValueError on mismatched-length inputs
    (the numba kernels loop on len(y_true) and index y_pred[i] with bounds
    checking off, so a mismatch silently read out-of-bounds garbage).
  * API13 -- fast_brier_score_loss returns NaN on out-of-[0,1] / NaN probs,
    mirroring fast_log_loss_binary.
  * API19 -- fast_aucs raises on sample_weight rather than silently ignoring it.
  * API20 -- ranking.py family accepts group_ids=None as a single group,
    matching the _ranking_extras family.
  * API8 -- _split_by_group returns a consistent 3-tuple on empty input.
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# API-P0-1: equal-length validation in fast_* wrappers
# --------------------------------------------------------------------------- #


def test_fast_roc_auc_mismatched_lengths_raises():
    from mlframe.metrics._core_auc_brier import fast_roc_auc

    yt = np.array([0, 1, 0, 1, 1])
    ys = np.array([0.1, 0.9, 0.2])  # shorter on purpose
    with pytest.raises(ValueError):
        fast_roc_auc(yt, ys)


def test_fast_brier_mismatched_lengths_raises():
    from mlframe.metrics._core_auc_brier import fast_brier_score_loss

    yt = np.array([0, 1, 0, 1])
    yp = np.array([0.1, 0.9])
    with pytest.raises(ValueError):
        fast_brier_score_loss(yt, yp)


def test_fast_mae_mismatched_lengths_raises():
    from mlframe.metrics.regression._regression_metrics import fast_mean_absolute_error

    with pytest.raises(ValueError):
        fast_mean_absolute_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_fast_mse_mismatched_lengths_raises():
    from mlframe.metrics.regression._regression_metrics import fast_mean_squared_error

    with pytest.raises(ValueError):
        fast_mean_squared_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_fast_rmsle_mismatched_lengths_raises():
    from mlframe.metrics.regression._regression_extras import fast_rmsle

    with pytest.raises(ValueError):
        fast_rmsle(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_fast_spearman_mismatched_lengths_raises():
    from mlframe.metrics.regression._regression_extras import fast_spearman_corr

    with pytest.raises(ValueError):
        fast_spearman_corr(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0]))


def test_fast_kendall_mismatched_lengths_raises():
    from mlframe.metrics.regression._regression_extras import fast_kendall_tau

    with pytest.raises(ValueError):
        fast_kendall_tau(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0]))


def test_fast_binary_confusion_block_mismatched_lengths_raises():
    from mlframe.metrics.classification._classification_extras_blocks import (
        fast_binary_confusion_metrics_block,
    )

    with pytest.raises(ValueError):
        fast_binary_confusion_metrics_block(np.array([0, 1, 0, 1]), np.array([0, 1]))


def test_fast_binary_probability_block_mismatched_lengths_raises():
    from mlframe.metrics.classification._classification_extras_blocks import (
        fast_binary_probability_metrics_block,
    )

    with pytest.raises(ValueError):
        fast_binary_probability_metrics_block(np.array([0, 1, 0, 1]), np.array([0.2, 0.8]))


def test_fast_multiclass_confusion_block_mismatched_lengths_raises():
    from mlframe.metrics.classification._classification_extras_blocks import (
        fast_multiclass_confusion_metrics_block,
    )

    with pytest.raises(ValueError):
        fast_multiclass_confusion_metrics_block(np.array([0, 1, 2, 1]), np.array([0, 1]), 3)


def test_equal_length_inputs_still_work():
    """Sanity: equal-length inputs are NOT broken by the new guard."""
    from mlframe.metrics.regression._regression_metrics import fast_mean_absolute_error

    val = fast_mean_absolute_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert abs(val - (1.0 / 3.0)) < 1e-9


# --------------------------------------------------------------------------- #
# API13: Brier NaN guard mirrors log-loss
# --------------------------------------------------------------------------- #


def test_brier_returns_nan_on_out_of_range_prob():
    from mlframe.metrics._core_auc_brier import fast_brier_score_loss
    from mlframe.metrics._log_loss_and_separation import fast_log_loss_binary

    yt = np.array([0.0, 1.0, 0.0, 1.0])
    yp_bad = np.array([0.2, 1.7, 0.1, 0.9])  # 1.7 out of [0,1]
    assert np.isnan(fast_brier_score_loss(yt, yp_bad))
    # Sibling contract for reference.
    assert np.isnan(fast_log_loss_binary(yt, yp_bad))


def test_brier_returns_nan_on_nan_prob():
    from mlframe.metrics._core_auc_brier import fast_brier_score_loss

    yt = np.array([0.0, 1.0, 0.0, 1.0])
    yp = np.array([0.2, np.nan, 0.1, 0.9])
    assert np.isnan(fast_brier_score_loss(yt, yp))


def test_brier_valid_range_unchanged():
    from mlframe.metrics._core_auc_brier import fast_brier_score_loss

    yt = np.array([0.0, 1.0, 0.0, 1.0])
    yp = np.array([0.0, 1.0, 0.0, 1.0])
    assert fast_brier_score_loss(yt, yp) == 0.0


# --------------------------------------------------------------------------- #
# API19: fast_aucs sample_weight is not silently ignored
# --------------------------------------------------------------------------- #


def test_fast_aucs_rejects_sample_weight():
    from mlframe.metrics._core_auc_brier import fast_aucs

    yt = np.array([0, 1, 0, 1, 1, 0])
    ys = np.array([0.1, 0.9, 0.3, 0.8, 0.7, 0.2])
    w = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(NotImplementedError):
        fast_aucs(yt, ys, sample_weight=w)


def test_fast_aucs_without_weight_still_works():
    from mlframe.metrics._core_auc_brier import fast_aucs

    yt = np.array([0, 1, 0, 1, 1, 0])
    ys = np.array([0.1, 0.9, 0.3, 0.8, 0.7, 0.2])
    roc, pr = fast_aucs(yt, ys)
    assert 0.0 <= roc <= 1.0
    assert 0.0 <= pr <= 1.0


# --------------------------------------------------------------------------- #
# API20: ranking.py family accepts group_ids=None as a single group
# --------------------------------------------------------------------------- #


def test_ranking_none_group_ids_treated_as_single_group():
    from mlframe.metrics.ranking import ndcg_at_k
    from mlframe.metrics._ranking_extras import dcg_at_k

    y_true = np.array([3.0, 2.0, 0.0, 1.0])
    y_score = np.array([0.9, 0.4, 0.1, 0.7])
    # Must NOT raise (was a TypeError on len(None) before the fix).
    val_none = ndcg_at_k(y_true, y_score, None, k=4)
    # Identical to passing a single explicit group spanning all rows.
    val_one = ndcg_at_k(y_true, y_score, np.zeros(4, dtype=int), k=4)
    assert np.isfinite(val_none)
    assert abs(val_none - val_one) < 1e-9
    # _ranking_extras already supported None -- confirm the two families agree on the contract.
    assert np.isfinite(dcg_at_k(y_true, y_score, None, k=4))


# --------------------------------------------------------------------------- #
# API8: _split_by_group consistent 3-tuple on empty input
# --------------------------------------------------------------------------- #


def test_split_by_group_empty_unpacks_as_triple():
    from mlframe.metrics._ranking_extras import _split_by_group

    boundaries, yt, ys = _split_by_group(
        np.empty(0), np.empty(0), np.empty(0, dtype=int)
    )  # must not ValueError on unpack
    assert boundaries.shape[0] == 0


def test_ranking_extras_empty_input_returns_nan_not_crash():
    from mlframe.metrics._ranking_extras import dcg_at_k

    val = dcg_at_k(np.empty(0), np.empty(0), np.empty(0, dtype=int), k=10)
    assert np.isnan(val)
