"""The parallel metric reductions must return the same number every time they are called on the same input.

They used to accumulate a bare ``prange`` float reduction, so the per-thread partials were combined in completion
order: five runs of the same 300k-row calibration report returned five distinct Brier values (…855, …885, …846, …990,
…888), ICE inherited the drift, and the report's "byte-identical" contract failed at random.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._core_brier import _fast_brier_checked_par, _fast_brier_score_loss_par, fast_brier_score_loss


@pytest.mark.parametrize("n", [300_000, 1_000_003])
def test_the_parallel_kernel_is_reproducible(n):
    rng = np.random.default_rng(11)
    y_true = (rng.random(n) < 0.5).astype(np.float64)
    y_prob = rng.random(n)
    values = {repr(fast_brier_score_loss(y_true, y_prob)) for _ in range(5)}
    assert len(values) == 1, f"the same input returned {len(values)} different Brier values: {sorted(values)}"


def test_both_parallel_kernels_agree_with_each_other():
    rng = np.random.default_rng(3)
    n = 400_000
    y_true = (rng.random(n) < 0.3).astype(np.float64)
    y_prob = rng.random(n)
    assert _fast_brier_checked_par(y_true, y_prob) == _fast_brier_score_loss_par(y_true, y_prob)


def test_the_parallel_kernel_stays_within_fp_tolerance_of_the_sequential_one():
    """Chunked and straight-line summation of the same 400k terms differ in the last ulps; they must not differ more."""
    from mlframe.metrics._core_brier import _fast_brier_checked_seq

    rng = np.random.default_rng(5)
    n = 400_000
    y_true = (rng.random(n) < 0.7).astype(np.float64)
    y_prob = rng.random(n)
    assert _fast_brier_checked_par(y_true, y_prob) == pytest.approx(float(_fast_brier_checked_seq(y_true, y_prob)), rel=1e-12)


def test_an_invalid_probability_still_reaches_the_nan_guard_from_any_chunk():
    rng = np.random.default_rng(7)
    n = 300_000
    y_true = (rng.random(n) < 0.5).astype(np.float64)
    for pos in (0, n // 3, n - 1):
        y_prob = rng.random(n)
        y_prob[pos] = 1.5
        assert np.isnan(fast_brier_score_loss(y_true, y_prob)), f"an out-of-range probability at {pos} must produce NaN"


@pytest.mark.parametrize("n", [300_000, 1_000_003])
def test_the_parallel_log_loss_is_reproducible(n):
    """Same defect, same shape: the reported `ll` moved in its last ulps between identical runs, and the calibration
    report's byte-identity check failed at random because of it."""
    from mlframe.metrics._log_loss_and_separation import fast_log_loss_binary

    rng = np.random.default_rng(11)
    y_true = (rng.random(n) < 0.4).astype(np.int64)
    y_pred = rng.random(n)
    values = {repr(fast_log_loss_binary(y_true, y_pred)) for _ in range(5)}
    assert len(values) == 1, f"the same input returned {len(values)} different log-loss values: {sorted(values)}"


def test_the_parallel_separation_score_is_reproducible():
    from mlframe.metrics._log_loss_and_separation import probability_separation_score

    rng = np.random.default_rng(13)
    n = 500_000
    y_true = (rng.random(n) < 0.6).astype(np.int64)
    y_prob = rng.random(n)
    values = {repr(probability_separation_score(y_true, y_prob)) for _ in range(5)}
    assert len(values) == 1, f"the same input returned {len(values)} different separation scores: {sorted(values)}"


def test_the_full_calibration_report_is_reproducible():
    """The end-to-end contract the drift broke: two reports on identical inputs must agree element by element."""
    from mlframe.metrics.classification._classification_report import fast_calibration_report

    rng = np.random.default_rng(11)
    n = 300_000
    z = rng.normal(0, 1, n)
    y_true = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.int64)
    y_pred = (1 / (1 + np.exp(-(z + rng.normal(0, 0.5, n))))).astype(np.float64)
    first = fast_calibration_report(y_true, y_pred, show_plots=False)[:-2]
    second = fast_calibration_report(y_true, y_pred, show_plots=False)[:-2]
    assert list(first) == list(second)
