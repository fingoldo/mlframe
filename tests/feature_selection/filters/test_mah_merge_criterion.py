"""MAH bin merging counts the code length of the OTHER variable given the bins, and uses the multinomial NML regret.

Before 2026-11-22 the merge criterion was the NML code length of the whole joint table and the regret had the wrong scale and signs: merging cells shortens that code whatever the data, so every column
was merged down to its first two quantile bins (one edge at the 1/16 quantile, even for pure noise, and 1-2 edges for the real signal). These tests pin the repaired behaviour on the synthetic the
benchmark uses: no edge for a column without signal, cuts for a monotone, a U-shaped and a weak column, and an MI close to the MDLP binning's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters import _mah
from mlframe.feature_selection.filters._adaptive_nbins import edges_fayyad_irani


def _mi(codes, y):
    """Plug-in MI (nats) of integer codes and a binary label."""
    joint = np.zeros((int(codes.max()) + 1, 2))
    np.add.at(joint, (codes, y), 1)
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log(p[m] / (px @ py)[m])).sum())


@pytest.fixture(scope="module")
def data():
    """The benchmark synthetic: y from x0*x1 + 0.7 x2 - 0.5 x3^2 + 0.3 x4 + noise, thresholded at the median; x5 and x6 are noise."""
    rng = np.random.default_rng(0)
    n = 20000
    X = rng.standard_normal((n, 8))
    signal = X[:, 0] * X[:, 1] + 0.7 * X[:, 2] - 0.5 * X[:, 3] ** 2 + 0.3 * X[:, 4]
    y = ((signal + 0.5 * rng.standard_normal(n)) > np.median(signal)).astype(np.int8)
    return X, y


@pytest.mark.parametrize("column", [0, 1, 5, 6])
def test_a_column_without_marginal_signal_gets_no_edge(data, column):
    """Noise and the two pure-synergy columns end in one bin, as under MDLP."""
    X, y = data
    assert _mah.mah_bin_edges(X[:, column], y).size == 0


@pytest.mark.parametrize("column,floor", [(2, 0.09), (3, 0.05), (4, 0.015)])
def test_a_signal_column_keeps_close_to_the_information_of_the_mdlp_binning(data, column, floor):
    """MI of the MAH binning against the MDLP binning of the same column: at least 90 % of it, and above an absolute floor."""
    X, y = data
    x = X[:, column]
    mah_codes = np.searchsorted(_mah.mah_bin_edges(x, y), x, side="right")
    mdlp_codes = np.searchsorted(edges_fayyad_irani(x, y), x, side="right")
    assert _mi(mah_codes, y) >= floor
    assert _mi(mah_codes, y) >= 0.9 * _mi(mdlp_codes, y)


def test_the_u_shaped_column_is_cut_on_both_sides_of_zero(data):
    """-0.5 x3^2 is symmetric: the information is in |x3|, so cuts must exist on both sides of zero."""
    X, y = data
    edges = _mah.mah_bin_edges(X[:, 3], y)
    assert (edges < 0).any() and (edges > 0).any()


def test_the_regret_is_the_multinomial_one():
    """``_nml_regret`` against the exact binary NML regret (log of the sum over the outcomes of the maximised likelihood) at N = 200, within 0.1 nat, and monotone in K and N."""
    n = 200
    exact = math.log(sum(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + (k * math.log(k / n) if k else 0.0) + ((n - k) * math.log((n - k) / n) if k < n else 0.0)) for k in range(n + 1)))
    assert abs(_mah._nml_regret(2, n) - exact) < 0.1
    assert _mah._nml_regret(3, n) > _mah._nml_regret(2, n) > _mah._nml_regret(1, n) == 0.0
    assert _mah._nml_regret(2, 2 * n) > _mah._nml_regret(2, n)
    # the formula it replaced understated the regret of two categories by more than a nat at N = 200
    assert exact - _mah._nml_regret_legacy(2, n) > 1.0


def test_two_rows_with_the_same_distribution_are_merged_and_two_that_differ_are_not():
    """The pair gain is positive for equal row distributions and negative for very different ones."""
    same = np.array([[500.0, 500.0], [498.0, 502.0]])
    different = np.array([[900.0, 100.0], [100.0, 900.0]])
    assert _mah._pair_gain_conditional(same, 0, 0) > 0.0
    assert _mah._pair_gain_conditional(different, 0, 0) < 0.0


def test_the_label_columns_are_not_merged_when_binning_a_feature():
    """``merge_columns=False`` keeps both classes: with the columns merged, a column with no signal merged the labels first and then nothing could merge."""
    joint = np.array([[400.0, 400.0]] * 6)
    merged, rows, cols = _mah._greedy_merge_with_history(joint, True, False)
    assert merged.shape[1] == 2 and cols == []
    assert merged.shape[0] == 1 and len(rows) == 5


def test_the_joint_criterion_is_still_reachable_as_the_reference():
    """``conditional=False`` is the earlier criterion with its floor of two bins."""
    joint = np.array([[400.0, 400.0]] * 6)
    merged, _, _ = _mah._greedy_merge_with_history(joint, False, False)
    assert merged.shape[0] == 2


def test_naming_the_cuts_removes_more_bins_and_is_off_by_default():
    """``name_cuts=True`` charges the code length of naming each kept cut: it merges at least as much as the default, which is unchanged."""
    rng = np.random.default_rng(4)
    joint = np.column_stack([rng.integers(280, 330, 16), rng.integers(260, 340, 16)]).astype(float)
    base = _mah._greedy_merge_with_history(joint, True, False)[0]
    named = _mah._greedy_merge_with_history(joint, True, False, True)[0]
    assert named.shape[0] <= base.shape[0]
    np.testing.assert_array_equal(_mah._greedy_merge_with_history(joint, True, False, False)[0], base)
