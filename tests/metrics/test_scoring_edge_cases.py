"""Edge coverage for ``mlframe.metrics.scoring``: fast_rmse numerical extremes / n=1 and the remaining
``ProbaScoreProxy`` bounds (negative index, index == n_classes boundary, valid low index).

Complements ``test_scoring`` (basic values / dtype pairs) and ``test_scoring_proba_proxy_guard``
(1-D input + far-out-of-range index).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.scoring import ProbaScoreProxy, fast_rmse, rmse_loss


def test_fast_rmse_single_element():
    # n=1: RMSE = |y - p|.
    assert fast_rmse(np.array([5.0]), np.array([8.0])) == pytest.approx(3.0, abs=1e-12)


def test_fast_rmse_large_magnitude_matches_reference():
    # Large magnitude squares (up to ~1e300) stay within float64 range; fastmath must not diverge
    # from the numpy reference beyond float64 reduction-order noise.
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1e150, 0.0, 0.0])
    got = fast_rmse(y, p)
    ref = float(rmse_loss(y, p))
    assert got == pytest.approx(ref, rel=1e-9)
    assert np.isfinite(got) and got > 0.0


@pytest.mark.parametrize("bad_idx", [-1, 2, 3])
def test_proba_score_proxy_rejects_out_of_range_index(bad_idx):
    # 2-class matrix -> valid indices are {0, 1}; negative and idx>=n_classes must raise.
    y_true = np.array([0, 1, 0])
    y_probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    with pytest.raises(ValueError, match="class_idx"):
        ProbaScoreProxy(y_true, y_probs, class_idx=bad_idx, proxied_func=lambda yt, ys: float(np.mean(ys)))


def test_proba_score_proxy_selects_low_index_column():
    # class_idx=0 must forward column 0 to the proxied scorer.
    y_true = np.array([0, 1])
    y_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    got = ProbaScoreProxy(y_true, y_probs, class_idx=0, proxied_func=lambda yt, ys: float(np.mean(ys)))
    assert got == pytest.approx((0.8 + 0.3) / 2.0, abs=1e-12)
