"""Regression test for the ndim/class_idx guard in ProbaScoreProxy (EDGE-P2).

Pre-fix ``ProbaScoreProxy`` did ``y_probs[:, class_idx]`` with no shape check -> a 1-D y_probs or a
class_idx >= n_classes raised an opaque IndexError deep inside numpy instead of a clear message.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.scoring import ProbaScoreProxy


def _dummy_scorer(y_true, y_score):
    """Helper: Dummy scorer."""
    return float(np.mean(y_score))


def test_proba_score_proxy_1d_probs_raises_clear_error():
    """Proba score proxy 1d probs raises clear error."""
    y_true = np.array([0, 1, 0])
    y_probs_1d = np.array([0.2, 0.8, 0.4])  # 1-D, should be (N, n_classes)
    with pytest.raises(ValueError, match="2-D"):
        ProbaScoreProxy(y_true, y_probs_1d, class_idx=1, proxied_func=_dummy_scorer)


def test_proba_score_proxy_out_of_range_class_idx_raises_clear_error():
    """Proba score proxy out of range class idx raises clear error."""
    y_true = np.array([0, 1, 0])
    y_probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])  # 2 classes
    with pytest.raises(ValueError, match="class_idx"):
        ProbaScoreProxy(y_true, y_probs, class_idx=5, proxied_func=_dummy_scorer)
