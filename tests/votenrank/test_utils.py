"""Smoke tests for mlframe.votenrank.utils (E-P1.4)."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


@pytest.mark.fast
def test_import_votenrank_utils_module():
    """Module imports cleanly and exposes expected public names."""
    from mlframe.votenrank import utils as umod

    for name in ("ranking2top", "kendall_tau", "agreement_rate", "tracker_filename"):
        assert callable(getattr(umod, name)), f"{name} not callable"


@pytest.mark.fast
def test_ranking2top_returns_max_indices():
    """ranking2top returns list of index labels where ranking equals the max value."""
    from mlframe.votenrank.utils import ranking2top

    s = pd.Series([1, 3, 3, 2], index=["a", "b", "c", "d"])
    top = ranking2top(s)
    assert set(top) == {"b", "c"}
    assert isinstance(top, list)


@pytest.mark.fast
def test_tracker_filename_format():
    """tracker_filename composes model/task/dirpath into expected pattern."""
    from mlframe.votenrank.utils import tracker_filename

    out = tracker_filename(model="lgb", task="auc", dirpath="/exp")
    assert out == "/exp/lgb_auc_0/"
