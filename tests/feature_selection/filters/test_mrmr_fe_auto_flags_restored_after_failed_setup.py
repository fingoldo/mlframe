"""fe_auto's temporary generator flags are put back when the fit fails during setup, and a stale recommendation does not outlive its fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters.mrmr._mrmr_class as mod
from mlframe.feature_selection.filters.mrmr import MRMR


def _data():
    """Tiny frame and binary target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=80), "b": rng.normal(size=80)})
    return X, (rng.random(80) < 0.5).astype(int)


def test_failed_setup_leaves_the_constructor_flag_as_the_caller_set_it(monkeypatch):
    """The recommender turns fe_pair_prewarp_enable on; the override step then raises; the flag must be False again."""
    monkeypatch.setattr(mod, "recommend_fe_flags_by_rules", lambda X, y: {"fe_pair_prewarp_enable": True})

    def _boom(self, X, st):
        """Simulate a failing override step."""
        raise RuntimeError("override failed")

    monkeypatch.setattr(MRMR, "_fit_body_apply_fit_overrides", _boom)
    X, y = _data()
    est = MRMR(fe_auto=True, fe_pair_prewarp_enable=False)
    with pytest.raises(RuntimeError, match="override failed"):
        est.fit(X, y)
    assert est.fe_pair_prewarp_enable is False


def test_recommendation_from_a_previous_fit_is_cleared(monkeypatch):
    """After a fe_auto fit, a fit with fe_auto off must not keep describing the old recommendation."""
    monkeypatch.setattr(mod, "recommend_fe_flags_by_rules", lambda X, y: {"fe_pair_prewarp_enable": True})
    X, y = _data()
    est = MRMR(fe_auto=True, verbose=0)
    est._fe_recommended_flags_ = {"fe_pair_prewarp_enable": True}
    est.set_params(fe_auto=False)
    try:
        est.fit(X, y)
    except Exception:  # only the attribute reset is under test
        pass
    assert est._fe_recommended_flags_ is None
