"""Cluster 24 P3 regressions: the duplicate-target guard runs before the MI work, and a wrong forward-select return raises TypeError (audit 2026-11-20, findings 24-F62 and 24-F63).

The duplicate-target check used to run after the full permutation MI loop, so a bad call waited for all of it. A non-list return from the forward search
was a bare AssertionError, which says nothing about what went wrong.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection import functional_adapters
from mlframe.feature_selection import general


def test_duplicate_target_is_refused_before_any_mi_work(monkeypatch):
    """A repeated target name raises at once; the MI estimator is never reached (24-F62)."""

    def _boom(*args, **kwargs):
        """Fail if the MI loop is reached: the guard must come first."""
        raise AssertionError("estimate_features_relevancy must not run for duplicate targets")

    monkeypatch.setattr(general, "estimate_features_relevancy", _boom)
    with pytest.raises(ValueError, match="duplicate"):
        general.run_efs(
            df=pd.DataFrame({"x": [1, 2, 3]}),
            target_columns=["y", "y"],
            exclude_columns=[],
            permuted_mutual_informations={},
            binned_targets=None,
            mi_algorithms_ranking=[],
            binning_params={},
            efs_params={},
        )


def test_forward_select_non_list_return_is_a_type_error(monkeypatch):
    """A non-list result from forward_select raises TypeError with the returned type named (24-F63)."""
    monkeypatch.setattr(functional_adapters, "forward_select", lambda *a, **k: np.array(["x0"]))
    X = pd.DataFrame({"x0": np.arange(20.0), "x1": np.arange(20.0)[::-1]})
    y = (np.arange(20) % 2).astype(int)
    sel = functional_adapters.ForwardSelectSelector()
    with pytest.raises(TypeError, match="ndarray"):
        sel.fit(X, y)
