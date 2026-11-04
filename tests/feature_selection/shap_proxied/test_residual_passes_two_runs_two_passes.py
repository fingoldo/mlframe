"""residual_passes=2 runs two residual SHAP passes, not one."""

from __future__ import annotations

import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from tests.feature_selection.shap_proxied.test_shap_proxied_residual_passes import _FAST_KW, _tiny_fixture


def _fit(passes):
    """Fit a small ShapProxiedFS with the given residual pass count and return its residual report."""
    X, y = _tiny_fixture()
    sel = ShapProxiedFS(
        random_state=0, verbose=False, n_jobs=1, trust_guard=False, run_importance_ablation=False, residual_passes=passes, residual_merge="rescue", **_FAST_KW
    )
    sel.fit(X, y)
    return sel.shap_proxy_report_["residual_pass"]


def test_two_requested_passes_are_both_run():
    """The report states how many passes actually ran; two requested means two ran."""
    assert _fit(2)["n_passes_run"] == 2


def test_one_requested_pass_runs_one():
    """One requested pass is unchanged."""
    assert _fit(1)["n_passes_run"] == 1
