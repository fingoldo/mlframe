"""biz_val for gt_02's ``refine_mode="auto"`` (now the default): a cheap pre-gate that predicts,
before paying core-refine's LP + honest-reverify cost, whether it's likely to find anything greedy
wouldn't.

Root-caused empirically (see ``test_biz_val_shap_proxied_core_refine.py``'s own docstring): core
degrades to greedy when the strong/confident units' honest loss is already near its achievable floor.
``auto_should_use_core_refine`` measures the marginal held-out loss gain of the weaker half of units
relative to the confident half -- near-zero/negative on the saturated bed, clearly positive on the bed
where core actually helps.
"""

from __future__ import annotations

import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
from tests.feature_selection.shap_proxied.test_biz_val_shap_proxied_parsimony_tol_recall import _make_mixed_strength_fixture


def _fit_selected(X, y, refine_mode, seed=0, **kw):
    """Fit ShapProxiedFS with the given refine_mode and return (selected_feature_names, fitted_estimator)."""
    s = ShapProxiedFS(classification=True, random_state=seed, verbose=False, prescreen_ladder_mode="off", n_jobs=1, refine_mode=refine_mode, **kw)
    s.fit(X, y)
    return set(s.selected_features_), s


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_auto_matches_greedy_on_saturated_bed():
    """On the saturated bed (6 strong w=1.0 + 6 weak w=0.25, where core measurably degrades to greedy),
    auto's pre-gate should route to greedy directly (never even invoking core's LP) -- verified both by
    selection matching greedy exactly and by the resolved mode recorded in
    ``shap_proxy_report_["within_cluster_refine"]``."""
    X, y, _strong, _weak = _make_mixed_strength_fixture(n_strong=6, strong_weight=1.0, weak_weight=0.25)

    sel_auto, s_auto = _fit_selected(X, y, refine_mode="auto")
    sel_greedy, _s_greedy = _fit_selected(X, y, refine_mode="greedy")

    assert sel_auto == sel_greedy, f"auto diverged from greedy on the saturated bed: auto={sel_auto}, greedy={sel_greedy}"
    refine_report = s_auto.shap_proxy_report_["within_cluster_refine"]
    assert refine_report["requested_mode"] == "auto"
    assert refine_report["mode"] == "greedy", f"expected auto to resolve to greedy on the saturated bed, got {refine_report['mode']!r}"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_biz_val_auto_matches_core_on_non_saturated_bed():
    """On the bed where core measurably recovers more weak-feature recall than greedy (3 strong w=0.8 +
    6 weak w=0.35), auto's pre-gate should route to core and recover the SAME weak-feature recall core
    achieves directly."""
    X, y, _strong, weak = _make_mixed_strength_fixture(n_strong=3, strong_weight=0.8, weak_weight=0.35)
    weak_names = {f"f{i}" for i in weak}

    sel_auto, s_auto = _fit_selected(X, y, refine_mode="auto")
    sel_greedy, _s_greedy = _fit_selected(X, y, refine_mode="greedy")

    auto_weak_recall = len(weak_names & sel_auto)
    greedy_weak_recall = len(weak_names & sel_greedy)
    assert auto_weak_recall > greedy_weak_recall, (
        f"auto recall ({auto_weak_recall}/6) did not exceed greedy recall ({greedy_weak_recall}/6) on the "
        "bed where core is known to help -- the pre-gate should have routed to core here"
    )
    refine_report = s_auto.shap_proxy_report_["within_cluster_refine"]
    assert refine_report["mode"] == "core", f"expected auto to resolve to core on the non-saturated bed, got {refine_report['mode']!r}"
