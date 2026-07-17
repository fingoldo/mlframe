"""Coverage for ``use_bias_corrector``: the proxy bias-corrector engages under the trust guard,
records itself in the report, and does not worsen recovery vs corrector-off.

The corrector re-ranks candidate subsets by a regression of honest-loss on (proxy-loss, cardinality,
redundancy) fitted on the trust anchors, so the expensive top-N honest retrain budget is spent on the
candidates most likely to be honestly best. NOTE: the quantitative selection-FLIP win (corrector
changes the FINAL chosen subset for the better) is NOT pinned here -- on small CI frames the honest
retrain budget covers all candidates, so the final pick is identical with/without the corrector and the
win only surfaces under a constrained reval budget + heavy proxy card/redundancy bias (high variance,
infeasible to floor in the test budget). Bucketed NEEDS-DEEPER-BENCH in the B7 audit report. This test
pins the engage-and-record contract + the no-regression guarantee so a regression that silently
disables the corrector trips.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")


def _frame(seed=0, n=1000):
    """Helper that frame."""
    rng = np.random.default_rng(seed)
    xi = rng.normal(size=(n, 3))
    decoy = rng.normal(size=(n, 1))  # continuous high-card column SHAP tends to over-credit
    noise = rng.normal(size=(n, 8))
    X = pd.DataFrame(np.hstack([xi, decoy, noise]), columns=[f"inf{i}" for i in range(3)] + ["decoy"] + [f"n{i}" for i in range(8)])
    y = (xi @ [1.2, 0.9, 0.7] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def _fit(use_bias_corrector):
    """Helper that fit."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _frame()
    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        max_features=4,
        top_n=10,
        n_splits=3,
        n_revalidation_models=1,
        trust_guard=True,
        cluster_features=False,
        use_bias_corrector=use_bias_corrector,
        random_state=0,
        verbose=False,
    )
    sel.fit(X, y)
    return sel


def test_biz_val_bias_corrector_engages_and_is_recorded():
    """Biz val bias corrector engages and is recorded."""
    sel = _fit(True)
    bc = sel.shap_proxy_report_.get("bias_corrector")
    assert bc and bc.get("applied") is True, f"corrector should engage under trust_guard; got {bc}"
    assert bc["n_anchors"] > 0


def test_biz_val_bias_corrector_off_does_not_record():
    """Biz val bias corrector off does not record."""
    assert _fit(False).shap_proxy_report_.get("bias_corrector") is None


def test_biz_val_bias_corrector_preserves_recovery():
    """Biz val bias corrector preserves recovery."""
    sel = _fit(True)
    selected = {str(c) for c in sel.selected_features_}
    assert {"inf0", "inf1", "inf2"} <= selected, f"corrector-on must keep the 3 informative cols; got {selected}"
    assert "decoy" not in selected
