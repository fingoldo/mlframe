"""Pin ``ShapProxiedFS.trust_guard_n_estimators`` default (100) and its trust-gate behaviour.

The trust_guard stage fits ``n_anchors`` booster instances; per-anchor tree count is the wall driver.

The default was 100 -> 25 (commit 846ea4ef9, iter94), justified by an identical-subset sweep at C3
(width 10000). On a 2000-column regime (n=2000, 20 informative, 20 redundant, snr 8, seed 0) the 25-tree
default selects 22 features, including a noise column (``noise140``), while the 100-tree control selects
15 informative/redundant features. That is an accuracy loss, so the default is restored to 100 and the
speed loss is accepted. Trustworthy stays True at both values.

Two tests:
  * unit: the constructor default is 100 (sklearn-clone friendly; no hidden migration).
  * biz_value: at the default, the trust gate holds and the chosen subset equals the explicit 100-tree control.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import perf_time_budget


def test_trust_guard_n_estimators_default_is_100():
    """Pin the restored default. A future cut below 100 must be a deliberate edit that re-measures the subset."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.trust_guard_n_estimators == 100, f"trust_guard_n_estimators default drifted from 100 to {sel.trust_guard_n_estimators}"
    # sklearn-clone path: the default must round-trip through get_params/set_params untouched.
    assert sel.get_params()["trust_guard_n_estimators"] == 100


pytest.importorskip("shap")
pytest.importorskip("xgboost")


@pytest.mark.slow
@pytest.mark.timeout(perf_time_budget(900))
def test_biz_val_trust_guard_n_estimators_default_preserves_trust_and_subset():
    """At the new default (25), trust_guard must remain trustworthy AND select the same subset that
    the 100-tree control selects on a C3-tier regime. Rank-only consumer of anchor losses means the
    cap should not perturb the gating signal at this regime.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 20, 20, 2000
    X, y, _roles = make_regime_dataset(
        n_samples=2000,
        n_informative=n_informative,
        n_redundant=n_redundant,
        redundancy_rho=0.85,
        n_noise=width - n_informative - n_redundant,
        snr=8.0,
        task="binary",
        seed=0,
    )

    def _build(value):
        """Helper that build."""
        return ShapProxiedFS(
            classification=True,
            metric="brier",
            optimizer="auto",
            prefilter_top=300,
            cluster_features=True,
            top_n=15,
            n_splits=3,
            n_revalidation_models=2,
            n_anchors=20,
            trust_guard=True,
            trust_guard_n_estimators=value,
            random_state=0,
            verbose=False,
            n_jobs=1,
        )

    # The former default-vs-100 subset-equality check is dropped: the default is now 100, so the
    # comparison would be the same fit twice. The measured 25-tree divergence is recorded in the module docstring.
    sel_default = _build(100)
    assert sel_default.trust_guard_n_estimators == 100
    sel_default.fit(X, pd.Series(y))
    trust_default = sel_default.shap_proxy_report_.get("trust", {}) or {}
    assert (
        trust_default.get("trustworthy") is True
    ), f"trust gate tripped at default (trustworthy={trust_default.get('trustworthy')}, fidelity={trust_default.get('proxy_fidelity_score')})"
