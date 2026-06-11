"""Pin ``ShapProxiedFS.trust_guard_n_estimators`` default + biz_value trustworthiness at the new default.

The trust_guard stage fits ``n_anchors`` (default 30) booster instances at random feature-subset
cardinalities; per-anchor tree count is the dominant wall driver. The trust report only consumes
RANKS of anchor losses (Spearman / Kendall / recall@k), so rank stability across capped vs.
uncapped boosters is what matters, not absolute loss reproduction.

iter94 measured at C3 (width=10000, n_rows=10000, n_inf=20, n_red=20, snr=8.0, seed=0):
trust_guard_n_estimators in {100, 50, 25} all produce IDENTICAL chosen subsets and trustworthy=True
everywhere; composite proxy_fidelity_score moves 0.9848 -> 0.9837 -> 0.9834 (-0.14% at n=25).
trust_guard wall 4.106s -> 1.953s -> 1.677s; e2e 69.91s -> 28.45s -> 26.37s. Default moved 100 -> 25.

Two tests:
  * unit: the constructor default is 25 (sklearn-clone friendly; no hidden migration).
  * biz_value: trust report on a smaller C3-tier regime stays trustworthy=True with chosen subset
    identical to the 100-tree control, demonstrating the cap preserves the gating signal.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_trust_guard_n_estimators_default_is_25():
    """Pin the iter94 default. A future bump back should be a deliberate edit, not silent drift."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    sel = ShapProxiedFS()
    assert sel.trust_guard_n_estimators == 25, (
        f"trust_guard_n_estimators default drifted from 25 to {sel.trust_guard_n_estimators}")
    # sklearn-clone path: the default must round-trip through get_params/set_params untouched.
    assert sel.get_params()["trust_guard_n_estimators"] == 25


pytest.importorskip("shap")
pytest.importorskip("xgboost")


@pytest.mark.slow
@pytest.mark.timeout(240)
def test_biz_val_trust_guard_n_estimators_default_preserves_trust_and_subset():
    """At the new default (25), trust_guard must remain trustworthy AND select the same subset that
    the 100-tree control selects on a C3-tier regime. Rank-only consumer of anchor losses means the
    cap should not perturb the gating signal at this regime.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    n_informative, n_redundant, width = 20, 20, 2000
    X, y, _roles = make_regime_dataset(
        n_samples=2000, n_informative=n_informative, n_redundant=n_redundant,
        redundancy_rho=0.85, n_noise=width - n_informative - n_redundant, snr=8.0,
        task="binary", seed=0,
    )

    def _build(value):
        return ShapProxiedFS(
            classification=True, metric="brier", optimizer="auto",
            prefilter_top=300, cluster_features=True,
            top_n=15, n_splits=3, n_revalidation_models=2, n_anchors=20,
            trust_guard=True, trust_guard_n_estimators=value,
            random_state=0, verbose=False, n_jobs=1,
        )

    sel_default = _build(25)
    assert sel_default.trust_guard_n_estimators == 25
    sel_default.fit(X, pd.Series(y))
    trust_default = sel_default.shap_proxy_report_.get("trust", {}) or {}
    assert trust_default.get("trustworthy") is True, (
        f"trust gate tripped at new default (trustworthy={trust_default.get('trustworthy')}, "
        f"fidelity={trust_default.get('proxy_fidelity_score')})")

    sel_full = _build(100)
    sel_full.fit(X, pd.Series(y))
    trust_full = sel_full.shap_proxy_report_.get("trust", {}) or {}
    assert trust_full.get("trustworthy") is True

    chosen_default = tuple(sorted(sel_default.selected_features_))
    chosen_full = tuple(sorted(sel_full.selected_features_))
    assert chosen_default == chosen_full, (
        f"chosen subset diverged between trust_guard_n_estimators=25 (new default) and 100 (control): "
        f"default={chosen_default} vs full={chosen_full}")
