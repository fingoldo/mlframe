"""ShapProxiedFS(report_holdout_fraction>0) scores the chosen subset once on holdout rows no candidate was ranked on."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("shap")


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] + 0.8 * X["f1"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, pd.Series(y)


def _fit(fraction):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _data()
    return ShapProxiedFS(classification=True, metric="brier", optimizer="bruteforce", max_features=4, top_n=8, n_splits=3,
                         n_revalidation_models=1, random_state=0, verbose=False, n_jobs=1, report_holdout_fraction=fraction).fit(X, y)


def test_default_carves_nothing():
    assert "report_holdout" not in _fit(0.0).shap_proxy_report_


def test_opt_in_reports_a_loss_on_the_untouched_slice():
    rep = _fit(0.25).shap_proxy_report_["report_holdout"]
    assert rep["fraction"] == 0.25 and rep["selection_optimistic"] is False
    assert 0.0 < rep["loss"] < 0.25 and 50 <= rep["n_rows"] <= 120  # 25% of the 375-row holdout, a real Brier


def test_split_is_disjoint_and_covers_the_holdout():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxied_report_slice import split_report_slice

    idx = np.arange(100, 500)
    X = pd.DataFrame({"a": np.arange(600.0)})
    keep, (X_rep, y_rep) = split_report_slice(idx, X, np.arange(600) % 2, 0.25, True, 0)
    assert len(keep) == 300 and len(y_rep) == 100
    assert set(keep).isdisjoint(set(X_rep["a"].astype(int))) and set(keep) | set(X_rep["a"].astype(int)) == set(idx)
