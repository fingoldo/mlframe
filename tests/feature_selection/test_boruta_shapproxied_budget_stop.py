"""Regression: BorutaShap + ShapProxiedFS honor a wall-clock budget (``max_runtime_mins``)
and a filesystem stop-flag (``stop_file``) -- parity with MRMR / RFECV.

Both must abort cleanly and still expose a valid (possibly partial) selection
(``support_`` / ``selected_features_``), so a runaway FS on a big frame can be bounded
without losing the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _data(n: int = 300, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = ((1.5 * X["f0"] - 0.8 * X["f1"] + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    return X, y


def test_boruta_shap_stop_file_aborts_with_valid_selection(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _data()
    stop = tmp_path / "stop"
    stop.write_text("")  # stop-flag present -> fit must bail out of the trial loop
    b = BorutaShap(
        model=lgb.LGBMClassifier(n_estimators=10, verbose=-1),
        importance_measure="permutation",
        n_trials=50,
        stop_file=str(stop),
        verbose=False,
    )
    b.fit(X, y)
    assert b.n_trials_run_ < 50, f"stop_file must abort before all 50 trials; ran {b.n_trials_run_}"
    assert hasattr(b, "selected_features_"), "fit must still produce a valid selection after stop"
    b.transform(X)  # selection is usable


def test_boruta_shap_runtime_budget_aborts(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    from mlframe.feature_selection.boruta_shap import BorutaShap

    X, y = _data()
    b = BorutaShap(
        model=lgb.LGBMClassifier(n_estimators=50, verbose=-1),
        importance_measure="permutation",
        n_trials=300,
        max_runtime_mins=1e-6,
        verbose=False,
    )
    b.fit(X, y)
    assert b.n_trials_run_ < 300, f"tiny runtime budget must abort early; ran {b.n_trials_run_}"
    assert hasattr(b, "selected_features_")


def test_shap_proxied_fs_stop_file_skips_revalidation(tmp_path):
    pytest.importorskip("shap")
    lgb = pytest.importorskip("lightgbm")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _data()
    stop = tmp_path / "stop"
    stop.write_text("")
    s = ShapProxiedFS(
        model=lgb.LGBMClassifier(n_estimators=20, verbose=-1),
        stop_file=str(stop),
        verbose=False,
        n_splits=2,
        top_n=5,
    )
    s.fit(X, y)
    assert hasattr(s, "support_") and s.support_.any(), "must finalize with a valid proxy-best subset"
    # The expensive optional phase (honest revalidation) must have been skipped by the stop-flag.
    assert s.shap_proxy_report_.get("budget_skipped", {}).get("phase") == "revalidation", "stop_file must skip the honest-revalidation phase"
