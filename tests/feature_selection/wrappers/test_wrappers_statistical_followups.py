"""Regression tests for the second statistical-wrapper audit wave (F9-F12 follow-ups).

F9  Kendall subsample threads the caller seed (was hard-coded rng(0)).
F10 noise-floor supports binary / multiclass / continuous targets (was binary-only, crashed otherwise).
F11 continuous-target chi2 path tolerates qcut NaN bins (heavily-tied / NaN y).
F12 chi2 small-cell warning (Cochran's rule) on sparse contingency tables.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers._univariate_ht import (
    calculate_relevance_table,
    _kendall_p_numeric_continuous,
    _chi2_independence_p,
)
from mlframe.feature_selection.wrappers._noise_floor import (
    select_features_noise_floor,
    _infer_task_scoring,
)


class TestKendallSubsampleSeed:
    """The Kendall-tau continuous path now tests at FULL n via tie-corrected ``scipy.stats.kendalltau`` (O(n log n)),
    NOT a 2000-row random subsample. Mixing a subsampled 2000-row p-value with full-n p-values in one BY-FDR family
    is invalid (heterogeneous effective-n breaks the procedure's homogeneity assumption) and loses power, so the
    subsample -- and with it the seed-dependence the F9 fix threaded -- was removed. ``random_state`` is retained for
    signature/back-compat only and has NO effect on the Kendall p-value. The pre-fix seed-perturbs-subsample
    behaviour was the stale proxy; the real contract is full-n determinism + more power."""

    def test_distinct_seeds_give_identical_full_n_p_values(self):
        """Distinct seeds give identical full n p values."""
        rng = np.random.default_rng(1)
        n = 6000  # full-n path; no subsample, so the seed cannot perturb the draw
        x = rng.standard_normal(n)
        y = 0.05 * x + rng.standard_normal(n)
        p0 = _kendall_p_numeric_continuous(x, y, random_state=0)
        p1 = _kendall_p_numeric_continuous(x, y, random_state=1)
        assert p0 == p1, "full-n Kendall is deterministic; random_state must not change the p-value"

    def test_fixed_seed_is_reproducible(self):
        """Fixed seed is reproducible."""
        rng = np.random.default_rng(2)
        n = 6000
        x = rng.standard_normal(n)
        y = 0.05 * x + rng.standard_normal(n)
        assert _kendall_p_numeric_continuous(x, y, random_state=7) == _kendall_p_numeric_continuous(x, y, random_state=7)

    def test_relevance_table_kendall_is_seed_independent(self):
        """Relevance table kendall is seed independent."""
        rng = np.random.default_rng(3)
        n = 6000
        f0 = rng.standard_normal(n)
        X = pd.DataFrame({"f0": f0})
        y = 0.05 * f0 + rng.standard_normal(n)
        p_a = calculate_relevance_table(X, y, ml_task="regression", random_state=0).loc["f0", "p_value"]
        p_b = calculate_relevance_table(X, y, ml_task="regression", random_state=1).loc["f0", "p_value"]
        assert p_a == p_b


class TestNoiseFloorTaskTypes:
    """F10: noise-floor must run on binary, multiclass AND continuous targets (was binary-only)."""

    def test_infer_task_scoring(self):
        """Infer task scoring."""
        assert _infer_task_scoring(np.array([0, 1, 0, 1])) == ("binary", "roc_auc")
        assert _infer_task_scoring(np.array([0, 1, 2, 0, 1, 2])) == ("multiclass", "roc_auc_ovr")
        assert _infer_task_scoring(np.linspace(0.0, 1.0, 50))[0] == "regression"

    def test_multiclass_target_does_not_crash(self):
        """Multiclass target does not crash."""
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        n, p = 300, 6
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
        y = rng.integers(0, 3, n)  # 3-class -> pre-fix roc_auc + StratifiedKFold crashed in cross_val_score
        X["f0"] = X["f0"] + y  # give f0 signal
        out = select_features_noise_floor(
            lambda: RandomForestClassifier(n_estimators=20, random_state=0), X, y, ranking=list(X.columns), n_grid=[1, 2, 3, 6], cv=3, n_perm=3
        )
        assert "n_star" in out and out["n_star"] >= 0

    def test_continuous_target_does_not_crash(self):
        """Continuous target does not crash."""
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(0)
        n, p = 300, 6
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
        y = X["f0"].to_numpy() * 2.0 + rng.standard_normal(n) * 0.1  # continuous -> pre-fix StratifiedKFold crashed
        out = select_features_noise_floor(
            lambda: RandomForestRegressor(n_estimators=20, random_state=0), X, y, ranking=list(X.columns), n_grid=[1, 2, 3, 6], cv=3, n_perm=3
        )
        assert "n_star" in out and out["n_star"] >= 0

    def test_binary_target_still_works(self):
        """Binary target still works."""
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        n, p = 300, 6
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
        y = (X["f0"].to_numpy() > 0).astype(int)
        out = select_features_noise_floor(
            lambda: RandomForestClassifier(n_estimators=20, random_state=0), X, y, ranking=list(X.columns), n_grid=[1, 2, 3, 6], cv=3, n_perm=3
        )
        assert "n_star" in out


class TestChi2QcutNaNBins:
    """F11: continuous-target chi2 path must tolerate NaN bins from qcut(duplicates='drop') / NaN in y."""

    def test_heavily_tied_continuous_target_no_crash(self):
        """Heavily tied continuous target no crash."""
        rng = np.random.default_rng(0)
        n = 600
        cat = rng.integers(0, 4, n).astype(object)
        # Heavily-tied y -> qcut drops duplicate edges -> fewer bins; must not raise / mis-align.
        y = np.where(rng.random(n) < 0.8, 0.0, rng.standard_normal(n))
        X = pd.DataFrame({"c": cat})
        out = calculate_relevance_table(X, y, ml_task="regression")
        assert np.isfinite(out.loc["c", "p_value"])

    def test_nan_in_continuous_target_no_crash(self):
        """Nan in continuous target no crash."""
        rng = np.random.default_rng(1)
        n = 600
        cat = rng.integers(0, 4, n).astype(object)
        y = rng.standard_normal(n)
        y[::50] = np.nan  # NaN bins in qcut output
        X = pd.DataFrame({"c": cat})
        out = calculate_relevance_table(X, y, ml_task="regression")
        assert np.isfinite(out.loc["c", "p_value"])


class TestChi2SmallCellWarning:
    """F12: warn (Cochran's rule) when many expected cells < 5."""

    def test_small_cell_emits_warning(self, caplog):
        # 3x3 table with tiny counts -> expected cells well below 5.
        """Small cell emits warning."""
        x = np.array(list("abcabcabc"))
        y = np.array(list("xyzxyzxyz"))
        with caplog.at_level(logging.WARNING):
            _chi2_independence_p(x, y)
        assert any("Cochran" in r.message for r in caplog.records)

    def test_large_table_no_warning(self, caplog):
        """Large table no warning."""
        rng = np.random.default_rng(0)
        n = 5000
        x = rng.integers(0, 2, n).astype(str)
        y = rng.integers(0, 2, n).astype(str)
        with caplog.at_level(logging.WARNING):
            _chi2_independence_p(x, y)
        assert not any("Cochran" in r.message for r in caplog.records)
