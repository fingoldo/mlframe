"""Regression tests for the statistical-wrapper audit fixes (knockoffs / univariate-HT / noise-floor / auto-tune)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers._univariate_ht import (
    calculate_relevance_table,
    _mann_whitney_u_z,
    _mann_whitney_u_z_v2,
    _kruskal_wallis_h,
    _kruskal_wallis_h_v2,
)
from mlframe.feature_selection.wrappers._auto_tune import DataFingerprint


class TestSingleSortKernelsBitIdentical:
    """The v2 single-argsort kernels must be bit-identical to the originals, including on heavily-tied data."""

    def test_mwu_v2_matches_on_ties(self):
        rng = np.random.default_rng(7)
        for _ in range(15):
            n = int(rng.integers(10, 400))
            x = rng.integers(0, 5, n).astype(np.float64)  # heavy ties stress the tie-correction
            g = (rng.random(n) > 0.45).astype(np.int64)
            np.testing.assert_array_equal(_mann_whitney_u_z(x, g), _mann_whitney_u_z_v2(x, g))

    def test_kw_v2_matches_on_ties(self):
        rng = np.random.default_rng(9)
        for _ in range(15):
            n = int(rng.integers(10, 400))
            x = rng.integers(0, 6, n).astype(np.float64)
            g = rng.integers(0, 4, n).astype(np.int64)
            np.testing.assert_array_equal(_kruskal_wallis_h(x, g, 4), _kruskal_wallis_h_v2(x, g, 4))


class TestClassificationNaNTargetRouting:
    """ml_task='classification' must drop NaN labels before counting -> a binary target stays binary."""

    def test_classification_nan_label_does_not_inflate_to_multiclass(self, monkeypatch):
        import mlframe.feature_selection.wrappers._univariate_ht as ht

        rng = np.random.default_rng(0)
        n = 300
        f0 = rng.standard_normal(n)
        X = pd.DataFrame({"f0": f0, "f1": rng.standard_normal(n)})
        y = (f0 > 0).astype(float)
        y[5] = np.nan  # pre-fix np.unique(y) counts {0.0, 1.0, nan} = 3 -> 'multiclass' route

        chosen = {}
        orig_bin = ht._mann_whitney_p_numeric_binary
        orig_kw = ht._kw_p_numeric_multiclass

        def _bin(*a, **k):
            chosen["route"] = "binary"
            return orig_bin(*a, **k)

        def _kw(*a, **k):
            chosen["route"] = "multiclass"
            return orig_kw(*a, **k)

        monkeypatch.setattr(ht, "_mann_whitney_p_numeric_binary", _bin)
        monkeypatch.setattr(ht, "_kw_p_numeric_multiclass", _kw)
        ht.calculate_relevance_table(X, y, ml_task="classification", fdr_level=0.5)
        assert chosen.get("route") == "binary"


class TestAutoTuneTargetTyping:
    def test_negative_integer_labels_do_not_crash(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((120, 3)), columns=list("abc"))
        y = rng.choice([-1, 1], 120)  # pre-fix np.bincount(astype(int)) raised on negative labels
        fp = DataFingerprint.from_xy(X, y)
        assert fp.target_type == "binary"
        assert 0.0 < fp.target_imbalance <= 0.5

    def test_high_card_integer_regression_not_multiclass_on_large_n(self):
        rng = np.random.default_rng(0)
        n = 200_000
        X = pd.DataFrame(rng.standard_normal((n, 2)), columns=list("ab"))
        # 200 distinct integer codes: old cap max(10, sqrt(2e5)=447) -> 200<=447 -> 'multiclass' (WRONG);
        # new absolute cap of 50 -> 200>50 -> 'regression'.
        y = rng.integers(0, 200, n)
        fp = DataFingerprint.from_xy(X, y)
        assert fp.target_type == "regression"
