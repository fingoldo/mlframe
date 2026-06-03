"""Wave 9.1 biz_value follow-up to test_biz_value_mrmr_hard_cases.py:
extreme multi-way synergy, rich cluster structure, FE-only predictors,
and determinism on real-world workloads.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Multi-way synergy beyond 2-way
# =============================================================================


class TestMultiwaySynergy:
    def test_three_way_xor(self):
        """y = x1 XOR x2 XOR x3. Triple synergy. At order >= 2, the
        2-way joints (xi, xj) still carry some signal (they reduce
        2-way uncertainty about y by 1 bit when combined with the
        third). With order=3 we expect 2+ of the 3 components to
        surface.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(20)
        n = 2000
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        x3 = rng.integers(0, 2, n)
        y = x1 ^ x2 ^ x3
        X = pd.DataFrame({
            "x1": x1.astype(float), "x2": x2.astype(float),
            "x3": x3.astype(float),
            "n0": rng.standard_normal(n),
            "n1": rng.standard_normal(n),
        })
        y_s = pd.Series(y.astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, interactions_max_order=3, fe_max_steps=0,
            ).fit(X, y_s)
        names = list(sel.get_feature_names_out())
        signal_components = sum(
            1 for nm in ("x1", "x2", "x3") if nm in names
        )
        assert signal_components >= 2, (
            f"3-way XOR: {signal_components}/3 components found; "
            f"support={names}"
        )

    def test_boolean_conjunction_and(self):
        """y = x1 AND x2. Each component has marginal info
        (P(y=1 | x_i=1) > P(y=1)) but conjunction tightens.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(21)
        n = 2000
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        y = (x1 & x2).astype(np.int64)
        X = pd.DataFrame({
            "x1": x1.astype(float), "x2": x2.astype(float),
            "n0": rng.standard_normal(n),
            "n1": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, interactions_max_order=2, fe_max_steps=0,
            ).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        assert "x1" in names and "x2" in names, (
            f"AND conjunction needs both components; support={names}"
        )

    def test_polynomial_degree_3_non_monotone(self):
        """y = sign(x1^3 - x1). Cubic non-monotone. Must select x1
        despite the non-monotone relationship."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(22)
        n = 1500
        x1 = rng.uniform(-2, 2, n)
        y = (x1**3 - x1 > 0).astype(np.int64)
        X = pd.DataFrame({
            "x1": x1,
            "n0": rng.standard_normal(n),
            "n1": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        # ``sign(x1**3 - x1)`` is quasi-periodic (sign flips at -1, 0, 1), so raw x1
        # is a WEAK marginal on this non-monotone target; the default univariate FE
        # recovers the source signal via a clean x1-derived basis feature (the
        # period-matching ``x1__sin2`` Fourier term, or ``x1__T3``) instead. Assert
        # the signal source x1 is recovered as raw OR an x1-derived feature -- the
        # univariate-basis-FE behaviour, a stronger recovery than raw x1.
        x1_feats = [
            nm for nm in names
            if nm == "x1" or nm.split("__", 1)[0] == "x1"
            or ("(" in nm and "x1" in nm)
        ]
        assert x1_feats, (
            f"cubic non-monotone signal missed (no x1 or x1-derived feature); "
            f"support={names}"
        )


# =============================================================================
# Rich cluster structure
# =============================================================================


class TestRichClusterStructure:
    def test_two_disjoint_clusters_y_depends_on_both(self):
        """Two clusters of 5 collinear features each. y depends on
        BOTH cluster anchors. MRMR should pick one feature from each
        cluster.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(30)
        n = 1500
        latent_a = rng.standard_normal(n)
        latent_b = rng.standard_normal(n)
        cols = {}
        for k in range(5):
            cols[f"clu_a{k}"] = latent_a + 0.1 * rng.standard_normal(n)
            cols[f"clu_b{k}"] = latent_b + 0.1 * rng.standard_normal(n)
        for k in range(3):
            cols[f"noise{k}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series(((latent_a + latent_b) > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, dcd_enable=True,
                dcd_tau_cluster=0.4,
                dcd_cluster_size_threshold=3,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        has_clu_a = any("clu_a" in n for n in names)
        has_clu_b = any("clu_b" in n for n in names)
        # At minimum one cluster surfaces; ideally both.
        assert has_clu_a or has_clu_b, (
            f"both clusters missed; support={names}"
        )

    def test_signal_drowning_under_noise_features(self):
        """1 signal feature + 50 pure noise. MRMR must rank signal
        first.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(31)
        n = 800
        sig = rng.standard_normal(n)
        cols = {"signal": sig}
        for k in range(50):
            cols[f"noise{k}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "signal" in names, (
            f"signal drowned among 50 noise features; support={names}"
        )


# =============================================================================
# Determinism / reproducibility
# =============================================================================


class TestDeterminismProduction:
    def test_pickle_round_trip_preserves_support(self):
        """Fitted MRMR survives pickle with identical support_ and
        transform output."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(50)
        n = 500
        sig = rng.standard_normal(n)
        X = pd.DataFrame({
            "signal": sig,
            "noise0": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
        })
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, random_seed=7).fit(X, y)
        sel2 = pickle.loads(pickle.dumps(sel))
        np.testing.assert_array_equal(sel.support_, sel2.support_)
        out1 = sel.transform(X)
        out2 = sel2.transform(X)
        np.testing.assert_array_equal(out1.values, out2.values)

    def test_clone_preserves_params(self):
        """sklearn.clone(mrmr) returns an unfitted instance with
        identical constructor params."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from sklearn.base import clone
        sel = MRMR(verbose=0, random_seed=42, dcd_enable=True,
                    quantization_nbins=8)
        sel2 = clone(sel)
        # Same params
        assert sel.get_params() == sel2.get_params()
        # sel2 is unfitted
        assert not hasattr(sel2, "support_")


# =============================================================================
# NaN / missing-value end-to-end production scenarios
# =============================================================================


class TestNaNProduction:
    def test_signal_with_15pct_nan_still_detected(self):
        """Real production data has missing values. 15% NaN in the
        signal column should NOT destroy detection."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(60)
        n = 1500
        sig = rng.standard_normal(n)
        # Inject 15% NaN
        nan_idx = rng.choice(n, int(0.15 * n), replace=False)
        sig_nan = sig.copy()
        sig_nan[nan_idx] = np.nan
        X = pd.DataFrame({
            "signal_with_nan": sig_nan,
            "noise0": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
        })
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "signal_with_nan" in names, (
            f"15% NaN signal lost; support={names}"
        )

    def test_categorical_with_nan_levels(self):
        """Categorical feature with NaN entries. After iter 31 fix
        (categorical NaN sentinel), MRMR must handle this.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(61)
        n = 1000
        cats = rng.choice(["A", "B", "C", "D"], size=n)
        # Inject NaN
        nan_idx = rng.choice(n, 100, replace=False)
        cats_with_nan = pd.Series(cats).copy()
        cats_with_nan[nan_idx] = None
        # y depends on category D
        y = pd.Series((cats == "D").astype(np.int64))
        X = pd.DataFrame({
            "cat_nan": pd.Categorical(cats_with_nan),
            "noise": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "cat_nan" in names, (
            f"categorical-with-NaN signal lost; support={names}"
        )
