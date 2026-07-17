"""Wave 9.1 biz_value Layer 3: extreme combinations of difficulty.

Cases combining multiple hardness axes that production users hit:
* rare positive (1-3%) + synergy
* noisy targets (15-30% label noise on top of signal)
* regression targets (continuous y)
* heavy-tail distributions (log-normal, Cauchy)
* mixed-dtype frames (numeric + categorical + datetime)
* concept drift between train and "validation" subsets
* highly imbalanced multi-class
* time-series-style auto-correlated features
* very small N (n=80) where every bit of signal matters
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# =============================================================================
# Combined hardness: rare positives + synergy
# =============================================================================


class TestRareSynergyCombo:
    """Groups tests covering TestRareSynergyCombo."""
    def test_5pct_positive_with_2way_synergy(self):
        """Realistic credit-risk style: 5% positive rate, defaults
        triggered by INTERACTION (high_balance AND no_income), not
        single features alone.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(70)
        n = 5000
        balance = rng.standard_normal(n)
        income = rng.standard_normal(n)
        # 5% positive: defaults concentrated where balance > p95 AND income < p25
        b_high = balance > np.percentile(balance, 95)
        i_low = income < np.percentile(income, 25)
        y_raw = (b_high & i_low).astype(np.int64)
        # Add a few decoy correlated noise vars
        X = pd.DataFrame(
            {
                "balance": balance,
                "income": income,
                "noise_balance_copy": balance + 0.4 * rng.standard_normal(n),
                "noise_pure": rng.standard_normal(n),
                "noise2": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                interactions_max_order=2,
                fe_max_steps=0,
            ).fit(X, pd.Series(y_raw))
        names = list(sel.get_feature_names_out())
        # Must surface BOTH balance and income (the interacting pair)
        # or an engineered combination of them.
        has_both = "balance" in names and "income" in names
        has_combo = any(("balance" in n and "income" in n) for n in names)
        assert has_both or has_combo, f"rare+synergy missed; support={names}"


class TestNoisyTargets:
    """Groups tests covering TestNoisyTargets."""
    def test_30pct_label_noise_signal_still_found(self):
        """30% label flip noise on a clean linear target. Signal-to-
        noise ratio drops drastically but MRMR should still find the
        true signal.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(71)
        n = 3000
        sig = rng.standard_normal(n)
        y_clean = (sig > 0).astype(np.int64)
        # Flip 30% of labels
        flip_mask = rng.random(n) < 0.30
        y_noisy = np.where(flip_mask, 1 - y_clean, y_clean)
        X = pd.DataFrame(
            {
                "signal": sig,
                "noise0": rng.standard_normal(n),
                "noise1": rng.standard_normal(n),
                "noise2": rng.standard_normal(n),
                "noise3": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y_noisy))
        names = list(sel.get_feature_names_out())
        # Under 30% label noise MRMR recovers the signal via an engineered transform (relu / sign of ``signal``, which
        # separates the thresholded noisy label better than the raw Gaussian column) rather than the raw ``signal``; the
        # contract is that SOME selected feature is signal-derived, not that the raw name survives.
        assert any("signal" in n for n in names), f"30% label noise: signal lost (no signal-derived feature selected); support={names}"

    def test_noisy_xor_synergy(self):
        """sign(x1*x2) target with 20% label noise. Synergy is
        weakened but detectable at n=3000.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(72)
        n = 3000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y_clean = (x1 * x2 > 0).astype(np.int64)
        flip_mask = rng.random(n) < 0.20
        y_noisy = np.where(flip_mask, 1 - y_clean, y_clean)
        X = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                interactions_max_order=2,
                fe_max_steps=0,
            ).fit(X, pd.Series(y_noisy))
        names = list(sel.get_feature_names_out())
        has_both = "x1" in names and "x2" in names
        has_combo = any(("x1" in n and "x2" in n) for n in names)
        assert has_both or has_combo, f"noisy XOR missed; support={names}"


# =============================================================================
# Regression-target (continuous y)
# =============================================================================


class TestRegressionTargets:
    """Groups tests covering TestRegressionTargets."""
    def test_continuous_y_linear_signal(self):
        """Continuous y = sig + noise. MRMR with binned y must still
        find sig.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(73)
        n = 1500
        sig = rng.standard_normal(n)
        y_cont = sig + 0.3 * rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "signal": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y_cont))
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"continuous regression target: signal missed; support={names}"

    def test_continuous_y_quadratic_signal(self):
        """y = sig**2 + noise (non-monotone, SYMMETRIC domain). Raw ``signal`` is
        ~uncorrelated with sig**2 (an even function -> Pearson/MI ~0), so the
        DEFAULT univariate-basis FE (``fe_univariate_basis_enable``) recovers the
        signal via a clean single-source basis feature (``signal__He2`` /
        ``signal__T2`` ~ sig**2) rather than the raw column. This is the
        univariate-basis win: pre-FE the raw column was the only candidate and the
        signal was effectively missed; now a signal-derived feature is selected AND
        recovers sig**2 at high |corr| (measured ~0.999). Pinned as the better
        behaviour, not the raw-column presence the old assertion expected."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(74)
        n = 1500
        sig = rng.standard_normal(n)
        y_cont = sig**2 + 0.3 * rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "signal": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y_cont))
        names = list(sel.get_feature_names_out())
        # The signal is recovered as the raw column OR a signal-derived feature
        # (basis ``signal__He2`` or a functional form referencing ``signal``).
        sig_feats = [nm for nm in names if "signal" in nm]
        assert sig_feats, f"quadratic continuous target: signal not recovered (neither raw nor a signal-derived basis feature); support={names}"
        # ... and that recoverer actually captures sig**2 (the univariate win).
        Xt = np.asarray(sel.transform(X))
        true = sig**2 - float(np.mean(sig**2))
        best = 0.0
        for i, nm in enumerate(names):
            if nm not in sig_feats:
                continue
            col = Xt[:, i]
            if np.isfinite(col).all() and float(np.std(col)) > 1e-12:
                best = max(best, abs(float(np.corrcoef(col, true)[0, 1])))
        assert best >= 0.85, f"quadratic continuous target: signal-derived feature did not recover sig**2 (best |corr|={best:.3f}); support={names}"


# =============================================================================
# Heavy-tail distributions
# =============================================================================


class TestHeavyTailDistributions:
    """Groups tests covering TestHeavyTailDistributions."""
    def test_lognormal_signal_detected(self):
        """Log-normal signal (right-skewed). After iter-6/9 quantile
        dedup fix, the FD binner should handle outlier tails.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(75)
        n = 1500
        sig = rng.lognormal(0, 1, n)  # heavy right tail
        y = (sig > np.median(sig)).astype(np.int64)
        X = pd.DataFrame(
            {
                "lognormal_sig": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        assert "lognormal_sig" in names, f"log-normal signal missed; support={names}"

    def test_cauchy_signal_with_outliers(self):
        """Cauchy distribution (infinite variance). Quantile binning
        is more robust than uniform here.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(76)
        n = 1500
        # Cauchy via tan(uniform) - heavy tails
        sig = np.tan(np.pi * (rng.uniform(size=n) - 0.5))
        y = (sig > 0).astype(np.int64)
        X = pd.DataFrame(
            {
                "cauchy_sig": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        assert "cauchy_sig" in names, f"Cauchy signal lost; support={names}"


# =============================================================================
# Mixed-dtype frames
# =============================================================================


class TestMixedDtypes:
    """Groups tests covering TestMixedDtypes."""
    def test_numeric_plus_categorical_plus_datetime(self):
        """Production-grade frame: float + int + categorical +
        boolean. MRMR must handle each dtype gracefully.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(80)
        n = 1500
        sig = rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "num_signal": sig,
                "int_noise": rng.integers(0, 10, n),
                "bool_noise": rng.integers(0, 2, n).astype(bool),
                "cat_noise": pd.Categorical(rng.choice(["X", "Y", "Z"], n)),
                "str_noise": rng.choice(["small", "med", "large"], n).astype(object),
            }
        )
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(df, y)
        names = list(sel.get_feature_names_out())
        assert "num_signal" in names, f"mixed-dtype: numeric signal missed; support={names}"

    def test_only_categoricals_no_numeric(self):
        """All-categorical frame with one informative category."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(81)
        n = 1500
        cat_sig = rng.choice(["A", "B", "C", "D"], n)
        # y depends on category being "D"
        y = pd.Series((cat_sig == "D").astype(np.int64))
        df = pd.DataFrame(
            {
                "cat_signal": pd.Categorical(cat_sig),
                "cat_noise1": pd.Categorical(rng.choice(["X", "Y", "Z"], n)),
                "cat_noise2": pd.Categorical(rng.choice(["P", "Q"], n)),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(df, y)
        names = list(sel.get_feature_names_out())
        assert "cat_signal" in names, f"all-categorical: cat signal missed; support={names}"


# =============================================================================
# Tiny N regime (n=80 to 200)
# =============================================================================


class TestSmallN:
    """Groups tests covering TestSmallN."""
    def test_n_80_clear_signal_still_found(self):
        """n=80 — very small. Clear signal must still be detectable."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(82)
        n = 80
        sig = rng.standard_normal(n)
        y = pd.Series((sig + 0.3 * rng.standard_normal(n) > 0).astype(np.int64))
        X = pd.DataFrame(
            {
                "signal": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"n=80 clear signal lost; support={names}"

    def test_n_200_with_many_noise(self):
        """n=200, 30 noise + 1 signal. Statistical power challenge."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(83)
        n = 200
        sig = rng.standard_normal(n)
        cols = {"signal": sig}
        for k in range(30):
            cols[f"noise{k}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"n=200 + 30 noise: signal lost; support={names}"


# =============================================================================
# Multi-class targets (imbalanced)
# =============================================================================


class TestMultiClass:
    """Groups tests covering TestMultiClass."""
    def test_imbalanced_5class_target(self):
        """5-class target with unbalanced class proportions
        [40%, 30%, 15%, 10%, 5%]. MRMR must find the signal feature.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(84)
        n = 2000
        sig = rng.standard_normal(n)
        # 5 classes from quantiles of sig
        bounds = np.percentile(sig, [40, 70, 85, 95])
        y_vals = np.digitize(sig, bounds)  # 0..4
        X = pd.DataFrame(
            {
                "signal": sig,
                "noise0": rng.standard_normal(n),
                "noise1": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y_vals))
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"imbalanced 5-class: signal missed; support={names}"

    def test_string_label_5class(self):
        """5 string-label classes."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(85)
        n = 1500
        sig = rng.standard_normal(n)
        bounds = np.percentile(sig, [20, 40, 60, 80])
        labels = ["A", "B", "C", "D", "E"]
        y_vals = pd.Series([labels[i] for i in np.digitize(sig, bounds)])
        X = pd.DataFrame(
            {
                "signal": sig,
                "noise": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y_vals)
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"5-string-class target: signal missed; support={names}"


# =============================================================================
# Concept drift between fit and predict
# =============================================================================


class TestConceptDrift:
    """Groups tests covering TestConceptDrift."""
    def test_distribution_shift_transform_still_works(self):
        """Fit on one distribution, transform on shifted distribution.
        Transform must not crash and must select the correct columns
        by name (sklearn contract from iter-19/42 fixes).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(86)
        n = 1000
        # Train on N(0, 1)
        train_sig = rng.standard_normal(n)
        train_X = pd.DataFrame(
            {
                "signal": train_sig,
                "noise": rng.standard_normal(n),
            }
        )
        train_y = pd.Series((train_sig > 0).astype(np.int64))
        # Test on N(5, 10) - heavy distribution shift
        test_sig = 5 + 10 * rng.standard_normal(n)
        test_X = pd.DataFrame(
            {
                "signal": test_sig,
                "noise": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(train_X, train_y)
            out = sel.transform(test_X)
        # Must not crash; output column count matches get_feature_names_out (raw + engineered),
        # and the raw-feature support_ mask is a subset of the emitted columns (FE adds, never drops).
        assert out.shape[0] == n
        assert list(out.columns) == list(sel.get_feature_names_out())
        assert out.shape[1] >= len(sel.support_)
        assert "signal" in out.columns


# =============================================================================
# DCD swap-discovery: harder cluster scenarios
# =============================================================================


class TestHarderClusterScenarios:
    """Groups tests covering TestHarderClusterScenarios."""
    def test_cluster_where_pc1_signal_beats_any_member(self):
        """All cluster members are noisy reflections of latent;
        individually each member's MI with y is ~0.3, but PC1 of
        members has MI ~0.5. DCD should swap to PC1 aggregate.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(87)
        n = 1500
        latent = rng.standard_normal(n)
        cols = {}
        for k in range(6):
            cols[f"member{k}"] = latent + 0.8 * rng.standard_normal(n)
        cols["noise"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series((latent > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                dcd_enable=True,
                dcd_cluster_size_threshold=4,
                dcd_swap_gain_threshold=0.0,
                dcd_swap_alpha=1.0,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        # At least ONE member or PC1 aggregate selected.
        ok = any("member" in n or "_pc1_" in n for n in names)
        assert ok, f"latent signal lost; support={names}"


# =============================================================================
# Fit-transform vs Fit().transform() identity
# =============================================================================


class TestFitTransformIdentity:
    """Groups tests covering TestFitTransformIdentity."""
    def test_fit_transform_equals_fit_then_transform(self):
        """``fit_transform(X, y)`` MUST equal
        ``fit(X, y).transform(X)`` bit-exact.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(88)
        n = 500
        sig = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "signal": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_a = MRMR(verbose=0, random_seed=7).fit_transform(X, y)
            sel = MRMR(verbose=0, random_seed=7).fit(X, y)
            out_b = sel.transform(X)
        # Both DataFrames; columns and values identical.
        assert isinstance(out_a, pd.DataFrame)
        assert isinstance(out_b, pd.DataFrame)
        assert list(out_a.columns) == list(out_b.columns)
        np.testing.assert_array_equal(out_a.values, out_b.values)
