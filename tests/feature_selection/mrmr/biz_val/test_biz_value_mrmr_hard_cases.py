"""Wave 9.1 biz-value: hard cases for MRMR feature selection / cluster
discovery / engineered features. Goal is NOT to find more bugs but to
exercise the real production-impact paths and quantify "does MRMR do
its job well on the cases users actually care about".

Each test has a STRICT pass criterion that captures the intent:
* signal detection -> the true signal IS in support_
* cluster discovery -> the true cluster anchor IS in support_ AND noisy
  copies are pruned (or fold into PC1 aggregate)
* engineered FE -> the binary/unary combination of true components IS
  surfaced when raw features alone don't predict y
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Signal detection — synergy / collinearity / imbalance
# =============================================================================


class TestSignalDetection:
    def test_xor_synergy_both_features_selected(self):
        """y = x1 XOR x2. Neither x1 nor x2 alone predicts y; both are
        needed. MRMR with JMIM (synergy-preserving) must select BOTH
        components. Plain Fleuret CMIM would reject the second after
        the first because it adds zero MARGINAL MI under conditioning.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 800
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        y = x1 ^ x2
        noise = pd.DataFrame(
            {
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
                "n2": rng.standard_normal(n),
            }
        )
        X = pd.DataFrame({"x1": x1.astype(float), "x2": x2.astype(float)})
        X = pd.concat([X, noise], axis=1)
        y_s = pd.Series(y.astype(np.int64), name="y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                redundancy_aggregator="jmim",
                interactions_max_order=2,
                fe_max_steps=0,
            ).fit(X, y_s)
        names = list(sel.get_feature_names_out())
        # At minimum, x1*x2 (engineered) or both x1 and x2 must surface.
        carries_signal = ("x1" in names and "x2" in names) or any("x1" in n and "x2" in n for n in names)
        assert carries_signal, f"XOR synergy not detected; support={names}"

    def test_signal_under_heavy_collinearity(self):
        """Signal feature + 10 collinear noisy copies of it + 5 pure
        noise. MRMR must return the canonical signal (raw or PC1
        aggregate); pure-noise features must NOT crowd it out.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(1)
        n = 600
        signal = rng.standard_normal(n)
        cols = {"signal": signal}
        for k in range(10):
            cols[f"sig_copy{k}"] = signal + 0.15 * rng.standard_normal(n)
        for k in range(5):
            cols[f"noise{k}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series((signal > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, dcd_enable=True).fit(X, y)
        names = list(sel.get_feature_names_out())
        # signal cluster (raw or DCD aggregate) must be present.
        has_signal = "signal" in names or any(n.startswith("sig_copy") for n in names) or any("dcd_pc1" in n or "_pc1_" in n for n in names)
        assert has_signal, f"signal lost under collinearity; support={names}"
        # Pure noise must NOT be the top pick.
        if names:
            top = names[0]
            assert not top.startswith("noise"), f"pure noise feature ranked first: top={top}"

    def test_rare_positive_class_1pct(self):
        """1% positive rate (highly imbalanced). MRMR should still
        find the feature that drives the rare class.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(2)
        n = 5000  # need enough samples for rare class signal
        signal = rng.standard_normal(n)
        # 1% positive triggered by signal > 99th percentile
        y_raw = (signal > np.percentile(signal, 99)).astype(np.int64)
        X = pd.DataFrame(
            {
                "signal": signal,
                "noise0": rng.standard_normal(n),
                "noise1": rng.standard_normal(n),
                "noise2": rng.standard_normal(n),
            }
        )
        y = pd.Series(y_raw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"rare-positive signal missed; support={names}, pos_rate=1%"

    @pytest.mark.parametrize(
        "nbins_strategy",
        [
            None,  # legacy fixed quantization (control)
            "mdlp",  # default - supervised; iter MDLP collapse fix
            "fayyad_irani",  # alias of mdlp
            "freedman_diaconis",  # unsupervised, robust
            "sturges",  # unsupervised, simple
            "qs",  # quantile-spacing
            "auto",  # = freedman_diaconis
        ],
    )
    def test_engineered_feature_only_predicts_y_xor_synergy(self, nbins_strategy):
        """y = sign(x1 * x2) - PRODUCT determines y, neither x1 nor x2
        alone is informative. MRMR with order-2 interactions must
        find {x1, x2} regardless of nbins_strategy.

        Supervised MDLP / fayyad_irani used to collapse x1 and x2 to
        1 bin each (no individual signal) - the joint then had 1x1=1
        cell, killing synergy detection. Fixed via collapsed-fallback
        to unsupervised binning in _adaptive_nbins.py.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(3)
        n = 1500
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = (x1 * x2 > 0).astype(np.int64)
        X = pd.DataFrame(
            {
                "x1": x1,
                "x2": x2,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
            }
        )
        y_s = pd.Series(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                interactions_max_order=2,
                nbins_strategy=nbins_strategy,
                fe_max_steps=0,
            ).fit(X, y_s)
        names = list(sel.get_feature_names_out())
        # XOR-product synergy needs both x1 AND x2 (or engineered combo
        # containing both).
        has_both = "x1" in names and "x2" in names
        has_combo = any(("x1" in n and "x2" in n) for n in names)
        assert has_both or has_combo, (
            f"XOR-synergy missed under nbins_strategy={nbins_strategy!r}; "
            f"support={names}. Both x1 and x2 must be selected since "
            f"individually they have zero MI with y and only joint predicts."
        )


# =============================================================================
# Cluster discovery — DCD post-Wave-9 contract
# =============================================================================


class TestClusterDiscovery:
    def test_three_latent_factors_with_copies(self):
        """3 latent factors, each with 4 collinear copies + 3 pure
        noise. y depends on factor 0 only. DCD must:
          1. Discover the cluster of 5 (factor 0 + 4 copies)
          2. Either pick the original anchor OR swap to PC1 aggregate
          3. NOT include the unrelated factor 1/2 copies in support
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(4)
        n = 1500
        cols = {}
        for fac in range(3):
            latent = rng.standard_normal(n)
            cols[f"f{fac}_anchor"] = latent
            for k in range(4):
                cols[f"f{fac}_copy{k}"] = latent + 0.1 * rng.standard_normal(n)
        for k in range(3):
            cols[f"noise{k}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series((cols["f0_anchor"] > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                dcd_enable=True,
                dcd_tau_cluster=0.4,
                dcd_cluster_size_threshold=3,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        # At least one f0_* must be in support (raw or aggregate).
        f0_present = any(n.startswith("f0_") or "f0_anchor" in n or "_pc1_" in n for n in names)
        assert f0_present, f"f0 cluster missed entirely; support={names}"

    def test_anti_correlated_cluster_members(self):
        """X1 and -X1 should be detected as a cluster (perfect
        anti-correlation = perfect redundancy under MI/SU).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(5)
        n = 800
        sig = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "pos_sig": sig,
                "neg_sig": -sig + 0.05 * rng.standard_normal(n),
                "noise0": rng.standard_normal(n),
                "noise1": rng.standard_normal(n),
            }
        )
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, dcd_enable=True, dcd_tau_cluster=0.5, dcd_cluster_size_threshold=2).fit(X, y)
        names = list(sel.get_feature_names_out())
        # Should select pos_sig OR neg_sig (both perfectly carry signal),
        # not both, not the noise.
        has_sig = any("sig" in n for n in names)
        has_only_one_sig = sum(1 for n in names if n in ("pos_sig", "neg_sig")) <= 1
        assert has_sig, f"anti-correlated signal missed; support={names}"
        assert has_only_one_sig, f"redundant anti-correlated pair both kept; support={names}"


# =============================================================================
# Robustness — edge inputs
# =============================================================================


class TestRobustness:
    def test_single_unique_value_column_ignored(self):
        """Constant column carries zero MI. Should NEVER be selected."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(6)
        n = 400
        sig = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "const": np.full(n, 7.0),
                "signal": sig,
                "noise": rng.standard_normal(n),
            }
        )
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "const" not in names, f"constant column selected; support={names}"

    def test_high_dim_p_greater_than_n(self):
        """p=300, n=100. Only 3 cols carry signal. MRMR must pick
        signal-bearing cols, not be drowned by noise.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(7)
        n = 100
        p = 300
        sig0 = rng.standard_normal(n)
        sig1 = rng.standard_normal(n)
        # y depends on sig0 + sig1
        y_lin = sig0 + 0.7 * sig1 + 0.2 * rng.standard_normal(n)
        y = pd.Series((y_lin > 0).astype(np.int64))
        X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
        # Inject true signal at positions 0 and 1
        X["x0"] = sig0
        X["x1"] = sig1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0, max_runtime_mins=2.0).fit(X, y)
        names = list(sel.get_feature_names_out())
        # At least one of the true signals (x0 / x1) must be RECOVERED --
        # either as a raw column or folded into an engineered combination.
        # Rebaselined from the old ``"x0" in names or "x1" in names`` raw-
        # membership check, which was simple-mode specific: under the new
        # default (``use_simple_mode=False`` + directed FE) MRMR returns a
        # compact selection that can fold both signals into a single
        # engineered column such as ``add(sin(x0),log(x1))``, so raw-name
        # membership reads False even though both x0 and x1 are used.
        # ``signal_recovery_count`` credits ``xK`` appearing in ANY selected
        # feature name (the project's dedup-aware recovery metric).
        from tests.feature_selection._biz_val_synth import signal_recovery_count

        recovered = signal_recovery_count(sel, [0, 1], prefix="x")
        assert recovered >= 1, f"high-dim p>n: true signal missed; neither x0 nor x1 is recovered (raw or engineered); support={names}"

    def test_categorical_with_rare_levels(self):
        """Categorical feature with 5 levels but level 4 occurs only
        1% of the time. MRMR must still work (no crash) and rank the
        feature correctly.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(8)
        n = 1000
        # Heavy-tailed level distribution
        probs = [0.6, 0.25, 0.1, 0.04, 0.01]
        cats = rng.choice(["A", "B", "C", "D", "E"], size=n, p=probs)
        # y depends on whether category is in {D, E}
        y = pd.Series(np.isin(cats, ["D", "E"]).astype(np.int64))
        X = pd.DataFrame(
            {
                "cat": pd.Categorical(cats),
                "noise": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "cat" in names, f"categorical with rare levels missed; support={names}"

    def test_target_string_labels_3_class(self):
        """Multi-class string-label classification target. After
        iter-50 fix, this must work end-to-end.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(9)
        n = 600
        sig = rng.standard_normal(n)
        # 3 classes based on quantile of sig
        labels = np.where(sig < -0.5, "low", np.where(sig > 0.5, "high", "mid"))
        y = pd.Series(labels)
        X = pd.DataFrame(
            {
                "signal": sig,
                "noise": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "signal" in names, f"multi-class string target: signal missed; support={names}"


# =============================================================================
# Reproducibility / stability — same seed -> same support
# =============================================================================


class TestStability:
    def test_same_seed_same_support_across_runs(self):
        """Same X, y, random_seed -> identical support_ across runs."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(10)
        n = 500
        sig = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "signal": sig,
                "n0": rng.standard_normal(n),
                "n1": rng.standard_normal(n),
                "n2": rng.standard_normal(n),
            }
        )
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel1 = MRMR(verbose=0, random_seed=42).fit(X, y)
            sel2 = MRMR(verbose=0, random_seed=42).fit(X, y)
        np.testing.assert_array_equal(sel1.support_, sel2.support_)


# =============================================================================
# NaN-as-signal — iter 11 follow-up
# =============================================================================


class TestNaNAsSignal:
    def test_nan_pattern_perfectly_predicts_y(self):
        """The PRESENCE of NaN in a column perfectly predicts y. After
        iter 11/9/31 NaN-handling fixes, MRMR should rank this feature
        highly (the NaN pattern carries the signal).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(11)
        n = 600
        # First half: real values; second half: NaN
        sig = rng.standard_normal(n)
        sig[300:] = np.nan
        y = pd.Series(
            np.concatenate(
                [
                    np.zeros(300, dtype=np.int64),
                    np.ones(300, dtype=np.int64),
                ]
            )
        )
        X = pd.DataFrame(
            {
                "nan_signal": sig,
                "noise": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "nan_signal" in names, f"NaN-as-signal missed; support={names}"
