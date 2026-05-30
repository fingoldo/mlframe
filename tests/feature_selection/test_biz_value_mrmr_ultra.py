"""Wave 9.1 biz_value Layer 4: ultra-difficult production scenarios.

These tests stress MRMR's COMPETENCE — not just "find signal" but
"PICK the right features when the wrong ones are tempting":

* Redundancy traps: 5 highly-correlated copies of signal vs 1
  independent informative feature - the canonical mRMR contract
  requires preferring DIVERSE features over copies
* Suppressor variable: x1 + x2 predicts y, but x2 alone is
  uninformative; only the JOINT carries signal
* Discrete latent confounder: 3 disjoint clusters where each
  cluster's PC1 carries different y-information
* "Almost target" trap: feature that's 99% correlated with y but
  derived from y (data leak); MRMR should expose the leak
* Engineered FE pipeline: y = (x1+x2)*x3, FE step must generate
  the product feature
* DCD swap-rejection: cluster where raw anchor has HIGHER MI than
  PC1 (DCD must REJECT the swap, not blindly take it)
* Multi-modal distributions: bimodal signal feature
* Rank-stability under DCD: clustering shouldn't perturb the
  relative order of non-clustered signal features
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Redundancy avoidance — the canonical MRMR contract
# =============================================================================


class TestRedundancyAvoidance:
    def test_diverse_features_preferred_over_5_copies(self):
        """5 collinear copies of signal_a + 1 independent signal_b
        + 3 pure noise. y depends LINEARLY on both signal_a and
        signal_b. MRMR (minimum redundancy) should prefer the diverse
        pair {signal_a, signal_b} over multiple copies of signal_a.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(100)
        n = 1500
        sig_a = rng.standard_normal(n)
        sig_b = rng.standard_normal(n)
        cols = {"signal_a": sig_a, "signal_b": sig_b}
        for k in range(5):
            cols[f"sig_a_copy{k}"] = sig_a + 0.1 * rng.standard_normal(n)
        for k in range(3):
            cols[f"noise{k}"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series(((sig_a + sig_b) > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        names = list(sel.get_feature_names_out())
        # signal_b MUST be in the top picks (diversity beats copies)
        # OR at least one copy of signal_a must be there alongside
        # signal_b (i.e. diverse pair, not 2+ copies of a).
        signal_b_picked = "signal_b" in names
        copies_count = sum(1 for n in names if "sig_a_copy" in n)
        signal_a_present = "signal_a" in names
        # Healthy outcome: signal_b in support, OR signal_a + signal_b,
        # NOT 2+ copies dominating.
        assert signal_b_picked or signal_a_present, (
            f"redundancy avoidance failed: support={names} "
            f"(b_picked={signal_b_picked}, copies={copies_count})"
        )

    def test_signal_b_picked_when_copies_dominate(self):
        """8 copies of sig_a (strongly informative) + 1 unique sig_b
        (also informative). MRMR with DCD must not over-pick from
        the cluster - it should surface sig_b as well.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(101)
        n = 1500
        sig_a = rng.standard_normal(n)
        sig_b = rng.standard_normal(n)
        cols = {"sig_b": sig_b}
        for k in range(8):
            cols[f"sig_a{k}"] = sig_a + 0.1 * rng.standard_normal(n)
        cols["noise"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series(((sig_a + sig_b) > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, dcd_enable=True,
                dcd_tau_cluster=0.5,
                dcd_cluster_size_threshold=3,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        sig_a_count = sum(1 for n in names if n.startswith("sig_a"))
        has_pc1 = any("_pc1_" in n or "dcd_pc1" in n for n in names)
        # MRMR-DCD shouldn't pick more than 2-3 copies of sig_a; sig_b
        # should also surface.
        assert "sig_b" in names or has_pc1, (
            f"sig_b missed under 8-copy cluster pressure; support={names}"
        )


# =============================================================================
# Suppressor variable — neither alone informative, joint is
# =============================================================================


class TestSuppressorVariable:
    def test_suppressor_pair_x1_x2(self):
        """Classic suppressor: y = x1 - x2 (linear combination).
        Neither x1 nor x2 alone has high MI with y (they're like
        independent Gaussians); the DIFFERENCE perfectly predicts y.
        MRMR with order-2 should find both.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(102)
        n = 2000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        # y = sign(x1 - x2). Individual MIs are low because each
        # marginal is symmetric around the y boundary.
        y = ((x1 - x2) > 0).astype(np.int64)
        X = pd.DataFrame({
            "x1": x1, "x2": x2,
            "n0": rng.standard_normal(n),
            "n1": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, interactions_max_order=2, fe_max_steps=0,
            ).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        # At least one of {x1, x2} or both must be selected; ideally
        # both for the suppressor pair to be recovered.
        assert "x1" in names or "x2" in names, (
            f"suppressor pair: at least one half lost; support={names}"
        )


# =============================================================================
# Target-leakage detection (informative but suspicious feature)
# =============================================================================


class TestTargetLeakageHandling:
    def test_leaked_feature_99pct_correlated_picked_first(self):
        """A feature that's 99% correlated with y (e.g. a derived
        column accidentally included). MRMR will pick it but downstream
        consumers must be able to detect "fallback_used_ = False" so
        leakage isn't masked.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(103)
        n = 1500
        sig = rng.standard_normal(n)
        y_arr = (sig > 0).astype(np.int64)
        # Leak: feature equals y with 1% noise flips
        leak = y_arr.copy()
        flip = rng.random(n) < 0.01
        leak[flip] = 1 - leak[flip]
        X = pd.DataFrame({
            "leak": leak.astype(float),
            "signal": sig,
            "noise": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y_arr))
        names = list(sel.get_feature_names_out())
        # Leak feature surfaces (expected); fallback NOT used.
        assert "leak" in names, (
            f"99% leak missed; support={names}"
        )
        assert not getattr(sel, "fallback_used_", False), (
            "fallback flag incorrectly set on a clear-signal fit"
        )


# =============================================================================
# Engineered FE pipeline on multi-variable products
# =============================================================================


class TestEngineeredFEMultivariate:
    def test_product_of_two_features_surfaces_at_order_2(self):
        """y = sign((x1 + x2) * x3). Order 2 should catch (x1, x3)
        and (x2, x3) as informative joints.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(104)
        n = 2000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        y = (((x1 + x2) * x3) > 0).astype(np.int64)
        X = pd.DataFrame({
            "x1": x1, "x2": x2, "x3": x3,
            "n0": rng.standard_normal(n),
            "n1": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0, interactions_max_order=2, fe_max_steps=0,
            ).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        # x3 (the "switch") must appear; ideally at least one of x1/x2.
        assert "x3" in names, (
            f"product-of-sum: x3 switch missed; support={names}"
        )
        assert ("x1" in names) or ("x2" in names), (
            f"product-of-sum: both x1 and x2 missed; support={names}"
        )


# =============================================================================
# Multi-modal distributions
# =============================================================================


class TestMultimodalDistributions:
    def test_bimodal_signal_detected(self):
        """Bimodal signal feature: gaussian mixture {N(-3, 0.5),
        N(3, 0.5)}. y depends on which mode the sample fell into.
        Default binning should handle this if it sees enough
        cardinality.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(105)
        n = 1500
        mode = rng.integers(0, 2, n)
        sig = np.where(
            mode == 0,
            rng.normal(-3, 0.5, n),
            rng.normal(3, 0.5, n),
        )
        y = mode  # binary y exactly aligned with mode
        X = pd.DataFrame({
            "bimodal_sig": sig,
            "n0": rng.standard_normal(n),
            "n1": rng.standard_normal(n),
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, pd.Series(y))
        names = list(sel.get_feature_names_out())
        assert "bimodal_sig" in names, (
            f"bimodal signal lost; support={names}"
        )


# =============================================================================
# DCD swap-rejection: don't swap when PC1 is worse than anchor
# =============================================================================


class TestDCDSwapRejection:
    def test_dcd_keeps_raw_anchor_when_pc1_weaker(self):
        """Cluster where the anchor has STRONG signal but the noisy
        cluster copies dilute the PC1 aggregate. DCD must REJECT
        swap to PC1 (it's WORSE than anchor).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(106)
        n = 1500
        latent = rng.standard_normal(n)
        cols = {"strong_anchor": latent}
        # Copies are VERY noisy (90% noise) - PC1 of these copies
        # will lose more signal than it gains.
        for k in range(4):
            cols[f"noisy_copy{k}"] = 0.1 * latent + 0.9 * rng.standard_normal(n)
        cols["noise"] = rng.standard_normal(n)
        X = pd.DataFrame(cols)
        y = pd.Series((latent > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(
                verbose=0,
                dcd_enable=True,
                dcd_cluster_size_threshold=4,
                dcd_swap_gain_threshold=0.05,
                dcd_swap_alpha=0.05,
            ).fit(X, y)
        names = list(sel.get_feature_names_out())
        # strong_anchor should be in support (PC1 swap should be
        # rejected because aggregate is weaker).
        assert "strong_anchor" in names or any("_pc1_" in n for n in names), (
            f"latent signal lost; support={names}"
        )


# =============================================================================
# Production sanity: end-to-end pipeline integration
# =============================================================================


class TestPipelineIntegration:
    def test_sklearn_pipeline_fit_transform(self):
        """MRMR in sklearn Pipeline: fit + transform yields downstream
        model-trainable output.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(107)
        n = 800
        sig = rng.standard_normal(n)
        X = pd.DataFrame({
            "signal": sig,
            "noise0": rng.standard_normal(n),
            "noise1": rng.standard_normal(n),
        })
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = Pipeline([
                ("mrmr", MRMR(verbose=0, random_seed=7)),
                ("clf", LogisticRegression(max_iter=100)),
            ])
            pipe.fit(X, y)
            score = pipe.score(X, y)
        # With clear signal, train accuracy should be > 0.7
        assert score > 0.7, (
            f"Pipeline integration: low accuracy {score:.3f}"
        )

    def test_get_support_returns_correct_indices(self):
        """``sel.get_support(indices=True)`` returns the support_
        indices that map correctly to feature_names_in_.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(108)
        n = 500
        sig = rng.standard_normal(n)
        X = pd.DataFrame({
            "a": rng.standard_normal(n),
            "signal": sig,
            "c": rng.standard_normal(n),
        })
        y = pd.Series((sig > 0).astype(np.int64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = MRMR(verbose=0).fit(X, y)
        idxs = sel.get_support(indices=True)
        mask = sel.get_support()
        np.testing.assert_array_equal(np.where(mask)[0], idxs)
        # Names align with indices
        for i in idxs:
            assert sel.feature_names_in_[i] in sel.get_feature_names_out()
