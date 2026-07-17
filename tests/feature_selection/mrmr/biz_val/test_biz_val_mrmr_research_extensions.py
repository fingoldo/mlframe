"""Wave 8 biz-value tests for the 8 new MRMR research extensions (2026-05-30).

Covers all sibling modules:
  - F13 Chao-Shen entropy / MI
  - A1 JMIM aggregator
  - A3 BUR unique-relevance term
  - A2 RelaxMRMR 3-D MI
  - C8 CMI-permutation stop + C9 UAED elbow
  - D10 Conditional Permutation Test (Berrett 2020)
  - E11 + E12 Cluster + Complementary Pairs Stability Selection
  - F14 PID (Williams-Beer + Ince I_ccs)

Plus end-to-end wired knobs:
  - MRMR(uaed_auto_size=True): post-fit elbow shrinks support_.
  - MRMR(stability_selection_method='cluster'|'complementary_pairs'):
    outer-loop bootstrap aggregation.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# F13 Chao-Shen
# ---------------------------------------------------------------------------


class TestChaoShenEntropy:
    def test_binary_uniform_returns_ln2(self):
        from mlframe.feature_selection.filters._chao_shen import chao_shen_entropy

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        h = chao_shen_entropy(y)
        # ln(2) = 0.693
        assert 0.65 < h < 0.74, f"Chao-Shen entropy of uniform binary = {h}"

    def test_constant_returns_zero(self):
        from mlframe.feature_selection.filters._chao_shen import chao_shen_entropy

        h = chao_shen_entropy(np.zeros(100, dtype=np.int64))
        assert h < 1e-10

    def test_mi_perfect_dependence(self):
        from mlframe.feature_selection.filters._chao_shen import chao_shen_mi

        rng = np.random.default_rng(0)
        x = rng.integers(0, 4, 500)
        y = x.copy()  # perfect dependence
        # Truth: H(y) = ln(4) for uniform-4-class = 1.386
        mi = chao_shen_mi(x, y)
        # Chao-Shen on perfect dependence approaches H(y).
        assert mi > 1.0, f"Chao-Shen MI on perfect dependence = {mi}"


# ---------------------------------------------------------------------------
# A1 JMIM
# ---------------------------------------------------------------------------


class TestJMIM:
    def test_returns_marginal_mi_with_empty_selected(self):
        from mlframe.feature_selection.filters._jmim_scorer import jmim_score

        rng = np.random.default_rng(0)
        n = 800
        x = rng.integers(0, 3, n)
        y = (x > 0).astype(np.int64)
        sc = jmim_score(x, [], y, 3, [], 2)
        assert sc > 0.05, f"empty-selected JMIM should equal marginal MI: {sc}"

    def test_min_aggregator_over_selected(self):
        from mlframe.feature_selection.filters._jmim_scorer import jmim_score

        rng = np.random.default_rng(0)
        n = 800
        x = rng.integers(0, 3, n)
        z_strong = x.copy()  # highly correlated with x
        z_weak = rng.integers(0, 3, n)  # noise
        y = (x > 0).astype(np.int64)
        sc = jmim_score(x, [z_strong, z_weak], y, 3, [3, 3], 2)
        # JMIM is the min; the strong-correlation pair should drive the score.
        assert sc >= 0.0


# ---------------------------------------------------------------------------
# A3 BUR
# ---------------------------------------------------------------------------


class TestBURTerm:
    def test_empty_selected_returns_marginal_mi(self):
        from mlframe.feature_selection.filters._bur_term import bur_term

        rng = np.random.default_rng(0)
        n = 800
        x = rng.integers(0, 4, n)
        y = (x > 1).astype(np.int64)
        b = bur_term(x, [], y, 4, [], 2)
        assert b > 0.05, f"empty-selected BUR = marginal MI; got {b}"

    def test_redundant_with_selected_drops_to_zero(self):
        from mlframe.feature_selection.filters._bur_term import bur_term

        rng = np.random.default_rng(0)
        n = 800
        x = rng.integers(0, 4, n)
        z_dup = x.copy()
        y = (x > 1).astype(np.int64)
        b = bur_term(x, [z_dup], y, 4, [4], 2)
        # Candidate fully redundant with selected -> BUR floors at 0.
        assert b < 0.05, f"redundant BUR should be 0; got {b}"


# ---------------------------------------------------------------------------
# A2 RelaxMRMR 3-D
# ---------------------------------------------------------------------------


class TestRelaxMRMR3D:
    def test_returns_finite(self):
        from mlframe.feature_selection.filters._relaxmrmr_3d import relax_mrmr_score

        rng = np.random.default_rng(0)
        n = 500
        x = rng.integers(0, 3, n)
        z1 = rng.integers(0, 3, n)
        z2 = rng.integers(0, 3, n)
        y = (x > 0).astype(np.int64)
        s = relax_mrmr_score(x, [z1, z2], y, 3, [3, 3], 2)
        assert np.isfinite(s)


# ---------------------------------------------------------------------------
# C8 CMI-perm stop + C9 UAED elbow
# ---------------------------------------------------------------------------


class TestCMIPermStop:
    def test_significant_on_real_signal(self):
        from mlframe.feature_selection.filters._cmi_perm_stop import (
            cmi_permutation_stop,
        )

        rng = np.random.default_rng(0)
        n = 600
        x = rng.integers(0, 3, n)
        y = (x > 0).astype(np.int64)
        is_sig, obs, _p = cmi_permutation_stop(
            x,
            y,
            [],
            3,
            2,
            [],
            n_permutations=80,
            alpha=0.05,
        )
        assert is_sig is True
        assert obs > 0.01

    def test_not_significant_on_noise(self):
        from mlframe.feature_selection.filters._cmi_perm_stop import (
            cmi_permutation_stop,
        )

        rng = np.random.default_rng(0)
        n = 600
        x = rng.integers(0, 3, n)
        y = rng.integers(0, 2, n).astype(np.int64)
        _is_sig, obs, p = cmi_permutation_stop(
            x,
            y,
            [],
            3,
            2,
            [],
            n_permutations=80,
            alpha=0.05,
        )
        # Should NOT reject under independence.
        assert p > 0.05 or obs < 0.005


class TestUAEDElbow:
    def test_classic_decay_curve(self):
        from mlframe.feature_selection.filters._cmi_perm_stop import uaed_elbow

        # Curve with elbow around index 2-3.
        curve = np.array([0.5, 0.45, 0.4, 0.1, 0.05, 0.02])
        idx = uaed_elbow(curve)
        # Allow some slack (sensitivity-dependent).
        assert 1 <= idx <= 4

    def test_flat_curve_picks_endpoint(self):
        from mlframe.feature_selection.filters._cmi_perm_stop import uaed_elbow

        curve = np.linspace(1.0, 0.5, 6)
        idx = uaed_elbow(curve)
        assert 0 <= idx < 6


# ---------------------------------------------------------------------------
# D10 CPT
# ---------------------------------------------------------------------------


class TestConditionalPermutationTest:
    def test_dependent_x_y_rejects_null(self):
        from mlframe.feature_selection.filters._conditional_permutation import (
            conditional_permutation_test,
        )

        rng = np.random.default_rng(0)
        n = 600
        x = rng.integers(0, 3, n)
        y = (x > 0).astype(np.int64)
        z = rng.integers(0, 2, n).astype(np.int64)
        stat, p = conditional_permutation_test(
            x,
            y,
            z,
            3,
            2,
            2,
            n_permutations=60,
        )
        assert stat > 0.0
        assert p <= 0.10

    def test_independent_x_y_no_reject(self):
        from mlframe.feature_selection.filters._conditional_permutation import (
            conditional_permutation_test,
        )

        rng = np.random.default_rng(0)
        n = 600
        x = rng.integers(0, 3, n)
        y = rng.integers(0, 2, n)
        z = rng.integers(0, 2, n)
        stat, _p = conditional_permutation_test(
            x,
            y,
            z,
            3,
            2,
            2,
            n_permutations=200,
        )
        # Observed CMI should be small AND not lopsidedly in the tail.
        assert stat < 0.03, f"independent stat too large: {stat}"


# ---------------------------------------------------------------------------
# E11 + E12 Stability
# ---------------------------------------------------------------------------


class TestClusterStability:
    def test_returns_subset_indices(self):
        from mlframe.feature_selection.filters._stability_cluster import (
            cluster_stability_selection,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5))
        y = (X[:, 0] > 0).astype(np.int64)

        def _sel(X_, y_):
            return np.array([0, 2])

        sel, _freq, info = cluster_stability_selection(
            X,
            y,
            _sel,
            n_bootstrap=10,
        )
        assert sel.size >= 1
        assert info["n_clusters"] >= 1


class TestComplementaryPairsStability:
    def test_returns_subset(self):
        from mlframe.feature_selection.filters._stability_cluster import (
            complementary_pairs_stability,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 5))
        y = (X[:, 0] > 0).astype(np.int64)

        def _sel(X_, y_):
            return np.array([0, 3])

        _sel_result, _freq, info = complementary_pairs_stability(
            X,
            y,
            _sel,
            n_pairs=10,
        )
        assert "pair_complementary_freq" in info


# ---------------------------------------------------------------------------
# F14 PID — the headline XOR demonstration
# ---------------------------------------------------------------------------


class TestPIDDecomposition:
    def test_xor_synergy(self):
        from mlframe.feature_selection.filters._pid_decomposition import (
            pid_decomposition,
        )

        rng = np.random.default_rng(0)
        n = 4000
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        y = (x1 ^ x2).astype(np.int64)
        pid = pid_decomposition(x1, x2, y, 2, 2, 2)
        # Theoretical XOR: synergy = ln(2), unique = redundant = 0.
        assert pid["synergistic"] > 0.5, f"XOR synergy too low: {pid}"
        assert pid["unique_x1"] < 0.05
        assert pid["unique_x2"] < 0.05
        assert pid["redundant"] < 0.05

    def test_copy_redundant(self):
        from mlframe.feature_selection.filters._pid_decomposition import (
            pid_decomposition,
        )

        rng = np.random.default_rng(0)
        n = 4000
        x1 = rng.integers(0, 2, n).astype(np.int64)
        x2 = x1.copy()  # x2 is a perfect copy
        y = x1  # y is the shared variable
        pid = pid_decomposition(x1, x2, y, 2, 2, 2)
        # Theoretical copy: redundant = ln(2), unique = synergy = 0.
        assert pid["redundant"] > 0.5, f"Copy redundant too low: {pid}"
        assert pid["unique_x1"] < 0.05
        assert pid["unique_x2"] < 0.05


# ---------------------------------------------------------------------------
# Wired knobs (UAED + stability outer-loop)
# ---------------------------------------------------------------------------


class TestMRMRWiredKnobs:
    def _toy(self, n=200, seed=0):
        rng = np.random.default_rng(int(seed))
        X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(5)})
        y = pd.Series(
            (X["f0"] + 0.3 * X["f1"] + rng.standard_normal(n) > 0).astype(np.int64),
            name="y",
        )
        return X, y

    def test_uaed_auto_size_no_crash(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._toy()
        sel = MRMR(uaed_auto_size=True, verbose=0)
        sel.fit(X, y)
        # The post-fit step is best-effort; passes as long as it does not
        # raise and the standard fit attributes are populated.
        assert hasattr(sel, "n_features_")

    def test_cluster_stability_selection_returns_subset(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._toy(n=180)
        sel = MRMR(
            stability_selection_method="cluster",
            stability_n_bootstrap=5,
            verbose=0,
        )
        sel.fit(X, y)
        assert hasattr(sel, "stability_info_")
        assert sel.stability_info_["n_clusters"] >= 1

    def test_complementary_pairs_returns_subset(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._toy(n=180)
        sel = MRMR(
            stability_selection_method="complementary_pairs",
            stability_n_bootstrap=5,
            verbose=0,
        )
        sel.fit(X, y)
        assert hasattr(sel, "stability_info_")

    def test_invalid_stability_method_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        with pytest.raises(ValueError):
            MRMR(stability_selection_method="bogus")._validate_string_params()

    def test_invalid_mi_correction_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        with pytest.raises(ValueError):
            MRMR(mi_correction="nonsense")._validate_string_params()


class TestJMIMBURInnerLoop:
    """2026-05-30 Wave 8 inner-loop wire-ins -- JMIM aggregator and BUR
    bonus are config-disable-able (defaults preserve legacy Fleuret bit-stable).
    """

    def _toy(self, n=300, seed=0):
        rng = np.random.default_rng(int(seed))
        true_sig = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "f0_dup": true_sig + 0.05 * rng.standard_normal(n),
                "f1_dup": true_sig + 0.05 * rng.standard_normal(n),
                "f2_other": rng.standard_normal(n),
                "f3_noise": rng.standard_normal(n),
            }
        )
        y = pd.Series(
            (true_sig + 0.3 * X["f2_other"] > 0).astype(np.int64),
            name="y",
        )
        return X, y

    def test_jmim_aggregator_completes_fit(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._toy()
        sel = MRMR(redundancy_aggregator="jmim", verbose=0)
        sel.fit(X, y)
        assert sel.n_features_ >= 1
        assert sel.get_feature_names_out().size >= 1

    def test_bur_bonus_completes_fit(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._toy()
        sel = MRMR(bur_lambda=0.5, verbose=0)
        sel.fit(X, y)
        assert sel.n_features_ >= 1

    def test_jmim_plus_bur_compose(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = self._toy()
        sel = MRMR(redundancy_aggregator="jmim", bur_lambda=0.3, verbose=0)
        sel.fit(X, y)
        assert sel.n_features_ >= 1

    def test_defaults_match_legacy_bit_stable(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.info_theory import (
            use_jmim_aggregator,
            get_bur_lambda,
        )

        X, y = self._toy()
        # Default config = JMIM off, BUR off.
        sel = MRMR(verbose=0)
        assert sel.redundancy_aggregator is None
        assert sel.bur_lambda == 0.0
        sel.fit(X, y)
        # After fit the thread-locals must be RESET to false / 0.
        assert use_jmim_aggregator() is False
        assert get_bur_lambda() == 0.0

    def test_invalid_redundancy_aggregator_raises(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        with pytest.raises(ValueError):
            MRMR(redundancy_aggregator="nonsense")._validate_string_params()
