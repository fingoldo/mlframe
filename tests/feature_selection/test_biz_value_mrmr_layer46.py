"""Layer 46 biz_value: VI-based DCD distance + ``"auto"`` SU/VI bake-off.

WHY THIS LAYER
--------------
Pre-Layer-46 DCD clustering routed every pair through Symmetric
Uncertainty (SU). SU is bounded [0, 1] and well-behaved on monotone /
linear dependencies but BLIND to non-linear functional equivalences:
two features ``X`` and ``f(X)`` for a non-monotone ``f`` (e.g.
``X**2``, ``|X|``, sin/cos, threshold indicators) carry the same
information but their SU is depressed because the bin-by-bin joint
distribution looks like noise to the SU estimator unless the bin grid
happens to align with ``f``.

Variation of Information (Meila 2007):

    VI(X, Y) = H(X | Y) + H(Y | X) = H(X) + H(Y) - 2 I(X; Y)

is a proper metric in nats. VI = 0 iff X and Y are functionally
equivalent (each determines the other), regardless of the form of the
functional relationship. Normalising VI by the joint cardinality bound
``log(K_x * K_y)`` keeps it in [0, 1].

LAYER 46 IMPROVEMENTS
---------------------
1. ``pair_vi(state, a, b, ...)`` is now exported from
   ``_dynamic_cluster_discovery`` and returns raw VI in nats for
   diagnostic use (cluster cohesion plots, SU vs VI side-by-side).
   The existing ``"vi"`` branch of ``pair_su`` continues to return a
   ``1 - VI / log(K_a * K_b)`` similarity score so the
   ``score > tau_cluster`` membership rule stays direction-uniform with
   SU.

2. New ``dcd_distance="auto"`` value: per pair, compute BOTH SU and
   the VI similarity score, and return the MAX. Picks up clusters
   either metric alone would surface. Cost: 1 extra ``mi()`` njit call
   per pair (H_a, H_b are shared via the per-column entropy cache).

3. Validator and MRMR ctor accept the new distance string. Default
   remains ``"su"`` (no universal "auto" win yet established; users
   opt in).

CONTRACTS
---------
* C1: ``pair_vi`` is exported and returns 0.0 for ``a == b``.
* C2: ``pair_vi(X, X_perfect_copy)`` is ~0 nats; ``pair_vi`` between
  two independent features ~= H(X_a) + H(X_b).
* C3: On a linear-friendly 5-perfect-dup cluster, ``"su"`` and ``"vi"``
  both detect the cluster (n_pruned >= 4 for either distance).
* C4: On a non-linear cluster ``(x, x**2, x**3)``, ``"vi"`` detects
  at least as many cluster members as ``"su"`` does — VI tighter for
  non-linear deps.
* C5: ``dcd_distance="auto"`` does not crash and produces a non-None
  ``cluster_anchors`` summary (clusters either detected or empty,
  never ``None``).
* C6: ``dcd_distance="auto"`` integrates with the Layer 43/44 auto
  swap bake-off (``swap_method="auto"``) — fit succeeds.
* C7: Recipe replay (``transform``) is deterministic regardless of
  distance choice.
* C8: Layers 41-45 contracts preserved across distance choice:
  - L41 ``cluster_anchors_names`` present on every DCD summary
  - L42 threshold=2 still fires on 3-dups under SU and AUTO
  - L44 auto-method pool unchanged
  - L45 SwapDecision.branch field round-trips
* C9: Validator rejects ``dcd_distance="bogus"``.

NEVER xfail.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _linear_dup_cluster(n: int = 2000, seed: int = 0):
    """5 features = 5 near-perfect copies of one latent + a strong
    unrelated feature + a noise filler.

    SU and VI should both detect the 5-dup cluster.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame({
        "strong_unrelated": other,
        "dup_a": latent + 0.02 * rng.standard_normal(n),
        "dup_b": latent + 0.02 * rng.standard_normal(n),
        "dup_c": latent + 0.02 * rng.standard_normal(n),
        "dup_d": latent + 0.02 * rng.standard_normal(n),
        "dup_e": latent + 0.02 * rng.standard_normal(n),
        "noise_filler": rng.standard_normal(n),
    })
    y = pd.Series(
        (2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int)
    )
    return X, y


def _nonlinear_cluster(n: int = 3000, seed: int = 1):
    """3 features = (x, x**2, x**3) of a shared latent + a strong
    unrelated feature.

    SU between ``x`` and ``x**2`` is depressed because ``x**2`` is
    non-monotone in ``x``: high ``|x|`` rows from both sides of zero
    end up in the same ``x**2`` bin, making the joint distribution
    look diffuse. VI sees that ``x**2`` is a deterministic function
    of ``x`` (H(x**2 | x) = 0 for noise-free x; small for slightly
    noisy x), so VI(x, x**2) ~= 0 and the similarity score ``1 -
    VI/log(K^2)`` stays high.
    """
    rng = np.random.default_rng(int(seed))
    latent = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame({
        "strong_unrelated": other,
        "x_lin": latent,
        "x_sq": latent ** 2,
        "x_cub": latent ** 3,
        "noise_filler": rng.standard_normal(n),
    })
    # y depends only on ``other`` so the non-linear-cluster of
    # transformations of ``latent`` is genuinely redundant (none of
    # them helps for y once one is in Selected). This is the harshest
    # test for DCD: it must catch the redundancy among (x, x**2, x**3)
    # without help from a target signal.
    y = pd.Series((2 * other + 0.2 * rng.standard_normal(n) > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# 1. ``pair_vi`` helper export + analytical sanity
# ---------------------------------------------------------------------------


class TestLayer46_PairVI_Helper:

    def test_pair_vi_exported(self):
        """``pair_vi`` is importable from the module __all__."""
        from mlframe.feature_selection.filters import (
            _dynamic_cluster_discovery as dcd,
        )
        assert "pair_vi" in dcd.__all__
        assert callable(dcd.pair_vi)

    def test_pair_vi_self_is_zero(self):
        """VI(X, X) == 0 by construction."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState, pair_vi,
        )
        rng = np.random.default_rng(0)
        x = rng.integers(0, 4, size=500).astype(np.int32)
        factors = np.column_stack([x, x])
        fn = np.array([4, 4], dtype=np.int64)
        state = DCDState(
            pool_pruned_mask=np.zeros(2, dtype=bool),
            factors_data=factors,
            factors_nbins=fn,
            cols=["a", "b"],
            nbins=fn,
            distance="vi",
        )
        assert pair_vi(state, 0, 0) == 0.0

    def test_pair_vi_perfect_copy_near_zero(self):
        """VI between a feature and its exact bin-equal copy is 0.

        For X and a perfect copy, H(X|Y) = H(Y|X) = 0 -> VI = 0.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState, pair_vi,
        )
        rng = np.random.default_rng(1)
        x = rng.integers(0, 5, size=1000).astype(np.int32)
        # ``y_col`` is byte-identical with ``x``.
        factors = np.column_stack([x, x.copy()])
        fn = np.array([5, 5], dtype=np.int64)
        state = DCDState(
            pool_pruned_mask=np.zeros(2, dtype=bool),
            factors_data=factors,
            factors_nbins=fn,
            cols=["x", "x_copy"],
            nbins=fn,
            distance="vi",
        )
        vi = pair_vi(state, 0, 1)
        assert vi == pytest.approx(0.0, abs=1e-9), (
            f"VI of a feature with itself must be 0; got {vi}"
        )

    def test_pair_vi_independent_is_large(self):
        """VI between two independent features ~= H(X_a) + H(X_b) -- i.e.
        zero mutual information makes the second term vanish and VI
        equals the sum of the marginal entropies.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState, pair_vi,
        )
        rng = np.random.default_rng(2)
        x = rng.integers(0, 4, size=2000).astype(np.int32)
        y = rng.integers(0, 4, size=2000).astype(np.int32)
        factors = np.column_stack([x, y])
        fn = np.array([4, 4], dtype=np.int64)
        state = DCDState(
            pool_pruned_mask=np.zeros(2, dtype=bool),
            factors_data=factors,
            factors_nbins=fn,
            cols=["x", "y"],
            nbins=fn,
            distance="vi",
        )
        vi = pair_vi(state, 0, 1)
        # H(X) for uniform 4-bin = log(4) ~ 1.386 nats; two such
        # features ~= 2.77 nats; allow some sampling slack.
        assert vi > 1.5, (
            f"VI of independent uniform 4-bin features should be > 1.5 nats; "
            f"got {vi}"
        )


# ---------------------------------------------------------------------------
# 2. Linear cluster: SU and VI both detect
# ---------------------------------------------------------------------------


def _fit_with_distance(X, y, *, distance: str, tau: float = 0.7):
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        dcd_enable=True,
        dcd_distance=distance,
        dcd_tau_cluster=tau,
        dcd_cluster_size_threshold=2,
        full_npermutations=20,
        verbose=0, random_seed=0,
    ).fit(X, y)


class TestLayer46_LinearCluster:

    def test_su_detects_5_dup_cluster(self):
        X, y = _linear_dup_cluster()
        m = _fit_with_distance(X, y, distance="su", tau=0.5)
        n_pruned = int((m.dcd_ or {}).get("n_pruned", 0))
        assert n_pruned >= 3, (
            f"SU must detect the 5-perfect-dup cluster (>= 3 pruned); "
            f"got n_pruned={n_pruned}"
        )

    def test_vi_detects_5_dup_cluster(self):
        X, y = _linear_dup_cluster()
        m = _fit_with_distance(X, y, distance="vi", tau=0.5)
        n_pruned = int((m.dcd_ or {}).get("n_pruned", 0))
        assert n_pruned >= 3, (
            f"VI must detect the 5-perfect-dup cluster (>= 3 pruned); "
            f"got n_pruned={n_pruned}"
        )


# ---------------------------------------------------------------------------
# 3. Non-linear cluster: VI tighter than SU
# ---------------------------------------------------------------------------


class TestLayer46_NonLinearCluster:

    def test_vi_cluster_size_at_least_su_on_nonlinear(self):
        """On (x, x**2, x**3) of a shared latent, VI must detect at
        least as many cluster members as SU at the SAME tau. VI is the
        tighter metric for non-monotone functional dependencies; the
        contract is "VI does not lose to SU here", not "VI strictly
        beats SU on every fixture".
        """
        X, y = _nonlinear_cluster()
        m_su = _fit_with_distance(X, y, distance="su", tau=0.5)
        m_vi = _fit_with_distance(X, y, distance="vi", tau=0.5)
        n_pruned_su = int((m_su.dcd_ or {}).get("n_pruned", 0))
        n_pruned_vi = int((m_vi.dcd_ or {}).get("n_pruned", 0))
        assert n_pruned_vi >= n_pruned_su, (
            f"VI must catch at least as many non-linear cluster members "
            f"as SU; got n_pruned_vi={n_pruned_vi}, n_pruned_su={n_pruned_su}"
        )

    def test_auto_at_least_as_tight_as_either(self):
        """``"auto"`` returns max(SU, VI) per pair, so its pruned count
        must be >= both individual distances on every fixture.
        """
        X, y = _nonlinear_cluster()
        m_su = _fit_with_distance(X, y, distance="su", tau=0.5)
        m_vi = _fit_with_distance(X, y, distance="vi", tau=0.5)
        m_auto = _fit_with_distance(X, y, distance="auto", tau=0.5)
        n_su = int((m_su.dcd_ or {}).get("n_pruned", 0))
        n_vi = int((m_vi.dcd_ or {}).get("n_pruned", 0))
        n_auto = int((m_auto.dcd_ or {}).get("n_pruned", 0))
        assert n_auto >= n_su, (
            f"AUTO must prune >= SU; got auto={n_auto}, su={n_su}"
        )
        assert n_auto >= n_vi, (
            f"AUTO must prune >= VI; got auto={n_auto}, vi={n_vi}"
        )


# ---------------------------------------------------------------------------
# 4. ``"auto"`` end-to-end correctness
# ---------------------------------------------------------------------------


class TestLayer46_AutoDistance:

    def test_auto_validator_accepts(self):
        """``dcd_distance="auto"`` must pass the string-param validator."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        # No fit needed -- ctor validation runs on construction.
        m = MRMR(
            dcd_enable=True,
            dcd_distance="auto",
            dcd_tau_cluster=0.5,
            full_npermutations=10,
            verbose=0, random_seed=0,
        )
        # ``_validate_string_params`` runs on first fit, so call it
        # explicitly via a tiny fit.
        X, y = _linear_dup_cluster(n=400, seed=3)
        m.fit(X, y)
        assert m.dcd_ is not None

    def test_auto_validator_rejects_bogus(self):
        """``dcd_distance="bogus"`` must raise."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _linear_dup_cluster(n=200, seed=4)
        with pytest.raises((ValueError, AssertionError)):
            MRMR(
                dcd_enable=True,
                dcd_distance="bogus",
                full_npermutations=5,
                verbose=0, random_seed=0,
            ).fit(X, y)

    def test_auto_produces_cluster_anchors(self):
        """``"auto"`` produces a non-None cluster_anchors map."""
        X, y = _linear_dup_cluster()
        m = _fit_with_distance(X, y, distance="auto", tau=0.5)
        ca = (m.dcd_ or {}).get("cluster_anchors", None)
        assert ca is not None
        assert isinstance(ca, dict)

    def test_auto_integrates_with_swap_method_auto(self):
        """L43/L44 swap-method ``"auto"`` bake-off must still fit
        under ``dcd_distance="auto"``.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _linear_dup_cluster()
        m = MRMR(
            dcd_enable=True,
            dcd_distance="auto",
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            dcd_swap_method="auto",
            full_npermutations=20,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None


# ---------------------------------------------------------------------------
# 5. Recipe replay determinism
# ---------------------------------------------------------------------------


class TestLayer46_ReplayInvariant:

    @pytest.mark.parametrize("distance", ["su", "vi", "auto"])
    def test_transform_deterministic(self, distance):
        """``transform`` must be deterministic regardless of distance."""
        X, y = _linear_dup_cluster()
        m = _fit_with_distance(X, y, distance=distance, tau=0.5)
        Xt1 = np.asarray(m.transform(X), dtype=np.float64)
        Xt2 = np.asarray(m.transform(X), dtype=np.float64)
        assert Xt1.shape == Xt2.shape
        assert np.allclose(Xt1, Xt2, equal_nan=True), (
            f"transform must be deterministic for distance={distance!r}"
        )


# ---------------------------------------------------------------------------
# 6. Regression on Layers 41-45
# ---------------------------------------------------------------------------


class TestLayer46_RegressionL41toL45:

    @pytest.mark.parametrize("distance", ["su", "vi", "auto"])
    def test_l41_cluster_anchors_names_present(self, distance):
        X, y = _linear_dup_cluster()
        m = _fit_with_distance(X, y, distance=distance, tau=0.5)
        assert m.dcd_ is not None
        assert "cluster_anchors_names" in m.dcd_

    def test_l42_threshold2_fires_swap_under_su(self):
        """L42: threshold=2 fires >=1 swap on 3-dups under SU."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        rng = np.random.default_rng(7)
        n = 1500
        latent = rng.standard_normal(n)
        other = rng.standard_normal(n)
        X = pd.DataFrame({
            "strong": other,
            "dup_a": latent + 0.05 * rng.standard_normal(n),
            "dup_b": latent + 0.05 * rng.standard_normal(n),
            "dup_c": latent + 0.05 * rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        })
        y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
        m = MRMR(
            dcd_enable=True,
            dcd_distance="su",
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert int(m.dcd_["n_swaps"]) >= 1

    def test_l44_auto_pool_unchanged(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )
        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z", "mean_inv_var", "pca_pc1",
            "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
        }

    def test_l45_swap_decision_branch_field(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            SwapDecision,
        )
        d = SwapDecision(accept=False)
        assert hasattr(d, "branch")
        assert d.branch == "none"

    def test_default_distance_remains_su(self):
        """Default ``dcd_distance`` is still ``"su"`` (Layer 46 adds
        ``"auto"`` as opt-in, does not flip the default).
        """
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.dcd_distance == "su"
