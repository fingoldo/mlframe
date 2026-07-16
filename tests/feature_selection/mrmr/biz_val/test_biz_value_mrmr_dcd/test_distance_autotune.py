"""DCD consolidation: Layer 46 biz_value: VI-based DCD distance + ``"auto"`` SU/VI bake-off.

Consolidated verbatim from test_biz_value_mrmr_layer46.py + test_biz_value_mrmr_layer47.py (per audit finding test_code_quality-16).
"""

from __future__ import annotations

import warnings
from functools import cache

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
    X = pd.DataFrame(
        {
            "strong_unrelated": other,
            "dup_a": latent + 0.02 * rng.standard_normal(n),
            "dup_b": latent + 0.02 * rng.standard_normal(n),
            "dup_c": latent + 0.02 * rng.standard_normal(n),
            "dup_d": latent + 0.02 * rng.standard_normal(n),
            "dup_e": latent + 0.02 * rng.standard_normal(n),
            "noise_filler": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
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
    X = pd.DataFrame(
        {
            "strong_unrelated": other,
            "x_lin": latent,
            "x_sq": latent**2,
            "x_cub": latent**3,
            "noise_filler": rng.standard_normal(n),
        }
    )
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
    """Unit tests for the pair_vi analytical helper (export + VI sanity checks)."""

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
            DCDState,
            pair_vi,
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
            DCDState,
            pair_vi,
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
        assert vi == pytest.approx(0.0, abs=1e-9), f"VI of a feature with itself must be 0; got {vi}"

    def test_pair_vi_independent_is_large(self):
        """VI between two independent features ~= H(X_a) + H(X_b) -- i.e.
        zero mutual information makes the second term vanish and VI
        equals the sum of the marginal entropies.
        """
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            DCDState,
            pair_vi,
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
        assert vi > 1.5, f"VI of independent uniform 4-bin features should be > 1.5 nats; " f"got {vi}"


# ---------------------------------------------------------------------------
# 2. Linear cluster: SU and VI both detect
# ---------------------------------------------------------------------------


def _fit_with_distance(X, y, *, distance: str, tau: float = 0.7):
    """Fit a DCD-enabled MRMR with the given pair-distance metric and cluster tau."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    return MRMR(
        dcd_enable=True,
        dcd_distance=distance,
        dcd_tau_cluster=tau,
        dcd_cluster_size_threshold=2,
        full_npermutations=20,
        verbose=0,
        random_seed=0,
    ).fit(X, y)


@cache
def _linear_dup_fit(distance: str):
    """Cached ``(X, y, m)`` for ``_fit_with_distance(_linear_dup_cluster(), distance=distance, tau=0.5)``.

    Shared across test_su_detects_5_dup_cluster / test_vi_detects_5_dup_cluster /
    test_auto_produces_cluster_anchors / test_l41_cluster_anchors_names_present /
    test_transform_deterministic -- 9 identical-config calls collapse to 3
    (one per distance). Nothing downstream mutates X/y/m in place.
    """
    X, y = _linear_dup_cluster()
    m = _fit_with_distance(X, y, distance=distance, tau=0.5)
    return X, y, m


@cache
def _nonlinear_fit(distance: str):
    """Cached ``(X, y, m)`` for ``_fit_with_distance(_nonlinear_cluster(), distance=distance, tau=0.5)``.

    Shared across test_vi_cluster_size_at_least_su_on_nonlinear and
    test_auto_at_least_as_tight_as_either -- 5 identical-config calls
    collapse to 3 (one per distance). Nothing downstream mutates X/y/m in place.
    """
    X, y = _nonlinear_cluster()
    m = _fit_with_distance(X, y, distance=distance, tau=0.5)
    return X, y, m


class TestLayer46_LinearCluster:
    """SU and VI distances both detect a 5-near-duplicate linear cluster."""

    def test_su_detects_5_dup_cluster(self):
        """SU-distance DCD prunes at least 3 of the 5 near-duplicate columns."""
        _X, _y, m = _linear_dup_fit("su")
        n_pruned = int((m.dcd_ or {}).get("n_pruned", 0))
        assert n_pruned >= 3, f"SU must detect the 5-perfect-dup cluster (>= 3 pruned); " f"got n_pruned={n_pruned}"

    def test_vi_detects_5_dup_cluster(self):
        """VI-distance DCD prunes at least 3 of the 5 near-duplicate columns."""
        _X, _y, m = _linear_dup_fit("vi")
        n_pruned = int((m.dcd_ or {}).get("n_pruned", 0))
        assert n_pruned >= 3, f"VI must detect the 5-perfect-dup cluster (>= 3 pruned); " f"got n_pruned={n_pruned}"


# ---------------------------------------------------------------------------
# 3. Non-linear cluster: VI tighter than SU
# ---------------------------------------------------------------------------


class TestLayer46_NonLinearCluster:
    """VI distance must not lose to SU on a non-monotone (x, x**2, x**3) redundancy cluster."""

    def test_vi_cluster_size_at_least_su_on_nonlinear(self):
        """On (x, x**2, x**3) of a shared latent, VI must detect at
        least as many cluster members as SU at the SAME tau. VI is the
        tighter metric for non-monotone functional dependencies; the
        contract is "VI does not lose to SU here", not "VI strictly
        beats SU on every fixture".
        """
        _X, _y, m_su = _nonlinear_fit("su")
        _X2, _y2, m_vi = _nonlinear_fit("vi")
        n_pruned_su = int((m_su.dcd_ or {}).get("n_pruned", 0))
        n_pruned_vi = int((m_vi.dcd_ or {}).get("n_pruned", 0))
        assert n_pruned_vi >= n_pruned_su, (
            f"VI must catch at least as many non-linear cluster members " f"as SU; got n_pruned_vi={n_pruned_vi}, n_pruned_su={n_pruned_su}"
        )

    def test_auto_at_least_as_tight_as_either(self):
        """``"auto"`` returns max(SU, VI) per pair, so its pruned count
        must be >= both individual distances on every fixture.
        """
        _X, _y, m_su = _nonlinear_fit("su")
        _X2, _y2, m_vi = _nonlinear_fit("vi")
        _X3, _y3, m_auto = _nonlinear_fit("auto")
        n_su = int((m_su.dcd_ or {}).get("n_pruned", 0))
        n_vi = int((m_vi.dcd_ or {}).get("n_pruned", 0))
        n_auto = int((m_auto.dcd_ or {}).get("n_pruned", 0))
        assert n_auto >= n_su, f"AUTO must prune >= SU; got auto={n_auto}, su={n_su}"
        assert n_auto >= n_vi, f"AUTO must prune >= VI; got auto={n_auto}, vi={n_vi}"


# ---------------------------------------------------------------------------
# 4. ``"auto"`` end-to-end correctness
# ---------------------------------------------------------------------------


class TestLayer46_AutoDistance:
    """``dcd_distance="auto"`` end-to-end correctness (validator, cluster anchors, swap-method integration)."""

    def test_auto_validator_accepts(self):
        """``dcd_distance="auto"`` must pass the string-param validator."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        # No fit needed -- ctor validation runs on construction.
        m = MRMR(
            dcd_enable=True,
            dcd_distance="auto",
            dcd_tau_cluster=0.5,
            full_npermutations=10,
            verbose=0,
            random_seed=0,
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
                verbose=0,
                random_seed=0,
            ).fit(X, y)

    def test_auto_produces_cluster_anchors(self):
        """``"auto"`` produces a non-None cluster_anchors map."""
        _X, _y, m = _linear_dup_fit("auto")
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
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None


# ---------------------------------------------------------------------------
# 5. Recipe replay determinism
# ---------------------------------------------------------------------------


class TestLayer46_ReplayInvariant:
    """transform() is deterministic regardless of the DCD distance metric used at fit."""

    @pytest.mark.parametrize("distance", ["su", "vi", "auto"])
    def test_transform_deterministic(self, distance):
        """``transform`` must be deterministic regardless of distance."""
        X, _y, m = _linear_dup_fit(distance)
        Xt1 = np.asarray(m.transform(X), dtype=np.float64)
        Xt2 = np.asarray(m.transform(X), dtype=np.float64)
        assert Xt1.shape == Xt2.shape
        assert np.allclose(Xt1, Xt2, equal_nan=True), f"transform must be deterministic for distance={distance!r}"


# ---------------------------------------------------------------------------
# 6. Regression on Layers 41-45
# ---------------------------------------------------------------------------


class TestLayer46_RegressionL41toL45:
    """Layer 46 must not regress the Layers 41-45 cluster-anchor / swap / pool-of-methods contracts."""

    @pytest.mark.parametrize("distance", ["su", "vi", "auto"])
    def test_l41_cluster_anchors_names_present(self, distance):
        """cluster_anchors_names remains populated in dcd_ regardless of the distance metric."""
        _X, _y, m = _linear_dup_fit(distance)
        assert m.dcd_ is not None
        assert "cluster_anchors_names" in m.dcd_

    def test_l42_threshold2_fires_swap_under_su(self):
        """L42: threshold=2 fires >=1 swap on 3-dups under SU."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(7)
        n = 1500
        latent = rng.standard_normal(n)
        other = rng.standard_normal(n)
        X = pd.DataFrame(
            {
                "strong": other,
                "dup_a": latent + 0.05 * rng.standard_normal(n),
                "dup_b": latent + 0.05 * rng.standard_normal(n),
                "dup_c": latent + 0.05 * rng.standard_normal(n),
                "noise": rng.standard_normal(n),
            }
        )
        y = pd.Series((2 * other + latent + 0.3 * rng.standard_normal(n) > 0).astype(int))
        m = MRMR(
            dcd_enable=True,
            dcd_distance="su",
            dcd_tau_cluster=0.5,
            dcd_cluster_size_threshold=2,
            full_npermutations=50,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert int(m.dcd_["n_swaps"]) >= 1

    def test_l44_auto_pool_unchanged(self):
        """The L44 auto-method candidate pool is unchanged by the Layer 46 distance work."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )

        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z",
            "mean_inv_var",
            "pca_pc1",
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        }

    def test_l45_swap_decision_branch_field(self):
        """SwapDecision's branch field defaults to "none" when accept=False."""
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bimodal_su_data(n: int = 1500, seed: int = 0):
    """Bimodal SU distribution: explicit clusters + independent fillers.

    8 dup-cluster features (5 around latent_A, 3 around latent_B) +
    6 independent noise fillers. Pair SU among same-latent dups is very
    high (~0.6-0.9); pair SU between latents and against fillers is near
    zero. The SU histogram should be bimodal: one peak at low SU
    (independent pairs), one peak at high SU (within-cluster pairs).
    """
    rng = np.random.default_rng(int(seed))
    latent_A = rng.standard_normal(n)
    latent_B = rng.standard_normal(n)
    other = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "strong_unrelated": other,
            # Cluster A: 5 noisy copies of latent_A
            "A_a": latent_A + 0.05 * rng.standard_normal(n),
            "A_b": latent_A + 0.05 * rng.standard_normal(n),
            "A_c": latent_A + 0.05 * rng.standard_normal(n),
            "A_d": latent_A + 0.05 * rng.standard_normal(n),
            "A_e": latent_A + 0.05 * rng.standard_normal(n),
            # Cluster B: 3 noisy copies of latent_B
            "B_a": latent_B + 0.05 * rng.standard_normal(n),
            "B_b": latent_B + 0.05 * rng.standard_normal(n),
            "B_c": latent_B + 0.05 * rng.standard_normal(n),
            # Independent fillers
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "f3": rng.standard_normal(n),
            "f4": rng.standard_normal(n),
            "f5": rng.standard_normal(n),
            "f6": rng.standard_normal(n),
        }
    )
    y = pd.Series((2 * other + latent_A + latent_B + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _unimodal_pure_noise(n: int = 1500, seed: int = 1):
    """Unimodal SU distribution: all features mutually independent.

    No clusters; SU histogram should be a single mode near zero. The
    auto-tau calibration should detect this and fall back to 0.7.
    """
    rng = np.random.default_rng(int(seed))
    n_features = 12
    cols = {f"noise_{i}": rng.standard_normal(n) for i in range(n_features)}
    X = pd.DataFrame(cols)
    # Target only weakly depends on a single feature -- the rest are pure noise.
    y = pd.Series((X["noise_0"] + 0.3 * rng.standard_normal(n) > 0).astype(int))
    return X, y


def _quantize_X(X, n_bins: int = 10) -> tuple:
    """Quantize a numeric DataFrame into integer bin codes matching the
    DCDState contract. Returns ``(factors_data, factors_nbins)``.
    """
    cols = []
    nbins = []
    for c in X.columns:
        col = X[c].to_numpy(dtype=np.float64)
        edges = np.quantile(col, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if edges.size < 3:
            binned = np.zeros(col.shape, dtype=np.int32)
            nb = 1
        else:
            binned = np.searchsorted(edges[1:-1], col, side="right").astype(np.int32)
            nb = int(binned.max()) + 1
        cols.append(binned)
        nbins.append(nb)
    factors_data = np.column_stack(cols)
    factors_nbins = np.asarray(nbins, dtype=np.int64)
    return factors_data, factors_nbins


# ---------------------------------------------------------------------------
# 1. Validator + ctor surface
# ---------------------------------------------------------------------------


class TestLayer47_ValidatorSurface:
    """``dcd_tau_cluster`` accepts 'auto' or a numeric value; rejects any other string."""

    def test_auto_string_accepted(self):
        """``dcd_tau_cluster='auto'`` must pass validator."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=600, seed=2)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=5,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None

    def test_bogus_string_rejected(self):
        """Non-'auto' strings must raise ValueError."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=400, seed=3)
        with pytest.raises((ValueError, AssertionError)):
            MRMR(
                dcd_enable=True,
                dcd_tau_cluster="bogus",
                full_npermutations=2,
                verbose=0,
                random_seed=0,
            ).fit(X, y)

    def test_numeric_tau_still_validates(self):
        """Numeric in-range tau still works."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=400, seed=4)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=2,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None


# ---------------------------------------------------------------------------
# 2. Bimodal SU distribution detection
# ---------------------------------------------------------------------------


class TestLayer47_BimodalDetection:
    """The valley-detection heuristic and its end-to-end auto-tau calibration wiring."""

    def test_valley_detector_on_synthetic_bimodal(self):
        """Direct unit test of ``_detect_valley_between_modes``: a clearly
        bimodal score array yields a valley in the gap between modes."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _detect_valley_between_modes,
        )

        rng = np.random.default_rng(0)
        low = rng.normal(0.15, 0.04, size=120).clip(0.0, 1.0)
        high = rng.normal(0.80, 0.04, size=80).clip(0.0, 1.0)
        scores = np.concatenate([low, high])
        tau = _detect_valley_between_modes(scores)
        assert tau is not None, "valley must be detected on clear bimodal data"
        # Valley should sit between the two modes (modes at 0.15 and 0.80).
        assert 0.30 <= tau <= 0.75, f"valley must sit between the two modes; got tau={tau}"

    def test_valley_detector_unimodal_returns_none(self):
        """Unimodal data -> detector returns None (no false-positive valley)."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _detect_valley_between_modes,
        )

        rng = np.random.default_rng(1)
        scores = rng.normal(0.30, 0.08, size=200).clip(0.0, 1.0)
        tau = _detect_valley_between_modes(scores)
        assert tau is None, f"unimodal data must NOT yield a valley; got {tau}"

    def test_valley_detector_small_high_su_tail_over_broad_bulk(self):
        """A SMALL high-SU redundancy mode over a BROAD decaying low-SU bulk must still be
        detected: this is the realistic sensor-mesh shape (few within-cluster near-duplicate
        pairs at SU~0.6-0.75, many irrelevant pairs spread 0.0-0.3 by plug-in MI bias). The
        legacy two-tallest-peaks rule paired two bulk-internal bins and missed the tail, reporting
        unimodal -> auto-tau fell back to 0.7 and under-clustered. The detector must instead place
        the valley BETWEEN the bulk and the high-SU tail (tau roughly in the 0.45-0.65 gap)."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _detect_valley_between_modes,
        )

        rng = np.random.default_rng(7)
        # Broad decaying low-SU bulk + a thin high-SU cluster tail (small count).
        bulk = np.abs(rng.normal(0.0, 0.12, size=190)).clip(0.0, 0.45)
        tail = rng.normal(0.70, 0.04, size=10).clip(0.0, 1.0)
        scores = np.concatenate([bulk, tail])
        tau = _detect_valley_between_modes(scores)
        assert tau is not None, "small high-SU tail over a broad bulk must be detected as bimodal"
        # Valley lands in the empty gap between the bulk (clipped at ~0.35) and the tail (~0.70);
        # the deepest (lowest-count) gap bin sits anywhere in that span. The contract is only that
        # it separates the two modes -- i.e. above the bulk top and below the cluster tail.
        assert 0.35 <= tau <= 0.68, f"valley must sit between the low-SU bulk and the high-SU cluster tail; got tau={tau}"

    def test_calibrate_tau_auto_on_bimodal_data(self):
        """End-to-end: ``_calibrate_tau_auto`` reports ``mode='bimodal'`` on
        synthetic bimodal cluster + noise data."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _calibrate_tau_auto,
        )

        X, _ = _bimodal_su_data(n=1500, seed=10)
        factors_data, factors_nbins = _quantize_X(X)
        tau, diag = _calibrate_tau_auto(
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            distance="su",
            n_pairs=100,
            seed=0,
        )
        assert diag["mode"] == "bimodal", (
            f"bimodal data must trigger bimodal detection; got mode="
            f"{diag['mode']!r}, valley_su={diag.get('valley_su')!r}, "
            f"su_mean={diag.get('su_mean')}, su_std={diag.get('su_std')}"
        )
        assert 0.3 <= tau <= 0.95
        assert diag["valley_su"] is not None

    def test_calibrate_tau_auto_on_pure_noise_falls_back(self):
        """Pure-noise data -> ``mode='unimodal'`` and tau falls back to 0.7."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _calibrate_tau_auto,
            _DCD_AUTO_TAU_FALLBACK,
        )

        X, _ = _unimodal_pure_noise(n=1500, seed=11)
        factors_data, factors_nbins = _quantize_X(X)
        tau, diag = _calibrate_tau_auto(
            factors_data=factors_data,
            factors_nbins=factors_nbins,
            distance="su",
            n_pairs=100,
            seed=0,
        )
        assert diag["mode"] == "unimodal", f"pure noise must NOT trigger bimodal detection; got mode=" f"{diag['mode']!r}, valley_su={diag.get('valley_su')!r}"
        assert tau == pytest.approx(_DCD_AUTO_TAU_FALLBACK)


# ---------------------------------------------------------------------------
# 3. End-to-end MRMR.fit with auto-tau
# ---------------------------------------------------------------------------


class TestLayer47_FitIntegration:
    """MRMR.fit end-to-end with dcd_tau_cluster='auto' -- diagnostics and fallback behaviour."""

    def test_auto_tau_records_diagnostics_on_dcd_summary(self):
        """``MRMR.dcd_['tau_calibration']`` is populated when auto-tau ran."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1500, seed=20)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert "tau_calibration" in m.dcd_
        cal = m.dcd_["tau_calibration"]
        assert cal is not None, "tau_calibration must be populated when dcd_tau_cluster='auto'"
        assert cal["requested"] == "auto"
        assert cal["mode"] in ("bimodal", "unimodal", "degenerate")
        # Effective tau gets reported on dcd_['tau_cluster'].
        assert 0.0 < m.dcd_["tau_cluster"] <= 1.0

    def test_default_fixed_tau_keeps_calibration_none(self):
        """Default numeric tau leaves ``tau_calibration`` at None
        (calibration didn't run -> legacy behaviour preserved)."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1000, seed=21)
        m = MRMR(
            dcd_enable=True,  # default tau=0.7
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert "tau_calibration" in m.dcd_  # key always present
        assert m.dcd_["tau_calibration"] is None, "Numeric tau must leave calibration None"

    def test_auto_tau_bimodal_data_produces_finite_tau(self):
        """On clear bimodal data, auto-tau picks a tau in [0.3, 0.95]."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1500, seed=22)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        tau = float(m.dcd_["tau_cluster"])
        assert 0.3 <= tau <= 0.95, f"auto-tau on bimodal data must produce a tau in [0.3, 0.95]; " f"got tau={tau} (mode={m.dcd_['tau_calibration']['mode']!r})"

    def test_auto_tau_unimodal_falls_back_to_default(self):
        """On pure-noise data, auto-tau falls back to 0.7."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _DCD_AUTO_TAU_FALLBACK,
        )

        X, y = _unimodal_pure_noise(n=1500, seed=23)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        cal = m.dcd_["tau_calibration"]
        # Either unimodal or degenerate (too few features) -> tau falls back.
        assert cal["mode"] in ("unimodal", "degenerate"), f"pure-noise data must NOT trigger bimodal; got mode={cal['mode']!r}"
        assert m.dcd_["tau_cluster"] == pytest.approx(_DCD_AUTO_TAU_FALLBACK)


# ---------------------------------------------------------------------------
# 4. Determinism + replay
# ---------------------------------------------------------------------------


class TestLayer47_Determinism:
    """Auto-tau calibration and its downstream transform are deterministic under a fixed seed."""

    def test_auto_tau_transform_deterministic(self):
        """``transform`` deterministic under auto-tau."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1500, seed=30)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        Xt1 = np.asarray(m.transform(X), dtype=np.float64)
        Xt2 = np.asarray(m.transform(X), dtype=np.float64)
        assert Xt1.shape == Xt2.shape
        assert np.allclose(Xt1, Xt2, equal_nan=True)

    def test_auto_tau_same_seed_reproducible(self):
        """Two fits with the same seed pick the same tau."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1500, seed=31)
        m1 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        m2 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m1.dcd_["tau_cluster"] == pytest.approx(m2.dcd_["tau_cluster"])


# ---------------------------------------------------------------------------
# 5. Regression on Layers 41-46
# ---------------------------------------------------------------------------


class TestLayer47_RegressionL41toL46:
    """Layer 47 must not regress the Layers 41-46 default-tau / cluster-anchor / swap contracts."""

    def test_default_tau_value_unchanged(self):
        """The default ``dcd_tau_cluster`` constructor value stays 0.7
        (Layer 47 adds 'auto' as opt-in, does not flip the default)."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        m = MRMR()
        assert m.dcd_tau_cluster == 0.7

    def test_l41_cluster_anchors_names_present_with_auto_tau(self):
        """cluster_anchors_names stays populated when dcd_tau_cluster='auto'."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1200, seed=40)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=5,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert "cluster_anchors_names" in m.dcd_

    def test_l46_distance_auto_with_tau_auto(self):
        """L46 ``dcd_distance='auto'`` composes with L47 ``dcd_tau_cluster='auto'``."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        X, y = _bimodal_su_data(n=1200, seed=41)
        m = MRMR(
            dcd_enable=True,
            dcd_distance="auto",
            dcd_tau_cluster="auto",
            full_npermutations=5,
            verbose=0,
            random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert m.dcd_["tau_calibration"] is not None

    def test_l45_swap_decision_branch_field_intact(self):
        """SwapDecision's branch field still defaults to "none" after the Layer 47 changes."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            SwapDecision,
        )

        d = SwapDecision(accept=False)
        assert hasattr(d, "branch")
        assert d.branch == "none"

    def test_l44_auto_method_pool_unchanged(self):
        """The L44 auto-method candidate pool is unchanged by the Layer 47 auto-tau work."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )

        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z",
            "mean_inv_var",
            "pca_pc1",
            "pca_pc2",
            "median_z",
            "signed_max_abs",
            "signed_l2_sum",
        }
