"""Layer 47 biz_value: AUTO-TUNE ``dcd_tau_cluster`` via small SU sweep.

WHY THIS LAYER
--------------
Pre-Layer-47 ``dcd_tau_cluster`` was a fixed hyperparameter (default 0.7).
On homogeneous data with many genuine clusters, 0.7 over-clusters
(prunes real signal). On heterogeneous data, 0.7 misses real clusters.
The user shouldn't have to hand-tune by hand for each fit.

LAYER 47 IMPROVEMENT
--------------------
``dcd_tau_cluster='auto'`` opts into a small calibration sweep that
runs at fit start (inside ``make_dcd_state``):

  1. Sample ~100 random feature pairs.
  2. Compute pair-SU on each via the same code path the cluster-membership
     rule consumes.
  3. Coarse-histogram the SU distribution into 20 bins over [0, 1].
  4. Detect a valley between two modes (cluster-similar pairs vs
     unrelated pairs). Bimodality criterion: two local maxima >= 3 bins
     apart and a valley between them <= 60% of the shallower peak.
  5. If bimodal -> tau = valley SU value (clamped to [0.3, 0.95]).
     If unimodal -> tau = 0.7 (fallback).

The chosen tau lives on ``state.tau_cluster``; the full diagnostics
(``mode``, ``valley_su``, ``su_mean``, ``su_std``, ``n_pairs_*``) are
surfaced on ``MRMR.dcd_["tau_calibration"]``.

CONTRACTS
---------
- C1: ``dcd_tau_cluster='auto'`` is accepted by the validator.
- C2: Bimodal data (clusters + noise) yields ``mode='bimodal'`` and the
  picked tau lies inside the valley between the two SU modes.
- C3: Unimodal data (pure noise) yields ``mode='unimodal'`` and the
  fallback tau (0.7).
- C4: Default ``dcd_tau_cluster=0.7`` keeps the legacy fixed-tau behaviour
  bit-identical (``state.tau_calibration`` is None).
- C5: ``MRMR.dcd_["tau_calibration"]`` is present (None when fixed tau,
  dict when auto).
- C6: Validator rejects ``dcd_tau_cluster='bogus'`` (non-'auto' string).
- C7: ``transform`` is deterministic regardless of tau mode.
- C8: Layers 41-46 contracts preserved across the new tau mode.

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
    X = pd.DataFrame({
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
    })
    y = pd.Series(
        (2 * other + latent_A + latent_B + 0.3 * rng.standard_normal(n) > 0).astype(int)
    )
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

    def test_auto_string_accepted(self):
        """``dcd_tau_cluster='auto'`` must pass validator."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _bimodal_su_data(n=600, seed=2)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=5,
            verbose=0, random_seed=0,
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
                verbose=0, random_seed=0,
            ).fit(X, y)

    def test_numeric_tau_still_validates(self):
        """Numeric in-range tau still works."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _bimodal_su_data(n=400, seed=4)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster=0.5,
            full_npermutations=2,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None


# ---------------------------------------------------------------------------
# 2. Bimodal SU distribution detection
# ---------------------------------------------------------------------------


class TestLayer47_BimodalDetection:

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
        assert 0.30 <= tau <= 0.75, (
            f"valley must sit between the two modes; got tau={tau}"
        )

    def test_valley_detector_unimodal_returns_none(self):
        """Unimodal data -> detector returns None (no false-positive valley)."""
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _detect_valley_between_modes,
        )
        rng = np.random.default_rng(1)
        scores = rng.normal(0.30, 0.08, size=200).clip(0.0, 1.0)
        tau = _detect_valley_between_modes(scores)
        assert tau is None, f"unimodal data must NOT yield a valley; got {tau}"

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
            _calibrate_tau_auto, _DCD_AUTO_TAU_FALLBACK,
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
        assert diag["mode"] == "unimodal", (
            f"pure noise must NOT trigger bimodal detection; got mode="
            f"{diag['mode']!r}, valley_su={diag.get('valley_su')!r}"
        )
        assert tau == pytest.approx(_DCD_AUTO_TAU_FALLBACK)


# ---------------------------------------------------------------------------
# 3. End-to-end MRMR.fit with auto-tau
# ---------------------------------------------------------------------------


class TestLayer47_FitIntegration:

    def test_auto_tau_records_diagnostics_on_dcd_summary(self):
        """``MRMR.dcd_['tau_calibration']`` is populated when auto-tau ran."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _bimodal_su_data(n=1500, seed=20)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert "tau_calibration" in m.dcd_
        cal = m.dcd_["tau_calibration"]
        assert cal is not None, (
            "tau_calibration must be populated when dcd_tau_cluster='auto'"
        )
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
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert "tau_calibration" in m.dcd_  # key always present
        assert m.dcd_["tau_calibration"] is None, (
            "Numeric tau must leave calibration None"
        )

    def test_auto_tau_bimodal_data_produces_finite_tau(self):
        """On clear bimodal data, auto-tau picks a tau in [0.3, 0.95]."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _bimodal_su_data(n=1500, seed=22)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0, random_seed=0,
        ).fit(X, y)
        tau = float(m.dcd_["tau_cluster"])
        assert 0.3 <= tau <= 0.95, (
            f"auto-tau on bimodal data must produce a tau in [0.3, 0.95]; "
            f"got tau={tau} (mode={m.dcd_['tau_calibration']['mode']!r})"
        )

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
            verbose=0, random_seed=0,
        ).fit(X, y)
        cal = m.dcd_["tau_calibration"]
        # Either unimodal or degenerate (too few features) -> tau falls back.
        assert cal["mode"] in ("unimodal", "degenerate"), (
            f"pure-noise data must NOT trigger bimodal; got mode={cal['mode']!r}"
        )
        assert m.dcd_["tau_cluster"] == pytest.approx(_DCD_AUTO_TAU_FALLBACK)


# ---------------------------------------------------------------------------
# 4. Determinism + replay
# ---------------------------------------------------------------------------


class TestLayer47_Determinism:

    def test_auto_tau_transform_deterministic(self):
        """``transform`` deterministic under auto-tau."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _bimodal_su_data(n=1500, seed=30)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0, random_seed=0,
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
            verbose=0, random_seed=0,
        ).fit(X, y)
        m2 = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=10,
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m1.dcd_["tau_cluster"] == pytest.approx(m2.dcd_["tau_cluster"])


# ---------------------------------------------------------------------------
# 5. Regression on Layers 41-46
# ---------------------------------------------------------------------------


class TestLayer47_RegressionL41toL46:

    def test_default_tau_value_unchanged(self):
        """The default ``dcd_tau_cluster`` constructor value stays 0.7
        (Layer 47 adds 'auto' as opt-in, does not flip the default)."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.dcd_tau_cluster == 0.7

    def test_l41_cluster_anchors_names_present_with_auto_tau(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _bimodal_su_data(n=1200, seed=40)
        m = MRMR(
            dcd_enable=True,
            dcd_tau_cluster="auto",
            full_npermutations=5,
            verbose=0, random_seed=0,
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
            verbose=0, random_seed=0,
        ).fit(X, y)
        assert m.dcd_ is not None
        assert m.dcd_["tau_calibration"] is not None

    def test_l45_swap_decision_branch_field_intact(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            SwapDecision,
        )
        d = SwapDecision(accept=False)
        assert hasattr(d, "branch")
        assert d.branch == "none"

    def test_l44_auto_method_pool_unchanged(self):
        from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
            _AUTO_METHOD_CANDIDATES,
        )
        assert set(_AUTO_METHOD_CANDIDATES) == {
            "mean_z", "mean_inv_var", "pca_pc1",
            "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
        }
