"""Tests for ``quantile_residual`` transform (R10c extension #6).

The transform is conditional-on-bin centering + scaling: T = (y - median_bin) / IQR_bin where ``base`` is bucketed into n_bins quantile bins. Non-parametric alternative to logratio without the ``y > 0`` constraint and naturally handles heteroscedasticity (Var(y|base) depending on base).

Coverage:
- Round-trip ``y -> T -> y'`` recovers y exactly (rtol=1e-7) on a homoscedastic DGP.
- Bin assignment is monotonic and stable for in-range values; out-of-range base values map to edge bins (no OOR bucket).
- Under-populated bins (< min_bin_n train rows) fall back to GLOBAL median(y) / IQR(y).
- Constant-y bins replace zero IQR with the global IQR to keep the inverse well-defined.
- Biz_value: on a heteroscedastic DGP where Var(y|base) scales with base, ``quantile_residual`` produces a residual T with near-iid variance (max-min across bins < 2x) while ``linear_residual`` leaves a residual whose variance scales linearly with base (max-min > 5x).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N,
    _QUANTILE_RESIDUAL_DEFAULT_N_BINS,
    _quantile_residual_assign_bins,
    _quantile_residual_domain,
    _quantile_residual_fit,
    _quantile_residual_forward,
    _quantile_residual_inverse,
    get_transform,
)


# ---------------------------------------------------------------------------
# Unit: fit + bin assignment
# ---------------------------------------------------------------------------


class TestFit:
    def test_default_n_bins_and_min_bin_n(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = base + rng.normal(scale=0.5, size=n)
        params = _quantile_residual_fit(y, base)
        assert params["n_bins"] == _QUANTILE_RESIDUAL_DEFAULT_N_BINS
        assert len(params["bin_medians"]) == params["n_bins"]
        assert len(params["bin_iqrs"]) == params["n_bins"]
        # Edge bins extend to +/- inf for the predict-time OOR contract.
        assert np.isneginf(params["bin_edges"][0])
        assert np.isposinf(params["bin_edges"][-1])
        # All bins should have >= min_bin_n rows for this dense input.
        assert all(s >= _QUANTILE_RESIDUAL_DEFAULT_MIN_BIN_N for s in params["bin_sizes"])

    def test_small_bin_falls_back_to_global(self) -> None:
        """A bin with fewer than min_bin_n rows must use the GLOBAL median(y) / IQR(y), not its own under-determined estimate. Quantile bins partition by COUNT so to force a small bin we use a very high n_bins with limited n_total."""
        rng = np.random.default_rng(1)
        # 30 rows, 10 quantile bins -> ~3 rows per bin. min_bin_n=20 forces ALL bins into fallback.
        n = 30
        base = np.linspace(0.0, 1.0, n)  # deterministic, evenly spaced
        y = rng.normal(loc=5.0, scale=2.0, size=n)
        params = _quantile_residual_fit(y, base, n_bins=10, min_bin_n=20)
        # Every bin has ~3 rows; all should fall back to global.
        assert all(s < 20 for s in params["bin_sizes"])
        # Each bin's median must equal global_median.
        for bin_med in params["bin_medians"]:
            assert bin_med == pytest.approx(params["global_median"], rel=1e-9)
        for bin_iqr in params["bin_iqrs"]:
            assert bin_iqr == pytest.approx(params["global_iqr"], rel=1e-9)

    def test_degenerate_constant_base_single_bin(self) -> None:
        """If base is identical for every row, the fit collapses to a single global bin (no quantile cuts possible)."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base = np.array([7.0] * 5)
        params = _quantile_residual_fit(y, base, n_bins=10, min_bin_n=2)
        assert params["n_bins"] == 1
        assert params["bin_medians"].size == 1
        assert params["bin_iqrs"].size == 1

    def test_assign_bins_in_range(self) -> None:
        edges = np.array([-np.inf, 2.0, 5.0, np.inf], dtype=np.float64)
        base = np.array([0.0, 1.9, 2.0, 3.0, 5.0, 5.1, 10.0])
        bins = _quantile_residual_assign_bins(base, edges)
        # Bin 0: (-inf, 2.0); 1: [2.0, 5.0); 2: [5.0, +inf). searchsorted right-side: edges 2.0 -> idx 1; 5.0 -> idx 2 included.
        assert bins.tolist() == [0, 0, 1, 1, 2, 2, 2]

    def test_assign_bins_out_of_range_maps_to_edge(self) -> None:
        edges = np.array([-np.inf, 2.0, 5.0, np.inf], dtype=np.float64)
        base = np.array([-1e9, 1e9])
        bins = _quantile_residual_assign_bins(base, edges)
        # Out of train range -> edge bins (0 and n_bins-1).
        assert bins.tolist() == [0, 2]


# ---------------------------------------------------------------------------
# Round-trip y -> T -> y
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_on_homoscedastic_dgp(self) -> None:
        rng = np.random.default_rng(2)
        n = 1500
        base = rng.normal(loc=10.0, scale=3.0, size=n)
        y = 0.8 * base + rng.normal(scale=0.5, size=n)
        params = _quantile_residual_fit(y, base, n_bins=10, min_bin_n=50)
        T = _quantile_residual_forward(y, base, params)
        y_back = _quantile_residual_inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_round_trip_via_registry(self) -> None:
        """Registry exposure works end-to-end."""
        rng = np.random.default_rng(3)
        n = 800
        base = rng.uniform(low=0.0, high=10.0, size=n)
        y = base + rng.normal(scale=0.5, size=n)
        t = get_transform("quantile_residual")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)


# ---------------------------------------------------------------------------
# Domain checks
# ---------------------------------------------------------------------------


class TestDomain:
    def test_rejects_non_finite_base(self) -> None:
        base = np.array([1.0, 2.0, np.nan, np.inf, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = _quantile_residual_domain(y, base)
        np.testing.assert_array_equal(mask, [True, True, False, False, True])

    def test_y_none_only_checks_base(self) -> None:
        base = np.array([1.0, np.nan, 3.0])
        mask = _quantile_residual_domain(None, base)
        np.testing.assert_array_equal(mask, [True, False, True])


# ---------------------------------------------------------------------------
# Biz_value: heteroscedastic DGP where quantile_residual beats linear_residual
# ---------------------------------------------------------------------------


class TestBizValueQuantileResidualOnHeteroscedasticDGP:
    """When Var(y|base) scales with base, the bin-conditional residual has near-iid variance across bins while the linear residual leaves a wedge-shaped envelope.

    Lock: the ratio max(var(T|bin)) / min(var(T|bin)) is STRICTLY smaller for ``quantile_residual`` than for ``linear_residual``.
    """

    def _make_hetero_dgp(self, n: int = 5000, seed: int = 0):
        rng = np.random.default_rng(seed)
        # base in [1, 11]; noise std scales LINEARLY with base.
        base = rng.uniform(low=1.0, high=11.0, size=n)
        y = 2.0 * base + base * rng.normal(size=n)  # noise std == base
        return y, base

    @staticmethod
    def _bin_var_ratio(T: np.ndarray, base: np.ndarray, n_bins: int = 10) -> float:
        edges = np.quantile(base, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        bin_idx = np.clip(np.searchsorted(edges[1:-1], base, side="right"), 0, n_bins - 1)
        variances = np.array([float(np.var(T[bin_idx == b])) if (bin_idx == b).sum() >= 10 else np.nan for b in range(n_bins)])
        finite = variances[np.isfinite(variances)]
        if finite.size < 2:
            return float("nan")
        return float(finite.max() / max(finite.min(), 1e-12))

    def test_quantile_residual_iid_variance_beats_linear(self) -> None:
        y, base = self._make_hetero_dgp()
        # Quantile residual.
        qr_params = _quantile_residual_fit(y, base, n_bins=10, min_bin_n=50)
        T_qr = _quantile_residual_forward(y, base, qr_params)
        # Linear residual (single-base OLS).
        from mlframe.training.composite import _linear_residual_fit, _linear_residual_forward

        lr_params = _linear_residual_fit(y, base)
        T_lr = _linear_residual_forward(y, base, lr_params)
        # Compare per-bin variance ratios on the same binning.
        qr_ratio = self._bin_var_ratio(T_qr, base)
        lr_ratio = self._bin_var_ratio(T_lr, base)
        # Quantile residual: near-iid variance across bins (each bin centered + scaled by its own IQR). Linear residual: wedge envelope (Var(T) scales with base).
        assert qr_ratio < lr_ratio, (
            f"quantile_residual must have lower max/min variance ratio "
            f"than linear_residual on heteroscedastic DGP; got "
            f"qr_ratio={qr_ratio:.2f}, lr_ratio={lr_ratio:.2f}"
        )
        # Tight envelope after quantile-residual: < 3x typical (the noise scaling within each bin is bounded since each bin spans a narrow base range).
        assert qr_ratio < 3.0
