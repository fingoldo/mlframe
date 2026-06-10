"""Regression sensors for the 2026-06-10 composite-discovery audit.

- D1: the vectorised leak-corr filter mean-imputed NaN cells, diluting |corr|
  by ~sqrt(frac_finite); an exact y-copy with a few NaN rows then slipped the
  forbidden-base threshold and could become the composite base.
- D2: iter_transform crashed on auto-promoted multi-base specs (extracted only
  base_column as a 1-D array; the multi forward expects (n,K)).
- D3: _linear_residual_multi_fit / forward_stepwise had no finite-row masking,
  so NaN-bearing lag bases silently disabled the default-ON multi-base promotion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import (
    CompositeSpec,
    CompositeTargetDiscovery,
    _linear_residual_multi_fit,
    forward_stepwise_multi_base,
)
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _make_disc(threshold=0.99999):
    cfg = CompositeTargetDiscoveryConfig(forbidden_base_corr_threshold=threshold)
    disc = CompositeTargetDiscovery(cfg)
    disc._target_col = "y"
    return disc


class TestLeakCorrNaNDilution:
    def test_y_copy_with_nan_rows_is_dropped(self) -> None:
        """D1: an exact y-copy carrying ~1% NaN must still be dropped by the
        leak-corr gate. Pre-fix the mean-imputation diluted |corr| to ~0.995
        < 0.99999 and the column passed."""
        rng = np.random.default_rng(0)
        n = 20_000
        y = rng.normal(0.0, 1.0, size=n)
        ycopy = y.copy()
        nan_idx = rng.choice(n, size=n // 100, replace=False)  # 1% NaN
        ycopy[nan_idx] = np.nan
        noise = rng.normal(0.0, 1.0, size=n)
        df = pd.DataFrame({"y": y, "ycopy": ycopy, "noise": noise})
        train_idx = np.arange(n)
        kept = _make_disc()._filter_features(
            df, ["ycopy", "noise"], y, train_idx,
        )
        assert "ycopy" not in kept, (
            "exact y-copy with sparse NaN slipped the leak-corr filter"
        )
        assert "noise" in kept, "independent noise column wrongly dropped"

    def test_clean_strong_correlate_still_dropped(self) -> None:
        """Control: a fully-finite near-perfect correlate is dropped (the
        fix must not weaken the gate on clean columns)."""
        rng = np.random.default_rng(1)
        n = 20_000
        y = rng.normal(0.0, 1.0, size=n)
        leak = y + rng.normal(0.0, 1e-9, size=n)
        df = pd.DataFrame({"y": y, "leak": leak})
        kept = _make_disc()._filter_features(df, ["leak"], y, np.arange(n))
        assert "leak" not in kept


class TestIterTransformMultiBase:
    def test_multi_base_spec_does_not_crash(self) -> None:
        """D2: iter_transform must apply a multi-base spec without raising the
        '(n,) vs (n,K)' shape error."""
        rng = np.random.default_rng(2)
        n = 500
        b1 = rng.normal(0.0, 1.0, size=n)
        b2 = rng.normal(0.0, 1.0, size=n)
        y = 1.0 * b1 + 0.5 * b2 + rng.normal(0.0, 0.1, size=n)
        df = pd.DataFrame({"y": y, "b1": b1, "b2": b2})
        spec = CompositeSpec(
            name="y-linresm-b1+b2",
            target_col="y",
            transform_name="linear_residual_multi",
            base_column="b1",
            fitted_params={"alphas": [1.0, 0.5], "beta": 0.0,
                           "collinear_fallback": False},
            mi_gain=0.1, mi_y=0.2, mi_t=0.3,
            valid_domain_frac=1.0, n_train_rows=n,
            extra_base_columns=("b2",),
        )
        disc = _make_disc()
        disc.specs_ = [spec]
        out = list(disc.iter_transform(df))  # pre-fix: ValueError
        assert len(out) == 1
        name, t = out[0]
        assert name == "y-linresm-b1+b2"
        assert np.all(np.isfinite(t))
        # T = y - (b1*1 + b2*0.5) - 0 == residual noise.
        np.testing.assert_allclose(t, y - (b1 + 0.5 * b2), rtol=1e-6, atol=1e-6)


class TestMultiBaseFiniteMasking:
    def test_fit_recovers_alphas_with_leading_nan(self) -> None:
        """D3: _linear_residual_multi_fit must recover the true alphas even
        when a base column carries leading NaN (lag/rolling base). Pre-fix the
        OLS raised LinAlgError -> all-zero collinear fallback."""
        rng = np.random.default_rng(3)
        n = 4000
        b1 = rng.normal(0.0, 1.0, size=n)
        b2 = rng.normal(0.0, 1.0, size=n)
        b1[:10] = np.nan  # leading NaN, as a lag-1 base has
        y = 2.0 * b1 + 3.0 * b2 + rng.normal(0.0, 0.05, size=n)
        params = _linear_residual_multi_fit(y, np.column_stack([b1, b2]))
        assert not params["collinear_fallback"], "fell back to collinear on NaN"
        np.testing.assert_allclose(params["alphas"][0], 2.0, rtol=0.05)
        np.testing.assert_allclose(params["alphas"][1], 3.0, rtol=0.05)

    def test_forward_stepwise_promotes_with_nan_lag_base(self) -> None:
        """D3/A9 biz_value: on y = b1 + b2 + eps where b1 has a leading NaN,
        forward-stepwise must still promote to BOTH bases (the benchmark-
        validated win that silently died on NaN lag bases)."""
        rng = np.random.default_rng(4)
        n = 5000
        b1 = rng.normal(0.0, 1.0, size=n)
        b2 = rng.normal(0.0, 1.0, size=n)
        b1[:5] = np.nan
        y = b1 + b2 + rng.normal(0.0, 0.05, size=n)
        kept, diag = forward_stepwise_multi_base(
            y,
            {"b1": b1, "b2": b2},
            seed_bases=["b2"],
            time_aware=False,
        )
        assert "b1" in kept and "b2" in kept, (
            f"promotion did not fire on a NaN lag base; kept={kept}"
        )
