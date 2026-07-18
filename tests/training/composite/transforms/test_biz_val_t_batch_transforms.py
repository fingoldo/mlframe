"""biz_value tests for the T-batch transforms: grouped-EWMA residual and volatility-normalised residual.

Quantitative claims (thresholds set with margin below the measured values, per repo policy):

- ``ewma_residual_grouped``: on an interleaved panel (very different per-group levels + random walk + a feature-driven residual) a small ridge model
  trained on the grouped-EWMA residual reconstructs y with an OOS y-scale RMSE far below BOTH the raw-y model and the same pipeline on the ungrouped
  ``ewma_residual`` (whose recursion mixes the group levels at every interleaved row).
- ``volatility_normalized_residual``: on a calm-then-turbulent series where the residual scales with the local volatility, normalising by the EWMA vol
  gives the ridge model a homoscedastic target; multiplying the prediction back by the per-row vol beats the plain ``ewma_residual`` pipeline whose
  single global coefficient over-scales the calm regime.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms import get_transform


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Rmse."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _ridge_1d(x_tr: np.ndarray, t_tr: np.ndarray, x_te: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Tiny closed-form ridge y ~ a*x + b (keeps the test dependency-light and <5s)."""
    X = np.column_stack([x_tr, np.ones_like(x_tr)])
    coef = np.linalg.solve(X.T @ X + lam * np.eye(2), X.T @ t_tr)
    return coef[0] * x_te + coef[1]


def _interleaved_panel(seed: int, n_per: int = 900, levels: tuple[float, ...] = (0.0, 1000.0, -500.0)):
    """Round-robin interleaved panel: y = group_level + within-group random walk + 0.6*f + eps; base = within-group lag-1 of y."""
    rng = np.random.default_rng(seed)
    K = len(levels)
    y = np.empty(n_per * K)
    base = np.empty(n_per * K)
    f = rng.standard_normal(n_per * K)
    groups = np.tile(np.arange(K), n_per).astype(np.int64)
    for gi, level in enumerate(levels):
        walk = np.cumsum(rng.normal(scale=1.0, size=n_per))
        resid = 0.6 * f[groups == gi] + rng.normal(scale=0.3, size=n_per)
        y_g = level + walk + resid
        idx = np.flatnonzero(groups == gi)
        y[idx] = y_g
        base[idx] = np.concatenate([[y_g[0]], y_g[:-1]])
    return y, base, f, groups


def test_biz_val_ewma_grouped_beats_raw_and_ungrouped_on_interleaved_panel() -> None:
    """Biz val ewma grouped beats raw and ungrouped on interleaved panel."""
    y, base, f, groups = _interleaved_panel(0)
    n = y.size
    ntr = int(n * 0.7)
    tr, te = slice(0, ntr), slice(ntr, n)

    def _pipeline(name: str) -> float:
        """Pipeline."""
        t = get_transform(name)
        kw_tr = {"groups": groups[tr]} if t.requires_groups else {}
        kw_te = {"groups": groups[te]} if t.requires_groups else {}
        p = t.fit(y[tr], base[tr], **kw_tr)
        T_tr = t.forward(y[tr], base[tr], p, **kw_tr)
        T_hat = _ridge_1d(f[tr], T_tr, f[te])
        y_hat = t.inverse(T_hat, base[te], p, **kw_te)
        return _rmse(y_hat, y[te])

    rmse_grouped = _pipeline("ewma_residual_grouped")
    rmse_ungrouped = _pipeline("ewma_residual")
    rmse_raw = _rmse(_ridge_1d(f[tr], y[tr], f[te]), y[te])

    # Measured (seed 0): grouped ~2.6, ungrouped ~570, raw ~630. Thresholds with wide margin.
    assert rmse_grouped <= 0.05 * rmse_ungrouped, f"grouped EWMA RMSE {rmse_grouped:.2f} should be <=0.05x ungrouped {rmse_ungrouped:.2f}"
    assert rmse_grouped <= 0.05 * rmse_raw, f"grouped EWMA RMSE {rmse_grouped:.2f} should be <=0.05x raw-y {rmse_raw:.2f}"


def test_biz_val_volatility_normalized_beats_plain_ewma_on_regime_switch() -> None:
    """Biz val volatility normalized beats plain ewma on regime switch."""
    rng = np.random.default_rng(1)
    n = 6000
    sigma = np.where(np.arange(n) < n // 2, 0.5, 10.0)
    level = np.cumsum(rng.normal(scale=0.2, size=n))
    f = rng.standard_normal(n)
    y = level + sigma * (0.8 * f + rng.normal(scale=0.3, size=n))
    base = np.concatenate([[y[0]], y[:-1]])
    ntr_lo, ntr_hi = int(n * 0.35), n // 2 + int(n * 0.35)
    # Train on a window covering BOTH regimes, test on the held-out remainder of both.
    tr_mask = np.zeros(n, dtype=bool)
    tr_mask[:ntr_lo] = True
    tr_mask[n // 2 : ntr_hi] = True
    te_mask = ~tr_mask
    # Recurrent transforms consume contiguous sequences; keep the full series for forward/inverse and mask afterwards.

    def _pipeline(name: str) -> float:
        """Pipeline."""
        t = get_transform(name)
        p = t.fit(y[tr_mask], base[tr_mask], k=10)
        T_full = t.forward(y, base, p)
        T_hat_te = _ridge_1d(f[tr_mask], T_full[tr_mask], f)
        y_hat_full = t.inverse(T_hat_te, base, p)
        return _rmse(y_hat_full[te_mask], y[te_mask])

    rmse_vnr = _pipeline("volatility_normalized_residual")
    rmse_ewma = _pipeline("ewma_residual")
    rmse_raw = _rmse(_ridge_1d(f[tr_mask], y[tr_mask], f)[te_mask], y[te_mask])

    # Measured (seed 1): vnr clearly below plain ewma (per-row vol rescaling) and far below raw.
    assert rmse_vnr <= 0.9 * rmse_ewma, f"vol-normalised RMSE {rmse_vnr:.3f} should be <=0.9x plain EWMA {rmse_ewma:.3f}"
    assert rmse_vnr <= 0.8 * rmse_raw, f"vol-normalised RMSE {rmse_vnr:.3f} should be <=0.8x raw-y {rmse_raw:.3f}"
