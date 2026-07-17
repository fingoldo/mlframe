"""Unit tests for the heavy-tail-gated ROBUST WARP-COEFFICIENT FITTING (backlog
idea #17, 2026-06-10).

Covers the Huber-IRLS solver (``_huber_irls_lstsq``) and its kept-under-its-own-
name OLS sibling (``_ols_lstsq``), the heavy-tail dispatcher
(``fit_basis_coef_robust``), and the wiring into the 1-D operand pre-warp
(``fit_operand_prewarp``). Central contract: a CLEAN / naturally-heavy-tailed
operand takes the byte-identical OLS path; only a spike-CONTAMINATED operand takes
the robust Huber path, and its winsor-bound provenance is recorded for leak-safe
replay. The env var ``MLFRAME_ROBUST_WARP_FIT`` is an independent gate from the
axis path's ``MLFRAME_ROBUST_AXIS`` (orthogonal-knob rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.hermite_fe import (
    _detect_heavy_tail,
    _huber_irls_lstsq,
    _ols_lstsq,
    _robust_warp_fit_enabled,
    apply_operand_prewarp,
    fit_basis_coef_robust,
    fit_operand_prewarp,
)


def _spike(rng, x, frac=0.015, scale_iqr=12.0):
    """Replace ``frac`` of x with gross outliers at +/- scale_iqr * IQR from median."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-9)
    idx = rng.choice(n, max(1, int(n * frac)), replace=False)
    x[idx] = med + rng.choice([-1.0, 1.0], idx.size) * scale_iqr * iqr
    return x


# ---------------------------------------------------------------------------
# Env gate: default ON, independent of the axis gate.
# ---------------------------------------------------------------------------


def test_warp_fit_gate_default_on(monkeypatch):
    monkeypatch.delenv("MLFRAME_ROBUST_WARP_FIT", raising=False)
    assert _robust_warp_fit_enabled() is True


def test_warp_fit_gate_env_override(monkeypatch):
    for val in ("0", "false", "off", "no"):
        monkeypatch.setenv("MLFRAME_ROBUST_WARP_FIT", val)
        assert _robust_warp_fit_enabled() is False
    monkeypatch.setenv("MLFRAME_ROBUST_WARP_FIT", "1")
    assert _robust_warp_fit_enabled() is True


def test_warp_fit_gate_independent_of_axis_gate(monkeypatch):
    """The two robustness knobs are orthogonal: turning the axis gate off must NOT
    turn the warp-fit gate off, and vice versa."""
    monkeypatch.setenv("MLFRAME_ROBUST_AXIS", "0")
    monkeypatch.delenv("MLFRAME_ROBUST_WARP_FIT", raising=False)
    assert _robust_warp_fit_enabled() is True


# ---------------------------------------------------------------------------
# Huber-IRLS solver: matches OLS on clean data, resists outliers on dirty data.
# ---------------------------------------------------------------------------


def test_huber_irls_matches_ols_on_clean_data():
    """On a clean (no-outlier) overdetermined system the Huber-IRLS solution must
    converge to essentially the OLS solution (all rows keep unit weight)."""
    rng = np.random.default_rng(0)
    n = 500
    B = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    true_c = np.array([0.3, -1.2, 0.8])
    y = B @ true_c + rng.normal(0, 0.1, n)
    c_ols = _ols_lstsq(B, y)
    c_rob = _huber_irls_lstsq(B, y)
    assert c_rob is not None
    np.testing.assert_allclose(c_rob, c_ols, atol=1e-3)


def test_huber_irls_resists_outliers_better_than_ols():
    """With a handful of gross outlier rows the Huber-IRLS coefficients stay close to
    the true coefficients while OLS is dragged away."""
    rng = np.random.default_rng(1)
    n = 500
    B = np.column_stack([np.ones(n), rng.standard_normal(n)])
    true_c = np.array([0.0, 2.0])
    y = B @ true_c + rng.normal(0, 0.1, n)
    # Corrupt 3% of the targets with huge outliers.
    idx = rng.choice(n, 15, replace=False)
    y[idx] += rng.choice([-1.0, 1.0], 15) * 50.0
    c_ols = _ols_lstsq(B, y)
    c_rob = _huber_irls_lstsq(B, y)
    err_ols = np.linalg.norm(c_ols - true_c)
    err_rob = np.linalg.norm(c_rob - true_c)
    assert err_rob < err_ols, f"robust err {err_rob} not < OLS err {err_ols}"


def test_huber_irls_degenerate_returns_none():
    """Underdetermined / malformed inputs return None (caller falls back to OLS)."""
    assert _huber_irls_lstsq(np.ones((2, 5)), np.ones(2)) is None  # rows < cols
    assert _huber_irls_lstsq(np.ones((5, 2)), np.ones(3)) is None  # shape mismatch


# ---------------------------------------------------------------------------
# Dispatcher: clean -> OLS byte-identical; heavy-tailed -> robust + winsor bounds.
# ---------------------------------------------------------------------------


def test_dispatcher_clean_operand_byte_identical_to_ols():
    """A clean operand must NOT trip the heavy-tail gate -> the dispatcher returns the
    BIT-IDENTICAL OLS coefficients, robust_used False, no winsor bounds."""
    rng = np.random.default_rng(2)
    n = 1000
    x = rng.standard_normal(n)
    assert _detect_heavy_tail(x) is False
    B = np.column_stack([np.ones(n), x, x**2])
    y = 0.5 + 1.5 * x - 0.3 * x**2 + rng.normal(0, 0.1, n)
    coef, robust_used, winsor = fit_basis_coef_robust(B, y, x)
    assert robust_used is False and winsor is None
    np.testing.assert_array_equal(coef, _ols_lstsq(B, y))


def test_dispatcher_heavy_tailed_operand_uses_robust_and_records_winsor():
    """A spike-contaminated operand trips the gate -> robust fit used, winsor bounds
    (MAD-anchored, tracking the clean core not the spikes) recorded for replay."""
    rng = np.random.default_rng(3)
    n = 1500
    x_clean = rng.uniform(-2.5, 2.5, n)
    x = _spike(rng, x_clean, frac=0.02, scale_iqr=15.0)
    assert _detect_heavy_tail(x) is True
    B = np.column_stack([np.ones(n), x, x**2])
    y = rng.normal(0, 1, n)
    _coef, robust_used, winsor = fit_basis_coef_robust(B, y, x)
    assert robust_used is True
    assert winsor is not None
    lo, hi = winsor
    # Winsor bounds anchored to the clean +/-2.5 core, not the +/-(15*IQR) spikes.
    assert -8.0 < lo < 0.0 < hi < 8.0, f"winsor leaked into spikes: {winsor}"


def test_dispatcher_gate_off_forces_ols(monkeypatch):
    """With the env gate off, even a heavy-tailed operand takes the OLS path."""
    monkeypatch.setenv("MLFRAME_ROBUST_WARP_FIT", "0")
    rng = np.random.default_rng(4)
    n = 1500
    x = _spike(rng, rng.uniform(-2.5, 2.5, n), frac=0.02, scale_iqr=15.0)
    B = np.column_stack([np.ones(n), x, x**2])
    y = rng.normal(0, 1, n)
    coef, robust_used, winsor = fit_basis_coef_robust(B, y, x)
    assert robust_used is False and winsor is None
    np.testing.assert_array_equal(coef, _ols_lstsq(B, y))


# ---------------------------------------------------------------------------
# 1-D operand pre-warp wiring: clean byte-identical; heavy-tail robust + provenance.
# ---------------------------------------------------------------------------


def _fit_with_gate(x, y, on):
    import os

    old = os.environ.get("MLFRAME_ROBUST_WARP_FIT")
    os.environ["MLFRAME_ROBUST_WARP_FIT"] = "1" if on else "0"
    try:
        return fit_operand_prewarp(x, y, basis="chebyshev", max_degree=4)
    finally:
        if old is None:
            os.environ.pop("MLFRAME_ROBUST_WARP_FIT", None)
        else:
            os.environ["MLFRAME_ROBUST_WARP_FIT"] = old


@pytest.mark.parametrize("seed", range(4))
def test_prewarp_clean_byte_identical(seed):
    """On a clean operand the prewarp coefficients are bit-identical with the gate on
    or off, and no robust provenance is attached."""
    rng = np.random.default_rng(50 + seed)
    n = 2000
    x = rng.uniform(-2.5, 2.5, n)
    y = 0.7 * x + 0.25 * x**3 + rng.normal(0, 0.3, n)
    s_off = _fit_with_gate(x, y, on=False)
    s_on = _fit_with_gate(x, y, on=True)
    assert s_off is not None and s_on is not None
    np.testing.assert_array_equal(s_off["coef"], s_on["coef"])
    assert not s_on.get("robust_fit", False)


def test_prewarp_heavy_tail_sets_robust_provenance():
    """On a spike-contaminated operand the robust prewarp attaches robust_fit + winsor
    bounds; replay via apply_operand_prewarp is closed-form on coef (leak-safe)."""
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(-2.5, 2.5, n)
    y = 0.7 * x + 0.25 * x**3 + rng.normal(0, 0.3, n)
    x_out = _spike(rng, x, frac=0.015, scale_iqr=12.0)
    spec = _fit_with_gate(x_out, y, on=True)
    assert spec is not None and spec.get("robust_fit") is True
    assert "winsor_lo" in spec and "winsor_hi" in spec
    # Replay must be deterministic + finite (closed-form on coef, no y).
    w1 = apply_operand_prewarp(x, spec)
    w2 = apply_operand_prewarp(x, spec)
    np.testing.assert_array_equal(w1, w2)
    assert np.all(np.isfinite(w1))


def test_prewarp_natural_heavy_tail_not_robustified():
    """A genuinely heavy-tailed-but-clean operand (lognormal, continuous tail) must NOT
    trip the gate -> byte-identical OLS prewarp, no robust provenance."""
    rng = np.random.default_rng(8)
    n = 2000
    x = rng.lognormal(0.0, 1.0, n)
    assert _detect_heavy_tail(x) is False
    y = np.log1p(x) + rng.normal(0, 0.2, n)
    s_off = _fit_with_gate(x, y, on=False)
    s_on = _fit_with_gate(x, y, on=True)
    np.testing.assert_array_equal(s_off["coef"], s_on["coef"])
    assert not s_on.get("robust_fit", False)
