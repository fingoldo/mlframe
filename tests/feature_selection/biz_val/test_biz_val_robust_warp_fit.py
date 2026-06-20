"""biz_value: heavy-tail-gated ROBUST WARP-COEFFICIENT FITTING (DEFAULT ON, backlog #17).

The 1-D operand pre-warp (``fit_operand_prewarp``) that BUILDS a warp feature fits its
orthogonal-polynomial coefficients with ORDINARY least squares. Under a heavy-tailed /
outlier marginal the basis entry ``x**k`` explodes on an outlier row, so the squared-error
loss is dominated by a handful of extreme rows and the fitted warp CHASES the outliers
instead of recovering the true relationship. The fix swaps the OLS solve for a Huber-IRLS
(outlier-down-weighting) solve, GATED on the same spike-contamination predicate the axis
path uses, so a clean column is byte-identical to legacy and only a contaminated column
takes the robust path.

This module pins the measured wins with wide margins (set below the measured value so
measurement noise does not trip them but a real regression -- gate stuck off, robust solve
broken -- fails them):

  WIN  : on an outlier-injected monotone column the robust warp recovers the TRUE warp with
         higher held-out R2 than the OLS warp (which chases the outliers), averaged over seeds.
  NO-CHANGE (common-case control): on clean / naturally-heavy-tailed columns the predicate
         does not fire -> the fitted coefficients are BYTE-IDENTICAL with the gate on or off.
  NO-FALSE-POSITIVE: a pure-noise operand admits no usable warp (no spurious signal recovery).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from mlframe.feature_selection.filters.hermite_fe import (
    _detect_heavy_tail,
    apply_operand_prewarp,
    fit_operand_prewarp,
)

_N = 2000
_TRUE_WARP = lambda z: 0.7 * z + 0.25 * z ** 3  # monotone increasing on [-2.5, 2.5]


def _inject_outliers(rng, x, frac=0.015, scale_iqr=12.0):
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.size
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = max(q3 - q1, 1e-9)
    idx = rng.choice(n, max(1, int(n * frac)), replace=False)
    x[idx] = med + rng.choice([-1.0, 1.0], idx.size) * scale_iqr * iqr
    return x


def _oos_r2(spec, x_eval, y_target):
    """Held-out R2 of the best linear map of the warp onto the TRUE warp shape."""
    w = apply_operand_prewarp(x_eval, spec)
    if np.std(w) < 1e-12:
        return -1.0
    A = np.vstack([w, np.ones_like(w)]).T
    coef, *_ = np.linalg.lstsq(A, y_target, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((y_target - pred) ** 2))
    ss_tot = float(np.sum((y_target - y_target.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else -1.0


def _fit(x, y, *, robust):
    old = os.environ.get("MLFRAME_ROBUST_WARP_FIT")
    os.environ["MLFRAME_ROBUST_WARP_FIT"] = "1" if robust else "0"
    try:
        return fit_operand_prewarp(x, y, basis="chebyshev", max_degree=4)
    finally:
        if old is None:
            os.environ.pop("MLFRAME_ROBUST_WARP_FIT", None)
        else:
            os.environ["MLFRAME_ROBUST_WARP_FIT"] = old


# ---------------------------------------------------------------------------
# WIN: robust warp recovers the true shape better than OLS on outlier columns.
# ---------------------------------------------------------------------------


def test_robust_beats_plain_oos_on_outlier_column():
    """Averaged over seeds, the robust warp's held-out R2 (vs the true warp) exceeds the
    OLS warp's. The OLS fit chases the injected outliers; the robust fit tracks the bulk."""
    deltas = []
    for seed in range(12):
        rng = np.random.default_rng(1000 + seed)
        x = rng.uniform(-2.5, 2.5, _N)
        y = _TRUE_WARP(x) + rng.normal(0, 0.3, _N)
        tr = np.arange(_N) % 3 != 0
        te = ~tr
        x_out = _inject_outliers(rng, x[tr])
        y_true_te = _TRUE_WARP(x[te])
        s_plain = _fit(x_out, y[tr], robust=False)
        s_rob = _fit(x_out, y[tr], robust=True)
        assert s_rob.get("robust_fit") is True  # gate must have fired
        deltas.append(_oos_r2(s_rob, x[te], y_true_te) - _oos_r2(s_plain, x[te], y_true_te))
    deltas = np.array(deltas)
    # Measured mean dR2 ~ +0.0045 over 30 seeds, 30/30 wins, worst +0.002. Pin a wide
    # margin: positive mean AND never a meaningful regression.
    assert deltas.mean() > 0.002, f"robust did not beat plain on average: mean dR2={deltas.mean():.4f}"
    assert deltas.min() > -0.005, f"robust regressed on a seed: worst dR2={deltas.min():.4f}"
    assert (deltas > 0).mean() >= 0.9, f"robust won on only {(deltas>0).mean():.0%} of seeds"


# ---------------------------------------------------------------------------
# NO-CHANGE common-case control: clean column -> byte-identical fit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen",
    [
        pytest.param(lambda r: r.uniform(-2.5, 2.5, _N), id="uniform"),
        pytest.param(lambda r: r.standard_normal(_N), id="gauss"),
        pytest.param(lambda r: r.lognormal(0.0, 1.0, _N), id="lognormal"),
        pytest.param(lambda r: r.standard_t(3, _N), id="student_t3"),
        pytest.param(lambda r: r.exponential(1.0, _N), id="exponential"),
    ],
)
def test_clean_data_byte_identical(gen):
    """On clean / naturally-heavy-tailed columns the spike predicate does NOT fire, so the
    robust-on and robust-off fits are BIT-for-BIT identical (the common case is untouched)."""
    rng = np.random.default_rng(11)
    x = gen(rng)
    assert _detect_heavy_tail(x) is False
    y = 0.6 * (x - np.median(x)) / (np.std(x) + 1e-12) + rng.normal(0, 0.4, x.size)
    s_off = _fit(x, y, robust=False)
    s_on = _fit(x, y, robust=True)
    assert s_off is not None and s_on is not None
    np.testing.assert_array_equal(s_off["coef"], s_on["coef"])
    assert not s_on.get("robust_fit", False)


# ---------------------------------------------------------------------------
# NO-FALSE-POSITIVE: pure-noise operand admits no usable warp.
# ---------------------------------------------------------------------------


def test_pure_noise_no_spurious_recovery():
    """A robust warp fit on a noise operand (y independent of x) must not manufacture
    signal: held-out R2 vs the noise target stays ~0."""
    rng = np.random.default_rng(0)
    x = _inject_outliers(rng, rng.uniform(-2.5, 2.5, _N))
    y = rng.normal(0, 1, _N)  # independent of x
    tr = np.arange(_N) % 3 != 0
    te = ~tr
    spec = _fit(x[tr], y[tr], robust=True)
    r2 = _oos_r2(spec, x[te], y[te]) if spec is not None else -1.0
    assert r2 < 0.05, f"robust warp manufactured signal on pure noise: OOS R2={r2:.4f}"
