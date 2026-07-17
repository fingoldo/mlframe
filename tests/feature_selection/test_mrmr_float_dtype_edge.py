"""Float / continuous-operand dtype hardening for the MRMR usability-signal + adaptive-nbins float paths.

Covers: f32-vs-f64 SELECTION-equivalence of the usability |corr| forms under heavy-tailed operands (the tail-concentrated regime this module
exists for), overflow/inf handling in the ``x0*x0`` / ratio forms, the ``_abs_pearson_njit`` variance guards + f64 accumulators on
constant / near-constant / 2-row / all-nan / large-mean columns, subsample determinism + outlier-proportion preservation, and the
regression pin for non-finite inner edges leaking from ``_edges_from_quantiles`` / ``_edges_from_uniform`` into searchsorted.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_usability_signal as U
from mlframe.feature_selection.filters import _adaptive_nbins as A


def _forms_gate(y, x0, x1, dtype, *, min_corr=0.6, margin=1.05):
    """Reproduce usability_form_corrs' form materialisation at a fixed dtype and return (cp, cs, gate-passes)."""
    eps = 1e-12
    _y = np.asarray(y, dtype=dtype).ravel()
    _x0 = np.asarray(x0, dtype=dtype).ravel()
    _x1 = np.asarray(x1, dtype=dtype).ravel()

    def sd(n, d):
        return n / np.where(np.abs(d) < eps, np.nan, d)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pf = [sd(_x0, _x1), sd(_x1, _x0), sd(_x0 * _x0, _x1), sd(_x1 * _x1, _x0), _x0 * _x1]
        sf = [_x0, _x1, _x0 * _x0, _x1 * _x1]
        cp = max((U.abs_pearson(_y, f) for f in pf), default=0.0)
        cs = max((U.abs_pearson(_y, f) for f in sf), default=0.0)
    return cp, cs, bool(cp >= min_corr and cp >= margin * cs)


def test_heavy_tail_f32_vs_f64_selection_equivalent():
    """The wide-margin usability |corr| gate must reach the SAME keep/reject decision in f32 and f64 across heavy-tailed operands
    spanning 1e2..1e6 (the tail-concentrated signal this module targets). Selection-equivalence is the bar, not bit-identical MI."""
    rng = np.random.default_rng(0)
    max_div = 0.0
    for _ in range(400):
        n = int(rng.integers(300, 1500))
        scale = 10.0 ** rng.integers(2, 7)
        x0 = rng.standard_t(3, n) * scale
        x1 = rng.standard_t(3, n) * scale
        frac = rng.uniform(0.3, 0.9)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sig = (x0 * x0) / np.where(np.abs(x1) < 1e-12, np.nan, x1)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        y = frac * sig + (1 - frac) * rng.normal(0, np.std(sig) + 1e-9, n) * rng.uniform(1, 20)
        cp32, cs32, g32 = _forms_gate(y, x0, x1, np.float32)
        cp64, cs64, g64 = _forms_gate(y, x0, x1, np.float64)
        max_div = max(max_div, abs(cp32 - cp64), abs(cs32 - cs64))
        assert g32 == g64, f"f32/f64 gate flip: cp32={cp32:.5f} cs32={cs32:.5f} | cp64={cp64:.5f} cs64={cs64:.5f}"
    assert max_div < 1e-3, f"max |corr| f32-vs-f64 divergence {max_div:.3e} exceeds selection-safe band"


def test_overflow_inf_dropped_not_poisoning_corr():
    """``x0*x0`` overflows to +inf in f32 for |x0|>~1.8e19; the njit finite-mask must DROP those rows, never poison |corr| with inf/nan."""
    rng = np.random.default_rng(1)
    n = 2000
    x0 = rng.normal(0, 1, n)
    rng.normal(1, 1, n)
    y = rng.normal(0, 1, n)
    x0[:5] = [2e19, -3e19, 5e19, 1e20, -2e19]
    with np.errstate(over="ignore", invalid="ignore"):
        sq32 = x0.astype(np.float32) * x0.astype(np.float32)
    assert np.isinf(np.asarray(sq32, np.float64)).sum() == 5  # confirm f32 overflow actually happened
    c = U.abs_pearson(y.astype(np.float32), sq32)
    assert np.isfinite(c) and 0.0 <= c <= 1.0  # inf rows dropped, corr stays a valid finite statistic


@pytest.mark.parametrize(
    "desc,a,b,expected",
    [
        ("constant", np.arange(100.0), np.ones(100), 0.0),
        ("near_const_1e-30", np.arange(100.0), 1.0 + np.linspace(0, 1e-15, 100), 0.0),
        ("one_row", np.array([1.0]), np.array([3.0]), 0.0),
        ("all_nan", np.arange(100.0), np.full(100, np.nan), 0.0),
    ],
)
def test_abs_pearson_degenerate_guards(desc, a, b, expected):
    """Constant / sub-1e-15-variance / <2-row / all-nan columns must return exactly 0.0, never a spurious huge |corr| from a
    catastrophic-cancellation variance that slips the ``va<=0`` / ``den<=0`` guards."""
    assert U.abs_pearson(a, b) == expected, desc


def test_abs_pearson_two_row_and_large_mean_f64_accumulators():
    """2-row perfect line -> 1.0 (n>=2 valid); a large-mean (1e8 offset) perfectly-correlated column -> ~1.0 only if the njit
    accumulators are f64 (f32 sums would cancel catastrophically and mis-report)."""
    assert U.abs_pearson([1.0, 2.0], [3.0, 4.0]) == pytest.approx(1.0)
    z = np.arange(100.0) + 1e8
    assert U.abs_pearson(z, z) == pytest.approx(1.0, abs=1e-9)


def test_subsample_deterministic_and_outlier_preserving():
    """Two calls give identical rows (selection reproducibility); the strided subsample preserves the outlier proportion
    (the whole reason the tail-concentration |corr| survives subsampling)."""
    n = 3 * U._ABS_PEARSON_MAX_ROWS if U._ABS_PEARSON_MAX_ROWS > 0 else 600000
    big = np.random.default_rng(0).normal(size=n)
    assert np.array_equal(U._subsample_for_corr(big.copy())[0], U._subsample_for_corr(big.copy())[0])
    z = np.zeros(n)
    z[np.random.default_rng(7).choice(n, size=n // 3, replace=False)] = 1e6  # random (non-periodic) so outliers don't alias the stride
    sub = U._subsample_for_corr(z.copy())[0]
    assert np.mean(z > 1e5) == pytest.approx(np.mean(sub > 1e5), abs=5e-3)
    # n below cap -> no stride (identity); exactly at cap -> no stride.
    small = np.arange(1000.0)
    assert U._subsample_for_corr(small)[0].shape[0] == 1000
    if U._ABS_PEARSON_MAX_ROWS > 0:
        assert U._corr_stride(U._ABS_PEARSON_MAX_ROWS) == 1


def test_edges_reject_non_finite_inner_edges():
    """REGRESSION: an inf-bearing column must not leak a non-finite INNER edge into searchsorted. ``_edges_from_uniform`` used
    ``np.nanmin/nanmax`` (inf-blind) -> all-NaN edges; ``_edges_from_quantiles`` used ``np.nanpercentile`` -> an inf inner edge on a
    mostly-inf tail. Both now drop non-finite values first (matching the sibling FD/QS/optimal_joint filters)."""
    x_one_inf = np.concatenate([np.random.default_rng(0).normal(size=200), [np.inf, -np.inf, np.nan]])
    x_many_inf = np.concatenate([np.random.default_rng(0).normal(size=50), [np.inf] * 40, [-np.inf] * 10, [np.nan] * 5])
    for x in (x_one_inf, x_many_inf):
        eu = A._edges_from_uniform(x, 10)
        eq = A._edges_from_quantiles(x, 10)
        assert np.isfinite(eu).all(), f"uniform leaked non-finite edge: {eu}"
        assert np.isfinite(eq).all(), f"quantile leaked non-finite edge: {eq}"


def test_edges_clean_data_unchanged():
    """The finite-filter must be a no-op on clean (all-finite) columns -> selection-equivalence preserved."""
    xc = np.random.default_rng(1).normal(size=500)
    ref = np.unique(np.percentile(xc, np.linspace(0, 100, 11)))[1:-1]
    assert np.allclose(A._edges_from_quantiles(xc, 10), ref)
