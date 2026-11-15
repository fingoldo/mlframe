"""The batched hinge precheck (numba kernel, fused CUDA kernel) gives the drops of the per-column QR precheck, and the detector's columns are the same with it."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe._numerical_floors import VARIANCE_FLOOR
from mlframe.feature_selection.filters import _hinge_basis_fe as B
from mlframe.feature_selection.filters import _hinge_precheck_gpu as H

QS = B._HINGE_PRECHECK_QS


def _reference_drop(x: np.ndarray, y: np.ndarray) -> float:
    """The best fractional SSE drop of the QR precheck loop of ``_hinge_slope_change_plausible`` (``-1`` when it rejects the column outright)."""
    n = x.size
    Q, r_y, sse = B._linear_qr_fit(x, y)
    if not np.isfinite(sse) or sse <= VARIANCE_FLOOR:
        return -1.0
    best = 0.0
    for c in np.unique(np.quantile(x, np.asarray(QS))):
        n_right = int(np.count_nonzero(x > c))
        if n_right < B._HINGE_MIN_SEG_ROWS or (n - n_right) < B._HINGE_MIN_SEG_ROWS:
            continue
        relu = np.maximum(x - c, 0.0)
        r_relu = relu - Q @ (Q.T @ relu)
        denom = float(r_relu @ r_relu)
        sse_h = sse if denom < VARIANCE_FLOOR else sse - float(r_relu @ r_y) ** 2 / denom
        if np.isfinite(sse_h):
            best = max(best, 1.0 - sse_h / sse)
    return best


def _block(n: int = 6000):
    """Twelve columns: a kinked one, noise, a tie-heavy one, a near-constant one and a pure line."""
    rng = np.random.default_rng(1)
    xs = rng.normal(size=(12, n))
    y = 1.5 * np.maximum(xs[0] - 0.2, 0.0) + 0.3 * rng.normal(size=n)
    xs[3] = np.round(xs[3], 1)
    xs[5] = 1.0 + 1e-3 * rng.normal(size=n)
    xs[7] = (y - y.mean()) * 2.0 + 3.0
    return xs, y


def test_the_host_kernel_gives_the_drops_of_the_qr_precheck():
    """Every column's drop equals the QR reference to the rounding of a double, and the kinked column clears the precheck bar while noise does not."""
    xs, y = _block()
    ref = np.array([_reference_drop(x, y) for x in xs])
    got = H._drops_host(xs, y, QS, B._HINGE_MIN_SEG_ROWS, VARIANCE_FLOOR)
    # the pure line (column 7) leaves a residual at rounding level: the QR path rejects it outright (-1), the closed form reads a drop of about 0; both are below the bar
    ordinary = np.arange(len(xs)) != 7
    assert np.abs(got - ref)[ordinary].max() < 1e-9
    assert got[7] < B._HINGE_PRECHECK_MIN_SSE_DROP and ref[7] < B._HINGE_PRECHECK_MIN_SSE_DROP
    assert got[0] >= B._HINGE_PRECHECK_MIN_SSE_DROP > got[1]
    assert ((got >= B._HINGE_PRECHECK_MIN_SSE_DROP) == (ref >= B._HINGE_PRECHECK_MIN_SSE_DROP)).all()


def test_the_device_kernel_gives_the_drops_of_the_host_kernel():
    """The CUDA kernel agrees with the numba kernel on the same block (float64 columns)."""
    pytest.importorskip("cupy")
    xs, y = _block()
    dev = H._drops_device(xs, y, QS, B._HINGE_MIN_SEG_ROWS, VARIANCE_FLOOR)
    if dev is None:
        pytest.skip("no usable device")
    host = H._drops_host(xs, y, QS, B._HINGE_MIN_SEG_ROWS, VARIANCE_FLOOR)
    assert np.abs(dev - host)[np.arange(len(xs)) != 7].max() < 1e-9
    assert dev[7] < B._HINGE_PRECHECK_MIN_SSE_DROP


@pytest.mark.parametrize("forced", ["0", "1"])
def test_the_forced_switch_picks_the_backend(monkeypatch, forced):
    """``MLFRAME_HINGE_PRECHECK_DEVICE`` forces a side whatever the row count and the cache say."""
    monkeypatch.setenv("MLFRAME_HINGE_PRECHECK_DEVICE", forced)
    assert H.hinge_precheck_on_device(10) is (forced == "1")


def test_the_survivors_keep_every_column_the_detector_finds_a_hinge_in(monkeypatch):
    """Filtering by the batched precheck loses no column the per-column detector would have given breakpoints for."""
    monkeypatch.setenv("MLFRAME_HINGE_PRECHECK_DEVICE", "0")
    xs, y = _block(4000)
    cols = [(f"c{i}", x) for i, x in enumerate(xs)]
    kept = {c for c, _x in B._precheck_survivors(cols, y)}
    for c, x in cols:
        if B._detect_hinge_breakpoints(x, y):
            assert c in kept
    assert "c1" not in kept and "c0" in kept


def test_a_narrow_block_budget_gives_the_same_drops(monkeypatch):
    """With a column block of two columns the drops equal the single-block ones (each column's drop depends on that column only)."""
    xs, y = _block(3000)
    whole = H.hinge_precheck_drops(xs, y, QS, B._HINGE_MIN_SEG_ROWS)
    monkeypatch.setattr(H, "_BLOCK_BYTES", 2 * 8 * xs.shape[1])
    assert H.column_block_width(xs.shape[1]) == 2
    assert np.array_equal(H.hinge_precheck_drops(xs, y, QS, B._HINGE_MIN_SEG_ROWS), whole)
