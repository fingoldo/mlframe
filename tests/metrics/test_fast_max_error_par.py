"""Regression sensor: ``fast_max_error`` parallel reduction twin + size dispatch (iter130).

Pre-fix ``fast_max_error`` dispatched only to the serial ``_fast_max_error_seq`` -- no ``parallel=True`` twin, unlike every
sibling regression metric (MAE/MSE/R2). At 10M this left a ~2-3x single-call win on the table. The fix adds
``_fast_max_error_par`` (prange ``max`` reduction, bit-identical) and routes large arrays to it via
``_MAX_ERROR_PAR_THRESHOLD``.

These sensors FAIL on pre-fix code: ``_fast_max_error_par`` does not exist (AttributeError), and the dispatcher never
calls a parallel twin.
"""

import numpy as np
import pytest

import mlframe.metrics.regression._regression_metrics as M


def test_fast_max_error_par_twin_exists_and_is_bit_identical():
    """Fast max error par twin exists and is bit identical."""
    par = getattr(M, "_fast_max_error_par", None)
    assert par is not None, "missing _fast_max_error_par parallel twin (iter130 regression)"
    rng = np.random.default_rng(0)
    for n in (1, 17, 1000, 250_000):
        yt = rng.standard_normal(n)
        yp = rng.standard_normal(n)
        seq_v = M._fast_max_error_seq(yt, yp)
        par_v = par(yt, yp)
        assert seq_v == par_v, f"par twin diverged from seq at n={n}: {seq_v} vs {par_v}"


def test_fast_max_error_dispatch_routes_to_par_above_threshold(monkeypatch):
    """Fast max error dispatch routes to par above threshold."""
    monkeypatch.setattr(M, "_MAX_ERROR_PAR_THRESHOLD", 1000)
    calls = {"seq": 0, "par": 0}
    real_seq = M._fast_max_error_seq
    real_par = M._fast_max_error_par

    def spy_seq(a, b):
        """Spy seq."""
        calls["seq"] += 1
        return real_seq(a, b)

    def spy_par(a, b):
        """Spy par."""
        calls["par"] += 1
        return real_par(a, b)

    monkeypatch.setattr(M, "_fast_max_error_seq", spy_seq)
    monkeypatch.setattr(M, "_fast_max_error_par", spy_par)

    rng = np.random.default_rng(1)
    big = (rng.standard_normal(5000), rng.standard_normal(5000))
    small = (rng.standard_normal(100), rng.standard_normal(100))

    M.fast_max_error(*big)
    assert calls["par"] == 1 and calls["seq"] == 0, "large 1-D input must route to parallel twin"

    calls["par"] = calls["seq"] = 0
    M.fast_max_error(*small)
    assert calls["seq"] == 1 and calls["par"] == 0, "small 1-D input must stay on the serial kernel"


def test_fast_max_error_matches_sklearn():
    """Fast max error matches sklearn."""
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    yt = rng.standard_normal(6000)
    yp = rng.standard_normal(6000)
    with_par = M.fast_max_error(yt, yp)  # 6000 >= patched? no -- uses default 5M -> seq; identity vs sklearn either way
    assert with_par == pytest.approx(sk.max_error(yt, yp))
