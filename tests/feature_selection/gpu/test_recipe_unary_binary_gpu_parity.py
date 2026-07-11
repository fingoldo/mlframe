"""GPU-resident ``unary_binary`` recipe replay must match the numpy replay.

Covers EACH GPU-mapped unary and binary operator: the cupy materialisation of
``binary(unary_a(X[a]), unary_b(X[b]))`` (``apply_unary_binary_gpu``) must equal
the numpy ``_apply_unary_binary`` within tolerance, so F2 selects the SAME
features under the resident opt-in. A wrong operator mapping is proven to FAIL
the parity assertion (negative control).

Auto-skips on CUDA-unavailable hosts via ``pytest.importorskip("cupy")``.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import pandas as pd

from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary import (
    _apply_unary_binary,
    build_unary_binary_recipe,
)
from mlframe.feature_selection.filters.engineered_recipes._recipe_unary_binary_gpu import (
    apply_unary_binary_gpu,
    _gpu_binary,
    _gpu_unary,
)


def _need_cuda():
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False


pytestmark = pytest.mark.skipif(not _need_cuda(), reason="CUDA not available")


_UNARY = [
    "identity", "neg", "abs", "sign", "rint",
    "sqr", "qubed", "reciproc", "invsquared", "invqubed",
    "cbrt", "sqrt", "invcbrt", "invsqrt",
    "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh", "log",
]
_BINARY = [
    "mul", "add", "sub", "div", "max", "min",
    "abs_diff", "hypot", "signed", "ratio_abs",
    "pow", "logaddexp", "heaviside", "greater", "less", "equal",
]


def _frame(seed=7, n=4000):
    rng = np.random.default_rng(seed)
    # Mixed-sign, includes zeros and small positives so div / log / reciprocal
    # guards are exercised exactly as the numpy registry handles them.
    a = rng.uniform(-3.0, 3.0, n)
    b = rng.uniform(-3.0, 3.0, n)
    b[::97] = 0.0           # exact-zero denominators for the _safe_div path
    a[::53] = 0.0           # zeros for reciproc / log non-positive shift
    return pd.DataFrame({"a": a.astype(np.float64), "b": b.astype(np.float64)})


def _recipe(u_a, u_b, binary, preset="maximal"):
    return build_unary_binary_recipe(
        name=f"{binary}({u_a}(a),{u_b}(b))",
        src_a_name="a", src_b_name="b",
        unary_a_name=u_a, unary_b_name=u_b,
        binary_name=binary,
        unary_preset=preset, binary_preset=preset,
        quantization_nbins=None, quantization_method=None,
        quantization_dtype=np.float32,
    )


@pytest.mark.parametrize("u", _UNARY)
def test_each_unary_gpu_matches_numpy(u, monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_VRAM_F32", raising=False)  # f64 -> tight tol
    X = _frame()
    rec = _recipe(u, "identity", "add")
    cpu = np.asarray(_apply_unary_binary(rec, X), dtype=np.float64)
    gpu = apply_unary_binary_gpu(rec, X)
    assert gpu is not None, f"unary {u!r} should be GPU-eligible"
    gpu = np.asarray(gpu, dtype=np.float64)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-6,
                               err_msg=f"unary {u!r} GPU != numpy")


@pytest.mark.parametrize("binary", _BINARY)
def test_each_binary_gpu_matches_numpy(binary, monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_VRAM_F32", raising=False)
    X = _frame()
    rec = _recipe("identity", "identity", binary)
    cpu = np.asarray(_apply_unary_binary(rec, X), dtype=np.float64)
    gpu = apply_unary_binary_gpu(rec, X)
    assert gpu is not None, f"binary {binary!r} should be GPU-eligible"
    gpu = np.asarray(gpu, dtype=np.float64)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-6,
                               err_msg=f"binary {binary!r} GPU != numpy")


def test_compound_div_sqr_matches_numpy(monkeypatch):
    """The F2 a/b half div(sqr(a),abs(b)) -- the operator chain the goal needs."""
    monkeypatch.delenv("MLFRAME_FE_VRAM_F32", raising=False)
    X = _frame()
    rec = _recipe("sqr", "abs", "div")
    cpu = np.asarray(_apply_unary_binary(rec, X), dtype=np.float64)
    gpu = np.asarray(apply_unary_binary_gpu(rec, X), dtype=np.float64)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-6)


def test_log_smart_shift_matches_numpy(monkeypatch):
    """smart_log's data-dependent (1e-5 - nanmin) shift must replay identically."""
    monkeypatch.delenv("MLFRAME_FE_VRAM_F32", raising=False)
    X = _frame()
    rec = _recipe("log", "sin", "mul")
    cpu = np.asarray(_apply_unary_binary(rec, X), dtype=np.float64)
    gpu = np.asarray(apply_unary_binary_gpu(rec, X), dtype=np.float64)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "u,x",
    [
        ("reciproc", [0.0, 2.0, -2.0]),
        ("invsquared", [0.0, 2.0, -2.0]),
        ("invqubed", [0.0, 2.0, -2.0]),
        # invcbrt / invsqrt: positive-only reference values -- a negative base with a fractional
        # exponent is an unrelated, pre-existing nan domain restriction (matches np.power itself),
        # out of scope for this eps-floor-at-zero fix (same carve-out as the numpy-side test).
        ("invcbrt", [0.0, 2.0, 4.0]),
        ("invsqrt", [0.0, 2.0, 4.0]),
    ],
)
def test_gpu_unary_reciprocal_power_zero_is_finite(u, x):
    """GPU eps-floor pin: a genuine zero operand must not produce +-inf on the GPU-resident replay path,
    mirroring the numpy registry's ``_safe_pow`` fix. Also what keeps ``test_each_unary_gpu_matches_numpy``
    green for these names now that the numpy side floors to a finite value at x=0 -- before this fix the
    GPU twin still produced +-inf at x=0 while the CPU side returned a finite eps-floored value, so the two
    would have diverged well outside the parity test's ``rtol=1e-6, atol=1e-6`` tolerance."""
    x_gpu = cp.asarray(x, dtype=cp.float64)
    out = cp.asnumpy(_gpu_unary(cp, u, x_gpu, None, "a"))
    assert np.all(np.isfinite(out)), f"{u} at x=0 must be finite on GPU, got {out}"


def test_wrong_operator_mapping_would_fail():
    """Negative control: a WRONG cupy binary mapping (mul instead of add) must
    NOT match the numpy add output -- proves the parity test has teeth."""
    X = _frame()
    a = cp.asarray(X["a"].to_numpy(), dtype=cp.float64)
    b = cp.asarray(X["b"].to_numpy(), dtype=cp.float64)
    right = cp.asnumpy(_gpu_binary(cp, "add", a, b))
    wrong = cp.asnumpy(a * b)  # deliberately the wrong operator
    assert not np.allclose(right, wrong, rtol=1e-6, atol=1e-6)


def test_pseudo_unary_returns_none_for_cpu_fallback():
    """prewarp / gate_med pseudo-unaries are NOT GPU-eligible -> None (CPU path)."""
    import dataclasses
    X = _frame()
    rec = _recipe("identity", "identity", "add")
    # gate_med pseudo-unary on side a -> GPU-ineligible (closed-form CPU path).
    rec = dataclasses.replace(rec, unary_names=("gate_med", "identity"))
    assert apply_unary_binary_gpu(rec, X) is None
