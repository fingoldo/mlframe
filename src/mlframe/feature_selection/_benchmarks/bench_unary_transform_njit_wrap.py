"""A/B: bare numpy ufunc vs an njit-wrapped ``lambda x: np.<ufunc>(x)`` for every unary-transform-registry
entry that was a bare ufunc (``create_unary_transformations`` in ``feature_engineering.py``).

``njit_functions_dict`` (in ``_internals.py``) can only wrap a plain Python function -- decorating a bare
ufunc directly (e.g. ``njit(np.cos)``) raises ``TypeError: The decorated object is not a function`` at
decoration time, silently swallowed by the registry's best-effort ``except: pass``, so every bare-ufunc
registry entry was NEVER actually jitted. Wrapping in a 1-line lambda lets numba compile it (numba lowers
most numpy ufunc calls inside a jitted body as intrinsics) -- but a jitted wrapper is not automatically
faster than numpy's already-vectorized C loop, so each entry needs an actual A/B before converting.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_unary_transform_njit_wrap``

Verdict (2026-07-16, median-of-50, warm dispatcher, n in {1e3, 3e4, 1e5}): CONVERTED (>=1.02x, consistent
across sizes) -- ``sign`` (numpy float64 sign has an unexpectedly slow path, ~18x), ``tanh`` (~1.8-2.7x),
``neg``/``rint``/``cbrt``/``arccos``/``arctan``/``cosh``/``arcsinh`` (~1.02-1.13x). REJECTED (flat-to-
regressing, 0.6-0.99x) -- ``abs``/``sin``/``exp``/``cos``/``tan``/``sinh``/``arcsin``/``arccosh``/
``arctanh``: numpy's C loop already wins; the njit wrapper only adds a dispatch layer. ``erf``/``gammaln``
(scipy.special) fail njit compilation entirely (no numba-scipy extension) and cannot be converted at all.
All conversions verified bit-identical or ~1 ULP FP-reorder (``cbrt``/``tanh``) on a 200k-row fuzz.
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

_CANDIDATES = {
    "neg": np.negative,
    "abs": np.abs,
    "sin": np.sin,
    "sign": np.sign,
    "rint": np.rint,
    "cbrt": np.cbrt,
    "exp": np.exp,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcsinh": np.arcsinh,
    "arccosh": np.arccosh,
    "arctanh": np.arctanh,
}

# Domain restriction per transform so every candidate gets a valid (non-NaN-producing) input.
_DOMAIN = {
    "arcsin": (-0.99, 0.99),
    "arccos": (-0.99, 0.99),
    "arctanh": (-0.99, 0.99),
    "arccosh": (1.01, 5.0),
}


def _bench_one(name: str, npfunc, n: int, reps: int = 50) -> tuple[float, float, bool]:
    """Time npfunc vs its njit-wrapped equivalent on n random rows; return (t_np, t_jit, bit_identical)."""
    rng = np.random.default_rng(0)
    lo, hi = _DOMAIN.get(name, (-2.0, 2.0))
    x = rng.uniform(lo, hi, n).astype(np.float64)
    jfunc = njit(lambda x, _f=npfunc: _f(x))
    jfunc(x)
    npfunc(x)  # warm both
    t_np = float(np.median([_time_call(npfunc, x) for _ in range(reps)]))
    t_jit = float(np.median([_time_call(jfunc, x) for _ in range(reps)]))
    a, b = npfunc(x), jfunc(x)
    bit_ok = bool(np.array_equal(a, b, equal_nan=True)) or bool(np.nanmax(np.abs(a - b)) < 1e-9)
    return t_np, t_jit, bit_ok


def _time_call(fn, x) -> float:
    """Return the wall-clock seconds to call fn(x) once."""
    t0 = time.perf_counter()
    fn(x)
    return time.perf_counter() - t0


def main() -> None:
    """Run the njit-wrap benchmark across candidate unary transforms and sample sizes, printing results."""
    for n in (1_000, 30_000, 100_000):
        print(f"\n=== n={n} ===")
        for name, npfunc in _CANDIDATES.items():
            t_np, t_jit, bit_ok = _bench_one(name, npfunc, n)
            speedup = t_np / t_jit if t_jit > 0 else float("nan")
            print(f"{name:10s} np={t_np * 1e6:8.1f}us  njit={t_jit * 1e6:8.1f}us  speedup={speedup:5.2f}x  identity_ok={bit_ok}")


if __name__ == "__main__":
    main()
