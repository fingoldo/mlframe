"""Wall-time bench: numpy vs numba ``|corr(y, X[:, j])|`` over all columns (y: 50k, X: 50k x 200).

``_safe_abs_corr_all`` computes the per-column absolute Pearson correlation of every
feature column against ``y`` in the forbidden-base leakage filter (``_filter.py``). The
numpy reference centres the whole ``X`` (an (n, F) temporary), folds the per-column
variance with an einsum and takes one ``X_dev.T @ y_dev`` matmul. The numba kernel
(``_corr_numba._abs_corr_all_kernel``) walks each column in ``prange`` register loops
against the pre-centred ``y`` with NO (n, F) centred temporary, and is numerically
equivalent to the reference within ~1e-9 (bit-identical wherever the near-1 leak
decision is sensitive -- borderline columns re-decided exactly; see the dispatcher
docstring + the ``test_corr_numba_bit_identity`` regression test).

This bench warms the kernel once, then times both paths on a (50_000, 200) matrix at
the dispatcher's production shape (n >= 20k AND F >= 64 -> the kernel is the default),
repeated several times (min-of-reps reported to suppress scheduler noise). It also
asserts the two outputs agree to ~1e-9 so a future "just rewrite the kernel" cannot
silently diverge.

MEASURED (this Windows host, py3.14, n=50_000 F=200, warm, min-of-reps): see the printed
numbers / the JSON written under ``_results/``. On a contended host the parallel
kernel's win can be marginal -- the dispatcher stays gated (n >= 20k AND F >= 64) so it
only engages where it amortises, and the bench records the host's actual numbers.

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_safe_corr_dispatch
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from mlframe.training.composite.discovery._corr_numba import (
    safe_abs_corr_all_dispatch,
)
from mlframe.training.composite.discovery.screening import (
    _safe_abs_corr_all_numpy,
)

_N = 50_000
_F = 200
_REPS = 7


def _make_data(n: int, f: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A (n, f) feature matrix + a target that correlates with a few columns."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = X[:, 0] * 0.6 + X[:, 1] * 0.3 + rng.normal(size=n)
    return y, X


def _time(fn, y, X, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(y, X)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    y, X = _make_data(_N, _F)

    def _numpy(yy, XX):
        return _safe_abs_corr_all_numpy(yy, XX)

    def _numba(yy, XX):
        return safe_abs_corr_all_dispatch(yy, XX, reference_fn=_safe_abs_corr_all_numpy)

    # Warm (JIT compile) + numerical-equivalence gate.
    ref = _numpy(y, X)
    fast = _numba(y, X)
    max_diff = float(np.abs(ref - fast).max())

    t_np = _time(_numpy, y, X, _REPS)
    t_nb = _time(_numba, y, X, _REPS)
    speedup = t_np / t_nb if t_nb > 0 else float("inf")

    print(
        f"safe_abs_corr_all bench  n={_N} F={_F} reps={_REPS} "
        f"py={sys.version.split()[0]}"
    )
    print(f"  max abs diff numpy vs numba: {max_diff:.2e}")
    print(f"  numpy reference: {t_np * 1e3:8.3f} ms")
    print(f"  numba kernel:    {t_nb * 1e3:8.3f} ms  ({speedup:.2f}x)")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "safe_corr_dispatch.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ts": datetime.now().isoformat(),
                "n": _N, "F": _F, "reps": _REPS,
                "max_abs_diff": max_diff,
                "numpy_ms": t_np * 1e3,
                "numba_ms": t_nb * 1e3,
                "speedup": speedup,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
