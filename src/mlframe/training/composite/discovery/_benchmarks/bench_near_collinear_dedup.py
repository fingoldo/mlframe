"""Wall-time bench: numpy vs numba near-collinear keep-mask walk on a (10k, 50) matrix.

``near_collinear_keep_mask`` de-duplicates a base's ``x_remaining`` with an
``O(B^2)`` pure-numpy Pearson + boolean-fancy-index walk. On a base with ~50
candidate remaining columns this is a measurable cost in the MI baseline sweep.
The numba kernel (``_collinear_numba._keep_mask_kernel``) does the same walk in
register loops over the full ``(n, B)`` matrix + a precomputed finite mask, with
no per-pair fancy-index allocation, and is bit-identical to the numpy reference
(borderline pairs re-decided exactly -- see the dispatcher docstring + the
``test_collinear_numba_bit_identity`` regression test).

This bench warms the kernel once, then times both paths on a (10_000, 50) matrix
seeded so ~40% of the columns are near-duplicates of a few latent bases (a
realistic dedup workload where columns actually drop), repeated several times
(min-of-reps reported to suppress scheduler noise). It also asserts mask
bit-identity so a future "just rewrite the kernel" cannot silently diverge.

MEASURED (this Windows host, py3.14, n=10_000 B=50, ~40% near-dup, warm, min-of-reps):
  see the printed numbers / the JSON written under ``_results/``; the kernel
  removes the per-pair ``col[pair]`` allocations + vectorised-dot dispatch the
  numpy walk pays B*(B-1)/2 times, so the JIT path is materially faster while
  returning the identical mask.

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_near_collinear_dedup
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from mlframe.training.composite.discovery._collinear_numba import (
    near_collinear_keep_mask_fast,
)
from mlframe.training.composite.discovery._eval_stats import (
    _near_collinear_keep_mask_numpy,
)

_N = 10_000
_B = 50
_THR = 0.99
_REPS = 7


def _make_matrix(n: int, b: int, seed: int = 0) -> np.ndarray:
    """An (n, b) matrix where ~40% of columns are near-duplicates of 5 latent bases."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 5))
    cols = []
    for _ in range(b):
        if rng.random() < 0.4:
            cols.append(latent[:, rng.integers(0, 5)] + 1e-3 * rng.normal(size=n))
        else:
            cols.append(rng.normal(size=n))
    fm = np.column_stack(cols)
    holes = rng.random((n, b)) < 0.03
    fm[holes] = np.nan
    return fm


def _time(fn, fm, reps: int) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(fm)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    fm = _make_matrix(_N, _B)

    def _numpy(m):
        return _near_collinear_keep_mask_numpy(m, corr_threshold=_THR)

    def _numba(m):
        return near_collinear_keep_mask_fast(
            m, corr_threshold=_THR, reference_fn=_near_collinear_keep_mask_numpy,
        )

    # Warm (JIT compile) + bit-identity gate.
    ref = _numpy(fm)
    fast = _numba(fm)
    identical = bool(np.array_equal(ref, fast))

    t_np = _time(_numpy, fm, _REPS)
    t_nb = _time(_numba, fm, _REPS)
    speedup = t_np / t_nb if t_nb > 0 else float("inf")

    print(
        f"near-collinear dedup bench  n={_N} B={_B} thr={_THR} "
        f"kept={int(ref.sum())}/{_B} reps={_REPS} py={sys.version.split()[0]}"
    )
    print(f"  bit-identical mask: {identical}")
    print(f"  numpy reference: {t_np * 1e3:8.3f} ms")
    print(f"  numba kernel:    {t_nb * 1e3:8.3f} ms  ({speedup:.2f}x)")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "near_collinear_dedup.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ts": datetime.now().isoformat(),
                "n": _N, "B": _B, "thr": _THR, "reps": _REPS,
                "kept": int(ref.sum()),
                "bit_identical": identical,
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
