"""Micro-bench: njit vs numpy ``_mi_from_binned_pair`` (the discovery pair-MI kernel).

``_mi_from_binned_pair`` (screening.py) computes MI from two already-binned integer code arrays
via ``np.bincount`` + log. It is the hottest pair-MI kernel in composite discovery -- called
~9.8k times per run (per-feature MI AND inside the per-permutation null loop in ``_auto_base``,
so the cost multiplies with ``n_targets x auto_base_top_k x npermutations``). It now dispatches
to a ``numba.njit(cache=True)`` single-pass histogram+MI kernel.

This bench times the numpy reference (``_mi_from_binned_pair_numpy``) vs the njit dispatch at the
production sample size, warm (JIT pre-compiled), over many iterations, and verifies bit-identity
(maxdiff << 1e-12; only the final-sum FP reduction order differs).

MEASURED (Windows host "OLL", py3.14, n=20k nbins=32, int16 codes):
  numpy: ~93.3 us/call
  njit:  ~35.4 us/call
  speedup: ~2.64x   (4.0-4.2x on faster-numpy hosts per the original lead micro-bench)
  bit-identical: maxdiff ~5e-15 across nbins {16,32,182,200}

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_mi_from_binned_pair_njit
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from mlframe.training.composite.discovery.screening import (
    _mi_from_binned_pair,
    _mi_from_binned_pair_numpy,
)

_N = 20_000
_NBINS = 32
_REPEATS = 2000


def _codes(rng: np.random.Generator, nbins: int) -> np.ndarray:
    dtype = np.int16 if nbins < 182 else np.int32
    return rng.integers(0, nbins, _N).astype(dtype)


def _time(fn, repeats: int = _REPEATS) -> float:
    fn()  # warm (JIT compile / cache load)
    samples = []
    for _ in range(7):
        t0 = timer()
        for _ in range(repeats):
            fn()
        samples.append((timer() - t0) / repeats)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    rng = np.random.default_rng(1)
    x = _codes(rng, _NBINS)
    y = _codes(rng, _NBINS)

    np_t = _time(lambda: _mi_from_binned_pair_numpy(x, y, nbins=_NBINS))
    nj_t = _time(lambda: _mi_from_binned_pair(x, y, nbins=_NBINS))

    # Bit-identity sweep across the int16->int32 storage boundary at nbins>=182.
    maxdiff = 0.0
    for nbins in (16, 32, 182, 200):
        xb = _codes(rng, nbins)
        yb = _codes(rng, nbins)
        a = _mi_from_binned_pair_numpy(xb, yb, nbins=nbins)
        b = _mi_from_binned_pair(xb, yb, nbins=nbins)
        maxdiff = max(maxdiff, abs(a - b))

    result = {
        "bench": "mi_from_binned_pair_njit",
        "host": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "?")),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n": _N,
        "nbins": _NBINS,
        "numpy_us": round(np_t * 1e6, 3),
        "njit_us": round(nj_t * 1e6, 3),
        "speedup_x": round(np_t / max(nj_t, 1e-12), 3),
        "bit_identity_maxdiff": maxdiff,
        "bit_identical_under_1e12": bool(maxdiff < 1e-12),
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mi_from_binned_pair_njit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}", file=sys.stderr)

    if maxdiff >= 1e-12:
        raise SystemExit(f"FAIL: njit MI diverges from numpy by {maxdiff} (>= 1e-12)")


if __name__ == "__main__":
    main()
