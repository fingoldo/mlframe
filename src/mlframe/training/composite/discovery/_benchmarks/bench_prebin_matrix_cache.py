"""Micro-bench: prebinned feature-matrix cache across two discovery runs on the SAME data.

``_prebin_feature_columns`` (screening.py) turns the small screen-sized float feature matrix into
an int16/int32 bin-code matrix via per-column ``np.quantile`` + ``np.searchsorted`` -- O(n*F*log n).
The codes are a DETERMINISTIC function of (matrix bytes, nbins) ONLY: nothing else in the discovery
config touches them. So a second discovery on the SAME data + sample + nbins but a DIFFERENT config
(e.g. a re-run that flips ``mi_estimator`` back to ``"bin"``, or varies transforms / rerank knobs)
recomputes the IDENTICAL codes. ``PrebinCache`` keys those codes by a content hash
(``prebin_matrix_signature``) so the second run skips the binning entirely.

This bench models exactly that: discovery run A on a screen matrix (config with ``mi_estimator=bin``)
populates the cache, then discovery run B on the SAME matrix + nbins (a different config, here
modelled as a second ``mi_estimator=bin`` pass) reuses the codes. It reports the second-run binning
wall time WITHOUT the cache (full recompute) vs WITH the cache (signature hash + dict hit), and
verifies the codes are BIT-IDENTICAL either way.

MEASURED (this Windows host "OLL", py3.14, n=80k F=40 nbins=32, int16 codes 6.1 MB):
  run-B binning, no cache (recompute): ~293.6 ms
  run-B binning, with cache (hit):     ~19.4 ms
  speedup:                             ~15.2 x   (binning time SAVED on re-discovery)
  bit-identical codes:                 True

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_prebin_matrix_cache
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from mlframe.training.composite.cache import PrebinCache, prebin_matrix_signature
from mlframe.training.composite.discovery.screening import _prebin_feature_columns

_N = 80_000
_F = 40
_NBINS = 32
_REPEATS = 7


def _make_screen_matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_N, _F)).astype(np.float32)


def _time(fn, repeats: int = _REPEATS) -> float:
    # Warm-up once, then take the median of ``repeats`` to dampen GC / scheduler jitter.
    fn()
    samples = []
    for _ in range(repeats):
        t0 = timer()
        fn()
        samples.append(timer() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def main() -> None:
    x = _make_screen_matrix()

    # Run A: first discovery pass populates the cache (this is the cost the FIRST discovery already
    # paid -- we are measuring what the SECOND discovery saves, not run A).
    cache = PrebinCache()
    sig = prebin_matrix_signature(x, _NBINS)
    codes_a = _prebin_feature_columns(x, nbins=_NBINS)
    cache.put(sig, codes_a)

    # Run B WITHOUT cache: the status-quo -- a second discovery recomputes the identical binning.
    t_recompute = _time(lambda: _prebin_feature_columns(x, nbins=_NBINS))

    # Run B WITH cache: signature hash + dict hit returns the stored codes.
    def _cached():
        s = prebin_matrix_signature(x, _NBINS)
        return cache.get(s)

    t_cached = _time(_cached)

    codes_b = cache.get(sig)
    bit_identical = bool(np.array_equal(codes_b, _prebin_feature_columns(x, nbins=_NBINS)))
    speedup = t_recompute / max(t_cached, 1e-12)

    result = {
        "bench": "prebin_matrix_cache",
        "host": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "?")),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n": _N,
        "n_features": _F,
        "nbins": _NBINS,
        "code_dtype": str(codes_a.dtype),
        "code_matrix_mb": round(codes_a.nbytes / 1024 / 1024, 2),
        "run_b_recompute_ms": round(t_recompute * 1e3, 3),
        "run_b_cached_ms": round(t_cached * 1e3, 3),
        "second_run_binning_speedup_x": round(speedup, 2),
        "bit_identical": bit_identical,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prebin_matrix_cache.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"\nwrote {out_path}", file=sys.stderr)

    if not bit_identical:
        raise SystemExit("FAIL: cached codes differ from recompute (must be bit-identical)")


if __name__ == "__main__":
    main()
