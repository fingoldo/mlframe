"""iter108 bench: serial vs prange _cyclical_sincos at 10M (date-cyclical FE, full-n).

Run: CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src python -m mlframe.feature_engineering._benchmarks.bench_cyclical_sincos_prange_iter108

Verdict (RESOLVED): par faster 15/15 paired @10M, 12.18x (serial min=324.3ms -> par min=26.6ms), byte-identical output (md5 match e2e via add_cyclical_date_features).
Each output element is independent (no reduction) so the prange twin is bit-identical to the serial loop. Gated at _CYCLICAL_PAR_THRESHOLD=1M (env-overridable):
below it the ~17ms prange thread-launch floor dwarfs the per-element sin/cos work; above it the split scales near-linearly. Small-N crossover sweep: par loses
below ~1M (fixed launch floor), wins decisively from 1M up.
"""
import sys
sys.modules["cupy"] = None
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")
import time
import numpy as np
import scipy.stats  # noqa: F401  (cold-import segfault guard on py3.14)
import numba  # noqa: F401
from mlframe.feature_engineering.basic import _cyclical_sincos_serial, _cyclical_sincos_parallel


def main():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(10_000_000) * 180.0
    scale = 2 * np.pi / 365.0
    for _ in range(8):
        _cyclical_sincos_serial(base[:2000], scale)
        _cyclical_sincos_parallel(base, scale)
    ser, par, wins = [], [], 0
    for _ in range(15):
        s = time.perf_counter(); _cyclical_sincos_serial(base, scale); ser.append(time.perf_counter() - s)
        s = time.perf_counter(); _cyclical_sincos_parallel(base, scale); par.append(time.perf_counter() - s)
        if par[-1] < ser[-1]:
            wins += 1
    print(f"10M paired: par faster {wins}/15  serial min={min(ser) * 1000:.1f}ms  par min={min(par) * 1000:.1f}ms  speedup(min)={min(ser) / min(par):.2f}x")


if __name__ == "__main__":
    main()
