"""Bench + parity: FE-step CMI-gate candidate marginal MI, per-candidate device path vs one batched device workload.

Per-candidate (current): for each engineered candidate column, ``_quantile_bin_gpu_resident`` then ``_cmi_from_binned(codes, y, None)``.
Batched: stack the K columns, ``batched_quantile_bin_gpu`` once, ``batched_cmi_gpu(codes, y, None, kx, ky)`` once.
Reports max |delta marginal MI|, whether the codes are identical, and warm best-of-N synchronized wall times per (n, K).

Run: PYTHONPATH=src python -m mlframe.feature_selection.filters._benchmarks.bench_step_score_cmi_cands
"""

from __future__ import annotations

import time

import numpy as np

NBINS = 10


def _sync():
    """Block until queued device work finishes, so wall times include it."""
    import cupy as cp

    cp.cuda.Stream.null.synchronize()


def _per_candidate(block, y):
    """Current loop: bin and score each column separately on the device."""
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin_gpu_resident

    out = np.empty(block.shape[1])
    for j in range(block.shape[1]):
        codes = _quantile_bin_gpu_resident(np.ascontiguousarray(block[:, j]), NBINS)
        out[j] = float(_cmi_from_binned(codes, y, None, kx=NBINS))
    return out


def _batched(block, y):
    """One device binning of the whole block, one device marginal-MI workload."""
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu, batched_quantile_bin_gpu

    codes = batched_quantile_bin_gpu(block, NBINS)
    return np.asarray(batched_cmi_gpu(codes, y, None, kx=NBINS, ky=int(y.max()) + 1), dtype=np.float64)


def _best(fn, reps):
    """Best synchronized wall time over ``reps`` calls."""
    best = float("inf")
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Print parity and timing per (n, K)."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_batched_mi import batched_quantile_bin_gpu
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _quantile_bin_gpu_resident

    rng = np.random.default_rng(0)
    print(f"{'n':>8} {'K':>5} {'max|dMI|':>10} {'codes_eq':>9} {'per_cand_ms':>12} {'batched_ms':>11} {'speedup':>8}")
    for n in (50_000, 250_000):
        y = rng.integers(0, 3, size=n).astype(np.int64)
        for K in (8, 32, 128, 512):
            if n * K > 64_000_000:
                continue
            latent = rng.normal(size=(n, 1))
            block = latent * rng.uniform(0.1, 2.0, size=(1, K)) + rng.normal(size=(n, K))
            block[:, ::4] = np.round(block[:, ::4], 1)  # tied columns too
            block[:, 1::7] = np.exp(block[:, 1::7] * 0.1) + y[:, None] * 0.3
            a = _per_candidate(block, y)
            b = _batched(block, y)
            codes_b = cp.asnumpy(batched_quantile_bin_gpu(block, NBINS))
            codes_eq = all(np.array_equal(cp.asnumpy(_quantile_bin_gpu_resident(np.ascontiguousarray(block[:, j]), NBINS)), codes_b[:, j]) for j in range(K))
            reps = 5 if n * K <= 8_000_000 else 3
            t_a = _best(lambda: _per_candidate(block, y), reps)
            t_b = _best(lambda: _batched(block, y), reps)
            print(f"{n:>8} {K:>5} {np.max(np.abs(a - b)):>10.2e} {codes_eq!s:>9} {t_a * 1e3:>12.1f} {t_b * 1e3:>11.1f} {t_a / t_b:>8.2f}")


if __name__ == "__main__":
    main()
