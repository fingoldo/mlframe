"""Measures validate_pair_codes against the whole batch_pair_mi_cuda call (same process, interleaved best-of-N, warm) at n = 200k, 1M, 2M; prints each cost and the validation share of the call wall."""

import time

import numpy as np

from mlframe.feature_selection.filters import _batch_pair_mi_cuda_kernels as K
from mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused import validate_pair_codes


def _best(fn, reps: int) -> float:
    """Best-of-``reps`` wall time of ``fn`` in milliseconds."""
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def main() -> None:
    """Print validation cost, kernel-call wall and the share per size."""
    rng = np.random.default_rng(0)
    nb, nf, npairs = 8, 32, 64
    for n in (200_000, 1_000_000, 2_000_000):
        f = rng.integers(0, nb, (n, nf)).astype(np.int32)
        y = rng.integers(0, 4, n).astype(np.int32)
        fy = np.bincount(y, minlength=4) / n
        nbins = np.full(nf, nb, dtype=np.int32)
        pa = rng.integers(0, nf, npairs).astype(np.int64)
        pb = ((pa + rng.integers(1, nf, npairs)) % nf).astype(np.int64)

        def call():
            return K.batch_pair_mi_cuda(f, pa, pb, nbins, y, fy)

        def val():
            return validate_pair_codes(f, pa, pb, nbins, y, 4)

        call()
        val()
        c = min(_best(call, 3) for _ in range(3))
        v = min(_best(val, 5) for _ in range(3))
        print(f"n={n} call_ms={c:.1f} validate_ms={v:.1f} share={100 * v / c:.2f}%", flush=True)


if __name__ == "__main__":
    main()
