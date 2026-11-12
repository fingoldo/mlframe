"""A/B of the public ``binned_mi_from_codes_gpu`` call: tiled path against the row-major route it replaced.

Usage: ``PYTHONPATH=src python bench_mi_from_codes_tiled.py <n> <K>`` (PYTHONPATH makes the worktree package the one imported). Times the call with the tiled path (the default) and with
the row-major kernel (its production launch rule) launched directly, best of 10 with the two calls interleaved (alternating order) to cancel thermal drift, on random 10-level X codes and 4-level y.
"""

import sys
import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _fe_batched_mi as m


def _timed_ms(fn):
    t = time.perf_counter()
    fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t) * 1e3


def main(n: int, K: int, Kx: int = 10, Ky: int = 4) -> None:
    rng = np.random.default_rng(0)
    codes = cp.asarray(rng.integers(0, Kx, size=(n, K), dtype=np.int64))
    y = cp.asarray(rng.integers(0, Ky, size=n, dtype=np.int64))

    def public_call():
        return m.binned_mi_from_codes_gpu(codes, y, ky=Ky, codes_trusted=True, as_device=True)

    tile_width = m._mi_from_codes_tile_width

    def row_major_route():
        # A tile width of 0 skips the tiled launch, so the call takes the row-major kernel exactly as production did before the tiling.
        m._mi_from_codes_tile_width = lambda cells: 0
        try:
            return public_call()
        finally:
            m._mi_from_codes_tile_width = tile_width

    diff = float(cp.abs(public_call() - row_major_route()).max())
    public_call()
    row_major_route()
    best = {"tiled": float("inf"), "row_major": float("inf")}
    for rep in range(10):
        order = (("tiled", public_call), ("row_major", row_major_route)) if rep % 2 == 0 else (("row_major", row_major_route), ("tiled", public_call))
        for name, fn in order:
            best[name] = min(best[name], _timed_ms(fn))
    print(f"n={n} K={K} tiled_ms={best['tiled']:.2f} row_major_ms={best['row_major']:.2f} maxdiff={diff:.3g}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
