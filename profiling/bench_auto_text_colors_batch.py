"""A/B bench for ``auto_text_colors_batch`` vs per-cell ``auto_text_color`` (2026-07-31).

Confirms bit-identity and measures the wall-time win from sampling the matplotlib colormap once for
a whole heatmap grid instead of once per cell -- surfaced by a reporting/charts cProfile
(``profile_one_combo.py --combo c0016_cbe1b080 --rows 2000000 --save-charts``:
``colors.py:auto_text_color`` 0.559s tottime / 396 calls).

Usage:
    python profiling/bench_auto_text_colors_batch.py
"""

from __future__ import annotations

import time

import numpy as np


def main():
    from mlframe.reporting.colors import auto_text_color, auto_text_colors_batch

    rng = np.random.default_rng(0)
    shape = (60, 60)  # a large-but-realistic heatmap grid (3600 cells)
    mat = rng.uniform(-2, 2, size=shape)
    colormap = "viridis"
    vmin, vmax = -1.5, 1.5
    filled = np.where(np.isfinite(mat), mat, vmin)

    t0 = time.perf_counter()
    scalar = np.array(
        [[auto_text_color(filled[i, j], colormap, vmin=vmin, vmax=vmax) for j in range(shape[1])] for i in range(shape[0])]
    )
    t_scalar = time.perf_counter() - t0

    t0 = time.perf_counter()
    batch = auto_text_colors_batch(filled, colormap, vmin=vmin, vmax=vmax)
    t_batch = time.perf_counter() - t0

    identical = np.array_equal(scalar, batch)
    print(f"shape={shape} cells={shape[0] * shape[1]}")
    print(f"per-cell:  {t_scalar:.4f}s")
    print(f"batched:   {t_batch:.4f}s")
    print(f"speedup: {t_scalar / t_batch:.2f}x")
    print(f"bit-identical: {identical}")
    assert identical, "auto_text_colors_batch must be bit-identical to per-cell auto_text_color"


if __name__ == "__main__":
    main()
