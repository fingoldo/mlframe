"""Sweep bench for GPU joint-histogram kernels across (block_size, joint_size,
n_samples). Two kernels are measured side-by-side:

* ``compute_joint_hist_batched_cuda`` (global-atomic, the fallback)
* ``compute_joint_hist_batched_shared_cuda`` (shared-mem atomic, the default
  at ``joint_size <= 4096``)

Output: CSV of per-config wall + a 2D heatmap PNG per kernel showing
log-walls indexed by ``block_size`` x ``n_samples``, for each ``joint_size``
slice. The heatmap files land under ``_results/`` next to the script.

Run::

    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \\
        -m mlframe.feature_selection._benchmarks.bench_gpu_kernel_sweep

The defaults sweep:
    block_size  in {64, 128, 256, 512, 1024}
    n_samples   in {100_000, 500_000, 1_000_000, 5_000_000}
    joint_size  in {6, 25, 100, 400} (= nbins_x * nbins_y for typical MRMR
                axes: 2x3, 5x5, 10x10, 20x20)
    batch_size  = 64 (matches mi_direct_gpu_batched default)
    npermutations = 256 (4 batches per config; cuts dispatch noise)

Total points: 5 * 4 * 4 = 80 configs * 2 kernels = 160 measurements.
Per-config wall ~50-500 ms; total runtime ~1-3 min on GTX 1050 Ti.
"""
from __future__ import annotations

import itertools
import os
import time
from typing import Iterable

import numpy as np


def _measure_one(
    kernel,
    grid_x: int,
    block_size: int,
    args: tuple,
    n_iters: int = 5,
    *,
    shared_mem_bytes: int = 0,
) -> float:
    """Run kernel ``n_iters`` times, return BEST (minimum) wall in ms.

    Always warms once (kernel JIT + first-call cache miss), then times each
    of the remaining ``n_iters`` iterations end-to-end including the
    device-side sync. We take ``min`` rather than mean / median because
    perf benches want the steady-state lower bound, not the noise floor --
    OS scheduler hiccups, driver-thread interference, and GC pauses push
    means upward but never push the min below the actual kernel time.
    """
    import cupy as cp

    # Warm-up call.
    if shared_mem_bytes > 0:
        kernel((grid_x,), (block_size,), args, shared_mem=shared_mem_bytes)
    else:
        kernel((grid_x,), (block_size,), args)
    cp.cuda.runtime.deviceSynchronize()

    best = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        if shared_mem_bytes > 0:
            kernel((grid_x,), (block_size,), args, shared_mem=shared_mem_bytes)
        else:
            kernel((grid_x,), (block_size,), args)
        cp.cuda.runtime.deviceSynchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        best = min(best, dt)
    return best


def _run_config(
    n_samples: int,
    block_size: int,
    nbins_x: int,
    nbins_y: int,
    *,
    seed: int,
    n_iters: int = 5,
) -> dict:
    import cupy as cp
    from mlframe.feature_selection.filters.gpu import (
        _ensure_kernels_inited,
        compute_joint_hist_batched_cuda,
        compute_joint_hist_batched_shared_cuda,
    )

    _ensure_kernels_inited()

    rng = np.random.default_rng(seed)
    classes_x = rng.integers(0, nbins_x, size=n_samples).astype(np.int32)
    classes_y_single = rng.integers(0, nbins_y, size=n_samples).astype(np.int32)
    # Single batch (b=1) to factor out batch-level effects; the per-batch
    # wall cost dominates the per-call dispatch overhead here.
    b = 1
    perms_y = classes_y_single.reshape(1, -1).copy()  # shape (1, n_samples)

    d_classes_x = cp.asarray(classes_x)
    d_perms_y = cp.asarray(perms_y)
    joint_counts_batch = cp.zeros((b, nbins_x * nbins_y), dtype=cp.int32)

    grid_x = (n_samples + block_size - 1) // block_size

    args = (
        d_classes_x, d_perms_y, joint_counts_batch,
        np.int32(n_samples), np.int32(nbins_x), np.int32(nbins_y),
    )

    joint_size = nbins_x * nbins_y
    # Global-atomic kernel
    joint_counts_batch[:] = 0
    t_global = _measure_one(
        compute_joint_hist_batched_cuda, grid_x, block_size, args,
        n_iters=n_iters,
    )

    # Shared-mem kernel
    joint_counts_batch[:] = 0
    t_shared = _measure_one(
        compute_joint_hist_batched_shared_cuda, grid_x, block_size, args,
        n_iters=n_iters, shared_mem_bytes=joint_size * 4,
    )

    return {
        "n_samples": n_samples,
        "block_size": block_size,
        "nbins_x": nbins_x,
        "nbins_y": nbins_y,
        "joint_size": joint_size,
        "grid_x": grid_x,
        "wall_ms_global": round(t_global, 3),
        "wall_ms_shared": round(t_shared, 3),
        "speedup_shared_vs_global": round(t_global / max(t_shared, 1e-6), 3),
    }


def _save_csv(rows, out_path: str) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV -> {out_path}")


def _save_heatmaps(rows, out_dir: str) -> None:
    """Per joint_size: one heatmap of speedup, one of each kernel's wall.

    Matplotlib is imported lazily so the CSV path still works on hosts
    without matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping heatmaps")
        return

    # Group rows by joint_size.
    by_js: dict[int, list[dict]] = {}
    for r in rows:
        by_js.setdefault(r["joint_size"], []).append(r)

    for js, rs in sorted(by_js.items()):
        # Axes: rows = n_samples (sorted asc), cols = block_size (sorted asc).
        ns_vals = sorted({r["n_samples"] for r in rs})
        bs_vals = sorted({r["block_size"] for r in rs})
        # Build matrices.
        speedup = np.full((len(ns_vals), len(bs_vals)), np.nan)
        wall_glob = np.full((len(ns_vals), len(bs_vals)), np.nan)
        wall_shar = np.full((len(ns_vals), len(bs_vals)), np.nan)
        for r in rs:
            i = ns_vals.index(r["n_samples"])
            j = bs_vals.index(r["block_size"])
            speedup[i, j] = r["speedup_shared_vs_global"]
            wall_glob[i, j] = r["wall_ms_global"]
            wall_shar[i, j] = r["wall_ms_shared"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for ax, mat, title, cmap in [
            (axes[0], wall_glob, f"global atomic wall (ms) joint_size={js}", "viridis"),
            (axes[1], wall_shar, f"shared atomic wall (ms) joint_size={js}", "viridis"),
            (axes[2], speedup, f"speedup shared/global joint_size={js}", "RdYlGn"),
        ]:
            im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="lower")
            ax.set_xticks(range(len(bs_vals)))
            ax.set_xticklabels(bs_vals)
            ax.set_yticks(range(len(ns_vals)))
            ax.set_yticklabels([f"{n:_}" for n in ns_vals])
            ax.set_xlabel("block_size")
            ax.set_ylabel("n_samples")
            ax.set_title(title)
            # Annotate cells with values.
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if not np.isnan(mat[i, j]):
                        # Auto-pick contrasting text colour per luminance.
                        normed = (mat[i, j] - np.nanmin(mat)) / max(np.nanmax(mat) - np.nanmin(mat), 1e-9)
                        text_color = "white" if 0.3 < normed < 0.7 else "black"
                        ax.text(
                            j, i, f"{mat[i, j]:.2f}",
                            ha="center", va="center", fontsize=8, color=text_color,
                        )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"bench_gpu_kernel_sweep_js{js:04d}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  PNG  -> {out_path}")


def main() -> None:
    print("=== GPU kernel sweep: compute_joint_hist_batched (global vs shared) ===")
    block_sizes: Iterable[int] = (64, 128, 256, 512, 1024)
    n_samples_grid: Iterable[int] = (100_000, 500_000, 1_000_000, 5_000_000)
    # (nbins_x, nbins_y) -> joint_size = nbins_x * nbins_y.
    nbins_grid: Iterable[tuple[int, int]] = (
        (2, 3),  # joint_size = 6,    typical binary classification with 2-bin x
        (5, 5),  # joint_size = 25,   default MRMR nbins=5
        (10, 10),  # joint_size = 100,  high-cardinality MRMR
        (20, 20),  # joint_size = 400,  very wide histograms
    )

    rows = []
    total = len(list(block_sizes)) * len(list(n_samples_grid)) * len(list(nbins_grid))
    done = 0
    for n_samples, block_size, (nbx, nby) in itertools.product(
        n_samples_grid, block_sizes, nbins_grid,
    ):
        done += 1
        try:
            row = _run_config(
                n_samples=n_samples, block_size=block_size,
                nbins_x=nbx, nbins_y=nby, seed=11, n_iters=5,
            )
        except Exception as e:
            print(f"  [{done}/{total}] FAIL n={n_samples:_} bs={block_size} " f"nbins=({nbx}, {nby}): {type(e).__name__}: {e}")
            continue
        rows.append(row)
        print(
            f"  [{done}/{total}] n={n_samples:_} bs={block_size:>4} "
            f"joint={row['joint_size']:>4}: global={row['wall_ms_global']:>7.2f}ms "
            f"shared={row['wall_ms_shared']:>7.2f}ms "
            f"speedup={row['speedup_shared_vs_global']:>5.2f}x"
        )

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    _save_csv(rows, os.path.join(out_dir, f"bench_gpu_kernel_sweep_{stamp}.csv"))
    _save_heatmaps(rows, out_dir)

    # Quick summary: best speedup per joint_size, worst speedup per joint_size,
    # block_size that wins most often.
    print("\n=== Summary ===")
    by_js: dict[int, list[float]] = {}
    bs_wins: dict[int, int] = {}
    for r in rows:
        by_js.setdefault(r["joint_size"], []).append(r["speedup_shared_vs_global"])
    print("  joint_size | min_speedup | median_speedup | max_speedup")
    for js, vals in sorted(by_js.items()):
        v = np.asarray(vals)
        print(f"  {js:>10} | {v.min():>11.2f} | {np.median(v):>14.2f} | {v.max():>11.2f}")

    # Per (n_samples, joint_size) find the winning block_size.
    by_nb_js: dict[tuple[int, int], dict[int, float]] = {}
    for r in rows:
        key = (r["n_samples"], r["joint_size"])
        by_nb_js.setdefault(key, {})[r["block_size"]] = r["wall_ms_shared"]
    bs_wins_acc: dict[int, int] = {}
    for key, bs_walls in by_nb_js.items():
        best_bs = min(bs_walls, key=bs_walls.get)
        bs_wins_acc[best_bs] = bs_wins_acc.get(best_bs, 0) + 1
    print("\n  block_size that wins most often (shared kernel):")
    for bs, count in sorted(bs_wins_acc.items(), key=lambda x: -x[1]):
        print(f"    block_size={bs:>4}: {count}/{len(by_nb_js)} wins")


if __name__ == "__main__":
    main()
