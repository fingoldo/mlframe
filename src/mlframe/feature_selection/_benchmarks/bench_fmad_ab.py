"""A/B of ``--fmad=false`` on the production kernels that ship with it: kernel time with the flag (as shipped) and with the compiler free to contract multiply-adds, and the largest value difference.

Usage (elevated shell with the CUPTI directory on ``PATH`` for the nvprof column; the wall-clock column needs no profiler)::

    python -m mlframe.feature_selection._benchmarks.bench_fmad_ab

Each case builds its inputs, runs the entry point once warm and then ``REPS`` times with ``cupy`` synchronisation (the result stays on the device: no download in the timing), once per setting (interleaved), and prints the median milliseconds of both and the
maximum absolute difference of the outputs. A kernel whose "free" column is not clearly faster keeps the flag: the equality with the cupy chain it replaces costs nothing. Kernels of the list
below that are not measured yet are named in ``PENDING`` so the next session can add a case.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

REPS = 25

#: production kernels compiled with ``--fmad=false`` that have no case below yet (module, kernel)
PENDING = (
    ("_honest_null_device", "root_split_stats / root_split_stats_v2 / root_split_stats_k2"),
    ("_honest_null_k3", "root_split_stats_k3"),
    ("_honest_null_k4", "root_split_stats_k4 / k4f / k5"),
    ("_offset_product_gpu", "scan_offset_products"),
)


def _median_ms(cp: Any, fn: Callable[[], Any]) -> float:
    """Median wall milliseconds of ``fn`` over ``REPS`` synchronised runs after one warm call."""
    fn()
    cp.cuda.Device().synchronize()
    t = []
    for _ in range(REPS):
        a = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        t.append(time.perf_counter() - a)
    return float(np.median(t)) * 1e3


def _case_bspline(cp: Any) -> "tuple[Callable[[bool], Callable[[], Any]], str]":
    """``bspline_block`` over the 36 spline columns of a four-source pool, 100000 rows, float32."""
    from mlframe.feature_selection.filters._kernel_types import render
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _extra_basis_resident as eb

    X, eng, meta, _y, _nb = eb._sweep_inputs({"n_samples": 100_000})
    names = [c for c in eng.columns if meta[c]["basis"] == "spline"]
    xs = [cp.asarray(X[meta[c]["src"]].to_numpy().astype(np.float32)) for c in names]
    metas = [meta[c] for c in names]

    def make(free: bool) -> Callable[[], Any]:
        eb._BSPLINE_KERNELS[np.dtype(np.float32).str] = cp.RawKernel(render(eb._BSPLINE_SRC, real=np.float32), "bspline_block", options=() if free else ("--fmad=false",))
        return lambda: eb._bspline_block_gpu(cp, xs, metas)

    return make, "bspline_block (36 columns x 100000 rows, float32)"


def _case_bf_apply(cp: Any) -> "tuple[Callable[[bool], Callable[[], Any]], str]":
    """``bf_apply`` logabs (the function with a multiply-add) on two (100000, 100) float32 matrices."""
    from mlframe.feature_selection.filters import _cupy_polynom_optimizer as cpo
    from mlframe.feature_selection.filters._kernel_types import render

    rng = np.random.default_rng(0)
    A = cp.asarray(rng.standard_normal((100_000, 100)).astype(np.float32))
    B = cp.asarray(rng.standard_normal((100_000, 100)).astype(np.float32))
    idx = cpo._BF_ORDER.index("logabs")

    def make(free: bool) -> Callable[[], Any]:
        src = render(cpo._BF_SRC, real=np.float32)
        for key, value in cpo._BF_FUNCS["float"].items():
            src = src.replace("{" + key + "}", value)
        cpo._BF_KERNELS["float"] = cp.RawKernel(src, "bf_apply", options=() if free else ("--fmad=false",))
        return lambda: cpo._bf_apply_checked(cp, idx, A, B)[0]

    return make, "bf_apply logabs (100000 x 100, float32)"


def _case_basis_fused(cp: Any) -> "tuple[Callable[[bool], Callable[[], Any]], str]":
    """``build_basis`` (fused hermite design, 7 columns) of a 1000000-row float32 column."""
    from mlframe.feature_selection.filters.hermite_fe import _basis_fused_gpu as bf

    x = cp.asarray(np.random.default_rng(0).standard_normal(1_000_000).astype(np.float32))

    def make(free: bool) -> Callable[[], Any]:
        options = ("-std=c++14",) if free else ("-std=c++14", "--fmad=false")
        bf._KERNELS_BY_REAL = {r: bf.LazyRawModule(bf._SRC.replace("{REAL}", r), options) for r in ("double", "float")}
        return lambda: bf.build_basis_fused(cp, "hermite", x, 7)

    return make, "build_basis_fused hermite (1000000 rows x 7, float32)"


CASES = (_case_bspline, _case_bf_apply, _case_basis_fused)


def main() -> None:
    """Run every case: interleaved medians of the shipped (flag) and free (contractable) builds and the largest output difference."""
    import cupy as cp

    for case in CASES:
        make, label = case(cp)
        shipped, free = make(False), None
        out_shipped = cp.asnumpy(shipped())
        free = make(True)
        out_free = cp.asnumpy(free())
        diff = float(np.nanmax(np.abs(np.asarray(out_shipped, dtype=np.float64) - np.asarray(out_free, dtype=np.float64))))  # Intentional float64: comparison statistic
        t_free = _median_ms(cp, free)
        shipped = make(False)
        t_ship = _median_ms(cp, shipped)
        print(f"{label}: --fmad=false {t_ship:.3f} ms, contractable {t_free:.3f} ms, max abs diff {diff:.3g}")
    for module, kernels in PENDING:
        print(f"not measured yet: {module}: {kernels}")


if __name__ == "__main__":
    main()
