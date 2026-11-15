"""A/B of ``--fmad=false`` on the production kernels that ship with it: kernel time with the flag (as shipped) and with the compiler free to contract multiply-adds, and the largest value difference.

Usage (elevated shell with the CUPTI directory on ``PATH`` for the nvprof column; the wall-clock column needs no profiler)::

    python -m mlframe.feature_selection._benchmarks.bench_fmad_ab

Each case builds its inputs, runs the entry point once warm and then ``REPS`` times with ``cupy`` synchronisation (the result stays on the device: no download in the timing), once per setting (interleaved), and prints the median milliseconds of both and the
maximum absolute difference of the outputs. A kernel whose "free" column is not clearly faster keeps the flag: the equality with the cupy chain it replaces costs nothing. Kernels that ship with the flag and have no case yet are named in ``PENDING`` so the next session can add one.
"""

from __future__ import annotations

import time
from typing import Any, Callable, cast

import numpy as np

REPS = 25

#: production kernels compiled with ``--fmad=false`` that have no case below yet (module, kernel)
PENDING: "tuple[tuple[str, str], ...]" = ()


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


def _case_honest_null(cp: Any) -> "tuple[Callable[[bool], Callable[[], Any]], str]":
    """The whole device root-split pipeline of the honest null (``root_split_stats`` v1 / v2 / k2 / k3 / k4 / k5 kernels as the dispatch picks them), n = 100000, 100 permutations, 2 classes.

    ``--fmad=false`` is stripped from every ``cp.RawKernel`` those modules compile (their kernel globals are reset so each setting recompiles)."""
    import tests.feature_selection.discretization.test_honest_null_device_equivalence as eq
    from mlframe.feature_selection.filters import _honest_null_device as hd
    from mlframe.feature_selection.filters import _honest_null_k3 as k3
    from mlframe.feature_selection.filters import _honest_null_k4 as k4

    n = 100_000
    args = eq._inputs(n, 2, 100, seed=2)
    no_nan = np.zeros(args[0].shape, dtype=bool)
    real_raw_kernel = cp.RawKernel
    modules = (hd, k3, k4)

    def make(free: bool) -> Callable[[], Any]:
        def raw_kernel(code, name, options=(), *a, **k):
            return real_raw_kernel(code, name, tuple(o for o in options if not (free and o == "--fmad=false")), *a, **k)

        cp.RawKernel = raw_kernel
        for mod in modules:  # every module-level kernel cache is dropped: the next launch compiles with the setting of this build
            for attr in [a for a in vars(mod) if a.startswith("_kernel")]:
                setattr(mod, attr, None)
        return lambda: cast(Any, hd).honest_null_columns_device(*args, no_nan)

    return make, "honest-null device pipeline (100000 rows, 100 permutations, 8 columns)"


def _case_offset_scan(cp: Any) -> "tuple[Callable[[bool], Callable[[], Any]], str]":
    """``scan_offset_products`` (the offset-product scan kernel) on the sweep inputs of the kernel-tuning entry: 100000 rows, a 12-column pool."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import _auto_tune_sweeps_c as sweeps
    from mlframe.feature_selection.filters import _offset_product_gpu as op

    U, tasks, yr, codes, ky, clips = sweeps._scan_inputs(100_000, 12)
    real_raw_kernel = cp.RawKernel

    def make(free: bool) -> Callable[[], Any]:
        def raw_kernel(code, name, options=(), *a, **k):
            return real_raw_kernel(code, name, tuple(o for o in options if not (free and o == "--fmad=false")), *a, **k)

        cp.RawKernel = raw_kernel
        op._KERNEL = None  # recompile with the setting of this build
        return lambda: op.scan_offset_products_gpu(U, tasks, yr, codes, ky, sweeps.N_BINS, clips)

    return make, "scan_offset_products (100000 rows, 12-column pool)"


CASES = (_case_bspline, _case_bf_apply, _case_basis_fused, _case_honest_null, _case_offset_scan)


def _host(cp: Any, out: Any) -> "tuple":
    """The outputs of a case as a tuple of host arrays (a case returns one device array or a tuple of host / device arrays)."""
    items = out if isinstance(out, tuple) else (out,)
    return tuple(cp.asnumpy(i) if hasattr(i, "get") else np.asarray(i) for i in items)


def main() -> None:
    """Run every case: interleaved medians of the shipped (flag) and free (contractable) builds and the largest output difference."""
    import cupy as cp

    for case in CASES:
        make, label = case(cp)
        shipped, free = make(False), None
        out_shipped = _host(cp, shipped())
        free = make(True)
        out_free = _host(cp, free())
        diff = max(float(np.nanmax(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))) for a, b in zip(out_shipped, out_free))  # Intentional float64: comparison statistic
        t_free = _median_ms(cp, free)
        shipped = make(False)
        t_ship = _median_ms(cp, shipped)
        print(f"{label}: --fmad=false {t_ship:.3f} ms, contractable {t_free:.3f} ms, max abs diff {diff:.3g}")
    for module, kernels in PENDING:
        print(f"not measured yet: {module}: {kernels}")


if __name__ == "__main__":
    main()
