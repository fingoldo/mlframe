"""The nested-parallel scanner treats a joblib ``Parallel`` on the threading backend as a thread fan-out.

The scanner used to root its walk only at ``ThreadPoolExecutor`` calls. Joblib's threading fan-outs (``backend="threading"`` or
``prefer="threads"``) were invisible, so an unguarded prange kernel behind one of them went unreported.
"""

from __future__ import annotations

from pathlib import Path

from mlframe._nested_parallel_scan import Package

_SOURCE = """\
from joblib import Parallel, delayed
from numba import njit, prange


@njit(parallel=True)
def kern(x):
    total = 0.0
    for i in prange(x.shape[0]):
        total += x[i]
    return total


def fan_out(xs):
    return Parallel(n_jobs=2, backend="threading")(delayed(kern)(x) for x in xs)
"""


def test_joblib_threading_fan_out_reaches_the_unguarded_kernel(tmp_path: Path):
    """The scanner records the joblib threading function as a pool and reports the prange kernel behind it."""
    (tmp_path / "pkgmod.py").write_text(_SOURCE, encoding="utf-8")

    pkg = Package(tmp_path)
    pool_names = {fname for _, fname in pkg.pools}

    assert "fan_out" in pool_names, f"joblib threading fan-out not seen as a thread pool: pools={pool_names}"
    found = pkg.reachable_kernels(tmp_path / "pkgmod.py", "fan_out")
    assert "kern" in found, f"unguarded prange kernel behind the joblib fan-out not reported: {found}"


_PARALLEL_RUN_SOURCE = """\
from joblib import delayed
from pyutilz.parallel import parallel_run
from numba import njit, prange


@njit(parallel=True)
def kern(x):
    total = 0.0
    for i in prange(x.shape[0]):
        total += x[i]
    return total


def fan_out_default_backend(xs):
    return parallel_run([delayed(kern)(x) for x in xs], n_jobs=2)
"""


def test_parallel_run_fan_out_without_explicit_backend_reaches_the_unguarded_kernel(tmp_path: Path):
    """``parallel_run`` with no explicit backend can fall back to threads, so the scanner must treat it as a fan-out too."""
    (tmp_path / "pkgmod.py").write_text(_PARALLEL_RUN_SOURCE, encoding="utf-8")

    pkg = Package(tmp_path)
    pool_names = {fname for _, fname in pkg.pools}

    assert "fan_out_default_backend" in pool_names, f"parallel_run fan-out not seen as a pool: pools={pool_names}"
    found = pkg.reachable_kernels(tmp_path / "pkgmod.py", "fan_out_default_backend")
    assert "kern" in found, f"unguarded prange kernel behind the parallel_run fan-out not reported: {found}"
