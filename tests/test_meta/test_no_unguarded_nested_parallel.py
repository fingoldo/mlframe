"""A thread fan-out must not let two threads into a numba ``parallel=True`` kernel at once.

numba's default threading layer is not safe to enter concurrently from several Python threads. Linux
mostly tolerates it; macOS aborts the process. mlframe's first three-OS CI run crashed 92 xdist workers
with ``Fatal Python error: Aborted``, every faulthandler dump showing multiple pool threads stopped at the
same ``parallel=True`` call site in ``per_feature_edges``.

That instance is fixed, and this gate exists because the NEXT one is easy to add by accident: the pattern
is ordinary and reads as obviously good -- fan work out over columns or chunks, and let each worker call
the fast kernel. Nothing about it looks wrong until it runs on macOS.

The rule: a module that BOTH starts a thread pool AND can REACH a prange kernel -- transitively, not just
by calling one directly -- has to import ``mlframe._numba_parallel_guard``. Transitively matters, and is
the whole reason this file is not two lines shorter: the crash that motivated it ran
``per_feature_edges`` -> ``edges_fayyad_irani`` -> ``mdlp_bin_edges`` -> ``_mdlp_recurse_validated_bfs``
-> the kernel, four hops from the pool, and a same-module check sees none of that. The walk goes six deep
rather than four: CI found a fifth-hop path after the first round of guarding, so a depth chosen to reach
the last known bug reaches only the last known bug.

Whether the guard is held at exactly the right call is not something a static check can decide, so this
asserts that the author was made to think about it, and the allowlist records the modules where the answer
was "this one cannot race" along with why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "mlframe"
_SKIP_DIRS = frozenset({"_benchmarks", "legacy", "benchmarks", "profiling", "__pycache__"})

#: Modules that start a pool and reach a prange kernel, but cannot have two threads inside one, with the
#: reason each is safe. A new entry needs that reason, not just a path.
_ALLOWLIST = {
    # Watchdog, not parallelism: the pool has one worker and the CALLING thread blocks on
    # ``_future.result(timeout=...)`` until it finishes, so exactly one thread is ever running work.
    "feature_selection/filters/_mrmr_fe_step/_step_pairmi.py",
}


@pytest.fixture(scope="module")
def package():
    """The parsed call graph, built by the scanner this gate shares its logic with.

    Imported rather than reimplemented: the walk has to know about dispatch tables, guarded modules and how
    deep to go, and two copies of that would drift. The first version of this file DID carry its own copy,
    and the copies disagreed within a day -- the scanner learned to follow ``{"name": kernel}`` lookup
    tables after CI found a crash through one, and this gate would still have been blind to it.
    """
    from mlframe._nested_parallel_scan import Package

    return Package(SRC)


def test_a_module_that_threads_into_a_prange_kernel_takes_the_guard(package):
    """Starting a pool whose workers can enter a prange kernel is what aborts the process on macOS."""
    offenders = []
    for path, fname in package.pools:
        rel = path.relative_to(SRC).as_posix()
        if rel in _ALLOWLIST:
            continue
        for kernel, trail in sorted(package.reachable_kernels(path, fname).items()):
            offenders.append(f"{rel}::{fname} -> {kernel}  ({trail})")

    assert not offenders, (
        "these start a thread pool and can reach an unguarded numba parallel=True kernel:\n  "
        + "\n  ".join(offenders)
        + "\n\nTwo threads inside one prange region aborts the process on macOS. Hold "
        "parallel_kernel_entry() around the kernel call -- preferably at a shared dispatcher, which covers "
        "every caller -- or add the module to _ALLOWLIST with the reason it cannot race. "
        "`python -m mlframe._nested_parallel_scan` prints these paths outside pytest."
    )


def test_the_allowlist_still_describes_real_modules():
    """An entry for a module that no longer exists is archaeology, not an exemption."""
    missing = sorted(rel for rel in _ALLOWLIST if not (SRC / rel).exists())
    assert not missing, f"allowlisted modules no longer exist: {missing}"


def test_the_guard_serialises_entry():
    """The guard is only worth importing if it actually excludes a second thread."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    lock = parallel_kernel_entry()
    assert lock is parallel_kernel_entry(), "each call handed back a different lock; nothing would be serialised"
    with lock:
        assert not lock.acquire(blocking=False), "a second thread could enter while the first was inside"
