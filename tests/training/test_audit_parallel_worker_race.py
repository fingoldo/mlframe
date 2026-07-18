"""Wave-27 sensors: race-condition fixes for parallel workers (3 sites).

#1 P1 feature_selection/filters/feature_engineering.py:246
   ``times_spent[bin_func_name] += timer() - start`` inside the worker.
   Dispatched via ``parallel_run(..., backend='threading')`` from
   mrmr.py:2257. Python's ``+=`` on a float is load-add-store and NOT
   atomic between threads under the GIL; concurrent workers dropped
   updates silently. Operator-visible at verbose>2 where the
   diagnostic ``logger.info('time spent by binary func')`` under-
   reported / silently zeroed.
   Fix: module-level ``_TIMES_SPENT_LOCK`` serialises the increment.

#2 P2 feature_selection/filters/gpu.py:73
   #2 P2 feature_engineering/transformer/_kernels_cupy.py:33
   ``_KERNEL_INIT_LOCK = multiprocessing.Lock()`` documented as
   "Cross-process safe ... picks up the host process's mutex on
   spawn". That's FALSE on Windows spawn; each child re-imports the
   module and constructs its own Lock.
   Fix: documentation only -- the kernel-init body is idempotent
   so the actual behaviour is fine; the misleading comment would
   bait future contributors to add non-idempotent init under the
   same "cross-process safe" guarantee.

#3 P2 feature_selection/wrappers/_rfecv.py:1643
   ``Parallel(..., prefer='threads')`` is a SOFT hint that an outer
   ``joblib.parallel_backend('loky')`` / sklearn ``parallel_config``
   can override. The day someone wraps ``RFECV.fit`` in a process
   backend, the closure-state mutations silently vanish in worker
   copies and ``final_score = nan`` with no exception.
   Fix: add ``require='sharedmem'`` so joblib RAISES when it can't
   satisfy threading.
"""

from __future__ import annotations

import pathlib

import mlframe as _mlframe

_ROOT = pathlib.Path(_mlframe.__file__).resolve().parent


def _read(rel: str) -> str:
    """Read a source file under src/mlframe.

    2026-05-21 rfecv monolith split: RFECV.fit body moved to
    ``_rfecv_fit.py``; concat so source-grep sensors for the relocated
    Parallel-call pattern still match.
    """
    primary = (_ROOT / rel).read_text(encoding="utf-8")
    if rel == "feature_selection/wrappers/rfecv/__init__.py":
        _dir = _ROOT / "feature_selection" / "wrappers" / "rfecv"
        # Concat every rfecv/ submodule so the source-grep sensor matches
        # the relocated code regardless of which submodule owns it now.
        for _sib_path in sorted(_dir.glob("*.py")):
            if _sib_path.name != "__init__.py":
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
    elif rel == "feature_selection/filters/feature_engineering.py":
        # check_prospective_fe_pairs lives in the ``_feature_engineering_pairs``
        # subpackage; concat every submodule so the source-grep sensor matches the
        # relocated ``_TIMES_SPENT_LOCK`` declaration + locked ``times_spent[...] +=``
        # regardless of which submodule owns each now.
        sibling_pkg = _ROOT / "feature_selection" / "filters" / "_feature_engineering_pairs"
        if sibling_pkg.is_dir():
            for _sib_path in sorted(sibling_pkg.glob("*.py")):
                primary = primary + "\n" + _sib_path.read_text(encoding="utf-8")
        else:
            sibling = _ROOT / "feature_selection" / "filters" / "_feature_engineering_pairs.py"
            if sibling.exists():
                primary = primary + "\n" + sibling.read_text(encoding="utf-8")
    return primary


# ---- #1 times_spent lock ------------------------------------------------


def test_feature_engineering_times_spent_lock_added():
    """Feature engineering times spent lock added."""
    src = _read("feature_selection/filters/feature_engineering.py")
    assert (
        "_TIMES_SPENT_LOCK = threading.Lock()" in src
    ), "Wave 27 P1 regression: _TIMES_SPENT_LOCK module-level lock removed; times_spent[k] += ... races silently between threading workers."
    assert "with _TIMES_SPENT_LOCK:" in src
    # The shared-dict increment MUST be guarded by the lock. Assert STRUCTURALLY
    # (tolerant of variable renames / indent / the batched-per-pair merge form)
    # that a ``times_spent[...] +=`` sits inside a ``with _TIMES_SPENT_LOCK:``
    # block, rather than pinning an exact variable name + indent: the increment
    # was legitimately renamed (``bin_func_name`` -> ``_bf``) and batched into one
    # locked merge per pair (fewer lock acquisitions, still race-safe). This fails
    # iff the lock no longer guards the increment (the actual regression to catch).
    import re

    locked_increment = re.search(
        r"with _TIMES_SPENT_LOCK:(?:\n[ \t]+[^\n]*)*?\n[ \t]+times_spent\[[^\]]+\]\s*\+=",
        src,
    )
    assert locked_increment is not None, (
        "Wave 27 P1 regression: no ``times_spent[...] +=`` found under a "
        "``with _TIMES_SPENT_LOCK:`` block -- the lock guarding the shared "
        "timing dict was removed or the increment moved outside it."
    )


def test_feature_engineering_times_spent_lock_behaves_threadsafe():
    """Behavioural: stress the lock with 100 concurrent threads each
    doing 1000 increments; assert no updates lost."""
    import threading as _t
    import time
    from collections import defaultdict
    from mlframe.feature_selection.filters.feature_engineering import _TIMES_SPENT_LOCK

    times_spent = defaultdict(float)

    def _worker():
        """Worker."""
        for _ in range(1000):
            with _TIMES_SPENT_LOCK:
                times_spent["bf"] += 0.001

    threads = [_t.Thread(target=_worker) for _ in range(100)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    # Exact total: 100 * 1000 * 0.001 = 100.0; allow rounding error.
    assert abs(times_spent["bf"] - 100.0) < 0.001, (
        f"Wave 27 P1 regression: lock didn't serialise increments. "
        f"Got {times_spent['bf']:.6f}, expected 100.0. "
        f"({elapsed:.2f}s under 100 threads x 1000 increments)"
    )


# ---- #2 _KERNEL_INIT_LOCK doc honesty -----------------------------------


def test_gpu_kernel_init_lock_doc_no_longer_claims_cross_process():
    """Gpu kernel init lock doc no longer claims cross process."""
    src = _read("feature_selection/filters/gpu.py")
    # Pre-fix claim must be gone:
    assert "Cross-process safe lock. Constructed on first import" not in src, (
        "Wave 27 P2 regression: gpu.py reverted to misleading "
        "'Cross-process safe' claim on a multiprocessing.Lock that "
        "is actually intra-process-only under Windows spawn."
    )
    # Post-fix honest documentation:
    assert "is FALSE on Windows ``spawn``" in src or "is FALSE on Windows" in src


def test_cupy_kernel_init_lock_doc_honest():
    """Cupy kernel init lock doc honest."""
    src = _read("feature_engineering/transformer/_kernels_cupy.py")
    assert "Cross-process safe lock (Windows spawn workers)." not in src
    # Post-fix marker:
    assert "intra-process lock" in src


# ---- #3 _rfecv require=sharedmem ---------------------------------------


def test_rfecv_parallel_requires_sharedmem():
    """Rfecv parallel requires sharedmem."""
    src = _read("feature_selection/wrappers/rfecv/__init__.py")
    # Pre-fix bare ``prefer='threads'`` without require must be gone:
    assert 'Parallel(n_jobs=n_jobs_effective, prefer="threads")(' not in src, (
        "Wave 27 P2 regression: _rfecv Parallel call reverted to "
        "prefer='threads'-only; an outer loky parallel_backend would "
        "silently break the closure-mutation pattern."
    )
    # Post-fix marker:
    assert 'prefer="threads", require="sharedmem"' in src
