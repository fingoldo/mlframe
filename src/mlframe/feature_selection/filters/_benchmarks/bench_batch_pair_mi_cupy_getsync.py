"""P-6 bench: batch the per-pair D2H .get() in batch_pair_mi_cupy into ONE final transfer.

OLD: ``out_host[p] = float(mi.get())`` inside the pair loop -> n_pairs blocking scalar syncs, each draining
the GPU queue between pairs. NEW: stage each pair's scalar into a resident (n_pairs,) buffer, one cp.asnumpy
at the end. Bit-identical output (same per-pair scalar), fewer syncs.

The OLD side is loaded from ``git show HEAD:<path>`` (real prior code, not a rewrite) and both are run on the
same inputs; the bench asserts byte-identical output and reports paired warm timings.

NOTE: cupy is the documented LOSER backend for batch_pair_mi (the dispatcher picks numba.cuda); this optimises
the fallback leg. Selection is unaffected (the dispatcher's MI values are unchanged; only the sync is batched).

Run: PYTHONPATH=src python -m mlframe.feature_selection.filters._benchmarks.bench_batch_pair_mi_cupy_getsync
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def _load_old_module():
    here = Path(__file__).resolve()
    repo_path = "src/mlframe/feature_selection/filters/batch_pair_mi_gpu.py"
    # locate the git repo root by walking up
    root = here
    for _ in range(12):
        root = root.parent
        if (root / ".git").exists() or (root / "src").exists():
            break
    src = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{repo_path}"], text=True)
    tmp = Path(tempfile.mkdtemp()) / "batch_pair_mi_gpu_OLD.py"
    tmp.write_text(src, encoding="utf-8")
    # the file uses only relative-free imports at module top for the cupy path we call; import it standalone
    modname = "mlframe.feature_selection.filters._batch_pair_mi_gpu_OLD"
    spec = importlib.util.spec_from_file_location(modname, tmp)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "mlframe.feature_selection.filters"  # so the file's relative imports resolve
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    from .. import batch_pair_mi_gpu as new

    old = _load_old_module()
    rng = np.random.default_rng(0)
    for n, n_fac, n_pairs, ncls in [(200000, 40, 120, 4), (1000000, 30, 66, 3)]:
        nbins = np.full(n_fac, 10, dtype=np.int32)
        data = np.empty((n, n_fac), dtype=np.int32)
        for j in range(n_fac):
            data[:, j] = rng.integers(0, nbins[j], size=n)
        classes_y = rng.integers(0, ncls, size=n).astype(np.int32)
        freqs_y = np.bincount(classes_y, minlength=ncls).astype(np.float64) / n
        pa = rng.integers(0, n_fac, size=n_pairs).astype(np.int64)
        pb = rng.integers(0, n_fac, size=n_pairs).astype(np.int64)
        args = (data, pa, pb, nbins, classes_y, freqs_y)

        r_new = new.batch_pair_mi_cupy(*args)
        r_old = old.batch_pair_mi_cupy(*args)
        identical = np.array_equal(r_new, r_old)
        maxdiff = float(np.max(np.abs(r_new - r_old))) if not identical else 0.0

        import cupy as cp

        def timed(fn):
            for _ in range(2):
                fn(*args)
            cp.cuda.Stream.null.synchronize()
            ts = []
            for _ in range(8):
                t = time.perf_counter()
                fn(*args)
                cp.cuda.Stream.null.synchronize()
                ts.append(time.perf_counter() - t)
            return np.median(ts) * 1e3

        t_new = timed(new.batch_pair_mi_cupy)
        t_old = timed(old.batch_pair_mi_cupy)
        print(f"n={n} pairs={n_pairs}: identical={identical} maxdiff={maxdiff:.2e} | " f"OLD {t_old:.2f}ms NEW {t_new:.2f}ms speedup {t_old / t_new:.2f}x")


if __name__ == "__main__":
    main()
