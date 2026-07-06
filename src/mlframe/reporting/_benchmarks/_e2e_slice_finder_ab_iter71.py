"""iter71 end-to-end A/B for find_weak_slices at the 200k diag regime (DIAG_ROW_CAP sub-sample n=100k).

NEW side imports the in-tree (F-order) module; OLD side loads HEAD:slice_finder.py (C-order codes) as a separate
module object so both run in ONE process against identical inputs. Paired, warm, best-of-N. Prints identity + speedup.

Run: python -m mlframe.reporting._benchmarks._e2e_slice_finder_ab_iter71
"""
import importlib.util
import os
import subprocess  # nosec B404 - subprocess used below with fixed list args, no shell=True
import sys
import tempfile
import time

import numpy as np

import mlframe.reporting.charts.slice_finder as NEW


def _load_old():
    here = os.path.dirname(os.path.abspath(NEW.__file__))
    repo = here
    for _ in range(6):
        repo = os.path.dirname(repo)
        if os.path.isdir(os.path.join(repo, ".git")) or os.path.isfile(os.path.join(repo, "pyproject.toml")):
            break
    src = subprocess.check_output(["git", "show", "HEAD:src/mlframe/reporting/charts/slice_finder.py"], cwd=repo)  # nosec B603, B607 - fixed/trusted executable (git) with list args, no untrusted input, resolved via PATH intentionally
    path = os.path.join(tempfile.gettempdir(), "_old_slice_finder_iter71.py")
    with open(path, "wb") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("_old_slice_finder_iter71", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_old_slice_finder_iter71"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    OLD = _load_old()
    rng = np.random.default_rng(0)
    n, p = 100_000, 30
    X = rng.standard_normal((n, p))
    y_true = rng.standard_normal(n)
    bad = X[:, 0] > 0.8
    y_pred = y_true + np.where(bad, rng.standard_normal(n) * 2.5, rng.standard_normal(n) * 0.1)
    names = [f"f{i}" for i in range(p)]
    kw = dict(task="regression", feature_names=names, seed=0)

    rn = NEW.find_weak_slices(X, y_true, y_pred, **kw)
    ro = OLD.find_weak_slices(X, y_true, y_pred, **kw)
    sn = rn.table["score"].to_numpy()
    so = ro.table["score"].to_numpy()
    assert np.array_equal(sn, so), "scores diverged between layouts!"  # nosec B101 - internal invariant check in src/mlframe/reporting/_benchmarks, not reachable with untrusted input
    assert rn.table["bounds"].tolist() == ro.table["bounds"].tolist()  # nosec B101 - internal invariant check in src/mlframe/reporting/_benchmarks, not reachable with untrusted input
    print(f"IDENTITY: bit-identical table ({len(sn)} rows), global_error {rn.global_error!r} == {ro.global_error!r}")

    N = 11
    tn = []; to = []
    for _ in range(N):
        t0 = time.perf_counter(); NEW.find_weak_slices(X, y_true, y_pred, **kw); tn.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); OLD.find_weak_slices(X, y_true, y_pred, **kw); to.append(time.perf_counter() - t0)
    mn, mo = min(tn), min(to)
    medn, medo = sorted(tn)[N // 2], sorted(to)[N // 2]
    faster = sum(1 for a, b in zip(to, tn) if b < a)
    print(f"n={n} p={p}  best-of-{N}  (full find_weak_slices)")
    print(f"  OLD (C-order codes): min={mo*1e3:.1f}ms median={medo*1e3:.1f}ms")
    print(f"  NEW (F-order codes): min={mn*1e3:.1f}ms median={medn*1e3:.1f}ms")
    print(f"  speedup min={mo/mn:.3f}x median={medo/medn:.3f}x  NEW faster in {faster}/{N} paired trials")


if __name__ == "__main__":
    main()
