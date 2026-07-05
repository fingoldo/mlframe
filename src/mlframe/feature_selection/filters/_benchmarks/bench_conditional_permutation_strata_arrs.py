"""Bench for _conditional_permutation.py: hoist per-stratum index-array
materialisation out of the permutation loop.

Hot path (cProfile, n=20k B=200): ``np.asarray(idx_list, dtype=int64)`` was
called once PER stratum PER permutation (B * n_strata calls, ~48% of wall),
rebuilding the SAME constant index arrays every iteration. Fix: build the
per-stratum int64 arrays once (and drop singletons once) before the loop.

Bit-identity: ``rng.permutation`` receives the IDENTICAL int64 arrays in the
IDENTICAL dict-iteration order -> same RNG draws -> same null distribution ->
same (observed, p_value). Verified exact == on (observed, p) here.

OLD side loaded from ``git show HEAD:<path>``; NEW from the live module.
Run:
    python src/mlframe/feature_selection/filters/_benchmarks/bench_conditional_permutation_strata_arrs.py
"""

import os
import subprocess
import sys
import time
import types

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
REL = "src/mlframe/feature_selection/filters/_conditional_permutation.py"


def _load_old_module() -> types.ModuleType:
    src = subprocess.run(["git", "show", f"HEAD:{REL}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    from mlframe.feature_selection.filters import _cmi_perm_stop  # noqa: F401

    pkg_name = "_cpt_old_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []
    sys.modules[pkg_name] = pkg
    sys.modules[pkg_name + "._cmi_perm_stop"] = _cmi_perm_stop
    mod = types.ModuleType(pkg_name + "._conditional_permutation")
    mod.__package__ = pkg_name
    sys.modules[pkg_name + "._conditional_permutation"] = mod
    exec(compile(src, "<HEAD:_conditional_permutation.py>", "exec"), mod.__dict__)
    return mod


def _median_time(fn, n_runs=7):
    fn()  # warm
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench(old_mod, new_mod, n=20_000, nbx=8, nby=8, nbz=10, B=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, nbx, n)
    y = rng.integers(0, nby, n)
    z = rng.integers(0, nbz, n)

    old_res = old_mod.conditional_permutation_test(x, y, z, nbx, nby, nbz, n_permutations=B, seed=7)
    new_res = new_mod.conditional_permutation_test(x, y, z, nbx, nby, nbz, n_permutations=B, seed=7)
    assert old_res == new_res, f"identity FAILED: {old_res} != {new_res}"

    t_old = _median_time(lambda: old_mod.conditional_permutation_test(x, y, z, nbx, nby, nbz, n_permutations=B, seed=7))
    t_new = _median_time(lambda: new_mod.conditional_permutation_test(x, y, z, nbx, nby, nbz, n_permutations=B, seed=7))
    print(f"[cpt] n={n} nbz={nbz} B={B}: OLD {t_old*1e3:8.2f} ms -> NEW {t_new*1e3:8.2f} ms" f"  ({t_old/t_new:.2f}x)  identity=exact (obs,p)={new_res}")


def main():
    sys.path.insert(0, os.path.join(REPO, "src"))
    from mlframe.feature_selection.filters import _conditional_permutation as new_mod

    old_mod = _load_old_module()
    print("=== _conditional_permutation strata-arr-hoist bench (OLD=HEAD, NEW=worktree) ===")
    bench(old_mod, new_mod, nbz=10)
    bench(old_mod, new_mod, nbz=50)


if __name__ == "__main__":
    main()
