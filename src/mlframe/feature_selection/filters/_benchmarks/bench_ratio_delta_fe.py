"""Bench for _ratio_delta_fe.py CPX4 optimizations.

Two hot-path inefficiencies:
  1. Pairwise ratio/log-ratio O(p^2) loop re-extracts the SAME column via
     ``.to_numpy()`` per inner-loop iteration. Fix: hoist per-column
     ``.to_numpy()`` into an outer dict computed once.
  2. grouped_delta_features / apply_grouped_delta apply a per-row Python
     ``dict.get`` listcomp to map group keys -> stats. Fix: vectorize via
     ``np.unique(return_inverse) + gather`` (per-unique map, then index).

OLD side loaded from ``git show HEAD:<path>`` (pre-edit); NEW from the live
module. Warm + median-of-N. Run:
    python src/mlframe/feature_selection/filters/_benchmarks/bench_ratio_delta_fe.py
"""

import importlib.util
import os
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
REL = "src/mlframe/feature_selection/filters/_ratio_delta_fe.py"


def _load_old_module() -> types.ModuleType:
    """Load pre-edit _ratio_delta_fe.py from git HEAD as a standalone module.

    The module imports ``from ._internals import group_key_strings``; we shim a
    package so the relative import resolves against the live _internals.
    """
    src = subprocess.run(
        ["git", "show", f"HEAD:{REL}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    from mlframe.feature_selection.filters import _internals

    pkg_name = "_rd_old_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []  # mark as package
    sys.modules[pkg_name] = pkg
    sys.modules[pkg_name + "._internals"] = _internals
    mod = types.ModuleType(pkg_name + "._ratio_delta_fe")
    mod.__package__ = pkg_name
    sys.modules[pkg_name + "._ratio_delta_fe"] = mod
    exec(compile(src, "<HEAD:_ratio_delta_fe.py>", "exec"), mod.__dict__)
    return mod


def _median_time(fn, n_runs=7):
    fn()  # warm
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def bench_pairwise(old_mod, new_mod, p=40, n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"c{i}" for i in range(p)]
    # Distinct-ish positive values so most ratio pairs survive redundancy gate.
    X = pd.DataFrame({c: rng.normal(5.0, 2.0, size=n) for c in cols})

    old_df, old_acc = old_mod.pairwise_ratio_features(X, cols)
    new_df, new_acc = new_mod.pairwise_ratio_features(X, cols)
    assert old_acc == new_acc, "ratio accepted pairs differ"
    assert np.array_equal(old_df.to_numpy(), new_df.to_numpy()), "ratio values differ"

    old_lr, old_lacc = old_mod.pairwise_log_ratio_features(X, cols)
    new_lr, new_lacc = new_mod.pairwise_log_ratio_features(X, cols)
    assert old_lacc == new_lacc, "log_ratio accepted pairs differ"
    assert np.array_equal(old_lr.to_numpy(), new_lr.to_numpy()), "log_ratio values differ"

    t_old = _median_time(lambda: old_mod.pairwise_ratio_features(X, cols))
    t_new = _median_time(lambda: new_mod.pairwise_ratio_features(X, cols))
    t_old_lr = _median_time(lambda: old_mod.pairwise_log_ratio_features(X, cols))
    t_new_lr = _median_time(lambda: new_mod.pairwise_log_ratio_features(X, cols))
    print(f"[pairwise_ratio]     p={p} n={n}: OLD {t_old*1e3:8.2f} ms -> NEW {t_new*1e3:8.2f} ms  ({t_old/t_new:.2f}x)  identity=exact")
    print(f"[pairwise_log_ratio] p={p} n={n}: OLD {t_old_lr*1e3:8.2f} ms -> NEW {t_new_lr*1e3:8.2f} ms  ({t_old_lr/t_new_lr:.2f}x)  identity=exact")


def bench_grouped(old_mod, new_mod, n=200_000, n_groups=500, n_num=4, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, size=n)
    data = {"grp": g}
    num_cols = [f"x{i}" for i in range(n_num)]
    for c in num_cols:
        data[c] = rng.normal(0.0, 1.0, size=n)
    X = pd.DataFrame(data)

    old_df, old_rec = old_mod.grouped_delta_features(X, "grp", num_cols)
    new_df, new_rec = new_mod.grouped_delta_features(X, "grp", num_cols)
    assert np.array_equal(old_df.to_numpy(), new_df.to_numpy()), "grouped_delta features differ"

    # apply path: build X_test with some unseen groups to exercise global fallback.
    g2 = rng.integers(0, n_groups + 50, size=n)
    data2 = {"grp": g2}
    for c in num_cols:
        data2[c] = rng.normal(0.0, 1.0, size=n)
    X_test = pd.DataFrame(data2)
    name_z = new_mod.engineered_name_grouped_delta_std(num_cols[0], "grp")
    rec = new_rec[name_z]
    old_app = old_mod.apply_grouped_delta(X_test, rec)
    new_app = new_mod.apply_grouped_delta(X_test, rec)
    assert np.array_equal(old_app, new_app), "apply_grouped_delta differs"

    t_old = _median_time(lambda: old_mod.grouped_delta_features(X, "grp", num_cols))
    t_new = _median_time(lambda: new_mod.grouped_delta_features(X, "grp", num_cols))
    t_old_a = _median_time(lambda: old_mod.apply_grouped_delta(X_test, rec))
    t_new_a = _median_time(lambda: new_mod.apply_grouped_delta(X_test, rec))
    print(f"[grouped_delta_features] n={n} groups={n_groups} num={n_num}: OLD {t_old*1e3:8.2f} ms -> NEW {t_new*1e3:8.2f} ms  ({t_old/t_new:.2f}x)  identity=exact")
    print(f"[apply_grouped_delta]    n={n} groups={n_groups}: OLD {t_old_a*1e3:8.2f} ms -> NEW {t_new_a*1e3:8.2f} ms  ({t_old_a/t_new_a:.2f}x)  identity=exact")


def main():
    sys.path.insert(0, os.path.join(REPO, "src"))
    from mlframe.feature_selection.filters import _ratio_delta_fe as new_mod

    old_mod = _load_old_module()
    print("=== _ratio_delta_fe CPX4 bench (OLD=HEAD, NEW=worktree) ===")
    bench_pairwise(old_mod, new_mod)
    bench_grouped(old_mod, new_mod)


if __name__ == "__main__":
    main()
