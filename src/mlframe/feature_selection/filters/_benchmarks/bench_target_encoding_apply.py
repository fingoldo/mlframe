"""Bench for CPX3: apply_target_encoding per-row dict.get -> vectorized map.

OLD side loaded from ``git show HEAD:<path>`` (pre-edit per-row loop); NEW from
the live module (pd.Series.map + fillna). Warm + median-of-N. Run:
    python src/mlframe/feature_selection/filters/_benchmarks/bench_target_encoding_apply.py
"""

import os
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
REL = "src/mlframe/feature_selection/filters/_target_encoding_fe.py"


def _load_old_module() -> types.ModuleType:
    src = subprocess.run(
        ["git", "show", f"HEAD:{REL}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    from mlframe.feature_selection.filters import _internals  # noqa: F401

    pkg_name = "_te_old_pkg"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = []
    sys.modules[pkg_name] = pkg
    sys.modules[pkg_name + "._internals"] = _internals
    mod = types.ModuleType(pkg_name + "._target_encoding_fe")
    mod.__package__ = pkg_name
    sys.modules[pkg_name + "._target_encoding_fe"] = mod
    exec(compile(src, "<HEAD:_target_encoding_fe.py>", "exec"), mod.__dict__)
    return mod


def _median_time(fn, n_runs=7):
    fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    sys.path.insert(0, os.path.join(REPO, "src"))
    from mlframe.feature_selection.filters import _target_encoding_fe as new_mod

    old_mod = _load_old_module()

    rng = np.random.default_rng(0)
    n = 500_000
    card = 300
    col = "cat"
    lookup = {str(k): float(rng.normal(0.3, 0.1)) for k in range(card)}
    recipe = {"lookup": lookup, "global_mean": 0.275}

    print("=== apply_target_encoding (OLD=HEAD .map, NEW=worktree per-unique fused int fast-path) ===")

    # INT source column (fast path): ~10% unseen -> global_mean fallback.
    cats_int = rng.integers(0, card + 30, size=n)
    X_int = pd.DataFrame({col: cats_int})
    old = old_mod.apply_target_encoding(X_int, col, recipe)
    new = new_mod.apply_target_encoding(X_int, col, recipe)
    assert np.array_equal(old, new), f"int values differ: max|d|={np.max(np.abs(old-new))}"
    t_old = _median_time(lambda: old_mod.apply_target_encoding(X_int, col, recipe))
    t_new = _median_time(lambda: new_mod.apply_target_encoding(X_int, col, recipe))
    print(f"int    n={n} card={card}: OLD {t_old*1e3:8.2f} ms -> NEW {t_new*1e3:8.2f} ms  ({t_old/t_new:.2f}x)  identity=exact")

    # OBJECT (string) source column (unchanged path): must stay neutral, identity exact.
    toks = np.array([f"c{k}" for k in range(card + 30)], dtype=object)
    cats_obj = toks[rng.integers(0, card + 30, size=n)]
    lookup_obj = {f"c{k}": float(rng.normal(0.3, 0.1)) for k in range(card)}
    recipe_obj = {"lookup": lookup_obj, "global_mean": 0.275}
    X_obj = pd.DataFrame({col: cats_obj})
    old_o = old_mod.apply_target_encoding(X_obj, col, recipe_obj)
    new_o = new_mod.apply_target_encoding(X_obj, col, recipe_obj)
    assert np.array_equal(old_o, new_o), f"obj values differ: max|d|={np.max(np.abs(old_o-new_o))}"
    t_old_o = _median_time(lambda: old_mod.apply_target_encoding(X_obj, col, recipe_obj))
    t_new_o = _median_time(lambda: new_mod.apply_target_encoding(X_obj, col, recipe_obj))
    print(f"object n={n} card={card}: OLD {t_old_o*1e3:8.2f} ms -> NEW {t_new_o*1e3:8.2f} ms  ({t_old_o/t_new_o:.2f}x)  identity=exact (unchanged path)")


if __name__ == "__main__":
    main()
