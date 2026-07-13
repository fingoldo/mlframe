"""iter75 e2e A/B: full kfold_target_encode_fit + apply at n=200k.

OLD side monkeypatches the pre-fix per-row _column_to_str into the live module;
NEW side restores the shipped per-unique factorize path. Paired interleaved.
"""
import time

import scipy.stats  # noqa: F401
import numba  # noqa: F401
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_encoding_fe as M
from mlframe.feature_selection.filters._internals import canonical_group_token

NEW_FN = M._column_to_str


def OLD_FN(col):
    arr = col.to_numpy() if hasattr(col, "to_numpy") else np.asarray(col)
    if arr.dtype.kind in ("i", "u", "b"):
        uniq, inv = np.unique(arr, return_inverse=True)
        toks = np.array([canonical_group_token(u) for u in uniq], dtype=object)
        return toks[inv]
    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if v is None:
            out[i] = "__nan__"
        elif isinstance(v, float) and v != v:
            out[i] = "__nan__"
        else:
            out[i] = canonical_group_token(v)
    return out


N = 200_000
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    obj_vals = np.array([f"cat_{k}" for k in range(200)], dtype=object)
    X = pd.DataFrame({
        "c_obj": obj_vals[rng.integers(0, 200, N)],
        "c_int": rng.integers(0, 1500, N),
    })
    y = rng.integers(0, 2, N).astype(np.float64)


    def run():
        te_df, recipes = M.kfold_target_encode_fit(X, y, ["c_obj", "c_int"])
        a = M.apply_target_encoding(X, "c_obj", recipes["c_obj"])
        b = M.apply_target_encoding(X, "c_int", recipes["c_int"])
        return te_df, a, b


    # Identity of the FULL e2e output.
    M._column_to_str = OLD_FN
    te_o, a_o, b_o = run()
    M._column_to_str = NEW_FN
    te_n, a_n, b_n = run()
    maxd = max(
        float(np.max(np.abs(te_o.to_numpy() - te_n.to_numpy()))),
        float(np.max(np.abs(a_o - a_n))),
        float(np.max(np.abs(b_o - b_n))),
    )
    print(f"e2e output maxdiff OLD vs NEW: {maxd:.3e}  (te_df + both applied cols)")

    # Warm both.
    for fn in (OLD_FN, NEW_FN):
        M._column_to_str = fn
        run()

    trials = 15
    old_t, new_t = [], []
    for _ in range(trials):
        M._column_to_str = OLD_FN
        t = time.perf_counter(); run(); old_t.append(time.perf_counter() - t)
        M._column_to_str = NEW_FN
        t = time.perf_counter(); run(); new_t.append(time.perf_counter() - t)
    M._column_to_str = NEW_FN

    wins = sum(n < o for o, n in zip(old_t, new_t))
    print(f"OLD min {min(old_t)*1e3:.1f}ms med {np.median(old_t)*1e3:.1f}ms")
    print(f"NEW min {min(new_t)*1e3:.1f}ms med {np.median(new_t)*1e3:.1f}ms")
    print(f"min {min(old_t)/min(new_t):.2f}x  med {np.median(old_t)/np.median(new_t):.2f}x  NEW faster {wins}/{trials}")
