"""iter75 A/B: _column_to_str per-unique factorize fast path vs per-row loop.

OLD = the pre-fix per-row implementation, reconstructed verbatim here (the only
changed function is _column_to_str; everything it calls is unchanged), wired in
by monkeypatching the live module. Identity + paired timing at n=200k.
"""
import time

import scipy.stats  # noqa: F401
import numba  # noqa: F401
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_encoding_fe as M
from mlframe.feature_selection.filters._internals import canonical_group_token

NEW = M._column_to_str


def OLD(col):
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


def _identity():
    rng = np.random.default_rng(1)
    cases = []
    # pure string object
    cases.append(pd.Series(np.array([f"c{k}" for k in rng.integers(0, 50, 5000)], dtype=object)))
    # object with None + nan + strings + ints + floats
    pool = np.array(["a", "b", None, float("nan"), 1, 1.0, 2, 2.5, "1"], dtype=object)
    cases.append(pd.Series(pool[rng.integers(0, len(pool), 3000)]))
    # int column (np.unique path)
    cases.append(pd.Series(rng.integers(0, 100, 4000)))
    # float column
    cases.append(pd.Series(rng.integers(0, 30, 4000).astype(float)))
    # bool-in-object (gated-out path)
    bpool = np.array([True, False, 1, 0, "x"], dtype=object)
    cases.append(pd.Series(bpool[rng.integers(0, len(bpool), 2000)]))
    # all-nan
    cases.append(pd.Series(np.array([None] * 1000, dtype=object)))
    for i, c in enumerate(cases):
        o, n = OLD(c), NEW(c)
        assert o.shape == n.shape, (i, o.shape, n.shape)
        assert np.array_equal(o.astype(str), n.astype(str)), f"case {i} MISMATCH\n{o[:20]}\n{n[:20]}"
    print("IDENTITY: all 6 cases bit-identical (incl bool-gated-out, all-nan)")


def _timing():
    N = 200_000
    rng = np.random.default_rng(0)
    obj_vals = np.array([f"cat_{k}" for k in range(200)], dtype=object)
    col = pd.Series(obj_vals[rng.integers(0, 200, N)])
    for f in (OLD, NEW):
        f(col)  # warm
    trials = 30
    old_t, new_t = [], []
    for _ in range(trials):
        t = time.perf_counter(); OLD(col); old_t.append(time.perf_counter() - t)
        t = time.perf_counter(); NEW(col); new_t.append(time.perf_counter() - t)
    wins = sum(n < o for o, n in zip(old_t, new_t))
    print(f"OLD min {min(old_t)*1e3:.2f}ms med {np.median(old_t)*1e3:.2f}ms")
    print(f"NEW min {min(new_t)*1e3:.2f}ms med {np.median(new_t)*1e3:.2f}ms")
    print(f"min speedup {min(old_t)/min(new_t):.2f}x  med {np.median(old_t)/np.median(new_t):.2f}x  NEW faster {wins}/{trials}")


if __name__ == "__main__":
    _identity()
    _timing()
