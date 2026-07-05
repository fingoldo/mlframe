"""iter123 A/B (@10M): ``_extra_fe_families._column_to_str`` per-unique factorize-gather vs the pre-fix per-ROW map.

The Family A/B/C copy of ``_column_to_str`` ran ``pandas.Series.map(canonical_group_token)`` once per ROW. It now delegates to the
already-optimized per-UNIQUE copy in ``_target_encoding_fe`` (one token call per distinct value; bool/0/1 collision gate falls back to the
exact per-row loop). Identical ``"__nan__"`` sentinel + ``canonical_group_token`` contract -> byte-identical output.

Run (store py3.14, CPU-only):
    PYTHONPATH=src MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 python -m mlframe.feature_selection._benchmarks.bench_extra_fe_column_to_str_iter123

Measured (10M rows, ~500 uniques, best-of-2/3, separate process):
    _column_to_str isolated:  int500      OLD 5.71s  -> NEW 0.235s  = 24.3x
                              float+nan   OLD 20.73s -> NEW 0.317s  = 65.3x
    apply_rare_category e2e:  is_rare     OLD 64.97s -> NEW 19.20s  = 3.38x   (str-build no longer the e2e bottleneck; np.unique over 10M
                                                                              object strings now dominates the residual 19s)
    Output byte-identical across int / float+NaN / str / bool / mixed-bool-int / mixed-int-float (bool cases hit the exact-loop fallback).
RESOLVED.
"""

import sys
import time
import gc

sys.modules.setdefault("cupy", None)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mlframe.feature_selection.filters._extra_fe_families as M  # noqa: E402
from mlframe.feature_selection.filters._internals import canonical_group_token  # noqa: E402

NEW = M._column_to_str


def OLD(col):
    s = pd.Series(col)
    if s.isna().any():
        return s.astype(object).map(lambda v: "__nan__" if (v is None or (isinstance(v, float) and v != v)) else canonical_group_token(v)).to_numpy()
    return s.astype(object).map(canonical_group_token).to_numpy()


def _best(fn, s, k=3):
    out = []
    for _ in range(k):
        gc.collect()
        t0 = time.perf_counter()
        fn(s)
        out.append(time.perf_counter() - t0)
    return min(out)


def main(n: int = 10_000_000):
    rng = np.random.default_rng(2)
    xi = rng.integers(0, 500, n)
    xf = rng.integers(0, 500, n).astype(float)
    xf[rng.random(n) < 0.05] = np.nan

    for name, arr in [("int500", xi), ("float+nan", xf)]:
        s = pd.Series(arr)
        assert np.array_equal(OLD(s).astype(str), NEW(s).astype(str)), f"identity FAIL {name}"
        to = _best(OLD, s)
        tn = _best(NEW, s)
        print(f"_column_to_str {name}: OLD {to:.3f}s NEW {tn:.3f}s speedup {to/tn:.1f}x  (identical)")

    X = pd.DataFrame({"c": xi})
    enc, recs = M.generate_rare_category_features(X, ["c"])
    rec = recs[[k for k in recs if k.startswith("is_rare")][0]]

    def e2e():
        gc.collect()
        t0 = time.perf_counter()
        M.apply_rare_category(X, rec)
        return time.perf_counter() - t0

    M._column_to_str = NEW
    tn = min(e2e() for _ in range(2))
    M._column_to_str = OLD
    to = min(e2e() for _ in range(2))
    M._column_to_str = NEW
    print(f"apply_rare_category e2e @10M: OLD {to:.3f}s NEW {tn:.3f}s speedup {to/tn:.2f}x")


if __name__ == "__main__":
    main()
