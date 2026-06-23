"""Bench: temporal_agg_fe history-build + expanding-replay optimizations.

Two algorithmic hotspots in ``_temporal_agg_fe.py`` are benched here against the
REAL prior code (loaded via ``git show HEAD:<path>`` into a baseline module):

1. **Per-group ``codes_sorted == g`` history-build masks** (fit-side, in
   ``generate_expanding/rolling/lag``): O(N * cardinality) -> O(N log N) via a
   single stable-argsort + bincount split (``_group_row_slices``).
2. **``apply_temporal_expanding`` per-row concatenate+reduce**: O(N^2) per
   entity -> O(N) via Welford running accumulators seeded from train history.

Identity gate: emitted columns must be exact-equal (history-build, lag, min,
max, count) or ~1e-9 (mean/std reduction-order). The bench asserts both.

Run:
    CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_selection/filters/_benchmarks/bench_temporal_agg_fe.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
# repo_root/src/mlframe/feature_selection/filters/_benchmarks/bench_*.py
REPO = HERE.parents[5]
REL = "src/mlframe/feature_selection/filters/_temporal_agg_fe.py"


def _load_baseline():
    """Load the prior (HEAD) version of the target as an isolated module."""
    src = subprocess.check_output(
        ["git", "show", f"HEAD:{REL}"], cwd=str(REPO)
    ).decode("utf-8")
    tmp = Path(tempfile.gettempdir()) / "_temporal_agg_fe_baseline.py"
    tmp.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("_temporal_agg_fe_baseline", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_temporal_agg_fe_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_new():
    sys.path.insert(0, str(REPO / "src"))
    from mlframe.feature_selection.filters import _temporal_agg_fe as mod
    return mod


def make_data(n_rows: int, n_entities: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    ent = rng.integers(0, n_entities, size=n_rows)
    t = rng.integers(0, n_rows, size=n_rows).astype(np.int64)
    val = rng.normal(size=n_rows)
    # sprinkle a few NaNs
    val[rng.integers(0, n_rows, size=n_rows // 50)] = np.nan
    df = pd.DataFrame({"ent": ent, "t": t, "val": val})
    return df


def best_of(fn, n=5):
    best = float("inf")
    res = None
    for _ in range(n):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return best, res


def main():
    base = _load_baseline()
    new = _load_new()

    for n_rows, n_ent in [(20_000, 200), (60_000, 400)]:
        df = make_data(n_rows, n_ent)
        ent_cols, val_cols, tcol = ["ent"], ["val"], "t"
        stats = ["mean", "std", "count", "min", "max"]

        # ---- fit side (history-build masks) ----
        t_old, (enc_old, rec_old) = best_of(
            lambda: base.generate_expanding_agg_features(df, ent_cols, val_cols, tcol, stats=stats)
        )
        t_new, (enc_new, rec_new) = best_of(
            lambda: new.generate_expanding_agg_features(df, ent_cols, val_cols, tcol, stats=stats)
        )
        # identity: fit-side encoded columns must be exact-equal
        assert list(enc_old.columns) == list(enc_new.columns)
        for c in enc_old.columns:
            np.testing.assert_array_equal(enc_old[c].to_numpy(), enc_new[c].to_numpy())

        print(f"[fit  n={n_rows} ent={n_ent}] OLD {t_old*1e3:8.2f}ms  NEW {t_new*1e3:8.2f}ms  "
              f"speedup {t_old/t_new:5.2f}x  (encoded EXACT-identical)")

        # ---- transform side (expanding replay O(N^2) -> O(N)) ----
        # build a test frame; pick the 'mean' and 'min' recipes
        df_test = make_data(n_rows, n_ent, seed=99)
        name_mean = new.engineered_name_expanding("val", "ent", "mean")
        name_min = new.engineered_name_expanding("val", "ent", "min")
        name_std = new.engineered_name_expanding("val", "ent", "std")
        name_cnt = new.engineered_name_expanding("val", "ent", "count")

        for nm, tol in [(name_min, 0.0), (name_cnt, 0.0), (name_mean, 1e-9), (name_std, 1e-9)]:
            extra_old = rec_old[nm]
            extra_new = rec_new[nm]
            t_o, r_o = best_of(lambda: base.apply_temporal_expanding(df_test, extra_old))
            t_n, r_n = best_of(lambda: new.apply_temporal_expanding(df_test, extra_new))
            if tol == 0.0:
                np.testing.assert_array_equal(r_o, r_n)
                idr = "EXACT"
            else:
                md = float(np.max(np.abs(r_o - r_n)))
                assert md <= tol, f"{nm}: max abs diff {md} > {tol}"
                idr = f"~{md:.1e}"
            print(f"[xform {nm:22s} n={n_rows}] OLD {t_o*1e3:8.2f}ms  NEW {t_n*1e3:8.2f}ms  "
                  f"speedup {t_o/t_n:6.2f}x  (identity {idr})")
        print()


if __name__ == "__main__":
    main()
