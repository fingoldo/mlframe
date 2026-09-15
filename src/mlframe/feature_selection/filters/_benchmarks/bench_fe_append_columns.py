"""Benchmark: appending engineered columns to a pandas frame without copying the base frame.

``_fe_frame_ops.fe_append_columns`` is the append seam every FE stage goes through. Its pandas branch is ``pd.concat([X, new], axis=1)``,
which under pandas 2.x without copy-on-write COPIES every existing column on every append, although its docstring says the base is not
duplicated. This bench compares that against a shallow copy of the frame plus column assignment, which shares the base blocks.

For each (n, base width, appended k) it reports, per strategy: warm best-of-N wall time, whether every base column still shares memory
with the input, whether the input frame is untouched, whether order / dtypes / values match the concat reference, and whether pandas
raised its block-fragmentation PerformanceWarning. It does not measure an end-to-end fit; run that A/B separately before changing the seam.

Usage: ``python -m mlframe.feature_selection.filters._benchmarks.bench_fe_append_columns [--repeats 5]``

Measured 2026-09-15, pandas 2.3.3 (copy-on-write off), best of 5, warm:
- concat never shares a base column; shallow_setitem always does, leaves the input untouched and equals the concat result.
- n >= 200k: shallow_setitem is faster at every k, e.g. n=2M p=60: k=1 454.5 -> 6.1 ms, k=8 529.5 -> 60.9 ms, k=150 2405 -> 1096 ms.
- n=10k, k >= 64: shallow_setitem is slower (k=64 5.8 -> 14.3 ms, k=150 12.5 -> 38.4 ms), trivial in absolute terms.
- k=150: shallow_setitem trips pandas' fragmentation PerformanceWarning at every n; its downstream cost is not measured here.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd


def _concat(X: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """The current seam: one concat along columns."""
    return pd.concat([X, pd.DataFrame(cols, index=X.index)], axis=1)


def _shallow_setitem(X: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Shallow copy of the frame, then one assignment per new column."""
    out = X.copy(deep=False)
    for name, values in cols.items():
        out[name] = values
    return out


STRATEGIES = {"concat": _concat, "shallow_setitem": _shallow_setitem}


def _frame(n: int, p: int, seed: int = 0) -> pd.DataFrame:
    """A mixed-dtype base frame: float columns plus a categorical and an int64 column."""
    rng = np.random.default_rng(seed)
    data: dict = {f"c{i}": rng.normal(size=n) for i in range(p)}
    data["cat"] = pd.Categorical(rng.integers(0, 5, size=n))
    data["i64"] = rng.integers(0, 100, size=n)
    return pd.DataFrame(data)


def run(repeats: int = 5) -> list[dict]:
    """Run the sweep and return one row per (n, p, k, strategy)."""
    rows = []
    for n in (10_000, 200_000, 2_000_000):
        for p in (10, 60):
            X = _frame(n, p)
            X_ref = X.copy()
            base_cols = [c for c in X.columns if X[c].dtype != "category"]
            for k in (1, 8, 64, 150):
                rng = np.random.default_rng(k)
                cols = {f"e{j}": rng.normal(size=n) for j in range(k)}
                reference = _concat(X, cols)
                for label, fn in STRATEGIES.items():
                    fn(X, cols)  # warm
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        best = float("inf")
                        out = None
                        for _ in range(repeats):
                            t0 = time.perf_counter()
                            out = fn(X, cols)
                            best = min(best, time.perf_counter() - t0)
                    assert out is not None
                    rows.append(
                        {
                            "n": n,
                            "p": p,
                            "k": k,
                            "strategy": label,
                            "best_ms": round(best * 1e3, 3),
                            "shares_base": all(np.shares_memory(out[c].to_numpy(), X[c].to_numpy()) for c in base_cols),
                            "input_untouched": X.equals(X_ref) and list(X.columns) == list(X_ref.columns),
                            "matches_concat": out.equals(reference) and list(out.columns) == list(reference.columns),
                            "fragmentation_warning": any("fragmented" in str(w.message) for w in caught),
                        }
                    )
    return rows


def main() -> None:
    """CLI entry point: print the sweep as a table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    print(f"pandas {pd.__version__}")
    for row in run(repeats=args.repeats):
        print(
            f"n={row['n']:>9} p={row['p']:>3} k={row['k']:>4} {row['strategy']:>16} {row['best_ms']:>10.3f} ms"
            f"  shares_base={row['shares_base']!s:5} input_untouched={row['input_untouched']!s:5}"
            f"  matches_concat={row['matches_concat']!s:5} fragmentation_warning={row['fragmentation_warning']}"
        )


if __name__ == "__main__":
    main()
