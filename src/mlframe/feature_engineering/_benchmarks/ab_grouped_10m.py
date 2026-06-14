"""E2E paired A/B + identity for grouped.py counting-sort segmentation @10M.
Loads NEW (worktree) and OLD (HEAD) grouped.py standalone, alternates calls, checks identity."""
import sys, importlib.util
sys.modules['cupy'] = None
import scipy.stats  # noqa
import numba  # noqa
import time
import numpy as np

NEW = "C:/Users/Admin/Machine learning/mlframe/.claude/worktrees/manual-perf102/src/mlframe/feature_engineering/grouped.py"
OLD = "C:/Users/Admin/AppData/Local/Temp/grouped_old.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    new = load(NEW, "grouped_new")
    old = load(OLD, "grouped_old")
    n = 10_000_000
    rng = np.random.default_rng(0)
    for n_groups in (200_000, 10_000):
        gids = rng.integers(0, n_groups, size=n).astype(np.int64)
        vals = rng.standard_normal(n)
        # warm both
        new.per_group_shift(vals[:5000], gids[:5000], 1)
        old.per_group_shift(vals[:5000], gids[:5000], 1)
        new.per_group_cum_reduce(vals[:5000], gids[:5000], "sum")
        old.per_group_cum_reduce(vals[:5000], gids[:5000], "sum")
        tn, to = [], []
        for i in range(4):
            t0 = time.perf_counter(); rn = new.per_group_shift(vals, gids, 1); tn.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); ro = old.per_group_shift(vals, gids, 1); to.append(time.perf_counter() - t0)
        wins = sum(1 for a, b in zip(tn, to) if a < b)
        print(f"[shift n_groups={n_groups}] NEW best={min(tn):.3f} OLD best={min(to):.3f} speedup={min(to)/min(tn):.2f}x faster {wins}/4")
        # identity (shift + cum_reduce + rolling + rank)
        a = new.per_group_shift(vals, gids, 1); b = old.per_group_shift(vals, gids, 1)
        print("  shift identity:", np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9)))
        a = new.per_group_cum_reduce(vals, gids, "sum"); b = old.per_group_cum_reduce(vals, gids, "sum")
        print("  cum_reduce max|diff|:", float(np.nanmax(np.abs(a - b))))
        a = new.per_group_rolling_reduce(vals, gids, 5, "mean"); b = old.per_group_rolling_reduce(vals, gids, 5, "mean")
        print("  rolling_mean max|diff|:", float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))))
        a = new.per_group_rank(vals, gids); b = old.per_group_rank(vals, gids)
        print("  rank max|diff|:", float(np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b)))))


if __name__ == "__main__":
    main()
