"""Profile per_group_shift / per_group_cum_reduce @10M with many groups."""
import sys
sys.modules['cupy'] = None
import scipy.stats  # noqa
import numba  # noqa
import cProfile, pstats, io, time
import numpy as np


def main():
    from mlframe.feature_engineering.grouped import per_group_shift, per_group_cum_reduce
    n = 10_000_000
    rng = np.random.default_rng(0)
    n_groups = 200_000  # ~50 rows/group
    gids = rng.integers(0, n_groups, size=n).astype(np.int64)
    vals = rng.standard_normal(n)
    # warm
    per_group_shift(vals[:10000], gids[:10000], 1)
    per_group_cum_reduce(vals[:10000], gids[:10000], "sum")

    for fn, args, name in [
        (per_group_shift, (vals, gids, 1), "per_group_shift"),
        (per_group_cum_reduce, (vals, gids, "sum"), "per_group_cum_reduce"),
    ]:
        pr = cProfile.Profile()
        t0 = time.perf_counter(); pr.enable()
        fn(*args)
        pr.disable(); wall = time.perf_counter() - t0
        print(f"=== {name} wall={wall:.3f}s (n_groups={n_groups}) ===")
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8)
        print("\n".join(s.getvalue().splitlines()[:16]))


if __name__ == "__main__":
    main()
