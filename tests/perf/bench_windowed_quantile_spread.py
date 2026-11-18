"""Profile of ``rolling_quantile_spread`` on 10 million values (5 groups of 2 million rows): a script, run as ``python tests/perf/bench_windowed_quantile_spread.py``."""

import cProfile
import pstats
import sys

import numpy as np

N = 10_000_000


def main() -> None:
    """Profile the rolling quantile spread with cupy blocked for the duration of the run, and put the import table back afterwards."""
    previous = sys.modules.get("cupy", sys.modules)  # the sentinel object means "was not imported"
    sys.modules["cupy"] = None  # type: ignore[assignment]
    try:
        from mlframe.feature_engineering.windowed_shape import rolling_quantile_spread

        rng = np.random.default_rng(0)
        vals = rng.standard_normal(N).astype(np.float64)
        gids = (np.arange(N) // 2_000_000).astype(np.int64)  # few groups, large segments: about N windows
        rolling_quantile_spread(vals[:1000], gids[:1000], window_K=20)  # warm the compiled kernel
        pr = cProfile.Profile()
        pr.enable()
        result = rolling_quantile_spread(vals, gids, window_K=20)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(20)
        print("checksum", float(np.nansum(result)))
    finally:
        if previous is sys.modules:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous


if __name__ == "__main__":
    main()
