import sys, time

sys.modules["cupy"] = None
import scipy.stats, numba  # noqa
import numpy as np
from mlframe.feature_engineering.windowed_shape import rolling_quantile_spread

N = 10_000_000
rng = np.random.default_rng(0)
# few groups, large segments -> ~N windows
vals = rng.standard_normal(N).astype(np.float64)
gids = (np.arange(N) // 2_000_000).astype(np.int64)  # 5 groups

# warm
_ = rolling_quantile_spread(vals[:1000], gids[:1000], window_K=20)

import cProfile, pstats

pr = cProfile.Profile()
pr.enable()
r = rolling_quantile_spread(vals, gids, window_K=20)
pr.disable()
st = pstats.Stats(pr).sort_stats("tottime")
st.print_stats(20)
print("checksum", float(np.nansum(r)))
