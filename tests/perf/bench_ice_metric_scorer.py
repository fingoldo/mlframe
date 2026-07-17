"""iter68 profiling harness: compute_probabilistic_multiclass_error scorer hot path.

This is the ICE scorer called on every CV-fold / model eval. Profiles the binary (K=1) and
multiclass (K=3/5) fastpath to find plain-Python orchestration overhead around the batched kernel.
"""

import cProfile
import pstats
import io
import numpy as np

from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error


def _make_binary(n, seed=0):
    """Helper that make binary."""
    rng = np.random.default_rng(seed)
    yt = (rng.random(n) < 0.3).astype(np.int64)
    p = rng.random(n)
    p = np.clip(p * 0.6 + yt * 0.3, 0, 1)
    return yt, p


def _make_multiclass(n, k, seed=0):
    """Helper that make multiclass."""
    rng = np.random.default_rng(seed)
    yt = rng.integers(0, k, size=n).astype(np.int64)
    score = rng.random((n, k))
    score = score / score.sum(axis=1, keepdims=True)
    return yt, score


def run(reps=4000):
    """Helper that run."""
    yt_b, p_b = _make_binary(20000)
    yt_m, s_m = _make_multiclass(20000, 3)
    # warm numba
    compute_probabilistic_multiclass_error(yt_b, p_b)
    compute_probabilistic_multiclass_error(yt_m, s_m)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(reps):
        compute_probabilistic_multiclass_error(yt_b, p_b)
        compute_probabilistic_multiclass_error(yt_m, s_m)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(35)
    print(s.getvalue())


if __name__ == "__main__":
    run()
