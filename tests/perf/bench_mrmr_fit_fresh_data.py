"""iter53 ad-hoc profile harness: fresh-data MRMR.fit at n=12000, FE default-on.

Pre-warms numba JIT + kernel_tuning_cache at the same size on throwaway data,
then profiles a FRESH-seed fit so the content-fingerprint replay cache is bypassed.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _make_mrmr(**overrides):
    """Returns ``MRMR(**kwargs)`` (after 3 setup steps)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(verbose=0, random_seed=0)
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _build(seed: int, n: int = 12000):
    """Builds seeded synthetic test data; returns ``(X, pd.Series(y, name='y'))``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=n)
    cols = {"a": x}
    for i in range(12):
        cols[f"n{i}"] = rng.standard_normal(n)
    # signal cols
    cols["s1"] = np.sign(rng.standard_normal(n)) * np.abs(x) + 0.1 * rng.standard_normal(n)
    cols["s2"] = np.sin(3.7 * x) + 0.05 * rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = np.sin(3.7 * x) + np.sin(5.3 * x) + np.abs(x) + 0.05 * rng.standard_normal(n)
    return X, pd.Series(y, name="y")


def run_fit(seed):
    """Returns ``m`` (after 3 setup steps)."""
    X, y = _build(seed)
    m = _make_mrmr()
    m.fit(X, y)
    return m


def main():
    # warm
    """Test helper: run_fit(99); run_fit(98); pr = cProfile.Profile()."""
    run_fit(99)
    run_fit(98)
    pr = cProfile.Profile()
    pr.enable()
    run_fit(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("tottime")
    st.print_stats(60)


if __name__ == "__main__":
    main()
