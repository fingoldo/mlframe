"""iter91 cProfile harness: composite-target discovery on a REGRESSION target at 1M rows.

Drives ``CompositeTargetDiscovery.fit`` with the suite-default regression discovery combo
(mi_estimator='bin', screening='hybrid', multi_base + stacked-residual surfaces on by default)
from a 1M-row synthetic regression frame. The discovery internally subsamples the MI screen
(mi_sample_n=100k) and the tiny-model rerank (tiny_model_sample_n=20k) -- this harness drives
those at their real working sizes via the 1M input, matching production.

Run:
    MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python -m mlframe.training.composite.discovery._benchmarks.bench_iter91_regression_discovery_1m [n]
"""
from __future__ import annotations

import sys

# py3.14 native-segfault workaround for cold mlframe.training.core / metrics.core import.
sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402
import numba  # noqa: F401,E402

import cProfile  # noqa: E402
import io  # noqa: E402
import pstats  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def synth_regression(n: int, seed: int = 0):
    """Realistic regression mix: ~30 numeric features (incl. an AR-style lag base and
    a couple correlated siblings) + 2 low-cardinality categorical (integer-coded) features."""
    rng = np.random.default_rng(seed)
    cols = {}
    # Dominant AR-style lag base + correlated siblings.
    base = rng.normal(0.0, 1.0, n).cumsum() / np.sqrt(n)
    cols["lag1"] = base
    cols["lag2"] = np.roll(base, 1) + rng.normal(0, 0.05, n)
    cols["smooth3"] = base + rng.normal(0, 0.1, n)
    # Numeric noise + signal features.
    for j in range(25):
        cols[f"f{j}"] = rng.normal(0.0, 1.0, n)
    # Two low-card integer categoricals.
    cols["cat_a"] = rng.integers(0, 5, n).astype(np.float64)
    cols["cat_b"] = rng.integers(0, 12, n).astype(np.float64)
    X = pd.DataFrame(cols)
    y = base + 0.5 * cols["f0"] + 0.3 * cols["f1"] + 0.2 * cols["cat_a"] + rng.normal(0.0, 0.3, n)
    feature_cols = list(cols.keys())
    return X.assign(y=y), feature_cols


def make_discovery():
    from ....configs import CompositeTargetDiscoveryConfig
    from .. import CompositeTargetDiscovery
    cfg = CompositeTargetDiscoveryConfig(enabled=True)  # suite-default regression combo
    return CompositeTargetDiscovery(cfg)


def run_fit(Xy, feature_cols, n):
    disco = make_discovery()
    train_idx = np.arange(n)
    return disco.fit(Xy, "y", feature_cols, train_idx)


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1_000_000
    Xy, feature_cols = synth_regression(n)
    # Warm numba kernels + JIT with a small pre-run (excluded from profile).
    Xy_w, fc_w = synth_regression(2000, seed=1)
    run_fit(Xy_w, fc_w, 2000)

    pr = cProfile.Profile()
    pr.enable()
    disco = run_fit(Xy, feature_cols, n)
    pr.disable()
    print(f"discovery @ n={n}: {len(disco.specs_)} specs")
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("tottime").print_stats(40)
    out = s.getvalue()
    # Filter to mlframe-own frames for the top-20 view.
    print("\n===== TOP by tottime (all) =====")
    print("\n".join(out.splitlines()[:50]))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
