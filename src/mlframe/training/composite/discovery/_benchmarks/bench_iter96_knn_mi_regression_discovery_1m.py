"""iter96 cProfile harness: composite-target discovery on a REGRESSION target at 1M rows,
VARIED CONFIG to escape the iter91-95 saturated default.

DIFFERS from the saturated default (``bench_iter91_regression_discovery_1m.py``) on:
  - ``mi_estimator='knn'`` (Kraskov kNN MI -- completely different code from the binned path
    that iters 76/91-94 mined; routes through the ``_mi_to_target`` knn branch + ``_auto_base``
    knn branch + ``_eval`` per-spec MI on the kNN estimator).
  - ``screening='mi'`` (MI-only -- removes the LightGBM tiny-model Phase B that made the default
    path ~73% external-LightGBM-bound, so the mlframe-own MI screening frames dominate instead).
  - heavier feature mix (50 numeric + 4 low-card categoricals) + more candidate bases
    (``auto_base_top_k=6``) so the per-feature MI sweep + dedup frames dominate differently.
  - ``auto_base_null_perms=20`` kept (the kNN null-permutation path is now real cost, not bin).

Run:
    MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python -m mlframe.training.composite.discovery._benchmarks.bench_iter96_knn_mi_regression_discovery_1m [n]
"""
from __future__ import annotations

import sys

sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402
import numba  # noqa: F401,E402

import cProfile  # noqa: E402
import io  # noqa: E402
import pstats  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def synth_regression(n: int, seed: int = 0):
    """Heavier regression mix: ~50 numeric features (incl. an AR-style lag base and
    correlated siblings) + 4 low-cardinality categorical (integer-coded) features."""
    rng = np.random.default_rng(seed)
    cols = {}
    base = rng.normal(0.0, 1.0, n).cumsum() / np.sqrt(n)
    cols["lag1"] = base
    cols["lag2"] = np.roll(base, 1) + rng.normal(0, 0.05, n)
    cols["smooth3"] = base + rng.normal(0, 0.1, n)
    for j in range(50):
        cols[f"f{j}"] = rng.normal(0.0, 1.0, n)
    cols["cat_a"] = rng.integers(0, 5, n).astype(np.float64)
    cols["cat_b"] = rng.integers(0, 12, n).astype(np.float64)
    cols["cat_c"] = rng.integers(0, 8, n).astype(np.float64)
    cols["cat_d"] = rng.integers(0, 3, n).astype(np.float64)
    X = pd.DataFrame(cols)
    y = base + 0.5 * cols["f0"] + 0.3 * cols["f1"] + 0.2 * cols["cat_a"] + rng.normal(0.0, 0.3, n)
    feature_cols = list(cols.keys())
    return X.assign(y=y), feature_cols


def make_discovery():
    from ....configs import CompositeTargetDiscoveryConfig
    from .. import CompositeTargetDiscovery
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        mi_estimator="knn",
        screening="mi",
        auto_base_top_k=6,
    )
    return CompositeTargetDiscovery(cfg)


def run_fit(Xy, feature_cols, n):
    disco = make_discovery()
    train_idx = np.arange(n)
    return disco.fit(Xy, "y", feature_cols, train_idx)


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 1_000_000
    Xy, feature_cols = synth_regression(n)
    Xy_w, fc_w = synth_regression(2000, seed=1)
    run_fit(Xy_w, fc_w, 2000)

    pr = cProfile.Profile()
    pr.enable()
    disco = run_fit(Xy, feature_cols, n)
    pr.disable()
    print(f"discovery @ n={n}: {len(disco.specs_)} specs")
    print("specs:", [s.spec_id if hasattr(s, "spec_id") else str(s) for s in disco.specs_][:20])
    s = io.StringIO()
    st = pstats.Stats(pr, stream=s)
    st.sort_stats("tottime").print_stats(45)
    out = s.getvalue()
    print("\n===== TOP by tottime (all) =====")
    print("\n".join(out.splitlines()[:60]))


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
