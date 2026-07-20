"""cProfile + wall-clock harness for gt_06's ``dro_reweight_fit`` (chi2-ball DRO game).

Run: python -m mlframe.data_valuation._benchmarks.profile_adversarial_reweighting

Measures ``dro_reweight_fit`` wall vs ``n_rounds`` at ``n`` in {5000, 50000} (plan section 5's grid).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def _make_fixture(n: int, n_features: int = 8, seed: int = 0):
    """Synthetic binary-classification fixture of the given (n, n_features) shape."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


def _fit_fn(X, y, w):
    """fit_fn for dro_reweight_fit: a small xgboost classifier respecting sample_weight."""
    from xgboost import XGBClassifier

    m = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", n_jobs=1, random_state=0)
    m.fit(X, y, sample_weight=w)
    return m


def _loss_fn(y, pred):
    """Per-row log-loss."""
    eps = 1e-7
    p = np.clip(pred, eps, 1.0 - eps)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def bench_wall_vs_rounds():
    """Wall-clock dro_reweight_fit at n in {5000, 50000} across n_rounds in {2, 4, 8}."""
    from mlframe.data_valuation._adversarial_reweighting import dro_reweight_fit

    print("n, n_rounds, wall_s")
    for n in (5000, 50000):
        X, y = _make_fixture(n)
        for n_rounds in (2, 4, 8):
            t0 = time.perf_counter()
            dro_reweight_fit(_fit_fn, _loss_fn, X, y, rho=0.5, n_rounds=n_rounds, n_splits=5, rng=np.random.default_rng(0))
            wall = time.perf_counter() - t0
            print(f"{n}, {n_rounds}, {wall:.4f}")


def bench_cprofile():
    """cProfile a mid-size dro_reweight_fit call, print top-25 by cumtime."""
    from mlframe.data_valuation._adversarial_reweighting import dro_reweight_fit

    X, y = _make_fixture(10000)
    pr = cProfile.Profile()
    pr.enable()
    dro_reweight_fit(_fit_fn, _loss_fn, X, y, rho=0.5, n_rounds=4, n_splits=5, rng=np.random.default_rng(0))
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    bench_wall_vs_rounds()
    bench_cprofile()
