"""Honest bench: region-adaptive transform selection vs single best global transform.

Synthetic with a region-DEPENDENT y-base relation:
    base ~ N(0,1.5)
    y = 2.0*base               + noise      where base < 0   (LINEAR regime)
    y = 1.5*base + 0.9*base^2  + noise      where base >= 0  (QUADRATIC regime)

A single global transform must compromise across the two regimes. A
region-adaptive spec can pick ``linear_residual`` in the negative region and a
curved transform (``monotonic_residual`` / ``polynomial_residual_deg2``) in the
positive region, so the residual ``T`` it hands the downstream learner carries
less base structure.

Honest scoring: fit each scheme TRAIN-ONLY, forward to ``T``, train a GBM to
predict ``T`` from a small feature matrix correlated with the noise, inverse-map
predictions back to ``y``, and report OOS RMSE on a held-out test split. Lower
RMSE wins. Run:

    cd "<repo>" && CUDA_VISIBLE_DEVICES="" python -m \\
      mlframe.training.composite.discovery._benchmarks.bench_region_adaptive
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.training.composite.discovery._region_adaptive import (
    DEFAULT_REGION_CANDIDATES,
    fit_region_adaptive,
)
from mlframe.training.composite.transforms.registry import _TRANSFORMS_REGISTRY


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, base, X) with a region-dependent y-base relation.

    ``X`` carries a feature linearly related to the noise term so a downstream
    learner has real (but imperfect) signal to recover ``T`` -- otherwise both
    schemes would tie at the irreducible-noise floor and the experiment would be
    uninformative.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.5, n)
    noise_signal = rng.normal(0.0, 1.0, n)
    noise = 0.4 * rng.normal(0.0, 1.0, n)
    y = np.where(base < 0.0, 2.0 * base, 1.5 * base + 0.9 * base * base)
    y = y + noise_signal + noise
    # X: the learnable part of the residual + 2 nuisance columns.
    X = np.column_stack([
        noise_signal + 0.1 * rng.normal(0, 1, n),
        rng.normal(0, 1, n),
        rng.normal(0, 1, n),
    ])
    return y, base, X


def _gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=120, max_depth=4, learning_rate=0.1, random_state=0)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def eval_global(y_tr, b_tr, X_tr, y_te, b_te, X_te) -> tuple[str, float]:
    """Pick the single global transform with lowest OOS RMSE (the strong baseline)."""
    best_name, best_rmse = None, np.inf
    for name in DEFAULT_REGION_CANDIDATES:
        tr = _TRANSFORMS_REGISTRY[name]
        params = tr.fit(y_tr, b_tr)
        t_tr = tr.forward(y_tr, b_tr, params)
        if not np.all(np.isfinite(t_tr)):
            continue
        m = _gbm().fit(X_tr, t_tr)
        t_hat = m.predict(X_te)
        y_hat = tr.inverse(t_hat, b_te, params)
        r = _rmse(y_hat, y_te)
        if r < best_rmse:
            best_name, best_rmse = name, r
    return best_name, best_rmse


def eval_region_adaptive(y_tr, b_tr, X_tr, y_te, b_te, X_te, k: int) -> tuple[float, tuple]:
    spec = fit_region_adaptive(y_tr, b_tr, k=k, random_state=0)
    t_tr = spec.forward(y_tr, b_tr)
    m = _gbm().fit(X_tr, t_tr)
    t_hat = m.predict(X_te)
    y_hat = spec.inverse(t_hat, b_te)
    return _rmse(y_hat, y_te), spec.region_transforms


def run(n: int = 8000, seeds: int = 5, k: int = 4) -> dict:
    g_rmses, ra_rmses = [], []
    last_global, last_regions = None, None
    t0 = time.time()
    for s in range(seeds):
        y, base, X = make_data(n, seed=100 + s)
        ntr = int(0.7 * n)
        y_tr, b_tr, X_tr = y[:ntr], base[:ntr], X[:ntr]
        y_te, b_te, X_te = y[ntr:], base[ntr:], X[ntr:]
        gname, grmse = eval_global(y_tr, b_tr, X_tr, y_te, b_te, X_te)
        rarmse, regions = eval_region_adaptive(y_tr, b_tr, X_tr, y_te, b_te, X_te, k)
        g_rmses.append(grmse)
        ra_rmses.append(rarmse)
        last_global, last_regions = gname, regions
    g_mean = float(np.mean(g_rmses))
    ra_mean = float(np.mean(ra_rmses))
    res = {
        "n": n, "seeds": seeds, "k": k,
        "global_best_transform": last_global,
        "global_rmse_mean": g_mean,
        "region_adaptive_rmse_mean": ra_mean,
        "region_transforms_lastseed": list(last_regions),
        "improvement_pct": 100.0 * (g_mean - ra_mean) / g_mean,
        "wins_of_seeds": int(np.sum(np.array(ra_rmses) < np.array(g_rmses))),
        "wall_s": round(time.time() - t0, 2),
    }
    return res


if __name__ == "__main__":
    import json

    out = run()
    print(json.dumps(out, indent=2))
