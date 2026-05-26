"""Bench: native FI methods for PyTorch MLP regressors.

Compares three approaches that can extract per-input importance from
a trained MLP WITHOUT the sklearn permutation_importance fallback:

  A) ``permutation_importance`` (sklearn, threading backend) - baseline
     correctness + timing reference. Always available.
  B) First-layer |W| magnitude proxy: |W1|.mean(axis=hidden) summed
     across all input rows. O(input * hidden_1) one-shot, no X/y,
     no predict calls. Requires the input column to feed directly
     into the first Linear layer (BatchNorm in front is fine because
     BN does NOT mix features; it normalises per-feature stats).
  C) Captum IntegratedGradients: per-row attribution via integrated
     gradient along a baseline-to-input path. Optional dep (captum).
     Aggregates absolute attribution across rows.

Metrics: wall time, agreement (Spearman) with A on the informative-
feature ranking. Score on a synthetic task with KNOWN ground truth
(first 10 features are informative, rest are noise).

Run:
    D:/ProgramData/anaconda3/python.exe profiling/bench_mlp_fi_methods.py
"""
from __future__ import annotations

import gc
import time

import numpy as np


_RNG = np.random.default_rng(20260526)


class _TorchMLPRegressor:
    """Same minimal stand-in as bench_permutation_fi_nn.py -- a
    BN + LeakyReLU + Linear stack quacking as a sklearn regressor."""

    def __init__(self, n_features: int, hidden: int = 128):
        import torch
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.net.eval()

    def fit(self, X, y, epochs: int = 30):
        import torch
        import torch.nn as nn
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        self.net.train()
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.net(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()
        self.net.eval()
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            return self.net(X_t).reshape(-1).numpy()


def _make_task(n_rows: int, n_features: int):
    X = _RNG.standard_normal((n_rows, n_features)).astype(np.float32)
    w = np.zeros(n_features, dtype=np.float32)
    w[:10] = _RNG.uniform(0.5, 2.0, size=10)
    y = X @ w + 0.1 * _RNG.standard_normal(n_rows).astype(np.float32)
    return X, y, w


# --------------------------------------------------------------------------
# A) Permutation (threading)
# --------------------------------------------------------------------------

def _r2_scorer(estimator, X, y):
    from sklearn.metrics import r2_score
    return float(r2_score(y, estimator.predict(X)))


def variant_permutation(model, X, y, n_repeats: int = 3):
    import joblib
    from sklearn.inspection import permutation_importance
    t0 = time.perf_counter()
    with joblib.parallel_backend("threading", n_jobs=-1):
        result = permutation_importance(
            model, X, y, scoring=_r2_scorer,
            n_repeats=n_repeats, random_state=0, n_jobs=-1,
        )
    return result.importances_mean, time.perf_counter() - t0


# --------------------------------------------------------------------------
# B) First-layer |W| magnitude proxy
# --------------------------------------------------------------------------

def extract_first_layer_importance(net) -> np.ndarray | None:
    """Locate the first nn.Linear inside a Sequential / Module and
    return ``|W|.sum(axis=hidden)`` -- one importance score per input
    feature. Skips leading BatchNorm / LayerNorm / Dropout layers
    (they don't mix features so they preserve the per-input mapping).
    Returns None when no Linear is found OR when the first Linear's
    in_features doesn't match the expected input count (Pipeline
    Polynomial / TargetEncoder layers ahead would break the mapping).
    """
    import torch.nn as nn
    if isinstance(net, nn.Sequential):
        layers = list(net.children())
    else:
        layers = list(net.modules())
    for layer in layers:
        if isinstance(layer, (nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, nn.Identity)):
            continue
        if isinstance(layer, nn.Linear):
            with_grad_off = layer.weight.detach().abs().cpu().numpy()
            # ``weight`` shape: (out_features, in_features).
            return with_grad_off.sum(axis=0).astype(np.float64)
        # Any non-linear / non-norm op breaks the per-feature mapping.
        if hasattr(layer, "weight"):
            break
    return None


def variant_first_layer_w(model, X, y):  # noqa: ARG001
    t0 = time.perf_counter()
    imp = extract_first_layer_importance(model.net)
    return imp, time.perf_counter() - t0


# --------------------------------------------------------------------------
# C) Captum IntegratedGradients
# --------------------------------------------------------------------------

def variant_captum_ig(model, X, y, n_samples: int = 500):  # noqa: ARG001
    import torch
    from captum.attr import IntegratedGradients
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    if X.shape[0] > n_samples:
        idx = rng.choice(X.shape[0], size=n_samples, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    model.net.eval()
    ig = IntegratedGradients(model.net)
    X_t = torch.as_tensor(np.asarray(X_sub), dtype=torch.float32)
    baseline = torch.zeros_like(X_t)
    attrs = ig.attribute(X_t, baselines=baseline, n_steps=20)
    importances = attrs.detach().abs().mean(axis=0).cpu().numpy().astype(np.float64)
    return importances, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def _top_recall(imp: np.ndarray, k: int = 10) -> float:
    """Fraction of true informative (indices 0..k-1) recovered in top-k."""
    top_k = np.argsort(imp)[-k:]
    return float(np.sum(top_k < k)) / k


def _run_scenario(n_rows: int, n_features: int):
    print(f"\nTask: n_rows={n_rows}, n_features={n_features}, truth = first 10 features")
    X, y, _ = _make_task(n_rows, n_features)
    model = _TorchMLPRegressor(n_features=n_features).fit(X, y)
    print(f"  train RMSE: {np.sqrt(((model.predict(X) - y) ** 2).mean()):.4f}")

    gc.collect()
    imp_a, t_a = variant_permutation(model, X, y, n_repeats=3)
    print(f"  A) permutation (threading)  : {t_a:7.3f}s  recall@10={_top_recall(imp_a):.2f}")

    gc.collect()
    imp_b, t_b = variant_first_layer_w(model, X, y)
    if imp_b is not None:
        print(f"  B) first-layer |W|          : {t_b:7.3f}s  recall@10={_top_recall(imp_b):.2f}  "
              f"speed={t_a/t_b:.1f}x  agree(A)={_spearman(imp_a, imp_b):.3f}")
    else:
        print("  B) first-layer |W|          : extraction failed")

    gc.collect()
    imp_c, t_c = variant_captum_ig(model, X, y, n_samples=500)
    print(f"  C) Captum IG (n_samples=500): {t_c:7.3f}s  recall@10={_top_recall(imp_c):.2f}  "
          f"speed={t_a/t_c:.1f}x  agree(A)={_spearman(imp_a, imp_c):.3f}")


def main():
    _run_scenario(n_rows=5000, n_features=200)
    _run_scenario(n_rows=2000, n_features=500)
    _run_scenario(n_rows=1000, n_features=50)


if __name__ == "__main__":
    main()
