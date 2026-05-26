"""Bench: feature-importance permutation strategies for a NN regressor.

Three variants on the same task:
  A) sklearn permutation_importance, n_jobs=1 (current default).
  B) sklearn permutation_importance, n_jobs=-1 (joblib loky workers).
  C) custom batched-permutation kernel -- single process, ONE predict
     call per feature with n_repeats × n_rows rows stacked, where each
     n_rows slice has column j shuffled by a distinct permutation.

Metrics: wall time, peak-process RSS delta, FI agreement (correlation
to variant A so we know B/C compute the same thing modulo noise).

Run:
    D:/ProgramData/anaconda3/python.exe profiling/bench_permutation_fi_nn.py
"""
from __future__ import annotations

import gc
import os
import time

import numpy as np
import psutil


_RNG = np.random.default_rng(20260526)


# --------------------------------------------------------------------------
# Fixture: tiny PyTorch MLP that quacks like a sklearn regressor + a
# realistic-shape synthetic regression task. Avoids the full Lightning
# stack so the bench measures the FI loop overhead, not the trainer.
# --------------------------------------------------------------------------

class _TorchMLPRegressor:
    """Minimal sklearn-style adapter around a torch.nn.Module."""

    def __init__(self, n_features: int, hidden: int = 128):
        import torch
        import torch.nn as nn
        self._torch = torch
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.net.eval()

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        self.net.train()
        for _ in range(5):
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
            # Batch huge tensors so a 200x5k forward pass doesn't blow
            # CPU cache; this mirrors what a real Lightning predict
            # loop would do (predict_batch_size resolver, etc).
            BATCH = 8192
            outs = []
            for i in range(0, X_t.shape[0], BATCH):
                outs.append(self.net(X_t[i:i + BATCH]).reshape(-1).numpy())
        return np.concatenate(outs) if outs else np.zeros(0)


def _make_task(n_rows: int, n_features: int):
    X = _RNG.standard_normal((n_rows, n_features)).astype(np.float32)
    # 10 informative features, rest noise.
    w = np.zeros(n_features, dtype=np.float32)
    w[:10] = _RNG.uniform(0.5, 2.0, size=10)
    y = X @ w + 0.1 * _RNG.standard_normal(n_rows).astype(np.float32)
    return X, y


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


# --------------------------------------------------------------------------
# Variant A + B: sklearn permutation_importance
# --------------------------------------------------------------------------

def _adaptive_r2_scorer(estimator, X, y):
    from sklearn.metrics import r2_score
    p = estimator.predict(X)
    return float(r2_score(y, p))


def variant_sklearn(model, X, y, *, n_repeats: int, n_jobs: int, seed: int = 0):
    from sklearn.inspection import permutation_importance
    t0 = time.perf_counter()
    rss0 = _rss_mb()
    result = permutation_importance(
        model, X, y,
        scoring=_adaptive_r2_scorer,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    rss1 = _rss_mb()
    return result.importances_mean, time.perf_counter() - t0, max(rss1 - rss0, 0.0)


# --------------------------------------------------------------------------
# Variant C: custom batched-permutation kernel
# --------------------------------------------------------------------------

def variant_batched(model, X, y, *, n_repeats: int, seed: int = 0):
    """One predict call per feature, with n_repeats stacked permutations.

    Memory per call: n_repeats × n_rows × n_features × dtype_size. For
    5k × 200 × 5 × 4B = 20 MB -- harmless. The big win is replacing
    n_features × n_repeats predict invocations with n_features (each
    batched n_repeats × larger). PyTorch forward time scales sub-
    linearly with batch size (constant per-call overhead), so wall
    time drops near-linearly in n_repeats.

    Invariant: at the start of every j-iteration the batched buffer
    holds the ORIGINAL X tiled n_repeats times. We permute column j
    only, predict, then restore column j to the original values before
    moving to j+1. Prior versions reset only column j between
    iterations and leaked permuted columns 0..j-1 forward, producing
    "all-features shuffled" scores instead of single-feature
    importances.
    """
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    n_rows, n_features = X.shape
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    baseline_pred = model.predict(X)
    baseline_score = r2_score(y, baseline_pred)
    importances = np.zeros(n_features, dtype=np.float64)

    # Pristine tiled buffer; column j gets temporarily permuted then
    # restored each iteration.
    batched_X = np.tile(X, (n_repeats, 1))
    y_tiled = np.tile(y, n_repeats)

    # Cache original column j contents for cheap restore.
    for j in range(n_features):
        original_col = batched_X[:, j].copy()
        for r in range(n_repeats):
            idx = rng.permutation(n_rows)
            batched_X[r * n_rows:(r + 1) * n_rows, j] = X[idx, j]
        preds = model.predict(batched_X)
        scores = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            scores[r] = r2_score(
                y_tiled[r * n_rows:(r + 1) * n_rows],
                preds[r * n_rows:(r + 1) * n_rows],
            )
        importances[j] = baseline_score - scores.mean()
        # Restore column j so next iteration starts from pristine X.
        batched_X[:, j] = original_col

    rss1 = _rss_mb()
    return importances, time.perf_counter() - t0, max(rss1 - rss0, 0.0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman to be insensitive to absolute scale (R^2 deltas can shift).
    from scipy.stats import spearmanr
    r, _ = spearmanr(a, b)
    return float(r)


def variant_thread_pool(model, X, y, *, n_repeats: int, seed: int = 0):
    """Custom permutation loop with joblib threading backend.

    PyTorch's matmul / conv ops release the GIL in their C++ kernels,
    so a thread pool runs N predicts truly in parallel WITHOUT pickling
    the model or X (no IPC overhead). Whether this wins depends on the
    BLAS configuration -- MKL / OpenBLAS already saturate cores on a
    single forward pass, so per-feature threading may not stack on top.
    """
    from joblib import Parallel, delayed
    from sklearn.metrics import r2_score

    n_rows, n_features = X.shape
    t0 = time.perf_counter()
    rss0 = _rss_mb()

    baseline_pred = model.predict(X)
    baseline_score = r2_score(y, baseline_pred)
    rng = np.random.default_rng(seed)
    feature_seeds = rng.integers(0, 2 ** 31 - 1, size=n_features)

    def _one_feature(j, seed_j):
        local_rng = np.random.default_rng(int(seed_j))
        scores = np.empty(n_repeats)
        col_orig = X[:, j].copy()
        X_local = X.copy()  # private buffer per task
        for r in range(n_repeats):
            X_local[:, j] = X[local_rng.permutation(n_rows), j]
            scores[r] = r2_score(y, model.predict(X_local))
        X_local[:, j] = col_orig
        return baseline_score - scores.mean()

    importances = np.array(
        Parallel(n_jobs=-1, backend="threading")(
            delayed(_one_feature)(j, feature_seeds[j]) for j in range(n_features)
        )
    )
    rss1 = _rss_mb()
    return importances, time.perf_counter() - t0, max(rss1 - rss0, 0.0)


def _run_scenario(n_rows: int, n_features: int, n_repeats: int, *, do_njobs: bool = False):
    print(f"\nTask: n_rows={n_rows}, n_features={n_features}, n_repeats={n_repeats}")
    X, y = _make_task(n_rows, n_features)
    model = _TorchMLPRegressor(n_features=n_features).fit(X, y)
    _ = model.predict(X[:128])  # warm

    gc.collect()
    imp_a, t_a, rss_a = variant_sklearn(model, X, y, n_repeats=n_repeats, n_jobs=1)
    print(f"  A) sklearn n_jobs=1       : {t_a:7.2f}s  RSS+{rss_a:6.1f}MB")

    if do_njobs:
        gc.collect()
        imp_b, t_b, rss_b = variant_sklearn(model, X, y, n_repeats=n_repeats, n_jobs=-1)
        print(f"  B) sklearn n_jobs=-1      : {t_b:7.2f}s  RSS+{rss_b:6.1f}MB   "
              f"speed={t_a/t_b:.2f}x   agree(A)={_agreement(imp_a, imp_b):.4f}")

    gc.collect()
    imp_c, t_c, rss_c = variant_batched(model, X, y, n_repeats=n_repeats)
    print(f"  C) batched single-process : {t_c:7.2f}s  RSS+{rss_c:6.1f}MB   "
          f"speed={t_a/t_c:.2f}x   agree(A)={_agreement(imp_a, imp_c):.4f}")

    gc.collect()
    imp_d, t_d, rss_d = variant_thread_pool(model, X, y, n_repeats=n_repeats)
    print(f"  D) thread-pool n_jobs=-1  : {t_d:7.2f}s  RSS+{rss_d:6.1f}MB   "
          f"speed={t_a/t_d:.2f}x   agree(A)={_agreement(imp_a, imp_d):.4f}")
    print(f"  top: A={sorted(np.argsort(imp_a)[-10:].tolist())} "
          f"C={sorted(np.argsort(imp_c)[-10:].tolist())} "
          f"D={sorted(np.argsort(imp_d)[-10:].tolist())}")


def main():
    # Production-shape scenario: TVT-like 206 features, 5k test rows.
    _run_scenario(n_rows=5000, n_features=200, n_repeats=5, do_njobs=True)

    # n_repeats sweep -- the larger n_repeats, the more per-call
    # overhead the batched variant amortises.
    for nr in (3, 10, 20):
        _run_scenario(n_rows=5000, n_features=200, n_repeats=nr)

    # n_features sweep -- batched call grows linearly with n_features.
    _run_scenario(n_rows=2000, n_features=500, n_repeats=5)

    # Small-test-set regime (FI on a tiny holdout).
    _run_scenario(n_rows=1000, n_features=200, n_repeats=10)


if __name__ == "__main__":
    main()
