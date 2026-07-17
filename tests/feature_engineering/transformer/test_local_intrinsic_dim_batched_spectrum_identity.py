"""Identity-pin for the local_intrinsic_dim batched-spectrum optimization.

``compute_local_intrinsic_dim_features`` was switched from a per-row Python loop
(``for q in range(n_q): cov = deviations[q].T @ deviations[q]; eigvalsh(cov)``) to a
batched form: one ``np.matmul`` building the whole (n_q, d, d) covariance stack + one
batched ``np.linalg.eigvalsh`` over the leading axis, with the spectrum math vectorized
over rows (7.3x @ d=8 .. 1.3x @ d=50; see
``feature_engineering/_benchmarks/bench_local_intrinsic_dim_batched_spectrum.py``).

The change is float32 reduction-order equivalent: the batched eigvalsh calls the SAME
LAPACK routine per (d,d) slice, and the matmul covariance equals ``dev^T @ dev``. This
test pins the public output against an independent OLD per-row reference that rebuilds the
identical neighbor deviations (deterministic RobustScaler + NearestNeighbors) and runs the
original loop -- so a future revert that breaks the numerics is caught.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.local_intrinsic_dim import (
    compute_local_intrinsic_dim_features,
)

N_FEATURES = 5


def _old_reference(X_train: np.ndarray, X_query: np.ndarray, k_neighbors: int) -> np.ndarray:
    """The pre-optimization per-row loop, reproduced from the prior source exactly."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler().fit(X_train)
    Xt_s = scaler.transform(X_train).astype(np.float32)
    Xq_s = scaler.transform(X_query).astype(np.float32)
    k_eff = min(k_neighbors, Xt_s.shape[0])
    nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
    _, idx = nn.kneighbors(Xq_s)
    neighbor_X = Xt_s[idx]
    deviations = neighbor_X - Xq_s[:, None, :]
    n_q, _, _d = deviations.shape
    out = np.zeros((n_q, N_FEATURES), dtype=np.float32)
    for q in range(n_q):
        cov = (deviations[q].T @ deviations[q]) / float(k_eff)
        lambdas = np.linalg.eigvalsh(cov)
        lambdas = np.clip(lambdas, 0.0, None) + 1e-9
        sum_l = float(lambdas.sum())
        sum_l_sq = float((lambdas**2).sum())
        out[q, 0] = (sum_l * sum_l) / sum_l_sq
        top1 = float(lambdas[-1])
        top2 = float(lambdas[-2]) if len(lambdas) >= 2 else 1e-9
        out[q, 1] = top1 / sum_l
        out[q, 2] = top2 / top1
        p = lambdas / sum_l
        spectrum_entropy = float(-np.sum(p * np.log(p + 1e-9)))
        out[q, 3] = spectrum_entropy
        out[q, 4] = float(np.exp(spectrum_entropy))
    return out


@pytest.mark.parametrize("n_train,n_query,d", [(800, 400, 8), (1200, 600, 20), (600, 300, 40)])
def test_batched_spectrum_matches_per_row_loop(n_train, n_query, d):
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    X_query = rng.standard_normal((n_query, d)).astype(np.float32)
    y_train = rng.standard_normal(n_train).astype(np.float32)

    df = compute_local_intrinsic_dim_features(
        X_train,
        y_train,
        X_query,
        seed=42,
        k_neighbors=30,
        standardize=True,
    )
    new = df.to_numpy().astype(np.float32)
    old = _old_reference(X_train, X_query, k_neighbors=30)

    max_abs = float(np.max(np.abs(old - new)))
    rel = max_abs / (float(np.max(np.abs(old))) + 1e-12)
    # float32 reduction-order equivalence: both call the same LAPACK eigvalsh per slice.
    assert rel < 1e-4, f"batched spectrum diverged from per-row loop: max_abs={max_abs:.3e} rel={rel:.3e}"
