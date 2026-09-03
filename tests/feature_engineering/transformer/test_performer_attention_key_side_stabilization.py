"""FE_TRANSFORMER_B-1 (2026-08-05 audit): ``_performer_features``'s numerical-stability shift subtracted
a PER-ROW max (the standard softmax-stabilization idiom) before ``exp()``. This is safe for genuine
row-wise softmax (normalization happens along the same axis, so any per-row constant cancels), but
Performer linear attention is structurally different: the key/training-side features are SUMMED ACROSS
ROWS into ``A = phi_k.T @ y_t`` / ``B = phi_k.sum(axis=0)``, so a per-row shift multiplies each training
row's contribution to that sum by an arbitrary, row-specific, target-unrelated factor -- silently
corrupting the kernel-weighted aggregate for every query. A single GLOBAL scalar shift factors out of
A/B uniformly and cancels exactly in the final ``numer/denom`` ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.performer_attention import compute_performer_attention_features


def _exact_performer_reference(X_train, y_train, X_query, seed, n_features):
    """Brute-force, UNSTABILIZED Performer linear-attention estimate computed directly in float64 (safe
    on this small, controlled fixture, no overflow risk) -- the ground truth _performer_features must
    match up to a numerical-stabilization-only tolerance."""
    d_input = X_train.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_input, n_features)) / np.sqrt(d_input)

    def phi(X):
        """Unstabilized Performer random-feature map exp(X@W - 0.5*||x||^2)."""
        x_norm_sq = (X**2).sum(axis=1, keepdims=True)
        return np.exp(X @ W - 0.5 * x_norm_sq)

    phi_k = phi(X_train.astype(np.float64))
    phi_q = phi(X_query.astype(np.float64))
    A = phi_k.T @ y_train.astype(np.float64)
    B = phi_k.sum(axis=0)
    numer = phi_q @ A
    denom = phi_q @ B + 1e-9
    return numer / denom


def test_performer_attention_matches_exact_kernel_on_varying_row_norms():
    """With training rows of deliberately VARYING norms (essential: equal-norm rows make the per-row max
    happen to be constant across rows, masking the bug), the production y_estimate must match the exact
    (unstabilized) Performer reference to a tight, stabilization-only tolerance -- not the ~30-40%
    relative corruption the per-row-shift bug produced."""
    rng = np.random.default_rng(0)
    n_train, n_query, d, M, seed = 40, 5, 5, 64, 7

    X_train = rng.standard_normal((n_train, d))
    scales = rng.uniform(0.3, 3.0, size=n_train)  # deliberately varying per-row norms
    X_train = X_train * scales[:, None]
    y_train = rng.standard_normal(n_train)
    X_query = rng.standard_normal((n_query, d))

    ref = _exact_performer_reference(X_train, y_train, X_query, seed, M)

    out = compute_performer_attention_features(
        X_train.astype(np.float32), y_train.astype(np.float32), X_query.astype(np.float32),
        seed=seed, n_features=M, standardize=False, column_prefix="perfattn",
    )
    got = out["perfattn_y_est"].to_numpy()

    max_abs_diff = float(np.max(np.abs(got - ref)))
    assert max_abs_diff < 1e-3, f"expected the production y_estimate to match the exact Performer reference (stabilization-only gap), got max_abs_diff={max_abs_diff:.4f}"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_performer_attention_query_side_shift_choice_is_close_regardless(seed):
    """Once the key/training side uses a global shift, the query-side shift choice (per-row vs. global)
    is mathematically exact-cancelling in each row's own numer/denom ratio -- pins that the two choices
    stay CLOSE (a generous floating-point tolerance, since a query row whose own max diverges sharply
    from the global max can lose some relative precision in finite arithmetic even though the two forms
    are algebraically identical), so a future regression that reintroduces a much LARGER divergence
    (the ~30-40% relative corruption class the fix addresses) is still caught."""
    rng = np.random.default_rng(seed)
    n_train, n_query, d, M = 30, 6, 4, 32

    X_train = rng.standard_normal((n_train, d)) * rng.uniform(0.3, 3.0, size=n_train)[:, None]
    y_train = rng.standard_normal(n_train)
    X_query = rng.standard_normal((n_query, d)) * rng.uniform(0.3, 3.0, size=n_query)[:, None]

    d_input = d
    W = np.random.default_rng(seed).standard_normal((d_input, M)) / np.sqrt(d_input)

    def phi_global_shift(X):
        """Performer feature map stabilized by a single global scalar shift (the fix)."""
        x_norm_sq = (X**2).sum(axis=1, keepdims=True)
        log_phi = X @ W - 0.5 * x_norm_sq
        log_phi = log_phi - log_phi.max()
        return np.exp(log_phi)

    def phi_per_row_shift(X):
        """Performer feature map stabilized by a per-row max shift (the pre-fix idiom)."""
        x_norm_sq = (X**2).sum(axis=1, keepdims=True)
        log_phi = X @ W - 0.5 * x_norm_sq
        log_phi = log_phi - log_phi.max(axis=1, keepdims=True)
        return np.exp(log_phi)

    phi_k = phi_global_shift(X_train)  # key side: global shift (the fix)
    A = phi_k.T @ y_train
    B = phi_k.sum(axis=0)

    y_est_global_q = (phi_global_shift(X_query) @ A) / (phi_global_shift(X_query) @ B + 1e-9)
    y_est_perrow_q = (phi_per_row_shift(X_query) @ A) / (phi_per_row_shift(X_query) @ B + 1e-9)

    assert np.allclose(
        y_est_global_q, y_est_perrow_q, atol=0.1
    ), f"query-side shift choice diverged well beyond floating-point noise: global={y_est_global_q} per_row={y_est_perrow_q}"
