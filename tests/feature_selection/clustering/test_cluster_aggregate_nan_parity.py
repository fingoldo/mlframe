"""Wave 9.1 loop-iter-8 regression: cluster_aggregate NaN preprocessing
parity between fit and replay.

Pre-fix: ``_cluster_aggregate._continuous_cols`` applied
``nan_to_num(NaN->0)`` BEFORE computing mean/std at fit time, but
``_apply_cluster_aggregate`` in ``engineered_recipes.py`` skipped that
preprocessing on replay. Result: rows with NaN in any member column
got two different values - fit recorded ``((0 - mean) / std * sign) @
weights`` (a specific number) while replay produced
``(NaN - mean) / std @ weights -> NaN -> nan_to_num -> 0.0``.

Effect: train/test parity broken whenever the train fold itself
contained NaN (the very common imputation-pipeline case). Also affects
the in-fit gate at ``_cluster_aggregate.py:286`` which uses
``_apply_cluster_aggregate`` to score the aggregate's MI vs y, biasing
acceptance downward.

Fix: ``_apply_cluster_aggregate`` now applies the same ``nan_to_num``
wrap at column extraction, matching the fit-time path. Train/replay
become bit-identical row-for-row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_cluster_aggregate_replay_matches_fit_on_nan_rows():
    """Pre-fix: NaN rows got different values at fit vs replay.
    Post-fix: train and replay agree bit-for-bit.
    """
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _continuous_cols, _standardize_align, _derive_weights,
    )
    from mlframe.feature_selection.filters.engineered_recipes import (
        EngineeredRecipe, _apply_cluster_aggregate,
    )

    rng = np.random.default_rng(0)
    n = 100
    latent = rng.standard_normal(n)
    m0 = latent + 0.1 * rng.standard_normal(n)
    m1 = latent + 0.1 * rng.standard_normal(n)
    m2 = latent + 0.1 * rng.standard_normal(n)
    # Inject NaN at several rows in m0 - the canonical real-world pattern.
    nan_rows = [5, 10, 17, 33, 84]
    m0[nan_rows] = np.nan
    X = pd.DataFrame({"m0": m0, "m1": m1, "m2": m2})

    M_fit = _continuous_cols(X, ["m0", "m1", "m2"])
    Z_fit, mean, std, signs = _standardize_align(M_fit, ref_col=0)
    weights = _derive_weights(Z_fit, "pca_pc1")
    agg_fit = Z_fit @ weights

    recipe = EngineeredRecipe(
        kind="cluster_aggregate", name="test",
        src_names=["m0", "m1", "m2"],
        extra={
            "method": "pca_pc1",
            "member_mean": mean.tolist(),
            "member_std": std.tolist(),
            "signs": signs.tolist(),
            "weights": weights.tolist(),
        },
        quantization=None,
    )
    agg_replay = _apply_cluster_aggregate(recipe, X)

    # Bit-identical row-for-row including the NaN rows.
    assert np.allclose(agg_fit, agg_replay, atol=1e-12), (
        f"fit/replay parity broken. Max |diff|={np.max(np.abs(agg_fit-agg_replay)):.6f}"
    )
    # Specifically the NaN rows: each must produce the fit-time value
    # (not the post-fix-trivial zero produced by the pre-fix path).
    for r in nan_rows:
        assert abs(agg_fit[r] - agg_replay[r]) < 1e-12, (
            f"row {r} (NaN in m0): fit={agg_fit[r]:.6f} != replay={agg_replay[r]:.6f}"
        )


def test_cluster_aggregate_replay_nan_free_data_unchanged():
    """Negative-control: data without NaN must replay identically
    pre- and post-fix.
    """
    from mlframe.feature_selection.filters._cluster_aggregate import (
        _continuous_cols, _standardize_align, _derive_weights,
    )
    from mlframe.feature_selection.filters.engineered_recipes import (
        EngineeredRecipe, _apply_cluster_aggregate,
    )

    rng = np.random.default_rng(1)
    n = 100
    latent = rng.standard_normal(n)
    X = pd.DataFrame({
        "m0": latent + 0.1 * rng.standard_normal(n),
        "m1": latent + 0.1 * rng.standard_normal(n),
        "m2": latent + 0.1 * rng.standard_normal(n),
    })

    M_fit = _continuous_cols(X, ["m0", "m1", "m2"])
    Z_fit, mean, std, signs = _standardize_align(M_fit, ref_col=0)
    weights = _derive_weights(Z_fit, "pca_pc1")
    agg_fit = Z_fit @ weights

    recipe = EngineeredRecipe(
        kind="cluster_aggregate", name="test",
        src_names=["m0", "m1", "m2"],
        extra={
            "method": "pca_pc1",
            "member_mean": mean.tolist(),
            "member_std": std.tolist(),
            "signs": signs.tolist(),
            "weights": weights.tolist(),
        },
        quantization=None,
    )
    agg_replay = _apply_cluster_aggregate(recipe, X)
    assert np.allclose(agg_fit, agg_replay, atol=1e-12)
