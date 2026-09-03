"""FE_TRANSFORMER_A-3 (2026-08-05 audit): ``compute_band_conditional_anchor_features``'s per-query softmax
scoring used to materialize a naive ``(n_query, n_anchors_total, d)`` broadcast-cube temporary
(``Xq_s[:, None, :] - all_anchors[None, :, :]`` then ``.sum(axis=-1)``) instead of the shared GEMM-
decomposition helper (``_squared_dists_shared.squared_dists``) already used by sibling files in the same
package -- a real OOM risk at this codebase's documented production scale (``d <= 32768``, 100+GB frames).
"""

from __future__ import annotations

import tracemalloc

import numpy as np

from mlframe.feature_engineering.transformer.band_conditional_anchor import compute_band_conditional_anchor_features


def test_band_conditional_anchor_peak_memory_does_not_scale_with_naive_cube():
    """At d=8192 (well above the _utils.py-documented d>16 OOM-onset threshold), the naive broadcast
    cube for 200 query rows x 20 anchors x 8192 dims would be ~131 MB (200*20*8192*4 bytes) of pure
    temporary allocation that is never even the return value. The GEMM-decomposition path's peak is a
    few hundred KB (the (200, 8192) query matrix, the (20, 8192) anchor matrix, and the (200, 20) output).
    """
    rng = np.random.default_rng(0)
    n_train, n_query, d = 300, 200, 8192
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = rng.standard_normal(n_train).astype(np.float32)
    X_query = rng.standard_normal((n_query, d)).astype(np.float32)

    tracemalloc.start()
    try:
        out = compute_band_conditional_anchor_features(
            X_train, y_train, X_query, seed=0, task="regression", n_bands=5, anchors_per_band=4, standardize=False,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert out.shape[0] == n_query
    peak_mb = peak / (1 << 20)
    assert peak_mb < 20, f"peak traced memory {peak_mb:.1f} MiB -- expected well under the ~125 MiB naive-cube size (GEMM path should stay under 20 MiB)"
