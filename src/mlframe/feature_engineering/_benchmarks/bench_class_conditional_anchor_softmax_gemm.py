"""Bench: squared-euclidean broadcast temporary vs GEMM decomposition in class_conditional_anchor.

``_softmax_similarity`` and the unified-mass block in ``_compute_class_anchor_features`` build a
(n_rows, n_anchors, d) broadcast temporary ``(X[:, None, :] - anchors[None, :, :]) ** 2`` then sum over
axis=2. The same squared distances follow from ``||x||^2 - 2 x.a + ||a||^2`` via one GEMM (X @ anchors.T),
allocating only (n_rows, n_anchors) instead of the 3D cube.

Shapes mirror Mode B at the documented N<10k scale (mammography ~4000 rows, d~30, K=16/class, 2K=32 for the
unified mass block). Reports broadcast-vs-GEMM wall (best-of-N, warm) and the max |softmax delta| so the
selection-equivalence claim is grounded.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_engineering._benchmarks._gemm_softmax_bench_shared import (
    best_of as _best_of,
    dists_broadcast as _dists_broadcast,
    dists_gemm as _dists_gemm,
    softmax_from_dists as _softmax_from_dists,
)


def main():
    rng = np.random.default_rng(0)
    temp = 1.0
    for n_rows, d, n_anchors in [(4000, 30, 16), (4000, 30, 32), (8000, 50, 32), (4000, 100, 16)]:
        X = rng.standard_normal((n_rows, d)).astype(np.float32)
        anchors = rng.standard_normal((n_anchors, d)).astype(np.float32)
        # warm
        _dists_broadcast(X, anchors)
        _dists_gemm(X, anchors)
        tb = _best_of(_dists_broadcast, X, anchors)
        tg = _best_of(_dists_gemm, X, anchors)
        sb = _softmax_from_dists(_dists_broadcast(X, anchors), temp)
        sg = _softmax_from_dists(_dists_gemm(X, anchors), temp)
        max_delta = float(np.max(np.abs(sb - sg)))
        print(f"n={n_rows} d={d} K={n_anchors}: broadcast={tb*1e3:.3f}ms gemm={tg*1e3:.3f}ms " f"speedup={tb/tg:.2f}x  max|softmax delta|={max_delta:.2e}")


if __name__ == "__main__":
    main()
