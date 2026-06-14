"""iter96 A/B: per-base knn ``mi_y`` baseline -- OLD per-base ``_mi_to_target`` recompute vs NEW compute-once.

On the knn path (``mi_estimator='knn'``) the per-base ``mi_y_for_base`` baseline = ``MI(y, X_remaining)`` was
re-run for EVERY base candidate (``_fit.py`` per-base loop), each call invoking sklearn's Kraskov
``mutual_info_regression`` once per remaining feature column (~50 columns at ~0.45s each on the 100k MI screen).
Since per-column ``MI(y, x_j)`` is base-INVARIANT, this is ``n_bases`` redundant sweeps of the same columns.

NEW: ``_mi_per_feature_knn`` computes the per-column vector ONCE; each base's baseline is aggregated over its
surviving (base-dropped, dedup-kept) original-column indices via ``_aggregate_mi_per_feature`` -- bit-identical to
the per-base ``_mi_to_target`` call.

This isolates the ``mi_y_for_base`` baseline cost (the sub-phase that shrinks ~n_bases-fold); the rest of the knn
discovery wall (per-spec ``mi_t`` / ``mi_y_compare``) is genuinely per-transform and unchanged.

Run:
    MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python -m mlframe.training.composite.discovery._benchmarks.bench_iter96_knn_mi_y_baseline_recompute [n] [n_cols] [n_bases]
"""
from __future__ import annotations

import sys

sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402
import numba  # noqa: F401,E402

import time  # noqa: E402

import numpy as np  # noqa: E402

from ..screening import (  # noqa: E402
    _aggregate_mi_per_feature,
    _mi_per_feature_knn,
    _mi_to_target,
)


def _old_per_base(full_x, y, drop_indices, *, n_neighbors, random_state, aggregation):
    """OLD path: per-base _mi_to_target over the base-dropped matrix (no dedup for the A/B isolation)."""
    out = []
    for d in drop_indices:
        x_rem = np.delete(full_x, d, axis=1)
        out.append(_mi_to_target(
            x_rem, y, n_neighbors=n_neighbors, random_state=random_state,
            estimator="knn", aggregation=aggregation,
        ))
    return out


def _new_compute_once(full_x, y, drop_indices, *, n_neighbors, random_state, aggregation):
    per_feat = _mi_per_feature_knn(full_x, y, n_neighbors=n_neighbors, random_state=random_state)
    out = []
    for d in drop_indices:
        surviving = np.delete(np.arange(full_x.shape[1]), d)
        out.append(_aggregate_mi_per_feature(per_feat[surviving], aggregation))
    return out


def main(argv):
    n = int(argv[1]) if len(argv) > 1 else 100_000
    n_cols = int(argv[2]) if len(argv) > 2 else 53
    n_bases = int(argv[3]) if len(argv) > 3 else 6
    rng = np.random.default_rng(0)
    full_x = rng.normal(0.0, 1.0, (n, n_cols))
    base = rng.normal(0, 1, n).cumsum() / np.sqrt(n)
    full_x[:, 0] = base
    y = base + 0.5 * full_x[:, 1] + 0.3 * full_x[:, 2] + rng.normal(0, 0.3, n)
    drop_indices = list(range(n_bases))

    old = _old_per_base(full_x, y, drop_indices, n_neighbors=3, random_state=42, aggregation="mean")
    new = _new_compute_once(full_x, y, drop_indices, n_neighbors=3, random_state=42, aggregation="mean")
    max_abs = max(abs(a - b) for a, b in zip(old, new))
    print(f"identity: max|OLD-NEW| over {n_bases} bases = {max_abs:.3e}  (bit-identical: {old == new})")

    def timed(fn):
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn(full_x, y, drop_indices, n_neighbors=3, random_state=42, aggregation="mean")
            best = min(best, time.perf_counter() - t0)
        return best

    t_old = timed(_old_per_base)
    t_new = timed(_new_compute_once)
    print(f"OLD per-base recompute: {t_old:.3f}s  |  NEW compute-once: {t_new:.3f}s  |  speedup {t_old/t_new:.2f}x  (n={n}, cols={n_cols}, bases={n_bases})")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv)
