"""A/B bench + quality validation for the njit port of ``iterstrat.ml_stratifiers.IterativeStratification``.

Ties are broken by an independent RNG stream (see the module docstring for why that's the accepted bar
here), so per-sample fold assignment is NOT expected to match the reference. What must match is
STRATIFICATION QUALITY: each fold's realised size and per-label positive-rate should track the requested
``r`` fractions as closely as the reference achieves (both are the same greedy algorithm, just breaking
ties differently). Validates that across many random multilabel datasets, then measures the real
wall-clock win at LTR-scale n.
"""

import time

import numpy as np
from iterstrat.ml_stratifiers import IterativeStratification as _iterstrat_ref
from sklearn.utils import check_random_state

from mlframe.training._iterative_stratification_njit import _iterative_stratification_njit


def _quality(labels, r, folds):
    """Max relative deviation of any (fold, label) positive-rate from its requested fraction r[fold]."""
    n, k = labels.shape
    worst = 0.0
    for j in range(len(r)):
        mask = folds == j
        fold_n = int(mask.sum())
        if fold_n == 0:
            continue
        for c in range(k):
            total_pos = int(labels[:, c].sum())
            if total_pos == 0:
                continue
            fold_pos = int(labels[mask, c].sum())
            expected = r[j] * total_pos
            worst = max(worst, abs(fold_pos - expected) / max(total_pos, 1))
    return worst


def _make_labels(n, k, seed):
    rng = np.random.default_rng(seed)
    return (rng.random((n, k)) < rng.uniform(0.02, 0.4, size=k)).astype(np.int8)


def main():
    r = np.array([0.7, 0.15, 0.15])

    # Quality sweep: many small/medium datasets, compare deviation-from-r against the reference.
    worst_ref, worst_new = 0.0, 0.0
    for seed in range(60):
        n = int(np.random.default_rng(seed + 1000).integers(200, 3000))
        k = int(np.random.default_rng(seed + 2000).integers(2, 12))
        labels = _make_labels(n, k, seed)
        ref_folds = _iterstrat_ref(labels=labels.astype(bool), r=r, random_state=check_random_state(seed))
        new_folds = _iterative_stratification_njit(labels, r, seed)
        assert set(np.unique(new_folds).tolist()) <= {0, 1, 2}
        assert (new_folds >= 0).all(), "every sample must be assigned"
        worst_ref = max(worst_ref, _quality(labels, r, ref_folds))
        worst_new = max(worst_new, _quality(labels, r, new_folds))

    print(f"quality (max per-fold/label deviation from r): reference={worst_ref:.4f} njit={worst_new:.4f}")

    # Speed at 2M-row / multilabel scale.
    n, k = 2_000_000, 15
    labels = _make_labels(n, k, seed=0)

    # warm JIT
    _iterative_stratification_njit(_make_labels(2000, k, 1), r, 1)

    t0 = time.perf_counter()
    ref_folds = _iterstrat_ref(labels=labels.astype(bool), r=r, random_state=check_random_state(0))
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_folds = _iterative_stratification_njit(labels, r, 0)
    t_new = time.perf_counter() - t0

    print(f"reference: {t_ref:.3f}s")
    print(f"njit:      {t_new:.3f}s")
    print(f"speedup:   {t_ref / t_new:.2f}x")
    print(f"ref fold sizes: {np.bincount(ref_folds)}")
    print(f"njit fold sizes: {np.bincount(new_folds)}")


if __name__ == "__main__":
    main()
