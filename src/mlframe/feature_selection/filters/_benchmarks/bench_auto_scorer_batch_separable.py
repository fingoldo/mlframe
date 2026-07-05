"""Bench for auto-scorer batched separable scorers (CPX12).

_compute_per_scorer_rank_table scored each column with each scorer via a
reshape(-1,1)/out[0] single-column call. The column-separable scorers (plug-in
MI, copula MI) are now batched across all columns in one kernel call.
Bit-identical to the per-column path (per-column quantile independent).
Run: python this.py
"""
from __future__ import annotations

import time

import numpy as np


def main():
    from mlframe.feature_selection.filters._orth_auto_scorer_fe import (
        _score_copula,
        _score_plug_in,
    )
    from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _copula_mi_batch
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _mi_classif_batch

    rng = np.random.default_rng(0)
    n, p = 1500, 60
    X = rng.standard_normal((n, p))
    y = rng.integers(0, 4, n)

    def old():
        mi = np.array([_score_plug_in(X[:, j], y, nbins=10) for j in range(p)])
        cop = np.array([_score_copula(X[:, j], y, n_bins=20) for j in range(p)])
        return mi, cop

    def new():
        mi = _mi_classif_batch(X, y.astype(np.int64), nbins=10)
        cop = _copula_mi_batch(X, y, n_bins=20)
        return np.asarray(mi), np.asarray(cop)

    o = old(); ne = new()
    assert np.array_equal(o[0], ne[0]) and np.array_equal(o[1], ne[1]), "identity"

    def _best(fn, reps=5):
        t = []
        for _ in range(reps):
            s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
        return min(t)

    t_old = _best(old)
    t_new = _best(new)
    print(f"plug_in+copula over p={p} n={n}: OLD(per-col) {t_old*1e3:.2f}ms -> " f"NEW(batched) {t_new*1e3:.2f}ms ({t_old/t_new:.2f}x) identity OK")


if __name__ == "__main__":
    main()
