"""Bench for _copula_mi_batch y-rank hoist (CPX11).

The docstring promised "uniformise y ONCE" but the per-column loop re-ranked
y_arr (O(n log n)) on every all-finite column. Hoisting v_full above the loop
removes that. Bit-identical scores. Run: python this.py
"""
from __future__ import annotations

import time

import numpy as np


def main():
    from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import (
        _bin_mi_uniform_pair,
        _copula_mi_batch,
        _rank_to_uniform,
    )

    rng = np.random.default_rng(0)
    n, p = 2000, 200
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    yf = np.isfinite(y)

    def old():
        out = np.empty(p)
        for j in range(p):
            col = X[:, j]
            fin = yf & np.isfinite(col)
            if fin.all():
                u = _rank_to_uniform(col)
                v = _rank_to_uniform(y)  # re-ranked per column (the waste)
            out[j] = _bin_mi_uniform_pair(u, v, n_bins=20)
        return out

    old()
    _copula_mi_batch(X, y)  # warm current impl
    assert np.allclose(old(), _copula_mi_batch(X, y))

    def _best(fn, reps=7):
        t = []
        for _ in range(reps):
            s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
        return min(t)

    t_old = _best(old)
    t_new = _best(lambda: _copula_mi_batch(X, y))
    print(f"n={n} p={p}: OLD(re-rank y/col) {t_old*1e3:.2f}ms -> "
          f"NEW(hoist) {t_new*1e3:.2f}ms ({t_old/t_new:.2f}x) identity OK")


if __name__ == "__main__":
    main()
