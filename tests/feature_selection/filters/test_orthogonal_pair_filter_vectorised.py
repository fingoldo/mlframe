"""The correlated-pair filters of the cluster and diff bases give the former double loop's pairs in its order (audit 2026-11-20-perf E-F07)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_diff_basis_fe as diff_fe


def _loop_pairs(abs_corr, thr):
    """The former implementation: row-major double loop with the finite / threshold / duplicate guards."""
    p = abs_corr.shape[0]
    out = []
    for i in range(p):
        for j in range(i + 1, p):
            c = float(abs_corr[i, j])
            if np.isfinite(c) and c >= thr and c < 1.0 - 1e-12:
                out.append((i, j, c))
    return out


def test_vectorised_upper_triangle_selection_equals_the_loop(monkeypatch):
    """Random symmetric matrix with NaN, exact 1.0 duplicates and values at the threshold: ``detect_correlated_pairs`` returns the loop's pairs (sorted by correlation, then the stable insertion order)."""
    rng = np.random.default_rng(0)
    p = 40
    m = np.abs(rng.uniform(0, 1, size=(p, p)))
    m = (m + m.T) / 2
    m[3, 9] = m[9, 3] = np.nan
    m[5, 6] = m[6, 5] = 1.0
    m[7, 8] = m[8, 7] = 0.8
    names = [f"c{i}" for i in range(p)]
    monkeypatch.setattr(diff_fe, "cached_dense_finite_corr_matrix", lambda X, cols, dtype=None: (names, m))
    got = diff_fe.detect_correlated_pairs(pd.DataFrame({n: np.zeros(3) for n in names}), names, corr_threshold=0.8, max_pairs=10_000)
    ref = [(names[i], names[j], c) for i, j, c in _loop_pairs(m, 0.8)]
    ref.sort(key=lambda t: t[2], reverse=True)
    assert got == ref
