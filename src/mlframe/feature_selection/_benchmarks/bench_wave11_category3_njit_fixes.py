"""Before/after wall-clock benchmarks for the Wave 11 (Category 3: njit/vectorization non-use) fixes.

Each section times the OLD (frozen reference copy of the pre-fix implementation) against the NEW (current
shipped) code path at a realistic scale, and asserts the equivalence already proven by the paired pytest
tests (this file focuses on the numbers, not exhaustive correctness coverage).

Run: ``python -m mlframe.feature_selection._benchmarks.bench_wave11_category3_njit_fixes``
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _bspline_basis_values_ref(z, knots, idx, degree=3):
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nk = len(knots)
    for i in range(n):
        zi = z[i]
        if zi >= knots[nk - degree - 1]:
            zi_eff = knots[nk - degree - 1] - 1e-12
        elif zi <= knots[degree]:
            zi_eff = knots[degree] + 1e-12
        else:
            zi_eff = zi
        k = degree
        for kk in range(degree, nk - degree - 1):
            if knots[kk] <= zi_eff < knots[kk + 1]:
                k = kk
                break
        else:
            k = nk - degree - 2
        N = np.zeros(degree + 1, dtype=np.float64)
        N[0] = 1.0
        for d in range(1, degree + 1):
            saved = 0.0
            for r in range(d):
                t_left = knots[k + 1 + r - d]
                t_right = knots[k + 1 + r]
                denom = t_right - t_left
                temp = 0.0 if denom <= 1e-12 else N[r] / denom
                N[r] = saved + (t_right - zi_eff) * temp
                saved = (zi_eff - t_left) * temp
            N[d] = saved
        rel = idx - (k - degree)
        if 0 <= rel <= degree:
            out[i] = N[rel]
    return out


def bench_bspline():
    from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import (
        _bspline_basis_values,
        _fit_spline_knots,
    )

    rng = np.random.default_rng(0)
    n_rows = 100_000
    x = rng.normal(size=n_rows)
    knots, lo, hi = _fit_spline_knots(x, 5, degree=3)
    span = max(hi - lo, 1e-12)
    z = np.clip((x - lo) / span, 0.0, 1.0)
    n_basis = len(knots) - 3 - 1

    for idx in range(n_basis):
        _bspline_basis_values_ref(z[:500], knots, idx, degree=3)
        _bspline_basis_values(z[:500], knots, idx, degree=3)

    t0 = time.perf_counter()
    for idx in range(n_basis):
        _bspline_basis_values_ref(z, knots, idx, degree=3)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    for idx in range(n_basis):
        _bspline_basis_values(z, knots, idx, degree=3)
    t_new = time.perf_counter() - t0

    print(f"H1 bspline: n_basis={n_basis} n={n_rows}: OLD={t_old:.3f}s NEW={t_new:.4f}s speedup={t_old / t_new:.1f}x")  # noqa: T201


def bench_ratio_delta_redundancy():
    from mlframe.feature_selection.filters._ratio_delta_fe import _passes_redundancy

    def ref(candidate, a_vals, b_vals, threshold):
        cand = candidate
        if cand.std() <= 1e-12:
            return False
        for src in (a_vals, b_vals):
            if src.std() <= 1e-12:
                continue
            mask = np.isfinite(cand) & np.isfinite(src)
            if not mask.any():
                continue
            c2, s2 = cand[mask], src[mask]
            if c2.std() <= 1e-12 or s2.std() <= 1e-12:
                continue
            rho = float(np.corrcoef(c2, s2)[0, 1])
            if abs(rho) > float(threshold):
                return False
        return True

    rng = np.random.default_rng(3)
    n, p = 5000, 60
    cols = [rng.normal(size=n) for _ in range(p)]
    pairs = [(i, j) for i in range(p) for j in range(p) if i != j]

    _a0 = np.nan_to_num(cols[0] / (cols[1] + 1e-9))
    _passes_redundancy(_a0.copy(), cols[0].copy(), cols[1].copy(), 0.99)

    t0 = time.perf_counter()
    for i, j in pairs:
        cand = np.nan_to_num(cols[i] / (cols[j] + 1e-9), nan=0.0, posinf=0.0, neginf=0.0)
        _passes_redundancy(cand, cols[i], cols[j], 0.99)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i, j in pairs:
        cand = np.nan_to_num(cols[i] / (cols[j] + 1e-9), nan=0.0, posinf=0.0, neginf=0.0)
        ref(cand, cols[i], cols[j], 0.99)
    t_old = time.perf_counter() - t0

    scale = (500 * 499) / len(pairs)
    print(  # noqa: T201
        f"H2 ratio-delta redundancy: p={p} n={n} ({len(pairs)} pairs): OLD={t_old:.3f}s NEW={t_new:.3f}s "
        f"speedup={t_old / t_new:.2f}x; extrapolated to p=500: OLD~{t_old*scale:.1f}s NEW~{t_new*scale:.1f}s"
    )


def bench_categorize_dataset_adaptive_searchsorted():
    import mlframe.feature_selection.filters._adaptive_nbins as _adaptive_nbins_mod
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.discretization._kernels import _searchsorted_2d_right_njit_parallel

    def old_loop(edges_per_col, arr, dtype):
        n_rows, n_cols = arr.shape
        data = np.empty((n_rows, n_cols), dtype=dtype)
        for j in range(n_cols):
            ej = edges_per_col[j]
            data[:, j] = 0 if ej.size == 0 else np.searchsorted(ej, arr[:, j].astype(np.float64), side="right").astype(dtype)
        return data

    rng = np.random.default_rng(99)
    n, p = 60_000, 400
    df = pd.DataFrame({f"c{j}": rng.normal(scale=rng.choice([0.01, 1.0, 100.0]), size=n) + j for j in range(p)})

    captured = {}
    orig_pfe = _adaptive_nbins_mod.per_feature_edges

    def _capture(arr, **kwargs):
        edges = orig_pfe(arr, **kwargs)
        captured["arr"] = arr.copy()
        captured["edges"] = [e.copy() for e in edges]
        return edges

    _adaptive_nbins_mod.per_feature_edges = _capture
    t0 = time.perf_counter()
    categorize_dataset(df, nbins_strategy="fd", missing_strategy="separate_bin", dtype=np.int32)
    t_new_full = time.perf_counter() - t0
    _adaptive_nbins_mod.per_feature_edges = orig_pfe

    arr, edges_per_col = captured["arr"], captured["edges"]
    n_edges_max = max((int(e.size) for e in edges_per_col), default=0)
    edges_padded = np.full((n_edges_max, p), np.inf, dtype=np.float64)
    for j, ej in enumerate(edges_per_col):
        if ej.size:
            edges_padded[: ej.size, j] = ej
    codes = np.empty((n, p), dtype=np.int64)
    _searchsorted_2d_right_njit_parallel(edges_padded, arr, codes)

    t0 = time.perf_counter()
    for _ in range(5):
        _searchsorted_2d_right_njit_parallel(edges_padded, arr, codes)
    t_new = (time.perf_counter() - t0) / 5

    t0 = time.perf_counter()
    for _ in range(5):
        old_loop(edges_per_col, arr, np.int32)
    t_old = (time.perf_counter() - t0) / 5

    print(f"M8 categorize_dataset searchsorted: n={n} p={p}: OLD={t_old:.4f}s NEW={t_new:.4f}s speedup={t_old / t_new:.2f}x")  # noqa: T201
    print(f"    full adaptive-path categorize_dataset wall: {t_new_full:.3f}s")  # noqa: T201


def bench_cat_interactions_pair_enum():
    rng = np.random.default_rng(0)

    def old_enum(cand, nbins, max_combined):
        pa, pb = [], []
        for ii in range(len(cand)):
            for jj in range(ii + 1, len(cand)):
                i, j = int(cand[ii]), int(cand[jj])
                nb_prod = int(nbins[i]) * int(nbins[j])
                if nb_prod > max_combined or nb_prod >= 2**31:
                    continue
                pa.append(i); pb.append(j)
        return np.asarray(pa, dtype=np.int64), np.asarray(pb, dtype=np.int64)

    def new_enum(cand, nbins, max_combined):
        n_cand = len(cand)
        if n_cand < 2:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        ii, jj = np.triu_indices(n_cand, k=1)
        i_arr = np.asarray(cand, dtype=np.int64)[ii]
        j_arr = np.asarray(cand, dtype=np.int64)[jj]
        nb64 = np.asarray(nbins, dtype=np.int64)
        nb_prod = nb64[i_arr] * nb64[j_arr]
        keep = (nb_prod <= int(max_combined)) & (nb_prod < 2**31)
        return i_arr[keep], j_arr[keep]

    n_cols, n_cand = 800, 500
    cand = rng.choice(np.arange(n_cols), size=n_cand, replace=False)
    nbins = rng.integers(2, 200, size=n_cols)
    max_combined = 20000

    t0 = time.perf_counter()
    a_old, b_old = old_enum(cand, nbins, max_combined)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    a_new, b_new = new_enum(cand, nbins, max_combined)
    t_new = time.perf_counter() - t0

    assert np.array_equal(a_old, a_new) and np.array_equal(b_old, b_new)
    print(f"M10 cat-interactions pair enum: {n_cand} candidates (~{n_cand*(n_cand-1)//2} pairs): OLD={t_old:.4f}s NEW={t_new:.4f}s speedup={t_old / t_new:.1f}x")  # noqa: T201


if __name__ == "__main__":
    bench_bspline()
    bench_ratio_delta_redundancy()
    bench_categorize_dataset_adaptive_searchsorted()
    bench_cat_interactions_pair_enum()
