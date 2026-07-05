"""Does the FULL ``compute_numaggs`` 53-stat panel per cell beat the lean mean/std/skew/kurt set, and at what
cost? Measures (a) OOS value on a quantile-shape-driven target where the rich panel (which carries q0.1..q0.9,
entropy, hurst, mode stats) should have an edge, and (b) the per-fit timing of the vectorised bincount path
(all cells at once) vs the per-cell compute_numaggs loop.

Answers the open questions: is the richer panel worth wiring; is the array return (compute_numaggs already
returns np.ndarray) a non-issue; and how much does the per-cell loop cost vs the vectorised moments.

Run:  python -m mlframe.feature_selection._benchmarks.bench_richer_cell_panel
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from mlframe.feature_engineering.numerical import compute_numaggs
from mlframe.feature_selection.filters._binned_numeric_agg_fe import (
    per_cell_stats_bincount, quantile_edges,
)

SEEDS = (0, 1, 2, 3, 4)


def _shape_target(n, seed):
    """Each cell of g has a 2-component Gaussian mixture whose SEPARATION s(cell) drives the target. The
    cell mean ~ 0 and the std grows mildly, but the q0.9-q0.1 spread / kurtosis track s(cell) most directly."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0, 1, n)
    sep = 1.0 + 3.0 * np.abs(g - 0.5)  # mixture separation by cell
    comp = rng.integers(0, 2, n) * 2 - 1  # +/-1
    aux = comp * sep + rng.normal(0, 0.5, n)  # bimodal within cell
    T = sep  # predict the separation
    return g, aux, T


def _panel_per_cell(codes, v, n_cells):
    """Full compute_numaggs panel per cell -> (n_cells, 53) array (per-cell Python loop, the rich path)."""
    rows = []
    for c in range(n_cells):
        vc = v[codes == c].astype(np.float32)
        rows.append(compute_numaggs(vc) if vc.size >= 2 else None)
    width = next((len(r) for r in rows if r is not None), 1)
    gl = np.zeros(width, dtype=np.float64)
    tbl = np.array([r if r is not None else gl for r in rows], dtype=np.float64)
    return np.nan_to_num(tbl, nan=0.0, posinf=0.0, neginf=0.0)


def run(n, seed):
    g, aux, T = _shape_target(n, seed)
    idx = np.random.default_rng(1000 + seed).permutation(n)
    tr, te = idx[: n // 2], idx[n // 2 :]
    edges = quantile_edges(g[tr], 10)
    ctr = np.searchsorted(edges, g[tr], side="right")
    cte = np.searchsorted(edges, g[te], side="right")
    n_cells = int(max(ctr.max(), cte.max())) + 1

    def _fit(feat_tr, feat_te):
        m = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(feat_tr, T[tr])
        return r2_score(T[te], m.predict(feat_te))

    # lean: mean/std/skew/kurt via bincount (timed)
    t0 = time.perf_counter()
    lean = per_cell_stats_bincount(ctr, aux[tr], n_cells, ("mean", "std", "skew", "kurt"))
    t_lean = time.perf_counter() - t0
    lean_tr = np.column_stack([np.nan_to_num(lean[s])[ctr] for s in ("mean", "std", "skew", "kurt")])
    lean_full = per_cell_stats_bincount(np.r_[ctr, cte], np.r_[aux[tr], aux[te]], n_cells, ("mean", "std", "skew", "kurt"))
    lean_te = np.column_stack([np.nan_to_num(lean_full[s])[cte] for s in ("mean", "std", "skew", "kurt")])

    # rich: full 53-panel per cell (timed)
    t0 = time.perf_counter()
    panel_tbl = _panel_per_cell(ctr, aux[tr], n_cells)
    t_rich = time.perf_counter() - t0
    rich_tr = panel_tbl[ctr]
    rich_te = panel_tbl[cte]

    return _fit(lean_tr, lean_te), _fit(rich_tr, rich_te), t_lean, t_rich


def main():
    print("\nRicher cell panel vs lean mean/std/skew/kurt -- shape-driven target, 5 seeds\n")
    for n in (8000, 30000):
        rows = [run(n, s) for s in SEEDS]
        lean = np.mean([r[0] for r in rows]); rich = np.mean([r[1] for r in rows])
        tl = np.mean([r[2] for r in rows]) * 1e3; tr_ = np.mean([r[3] for r in rows]) * 1e3
        print(f"n={n:>6}  lean(4)=R2 {lean:.3f} ({tl:.2f} ms)   rich(53)=R2 {rich:.3f} ({tr_:.2f} ms)   "
              f"value d {rich-lean:+.3f}   cost x{tr_/max(tl,1e-6):.0f}")
    print()


if __name__ == "__main__":
    main()
