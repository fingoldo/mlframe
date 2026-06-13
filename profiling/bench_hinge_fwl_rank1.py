"""Microbench: _detect_hinge_breakpoints per-cut SSE via Frisch-Waugh-Lovell rank-1 update vs the legacy full lstsq-per-cut.

The fixed design block ``B = [1, x, *extra_legs]`` is identical across every candidate cut in a round; only the ``relu`` column varies. So QR-factor B once
and score each cut by the partitioned-regression identity ``SSE_B - (r_relu . r_y)^2 / (r_relu . r_relu)`` (one O(n*k) projection per cut, no per-cut SVD).
Mathematically identical to the full lstsq SSE; the argmin tau is bit-identical (FP reduction ~1e-12, far below the cut spacing).

Measured (this box, n=4000 / 24 cuts): ~2.4x. The QR cost amortises across all cuts in the round; the per-cut work drops from an n*3 SVD to two O(n) projections.

Run: PYTHONPATH=src python profiling/bench_hinge_fwl_rank1.py
"""
from __future__ import annotations
import time
import numpy as np
from mlframe.feature_selection.filters import _hinge_basis_fe as H
from mlframe.feature_selection.filters._hinge_basis_fe import _detect_hinge_breakpoints


def legacy(x, y, *, max_breakpoints=2, min_heldout_r2_uplift=0.02):
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel(); n = x.size
    if n < H._HINGE_MIN_ROWS:
        return []
    f = np.isfinite(x) & np.isfinite(y)
    if not f.all():
        x = x[f]; y = y[f]; n = x.size
    if n < H._HINGE_MIN_ROWS:
        return []
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return []
    if not H._hinge_slope_change_plausible(x, y, min_sse_drop=H._HINGE_PRECHECK_MIN_SSE_DROP):
        return []
    qs = np.linspace(H._HINGE_CAND_Q_LO, H._HINGE_CAND_Q_HI, H._HINGE_N_CANDIDATES)
    cand = np.unique(np.quantile(x, qs))
    found = []; extra = []
    for _ in range(max(1, int(max_breakpoints))):
        bt = None; bs = float("inf")
        for c in cand:
            nr = int(np.count_nonzero(x > c))
            if nr < H._HINGE_MIN_SEG_ROWS or (n - nr) < H._HINGE_MIN_SEG_ROWS:
                continue
            if any(abs(c - t) < 1e-9 for t in found):
                continue
            relu = np.maximum(x - c, 0.0)
            A = np.column_stack([np.ones_like(x), x, relu] + extra)
            try:
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            except Exception:
                continue
            resid = y - A @ coef
            sse = float(resid @ resid)
            if sse < bs:
                bs = sse; bt = float(c)
        if bt is None:
            break
        if H._heldout_hinge_r2_uplift(x, y, bt) < min_heldout_r2_uplift:
            break
        found.append(bt); extra.append(np.maximum(x - bt, 0.0))
    return found


def main():
    import timeit
    rng = np.random.default_rng(1)
    for n in (200, 500, 1200, 4000, 10000):
        x = np.sort(rng.uniform(-3, 3, n))
        y = np.where(x < 0, 0.5 * x, 2.0 * x) + 0.2 * rng.standard_normal(n)
        rl = legacy(x, y); rn = _detect_hinge_breakpoints(x, y)
        ident = (len(rl) == len(rn)) and all(abs(a - b) <= 1e-9 for a, b in zip(rl, rn))
        reps = 50
        tl = min(timeit.repeat(lambda: legacy(x, y), number=reps, repeat=9)) / reps
        tn = min(timeit.repeat(lambda: _detect_hinge_breakpoints(x, y), number=reps, repeat=9)) / reps
        print(f"n={n:6d}  legacy(lstsq/cut)={tl*1e3:7.3f}ms  fwl(QR+rank1)={tn*1e3:7.3f}ms  "
              f"speedup={tl/tn:.3f}x  tau_bit_identical={ident}")


if __name__ == "__main__":
    main()
