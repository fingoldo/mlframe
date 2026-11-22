"""Correlation clustering and pandas bridging for :mod:`hybrid_selector`.

``corr_clusters`` groups near-duplicate columns, ``_as_pandas_view`` bridges a polars frame to pandas, and
``_TREE_OPS`` names the co-occurrence operators the tree stage applies. ``hybrid_selector`` re-exports every name.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.shared import crit_np_dtype as _crit_np_dtype

logger = logging.getLogger("mlframe.feature_selection.hybrid_selector")


def _as_pandas_view(X):
    """Bridge a polars frame to an Arrow-backed pandas VIEW (zero-copy for numeric/bool/string columns) so HybridSelector's pandas-only glue
    (corr_clusters on ``X.values``, ``X.iloc`` / ``X[col]`` member feeds, ``pd.concat`` augment) operates on a real pandas frame. The internal
    members (LightGBM FI, ShapProxiedFS, BorutaShap) want pandas / ndarray anyway and would each convert independently; bridging once at the fit
    boundary captures the real column names from the polars schema and is NOT a full materialisation (numeric columns stay zero-copy Arrow views).
    Non-polars inputs pass through untouched."""
    try:
        import polars as pl
    except ImportError as _exc:
        logger.debug("_as_pandas_view fell back to its default after %s", type(_exc).__name__)
        return X
    if not isinstance(X, pl.DataFrame):
        return X
    from mlframe.training.utils import get_pandas_view_of_polars_df
    return get_pandas_view_of_polars_df(X)


# Tree-member feature-engineering operators applied per co-occurrence pair (a, b). "mul" is the product; the rich
# operators (absdiff/signed/ratio) recover NON-product interaction signal a bilinear term cannot linearize (measured:
# |a-b| logit 0.49->0.88; sign(a)|b| 0.79->0.88; products+rich on madelon +0.020 3-seed, variance halved). All are
# leak-free pure functions of X (no y, no fit) so they replay exactly at transform time. ratio uses +1 eps; outputs
# are nan/inf-sanitised. Each op-column is synergy-gated independently (kept only if more informative than both operands).
_TREE_OPS = {
    "mul": lambda a, b: a * b,
    "absd": lambda a, b: np.abs(a - b),
    "sign": lambda a, b: np.sign(a) * np.abs(b),
    "rat": lambda a, b: a / (np.abs(b) + 1.0),
}


# corr_block_threshold: above this many columns, corr_clusters switches from the full p x p correlation matrix to a
# row-block path that never materializes the dense p x p (the wide-data memory blocker: p=10k -> ~800 MB float64).
# The blocked path standardizes columns once (z-scores) and computes each rep's above-threshold |corr| neighbours by a
# single (1 x p) row-block GEMV against the standardized matrix, so peak extra memory is O(n*p + p) not O(p^2). The
# greedy assignment order is byte-identical to the full-matrix path (same rep order = first unassigned column; same
# member order = ascending column index): both read the same |corr| values, only the access pattern differs.
CORR_BLOCK_THRESHOLD = 1500


def corr_clusters(X: pd.DataFrame, thr: float = 0.92, block_threshold: int = CORR_BLOCK_THRESHOLD) -> tuple[list, dict[Any, list]]:
    """Greedy |Pearson| >= thr clustering. Returns (reps, members) where reps is the representative list (first
    column of each cluster) and members maps rep -> [member columns incl. the rep]. Computed once and shared.

    Vectorized inner scan: the per-pair Python ``abs(C[i,j]) >= thr`` test is replaced by a single boolean adjacency
    matrix ``A = |C| >= thr`` and an ``np.flatnonzero`` of each rep's upper row - byte-identical to the prior greedy
    (same rep order = first unassigned column; same member order = ascending column index), but skips the full j-loop
    for the common all-singleton frame where each row has few/no above-threshold candidates (measured 3.83x on the
    p=240 mostly-singleton hybrid frame; the glue floor's single largest item, ~86% of the hybrid's own tottime).

    WIDE-DATA path (p > ``block_threshold``): the full p x p ``np.corrcoef`` materializes an O(p^2) dense matrix
    (~800 MB at p=10k) - the one uncapped super-linear-in-p hotspot among the FS paths. Above the threshold we never
    build the dense matrix: columns are z-standardized once, then for each rep we compute ONLY its row of correlations
    (a single 1 x p GEMV: z_i @ Z / (n-1)) and threshold that, so peak extra memory is O(n*p) for Z plus O(p) per row.
    The greedy is identical (it reads the same |corr| values for the same (i, j>i) pairs), so reps + members are
    set-identical to the full-matrix result (verified bit/set-identical on small frames in the test suite).

    cProfile/tracemalloc (2026-06-19, dev box, n*p random frame): p=6000,n=800 -> FULL np.corrcoef path 1141 ms /
    756 MB peak; BLOCKED path 1344 ms / 132 MB peak == 5.8x less memory at ~equal wall. The gap widens with p (the
    full path is O(p^2): ~800 MB at p=10k before corrcoef's internal deviation-matrix copy ~doubles it -> the OOM
    blocker), while the blocked path stays O(n*p) for Z + O(bs*p) for the slab. Small frames (p <= block_threshold)
    keep the faster full path. (p=2000: FULL 187 ms vs BLOCKED 311 ms - the per-block matmul overhead, paid only
    above the threshold where the memory bound matters more than the sub-second wall.)"""
    cols = list(X.columns)
    p = len(cols)
    if p == 1:  # single column (mirrors the C.ndim == 0 corrcoef scalar case)
        return [cols[0]], {cols[0]: [cols[0]]}
    if p <= block_threshold:
        C = np.nan_to_num(np.corrcoef(X.values, rowvar=False))
        if C.ndim == 0:  # single column (defensive; p==1 already returned above)
            return [cols[0]], {cols[0]: [cols[0]]}
        A = np.abs(C) >= thr  # adjacency once (vectorized), replaces the per-pair Python compare
        reps, members, assigned = [], {}, np.zeros(p, dtype=bool)
        for i in range(p):
            if assigned[i]:
                continue
            assigned[i] = True
            reps.append(cols[i]); ms = [cols[i]]
            cand = np.flatnonzero(A[i, i + 1:]) + (i + 1)   # only the above-threshold j>i (ascending index preserved)
            for j in cand:
                if not assigned[j]:
                    assigned[j] = True; ms.append(cols[j])
            members[cols[i]] = ms
        return reps, members

    # --- wide-data blocked path: standardize once, compute correlations in COLUMN BLOCKS (no dense p x p) ---
    # Columns are z-standardized once (Z is O(n*p)); then a block of ``bs`` rep-rows is correlated against all p columns
    # in a single (bs x p) GEMV slab (Z[:, blk].T @ Z), so peak transient memory is O(bs*p) - a small constant slab,
    # never the full p x p. The greedy reads each rep's slab row exactly as it would the dense matrix row, so the
    # assignment is identical to the full-matrix path. bs trades speed (bigger = fewer, larger matmuls) vs the slab's
    # memory (bs*p*8 bytes); 512 keeps the slab modest (e.g. 512*10000*8 ~= 41 MB) while recovering near-BLAS speed.
    V = X.values
    n = V.shape[0]
    zdt = _crit_np_dtype()
    Z = np.empty(V.shape, dtype=zdt)
    nz = np.zeros(p, dtype=bool)
    # Centre and scale each column block in float64 and narrow to the working dtype only afterwards: casting first would round a large-offset column to
    # the float32 grid before the mean is removed, and the correlations of such columns would differ from the dense path.
    # Zero-variance columns -> corr undefined; np.corrcoef yields nan there (nan_to_num -> 0). Mirror that by leaving their standardized column at 0, so every
    # correlation INVOLVING them is 0 (< thr) -> they stay singletons. A constant column is divided by 1, which leaves it as it is.
    step = max(1, min(512, (1 << 18) // (8 * max(n, 1))))  # keeps the double-precision transients to about 256 KiB per block
    for c0 in range(0, p, step):
        c1 = min(c0 + step, p)
        blk = np.asarray(V[:, c0:c1], dtype=np.float64)  # Intentional float64: centring/scaling before the narrowing to the working dtype
        sd_b = blk.std(axis=0)
        nz[c0:c1] = sd_b > 0
        centred = blk - blk.mean(axis=0)
        centred /= np.where(sd_b > 0, sd_b, 1.0)
        Z[:, c0:c1] = centred
    # Columns were standardised with the POPULATION sd (ddof=0), so Z.T @ Z / n is the exact correlation; dividing by n-1
    # inflated every |corr| by n/(n-1) and shifted the cluster threshold (audit 24-F21).
    denom = float(n) if n > 0 else 1.0
    bs = 512
    reps, members, assigned = [], {}, np.zeros(p, dtype=bool)
    for start in range(0, p, bs):
        stop = min(start + bs, p)
        # slab[r, j] = corr(col start+r, col j); only used for the upper triangle (j > rep index) below.
        slab = np.nan_to_num((Z[:, start:stop].T @ Z) / denom)  # (bs x p) transient, never p x p
        adj = np.abs(slab) >= thr
        for r in range(stop - start):
            i = start + r
            if assigned[i]:
                continue
            assigned[i] = True
            reps.append(cols[i]); ms = [cols[i]]
            if nz[i]:
                cand = np.flatnonzero(adj[r, i + 1 :]) + (i + 1)  # ascending j>i preserved (identical to full path)
                for j in cand:
                    if not assigned[j]:
                        assigned[j] = True; ms.append(cols[j])
            members[cols[i]] = ms
    return reps, members


def _selection_summary(sel):
    """Selection-summary fragment of this selector for the end-of-fit log."""
    return sel.selected_features_, int(sel.n_features_in_), f"+{sel.n_engineered_} engineered", len(sel.selected_features_) - int(sel.n_engineered_)
