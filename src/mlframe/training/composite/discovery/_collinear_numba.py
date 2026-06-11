"""Numba-JIT inner loop for the per-base near-collinear keep-mask walk.

``near_collinear_keep_mask`` (in ``_eval_stats``) de-duplicates a base's
``x_remaining`` by walking columns left to right and dropping any column whose
absolute Pearson correlation with an already-kept column exceeds a threshold.
That walk is ``O(B^2)`` pure-numpy Pearson + boolean fancy-indexing; on a base
with 50+ candidate remaining columns it is a measurable hot spot inside the MI
baseline sweep ``fit`` runs once per screened base.

This module carves the pair-walk into a ``numba.njit`` kernel
(:func:`_keep_mask_kernel`) that operates on the full ``(n, B)`` matrix plus a
precomputed finite mask, computing each pair's correlation in registers with no
per-pair fancy-index allocation. The public entry point is
:func:`near_collinear_keep_mask_fast`, a size-aware dispatcher: it routes tiny
inputs (``B < _MIN_COLS`` or ``n < _MIN_ROWS``) to the numpy reference (the JIT
overhead never amortises there per the numba ladder) and larger inputs to the
JIT kernel.

Bit-identity contract
---------------------
The kernel's per-pair reductions (``mean`` / ``dot`` lowered by numba) can differ
from numpy's BLAS / pairwise-reduce by ~1 ULP. That ULP can only flip a
keep/drop decision when a pair's correlation lands within ~1e-12 of the
threshold -- which is measure-zero on continuous data but reachable on tied /
discrete columns. The kernel therefore reports any pair whose correlation lands
in a tiny band around the threshold as "borderline"; the wrapper re-decides ONLY
those borderline pairs with the EXACT numpy primitives the reference uses
(``a.mean()`` / ``np.dot``). Every non-borderline decision is already separated
from the threshold by far more than the ULP noise, so the resulting mask is
bit-identical to :func:`near_collinear_keep_mask` for all inputs -- continuous,
discrete, tied, NaN-holed, and exact-duplicate (corr exactly 1.0). The
regression test pins this across seeds plus a degenerate constant column.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _numba

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep; allow graceful skip.
    _numba = None  # type: ignore
    _HAS_NUMBA = False

# Numba JIT wins the O(B^2) pair walk only once both the column count and the
# row count are large enough to amortise the ~register-loop setup vs numpy's
# vectorised dot; below these the numpy reference's lower per-call overhead wins
# (and the one-shot JIT compile never pays back on a handful of columns).
_MIN_COLS: int = 10
_MIN_ROWS: int = 256

# Half-width of the "borderline" band around the threshold. A pair whose
# kernel-computed |corr| lands within this of the threshold is re-decided with
# the exact numpy primitives so a ~1 ULP kernel/BLAS reduction difference can
# never flip the keep/drop decision. Far wider than the ~1e-15 ULP noise yet
# narrow enough that borderline pairs are vanishingly rare on real data.
_BORDERLINE_BAND: float = 1e-9

# Variance floor below which a column is "constant on the joint support" -> no
# correlation (kept). Matches the numpy reference exactly.
_VAR_FLOOR: float = 1e-24


if _HAS_NUMBA:

    @_numba.njit(cache=True, fastmath=False)
    def _keep_mask_kernel(fm, finite, thr, band):  # type: ignore[no-untyped-def]
        """Walk columns left to right, return (keep, borderline) int8 arrays.

        ``keep[j]`` is 1 when column j survives the kernel's own decision.
        ``borderline[j]`` is 1 when j's DROP/keep hinged on a pair whose |corr|
        was within ``band`` of ``thr`` (the wrapper rechecks those exactly).

        Reductions here mirror the numpy reference's two-pass mean + dot but are
        lowered by numba, so a non-borderline decision is trusted directly while
        a borderline one is flagged for exact numpy re-decision.
        """
        n = fm.shape[0]
        n_cols = fm.shape[1]
        keep = np.ones(n_cols, dtype=np.int8)
        borderline = np.zeros(n_cols, dtype=np.int8)
        # Indices of columns kept so far (compacted into the front of kept_idx).
        kept_idx = np.empty(n_cols, dtype=np.int64)
        n_kept = 0
        for j in range(n_cols):
            drop = False
            j_borderline = False
            for ki in range(n_kept):
                k = kept_idx[ki]
                # First pass: count jointly-finite rows + accumulate the two
                # means over exactly those rows.
                n_pair = 0
                sa = 0.0
                sb = 0.0
                for i in range(n):
                    if finite[i, j] and finite[i, k]:
                        sa += fm[i, j]
                        sb += fm[i, k]
                        n_pair += 1
                if n_pair < 3:
                    continue
                ma = sa / n_pair
                mb = sb / n_pair
                va = 0.0
                vb = 0.0
                vab = 0.0
                for i in range(n):
                    if finite[i, j] and finite[i, k]:
                        da = fm[i, j] - ma
                        db = fm[i, k] - mb
                        va += da * da
                        vb += db * db
                        vab += da * db
                if va < _VAR_FLOOR or vb < _VAR_FLOOR:
                    continue
                corr = abs(vab / np.sqrt(va * vb))
                if abs(corr - thr) <= band:
                    # Too close to the threshold to trust the lowered reduction;
                    # the wrapper will re-decide this column exactly.
                    j_borderline = True
                    drop = True
                    break
                if corr > thr:
                    drop = True
                    break
            if drop:
                keep[j] = 0
                if j_borderline:
                    borderline[j] = 1
            else:
                kept_idx[n_kept] = j
                n_kept += 1
        return keep, borderline


def _ref_pair_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    """Exact numpy |Pearson| for one jointly-finite pair, matching the reference.

    Returns ``None`` for the "skip" cases (constant on the joint support) so the
    caller treats the pair as uncorrelated, identical to the numpy reference.
    """
    a_dev = a - a.mean()
    b_dev = b - b.mean()
    va = float(np.dot(a_dev, a_dev))
    vb = float(np.dot(b_dev, b_dev))
    if va < _VAR_FLOOR or vb < _VAR_FLOOR:
        return None
    return abs(float(np.dot(a_dev, b_dev)) / np.sqrt(va * vb))


def _recheck_column_exact(
    feature_matrix: np.ndarray,
    finite: np.ndarray,
    j: int,
    kept_idx: list[int],
    thr: float,
) -> bool:
    """Exact numpy re-decision for one borderline column ``j``.

    Returns True when ``j`` is DROPPED (correlates above ``thr`` with a kept
    column), using the identical arithmetic of the numpy reference so the
    decision is bit-identical to ``near_collinear_keep_mask``.
    """
    fin_j = finite[:, j]
    col_j = feature_matrix[:, j]
    for k in kept_idx:
        pair = fin_j & finite[:, k]
        if int(pair.sum()) < 3:
            continue
        corr = _ref_pair_corr(col_j[pair], feature_matrix[:, k][pair])
        if corr is None:
            continue
        if corr > thr:
            return True
    return False


def near_collinear_keep_mask_fast(
    feature_matrix: np.ndarray,
    *,
    corr_threshold: float,
    reference_fn,
) -> np.ndarray:
    """Size-aware dispatcher for the near-collinear keep-mask walk.

    Routes tiny inputs to ``reference_fn`` (the pure-numpy reference) and larger
    inputs to the numba kernel, then re-decides any borderline column exactly so
    the returned mask is bit-identical to the reference for all inputs.
    ``reference_fn`` is injected (rather than imported) to avoid an import cycle
    with ``_eval_stats``.
    """
    if feature_matrix.ndim != 2:
        raise ValueError("near_collinear_keep_mask expects a 2-D matrix")
    n_rows, n_cols = feature_matrix.shape
    thr = float(corr_threshold)
    # Cheap exits identical to the reference (also avoids JIT on trivial inputs).
    if n_cols < 2 or n_rows < 3 or not (thr < 1.0):
        return np.ones(n_cols, dtype=bool)
    if not _HAS_NUMBA:
        return reference_fn(feature_matrix, corr_threshold=thr)
    # Backend choice goes through the kernel_tuning_cache (per-host measured crossover)
    # with the hardcoded _MIN_COLS / _MIN_ROWS gate as the fallback + an env-var
    # force-override; bit-identical either way (borderline columns re-decided exactly).
    from ._ktc_dispatch import choose_collinear_backend

    if choose_collinear_backend(n_rows, n_cols, min_rows=_MIN_ROWS, min_cols=_MIN_COLS) == "numpy":
        return reference_fn(feature_matrix, corr_threshold=thr)

    fm = np.ascontiguousarray(feature_matrix, dtype=np.float64)
    finite = np.isfinite(fm)
    keep_i8, borderline_i8 = _keep_mask_kernel(fm, finite, thr, _BORDERLINE_BAND)
    keep = keep_i8.astype(bool)
    if not borderline_i8.any():
        return keep
    # Replay the left-to-right walk, re-deciding only the borderline columns with
    # the exact numpy primitives. The kept set is rebuilt in column order so a
    # re-decided KEEP can correctly shadow a later borderline column too.
    kept_idx: list[int] = []
    for j in range(n_cols):
        if borderline_i8[j]:
            dropped = _recheck_column_exact(fm, finite, j, kept_idx, thr)
            keep[j] = not dropped
        if keep[j]:
            kept_idx.append(j)
    return keep


def _warm_collinear_kernel() -> None:
    """Compile the kernel at import on a tiny matrix so the first real call is hot."""
    if not _HAS_NUMBA:
        return
    try:
        warm = np.zeros((4, 2), dtype=np.float64)
        warm[:, 0] = np.arange(4.0)
        warm[:, 1] = np.arange(4.0)
        fin = np.isfinite(warm)
        _keep_mask_kernel(warm, fin, 0.99, _BORDERLINE_BAND)
    except Exception:  # pragma: no cover - warming is best-effort.
        pass


_warm_collinear_kernel()
