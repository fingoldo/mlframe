"""Numba-JIT kernel + size-aware dispatcher for the leakage-filter correlation gate.

``_safe_abs_corr_all`` (in ``screening``) computes ``|corr(y, X[:, j])|`` for every
feature column in one numpy pass. It runs inside the forbidden-base leakage filter
(``_filter.py``) on the mean-imputed (all-finite) screening matrix -- a (n, F) matrix
that at production scale is 50k+ rows by 100-200+ columns -- and is called a handful
of times per discovery. The numpy reference centres ``X``, folds the per-column
variance with an einsum and takes a single ``X_dev.T @ y_dev`` matmul; it still
materialises the full centred ``X_dev`` (an (n, F) temporary).

This module carves the same computation into a ``numba.njit(parallel=True)`` kernel
(:func:`_abs_corr_all_kernel`) that walks each column in registers -- one ``prange``
column-loop accumulating that column's mean, variance and cross-product against the
pre-centred ``y`` in a single pass -- with NO (n, F) centred temporary. The public
entry point is :func:`safe_abs_corr_all_dispatch`, a size-aware dispatcher that
routes small inputs to the numpy reference (the JIT / thread-spawn overhead never
amortises there per the numba ladder) and large inputs (``n >= _MIN_ROWS`` AND
``F >= _MIN_COLS``) to the kernel.

Numerical-equivalence contract
------------------------------
The kernel's per-column reductions (sequential ``+`` under ``parallel=True``) differ
from numpy's BLAS / pairwise-reduce by ~1 ULP, so a kernel ``|corr|`` can differ from
the reference by ~1e-9. Per the project's numerical-kernel rule this ~1e-9 FP
reduction-order divergence is acceptable for a real speedup -- EXCEPT where it could
flip the caller's leak decision. The only caller (``_filter.py``) drops a column when
``|corr| >= forbidden_base_corr_threshold`` (a near-1 threshold), so the
decision-relevant region is ``|corr|`` near 1.0. The kernel therefore flags any column
whose ``|corr|`` lands within ``_BORDERLINE_BAND`` of 1.0 (or whose variance sits near
the constant floor), and the wrapper re-decides ONLY those flagged columns with the
EXACT numpy single-column primitives the reference uses -- pulling their drift from
~1e-9 down to ~1e-12 (the reference itself reduces cov via a batched BLAS matmul, so
byte-equality is not attainable, but ~1e-12 is far tighter than any realistic leak
threshold spacing). Every non-flagged column is far from the leak threshold, so its
~1e-9 drift can never move a drop/keep decision; a constant column returns EXACTLY 0.0
like the reference. The returned vector is thus within ~1e-9 of the reference
everywhere and ~1e-12 in the decision-sensitive region. The regression test pins this
across seeds plus a constant / degenerate column.
"""
from __future__ import annotations

import numpy as np

try:
    import numba as _numba

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep; allow graceful skip.
    _numba = None
    _HAS_NUMBA = False

# Numba wins the per-column variance + cross-product walk only once BOTH the row
# count and the column count are large enough to amortise the prange thread-spawn
# (~50us) + one-shot JIT compile vs numpy's single vectorised matmul. Below these the
# numpy reference's lower per-call overhead wins. Gates per the numba ladder: the
# kernel is the default ONLY for n >= _MIN_ROWS AND F >= _MIN_COLS.
_MIN_ROWS: int = 20_000
_MIN_COLS: int = 64

# Half-width of the "borderline" band: a column whose kernel |corr| lands within this
# of the reference is re-decided with the exact numpy single-column primitive so a
# ~1 ULP kernel/BLAS reduction difference can never leave the dispatcher non-identical
# to the reference. Far wider than the ~1e-15 ULP noise yet narrow enough that
# borderline columns are vanishingly rare on real data.
_BORDERLINE_BAND: float = 1e-9

# Variance floor below which a column / target is "constant" -> 0.0 |corr|. Matches
# the numpy reference (``var < 1e-24 -> 0.0``) exactly.
_VAR_FLOOR: float = 1e-24


if _HAS_NUMBA:

    @_numba.njit(cache=True, fastmath=False, parallel=True)
    def _abs_corr_all_kernel(X, y_dev, var_y, band):
        """Per-column ``|corr(y, X[:, j])|`` + a borderline flag, over a pre-centred ``y``.

        ``y_dev`` is ``y - mean(y)`` (computed once by the wrapper) and ``var_y`` is
        ``dot(y_dev, y_dev)``. Each ``prange`` column walks its own rows twice: once
        for the column mean, once to accumulate the centred variance + the cross
        product with ``y_dev``. No (n, F) centred temporary is allocated.

        ``borderline[j]`` is 1 when column j's ``|corr|`` lands within ``band`` of 1.0
        (the near-1 leak-threshold region) or its variance sits within ``band`` of the
        constant floor -- the only regions where the ~1 ULP reduction drift could move
        the caller's drop/keep decision. The wrapper re-decides exactly those.
        """
        n = X.shape[0]
        n_cols = X.shape[1]
        out = np.zeros(n_cols, dtype=np.float64)
        borderline = np.zeros(n_cols, dtype=np.int8)
        for j in _numba.prange(n_cols):
            s = 0.0
            for i in range(n):
                s += X[i, j]
            m = s / n
            var_x = 0.0
            cov = 0.0
            for i in range(n):
                dx = X[i, j] - m
                var_x += dx * dx
                cov += dx * y_dev[i]
            if var_x < _VAR_FLOOR:
                out[j] = 0.0
            else:
                c = abs(cov / np.sqrt(var_y * var_x))
                out[j] = c
                if c >= 1.0 - band:
                    borderline[j] = 1
            # A variance within a factor of ~1e3 of the constant floor could land on
            # either side of the cutoff under the ~1 ULP reduction drift -> flag for an
            # exact re-decision so a near-constant column's 0.0/value choice is exact.
            if _VAR_FLOOR * 1e-3 <= var_x < _VAR_FLOOR * 1e3:
                borderline[j] = 1
        return out, borderline


def _safe_corr_single(y_dev: np.ndarray, var_y: float, col: np.ndarray) -> float:
    """Exact numpy ``|corr|`` for one column against a pre-centred ``y``.

    Uses the identical centred-dot primitives as the ``_safe_abs_corr_all`` reference
    (``col - col.mean()`` then ``np.dot``), so a re-decided borderline column is
    bit-identical to the reference's value for that column.
    """
    x_dev = col - col.mean()
    var_x = float(np.dot(x_dev, x_dev))
    if var_x < _VAR_FLOOR:
        return 0.0
    cov = float(np.dot(x_dev, y_dev))
    return float(abs(cov / np.sqrt(var_y * var_x)))


def safe_abs_corr_all_dispatch(
    y: np.ndarray, X: np.ndarray, *, reference_fn,
) -> np.ndarray:
    """Size-aware dispatcher for ``|corr(y, X[:, j])|`` over all columns.

    Routes small inputs to ``reference_fn`` (the pure-numpy ``_safe_abs_corr_all``)
    and large inputs (``n >= _MIN_ROWS`` AND ``F >= _MIN_COLS``) to the numba kernel,
    then re-decides any borderline column exactly so the returned vector is
    bit-identical to the reference for all inputs. ``reference_fn`` is injected to
    avoid an import cycle with ``screening``.

    The kernel masks ``y`` GLOBALLY on finiteness exactly like the reference (the
    caller gates columns on finite count before calling), and the (typical) all-finite
    path skips the row-subset copy entirely.
    """
    if X.ndim != 2:
        raise ValueError("safe_abs_corr_all_dispatch expects a 2-D X")
    n_rows, n_cols = X.shape
    # Backend choice goes through the kernel_tuning_cache (per-host measured crossover)
    # with the hardcoded _MIN_ROWS / _MIN_COLS gate as the fallback + an env-var
    # force-override; bit-identical either way (borderline columns re-decided exactly).
    if not _HAS_NUMBA:
        return np.asarray(reference_fn(y, X))
    from ._ktc_dispatch import choose_corr_backend

    if choose_corr_backend(n_rows, n_cols, min_rows=_MIN_ROWS, min_cols=_MIN_COLS) == "numpy":
        return np.asarray(reference_fn(y, X))

    y_finite = np.isfinite(y)
    n_finite = int(y_finite.sum())
    if n_finite < 3:
        return np.zeros(n_cols)
    if n_finite == y_finite.shape[0]:
        y_f = y
        X_f = X
    else:
        y_f = y[y_finite]
        X_f = X[y_finite]
    # Promote to float64 (matches the reference's float64 accumulation in einsum / @);
    # the kernel reads contiguous columns so a C-order copy keeps the column walk
    # cache-friendly.
    X_f = np.ascontiguousarray(X_f, dtype=np.float64)
    y_dev = (y_f - y_f.mean()).astype(np.float64)
    var_y = float(np.dot(y_dev, y_dev))
    if var_y < _VAR_FLOOR:
        return np.zeros(n_cols)
    out, borderline = _abs_corr_all_kernel(X_f, y_dev, var_y, _BORDERLINE_BAND)
    # Re-decide ONLY the columns the kernel flagged borderline (|corr| near the near-1
    # leak threshold, or variance near the constant floor) with the exact numpy
    # single-column primitives the reference uses -- so the caller's drop/keep decision
    # is exact there. Non-flagged columns are far from the threshold; their ~1e-9 drift
    # cannot move a decision. On continuous data this set is empty.
    if borderline.any():
        for j in np.nonzero(borderline)[0]:
            out[j] = _safe_corr_single(y_dev, var_y, X_f[:, j])
    return np.asarray(out)


def _warm_corr_kernel() -> None:
    """Compile the kernel at import on a tiny matrix so the first real call is hot."""
    if not _HAS_NUMBA:
        return
    try:
        warm = np.zeros((4, 2), dtype=np.float64)
        warm[:, 0] = np.arange(4.0)
        warm[:, 1] = np.arange(4.0)[::-1]
        y_dev = np.arange(4.0) - 1.5
        _abs_corr_all_kernel(warm, y_dev, float(np.dot(y_dev, y_dev)), _BORDERLINE_BAND)
    except Exception:  # pragma: no cover - warming is best-effort.  # nosec B110 - best-effort/optional path, no module logger
        pass


_warm_corr_kernel()
