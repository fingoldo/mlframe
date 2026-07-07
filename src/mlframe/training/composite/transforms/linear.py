"""Linear-residual + logratio composite transforms carved out of
``mlframe.training.composite_transforms``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _linear_residual_fit``
resolves transparently.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

try:
    import numba as _numba  # type: ignore
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _numba = None  # type: ignore
    _HAS_NUMBA = False

logger = logging.getLogger("mlframe.training.composite_transforms")

# Parent-resident constants referenced as default-arg values in function
# signatures below. Function-signature defaults evaluate at module load, so
# these MUST be top-level (lazy import inside the body wouldn't see them).
# The parent defines all four BEFORE its bottom-of-module sibling import,
# so this static cycle resolves at runtime. Whitelisted in
# tests/test_meta/test_no_import_cycles.py.
from . import (  # noqa: E402
    _GROUPED_MIN_GROUP_SIZE,
    _LINRES_ROBUST_MAD_K,
    _LINRES_ROBUST_MIN_KEEP_FRAC,
    _MULTI_BASE_COND_NUMBER_MAX,
    _THEILSEN_MAX_PAIRS,
)


logger = logging.getLogger("mlframe.training.composite_transforms")

def _logratio_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    # T_train computed in the valid domain (caller has already filtered).
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _MAD_FLOOR_FRAC, _MAD_SOFT_CAP_K
    t_train = np.log(y) - np.log(base)
    median_t = float(np.median(t_train))
    mad_train = float(np.median(np.abs(t_train - median_t)))
    # Floor against degenerate T_train (constant on train) -- otherwise
    # MAD = 0 collapses every prediction to ``base * exp(median_t)``,
    # which still ranks as "naive baseline" at predict time but at
    # least does not distort in-distribution predictions.
    std_y = float(np.std(y))
    mad_floor = _MAD_FLOOR_FRAC * std_y if std_y > 0 else 1e-6
    mad_eff = max(mad_train, mad_floor)
    return {
        "median_t": median_t,
        "mad_train": mad_train,
        "mad_eff": mad_eff,
        "soft_cap_k": _MAD_SOFT_CAP_K,
    }
def _logratio_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return np.log(y) - np.log(base)
def _logratio_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    median_t = float(params["median_t"])
    mad = float(params["mad_eff"])
    k = float(params["soft_cap_k"])
    # Soft-cap is centred on median(T_train), NOT on zero -- otherwise
    # any T distribution offset from zero (the typical case for
    # logratio when y and base have similar scale) gets clobbered by
    # the cap and inverse predictions collapse to ``base``.
    cap = k * mad
    t_capped = np.clip(t_hat, median_t - cap, median_t + cap)
    return base * np.exp(t_capped)
def _logratio_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base) & (base > 0)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y) & (y > 0)
def _linear_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """OLS fit with optional sample weights.

    Weighted least squares is implemented in closed form via
    ``np.linalg.lstsq`` on the row-scaled system
    ``sqrt(w) * X * beta = sqrt(w) * y`` (standard reformulation).
    Weights are normalised to sum to ``len(y)`` so the fit's
    numerical scale matches the unweighted case (avoids tiny
    coefficients on small w values).
    """
    n = len(y)
    if n < 2:
        return {"alpha": 0.0, "beta": float(np.mean(y)) if n > 0 else 0.0}
    X = np.column_stack([base.astype(np.float64), np.ones(n, dtype=np.float64)])
    y_f = y.astype(np.float64)

    if sample_weight is None:
        coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError(
                f"_linear_residual_fit: sample_weight length {w.size} != y length {n}"
            )
        # Drop non-positive weights silently; warn if all zero.
        finite = np.isfinite(w) & (w > 0)
        if not finite.any():
            return {"alpha": 0.0, "beta": float(np.mean(y_f))}
        # Normalise weights to mean 1 so the system has the same
        # numerical scale as the unweighted version. lstsq handles
        # rank-deficient cases.
        w_norm = w[finite]
        w_norm = w_norm * (n / w_norm.sum())
        sw = np.sqrt(w_norm)
        X_w = X[finite] * sw[:, None]
        y_w = y_f[finite] * sw
        coef, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    return {"alpha": alpha, "beta": beta}
def _linear_residual_fit_closed(
    x: np.ndarray, y: np.ndarray,
) -> tuple[float, float]:
    """Single-fold unweighted OLS via the closed-form normal equations.

    Reference scalar path the batched solver below is bit-identical to. For
    ``base = x``, ``y`` (length n) this returns ``(alpha, beta)`` of
    ``y ~ alpha*x + beta`` using
    ``alpha = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)``, ``beta = mean_y - alpha*mean_x``
    -- exact OLS, no ``lstsq`` / SVD dispatch. Degenerate folds are guarded
    EXACTLY as ``_linear_residual_fit``'s scalar path: ``n < 2`` returns
    ``(0.0, mean(y))`` (or ``(0.0, 0.0)`` for empty), and a zero-variance base
    (``den == 0``) returns ``(0.0, mean(y))``.

    NOTE: this is the normal-equations path, which differs from
    ``np.linalg.lstsq`` by ~1 ULP. It is used by the per-fold CV refit loops
    (alpha-drift gate, bootstrap, grouped per-segment) where the K systems are
    solved together via :func:`_linear_residual_fit_batched`; the public
    ``_linear_residual_fit`` keeps the ``lstsq`` path for selection stability.
    """
    n = len(y)
    if n < 2:
        return 0.0, (float(np.mean(y)) if n > 0 else 0.0)
    x_f = x.astype(np.float64)
    y_f = y.astype(np.float64)
    sx = float(x_f.sum())
    sy = float(y_f.sum())
    sxx = float((x_f * x_f).sum())
    sxy = float((x_f * y_f).sum())
    den = n * sxx - sx * sx
    if den == 0.0:
        return 0.0, float(np.mean(y_f))
    alpha = (n * sxy - sx * sy) / den
    beta = float(np.mean(y_f)) - alpha * float(np.mean(x_f))
    return float(alpha), float(beta)
def _linear_residual_fit_batched(
    x_segments: Sequence[np.ndarray],
    y_segments: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve K independent single-base OLS systems in one batched pass.

    ``x_segments`` / ``y_segments`` are the K per-fold ``(base, y)`` arrays. Each
    fold's five reductions (``n, Sx, Sy, Sxx, Sxy``) are computed on the
    contiguous segment with the SAME ``.sum()`` order as the scalar
    :func:`_linear_residual_fit_closed`, then the K ``(alpha, beta)`` pairs are
    derived in a SINGLE vectorised arithmetic pass over length-K arrays. This
    pays K x dispatch overhead exactly once (one set of ufunc calls) instead of
    K ``lstsq`` / SVD launches -- bit-identical to calling
    ``_linear_residual_fit_closed`` per fold (verified: the per-segment ``.sum()``
    reduction order is preserved, and the zero-variance / ``n<2`` guards match).

    Returns ``(alphas, betas)`` as length-K float64 ndarrays aligned to the
    input segment order. Empty input returns two empty arrays.
    """
    k = len(x_segments)
    if k != len(y_segments):
        raise ValueError(
            f"_linear_residual_fit_batched: {k} x-segments != {len(y_segments)} y-segments"
        )
    alphas = np.zeros(k, dtype=np.float64)
    betas = np.zeros(k, dtype=np.float64)
    if k == 0:
        return alphas, betas
    counts = np.empty(k, dtype=np.float64)
    sx = np.empty(k, dtype=np.float64)
    sy = np.empty(k, dtype=np.float64)
    sxx = np.empty(k, dtype=np.float64)
    sxy = np.empty(k, dtype=np.float64)
    small = np.zeros(k, dtype=bool)  # folds with n<2 -> (0, mean(y)|0).
    for i in range(k):
        xi = x_segments[i].astype(np.float64)
        yi = y_segments[i].astype(np.float64)
        n = yi.size
        counts[i] = n
        if n < 2:
            small[i] = True
            betas[i] = float(np.mean(yi)) if n > 0 else 0.0
            sx[i] = sy[i] = sxx[i] = sxy[i] = 0.0
            continue
        sx[i] = xi.sum()
        sy[i] = yi.sum()
        sxx[i] = (xi * xi).sum()
        sxy[i] = (xi * yi).sum()
    big = ~small
    if big.any():
        den = counts * sxx - sx * sx
        # Zero-variance base (den==0) degenerates to (0, mean(y)) exactly as
        # the scalar guard. mean(y) = Sy / n is the same value the scalar path
        # returns via np.mean for a finite-only segment.
        nz = big & (den != 0.0)
        safe_den = np.where(nz, den, 1.0)
        a = (counts * sxy - sx * sy) / safe_den
        mean_x = sx / np.where(counts > 0.0, counts, 1.0)
        mean_y = sy / np.where(counts > 0.0, counts, 1.0)
        alphas = np.where(nz, a, alphas)
        betas = np.where(big, np.where(nz, mean_y - alphas * mean_x, mean_y), betas)
    return alphas, betas
def _linear_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return y - alpha * base - beta
def _linear_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    alpha = float(params["alpha"])
    beta = float(params["beta"])
    return t_hat + alpha * base + beta
def _linear_residual_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base_ok = np.isfinite(base)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)
def _linear_residual_robust_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Trimmed-LS fit: OLS -> drop |resid|>3*MAD -> refit OLS.

    Returns the same ``{"alpha", "beta"}`` dict as :func:`_linear_residual_fit` so the existing forward / inverse functions work unchanged. ``sample_weight`` is honoured in BOTH passes.

    When the MAD-trim step doesn't drop any rows (no outliers above
    3*sigma_MAD), the second-pass OLS produces alpha/beta IDENTICAL to
    the first pass — the transform is mathematically equivalent to
    plain ``linear_residual``. We stamp ``is_redundant_with_linres=True``
    on the returned dict so composite discovery can skip the duplicate
    evaluation (observed in a prod log: y-linres-Y and
    y-linresR-Y produced identical RMSE=21.5433 because no outliers
    were trimmed -- pure duplicate compute).
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    n = len(y)
    if n < 2:
        result = {"alpha": 0.0, "beta": float(np.mean(y)) if n > 0 else 0.0}
        result["is_redundant_with_linres"] = True  # constant fit, OLS would do the same
        return result

    # Pass 1: standard OLS.
    first_pass = _linear_residual_fit(y, base, sample_weight=sample_weight)
    alpha1 = float(first_pass["alpha"])
    beta1 = float(first_pass["beta"])

    # Residuals + robust scale via MAD (sigma-equivalent multiplier 1.4826).
    resid = y.astype(np.float64) - alpha1 * base.astype(np.float64) - beta1
    if not np.all(np.isfinite(resid)):
        first_pass["is_redundant_with_linres"] = True
        return first_pass
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma_mad = mad * 1.4826
    if sigma_mad <= 0.0 or not np.isfinite(sigma_mad):
        # Constant residual or numerical pathology -- OLS already covers it.
        first_pass["is_redundant_with_linres"] = True
        return first_pass

    keep = np.abs(resid - med) <= _LINRES_ROBUST_MAD_K * sigma_mad
    n_kept = int(keep.sum())
    if n_kept < max(2, int(_LINRES_ROBUST_MIN_KEEP_FRAC * n)):
        first_pass["is_redundant_with_linres"] = True
        return first_pass
    if n_kept == n:
        # No outliers trimmed -> pass-2 OLS on inlier set IS pass-1 OLS.
        # Skip the redundant refit and mark for discovery-side dedup.
        first_pass["is_redundant_with_linres"] = True
        return first_pass

    # Pass 2: OLS on the inlier set.
    sw2 = None if sample_weight is None else np.asarray(sample_weight)[keep]
    return _linear_residual_fit(y[keep], base[keep], sample_weight=sw2)
def _theilsen_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Theil-Sen (median-of-pairwise-slopes) robust line fit.

    The slope ``alpha`` is the median of the pairwise slopes
    ``(y_j - y_i) / (base_j - base_i)`` over all point pairs ``i < j``; the
    intercept ``beta`` is the median of ``y - alpha * base``. Both are
    high-breakdown estimators (~29.3% breakdown for the slope), so up to
    ~29% of the rows can be gross outliers in EITHER ``y`` or ``base``
    without corrupting the fit -- unlike OLS (``linear_residual``, breakdown
    0%) or the one-shot trimmed-LS (``linear_residual_robust``, which still
    seeds from an OLS pass that a clustered outlier mass can already drag).

    Returns the same ``{"alpha", "beta"}`` dict as
    :func:`_linear_residual_fit`, so the shared ``linear_residual``
    forward / inverse / domain functions apply unchanged.

    Complexity. Theil-Sen is O(n^2) in the number of point pairs. For large
    ``n`` we cap the work at ``_THEILSEN_MAX_PAIRS`` by drawing a random
    subsample of pairs with a fixed-seed generator (deterministic across
    runs / processes). The subsampled estimator is the standard scalable
    Theil-Sen variant and keeps the breakdown-point robustness; only the
    finite-sample slope variance grows slightly versus the full O(n^2) fit.
    ``sample_weight`` is accepted for API symmetry but does NOT reweight the
    median (weighting a median is ill-defined for small clusters); it is
    ignored here, matching the per-group OLS weight handling.
    """
    n = len(y)
    if n < 2:
        return {"alpha": 0.0, "beta": float(np.mean(y)) if n > 0 else 0.0}

    y_f = y.astype(np.float64)
    base_f = base.astype(np.float64)
    finite = np.isfinite(y_f) & np.isfinite(base_f)
    if not finite.all():
        y_f = y_f[finite]
        base_f = base_f[finite]
    n = y_f.size
    if n < 2:
        return {"alpha": 0.0, "beta": float(np.mean(y_f)) if n > 0 else 0.0}

    # Full pair count is n*(n-1)/2. Subsample when it exceeds the cap.
    total_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(0)
    if total_pairs <= _THEILSEN_MAX_PAIRS:
        i_idx, j_idx = np.triu_indices(n, k=1)
    else:
        # Draw random ordered index pairs and reject self-pairs. A small
        # oversample factor compensates for the rejected diagonal so we end
        # up with ~_THEILSEN_MAX_PAIRS usable pairs in one pass.
        m = int(_THEILSEN_MAX_PAIRS * 1.05) + 1
        i_idx = rng.integers(0, n, size=m)
        j_idx = rng.integers(0, n, size=m)
        keep = i_idx != j_idx
        i_idx = i_idx[keep][:_THEILSEN_MAX_PAIRS]
        j_idx = j_idx[keep][:_THEILSEN_MAX_PAIRS]

    db = base_f[j_idx] - base_f[i_idx]
    dy = y_f[j_idx] - y_f[i_idx]
    # Drop pairs with (near-)tied base (vertical slope / 0/0); the median
    # over the remaining finite slopes is the Theil-Sen estimate.
    valid = np.abs(db) > 0.0
    slopes = dy[valid] / db[valid]
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        # Degenerate: base is (near-)constant -> no slope information.
        return {"alpha": 0.0, "beta": float(np.median(y_f))}
    alpha = float(np.median(slopes))
    beta = float(np.median(y_f - alpha * base_f))
    return {"alpha": alpha, "beta": beta}
def _linear_residual_multi_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Joint OLS fit for ``T = y - Σⱼ αⱼ·baseⱼ - β``.

    Parameters
    ----------
    y
        Training target (``shape=(n,)``).
    base
        Either 1-D ``(n,)`` (K=1, equivalent to single-base linear_residual)
        or 2-D ``(n, K)`` for K-base joint fit.
    sample_weight
        Optional row weights - same weighted-LS reformulation as
        single-base ``_linear_residual_fit``.

    Returns
    -------
    dict with keys:
    - ``alphas``: list of K floats (one per base column).
    - ``beta``: float intercept.
    - ``condition_number``: float - design-matrix condition number;
      diagnostic only.
    - ``collinear_fallback``: bool - True if condition number > the
      gate, in which case ``alphas`` is all-zero and ``beta`` is the
      train mean of ``y`` (transform degenerates to ``T = y - mean(y)``).
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    # asarray (not astype) avoids a redundant full (n, K) / (n,) copy when the caller already passes float64 (the CV-fold path supplies a preallocated float64 trial buffer); the body never mutates base_f / y_f in place (centering, column_stack and the row-mask all allocate fresh arrays), so a view is safe and bit-identical.
    base_f = np.asarray(base, dtype=np.float64)
    y_f = np.asarray(y, dtype=np.float64)
    # Drop non-finite rows (non-finite y OR any non-finite base column) before
    # the OLS. Lag / rolling bases carry leading NaN; np.linalg.lstsq / svd
    # raise LinAlgError on NaN, which the callers' broad except silently
    # degrades to a single-base / collinear (alphas=0) fallback -- so the
    # default-ON multi-base promotion died precisely on the canonical lag-base
    # case, losing the benchmark-validated win. Masking here makes every caller
    # NaN-safe and mirrors _linear_residual_multi_domain's finite gate.
    row_finite = np.isfinite(y_f) & np.all(np.isfinite(base_f), axis=1)
    if not bool(row_finite.all()):
        base_f = base_f[row_finite]
        y_f = y_f[row_finite]
        if sample_weight is not None:
            sample_weight = np.asarray(
                sample_weight, dtype=np.float64,
            ).reshape(-1)[row_finite]
    n, k = base_f.shape
    if n < k + 1:
        return {
            "alphas": [0.0] * k,
            "beta": float(np.mean(y_f)) if n > 0 else 0.0,
            "condition_number": float("nan"),
            "collinear_fallback": True,
        }
    X = np.column_stack([base_f, np.ones(n, dtype=np.float64)])

    # Condition-number gate (Belsley/Kuh/Welsch). Computed on the
    # CENTERED base columns ONLY: including the intercept column would
    # conflate uncentered scale (e.g. ``base ~ N(10, 2)`` gives cond
    # >> 30 just from the offset, not from real multicollinearity).
    # For K=1 the centered base is just a single column => cond = 1
    # by construction (still go through the path so the code shape
    # matches the K>1 branch).
    try:
        if k == 1:
            cond = 1.0
        else:
            base_centered = base_f - base_f.mean(axis=0, keepdims=True)
            # Guard against an exactly-constant base column (zero
            # singular value); cond is infinite by definition there.
            col_norms = np.linalg.norm(base_centered, axis=0)
            if np.any(col_norms < 1e-12):
                cond = float("inf")
            else:
                # BKW scaled condition index: equilibrate columns to unit
                # norm BEFORE the SVD. Without this the condition number is
                # scale-VARIANT -- two independent bases in different units
                # (e.g. N(0,1) and N(0,1e6)) read cond ~ 1e6 and get falsely
                # rejected as collinear (alphas=0), silently killing a valid
                # multi-base residual. Unit-norm scaling measures genuine
                # multicollinearity (column angle), not unit mismatch.
                base_scaled = base_centered / col_norms
                # The cond gate only needs the singular VALUES of the tall (n, K) scaled base. For K << n the values are the sqrt of the eigenvalues of the tiny (K, K) Gram matrix, which `eigvalsh` returns ~2.7x faster than a full tall-matrix `svd` (1.03ms -> 0.38ms @ n=100k, K=3) with no change to the lstsq inputs, so the fitted alphas/beta stay bit-identical. Squaring the matrix doubles the relative error on the condition number (~3.5e-8 near-collinear), so when the fast value lands inside a +-0.01% band of the gate threshold we recompute the exact SVD condition number -- a band ~6 orders of magnitude wider than the error, so the gate decision and the stored diagnostic are exact wherever they could matter.
                gram = base_scaled.T @ base_scaled
                ev = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
                sv = np.sqrt(ev)
                cond = float(sv.max() / max(sv.min(), np.finfo(np.float64).tiny))
                _band = _MULTI_BASE_COND_NUMBER_MAX
                if _band * (1.0 - 1e-4) <= cond <= _band * (1.0 + 1e-4):
                    sv = np.linalg.svd(base_scaled, compute_uv=False)
                    cond = float(sv.max() / max(sv.min(), np.finfo(np.float64).tiny))
    except np.linalg.LinAlgError:
        cond = float("inf")
    if cond > _MULTI_BASE_COND_NUMBER_MAX or not np.isfinite(cond):
        return {
            "alphas": [0.0] * k,
            "beta": float(np.mean(y_f)),
            "condition_number": cond,
            "collinear_fallback": True,
        }

    if sample_weight is None:
        coef, *_ = np.linalg.lstsq(X, y_f, rcond=None)
    else:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError(
                f"_linear_residual_multi_fit: sample_weight length {w.size} "
                f"!= y length {n}"
            )
        finite = np.isfinite(w) & (w > 0)
        if not finite.any():
            return {
                "alphas": [0.0] * k,
                "beta": float(np.mean(y_f)),
                "condition_number": cond,
                "collinear_fallback": True,
            }
        w_norm = w[finite] * (n / w[finite].sum())
        sw = np.sqrt(w_norm)
        X_w = X[finite] * sw[:, None]
        y_w = y_f[finite] * sw
        coef, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    return {
        "alphas": [float(c) for c in coef[:k]],
        "beta": float(coef[-1]),
        "condition_number": cond,
        "collinear_fallback": False,
    }
def _linear_residual_multi_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alphas = np.asarray(params["alphas"], dtype=np.float64)
    beta = float(params["beta"])
    if base.shape[1] != alphas.size:
        raise ValueError(
            f"linear_residual_multi: base has {base.shape[1]} columns but "
            f"fitted alphas has {alphas.size} entries"
        )
    return y - (np.ascontiguousarray(base, dtype=np.float64) @ alphas) - beta
def _linear_residual_multi_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    alphas = np.asarray(params["alphas"], dtype=np.float64)
    beta = float(params["beta"])
    if base.shape[1] != alphas.size:
        raise ValueError(
            f"linear_residual_multi: base has {base.shape[1]} columns but "
            f"fitted alphas has {alphas.size} entries"
        )
    # Canonical C-contiguous layout so the (n,K)@(K,) matvec rounds identically regardless of the caller's
    # array order (C vs F differ by ~1 ULP in BLAS dgemv); keeps predict and the serving spec byte-identical.
    return t_hat + (np.ascontiguousarray(base, dtype=np.float64) @ alphas) + beta
def _linear_residual_multi_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    if base.ndim == 1:
        base = base.reshape(-1, 1)
    base_ok = np.all(np.isfinite(base), axis=1)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(y)
def _linear_residual_grouped_fit(
    y: np.ndarray, base: np.ndarray,
    groups: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    min_group_size: int = _GROUPED_MIN_GROUP_SIZE,
) -> dict[str, Any]:
    """Per-group OLS fit with James-Stein shrinkage toward global α.

    Parameters
    ----------
    y, base
        Training target and base (1-D arrays, length n).
    groups
        Per-row group labels (1-D ndarray, length n). Required; raised
        as ValueError if None.
    sample_weight
        Optional row weights (currently unused for the per-group OLS
        because weight semantics within small groups are unstable; the
        global fit honours weights via standard OLS).
    min_group_size
        Rows-per-group threshold below which a group skips its own
        OLS and uses the global α/β.

    Returns
    -------
    fitted_params with keys:
    - ``alpha_global`` / ``beta_global``: global OLS fit on all rows.
    - ``per_group_alphas`` / ``per_group_betas``: dict[group_label ->
      float]. Keys are str(group_label) for JSON-serialisability.
    - ``shrinkage_factor``: float in [0, 1] from James-Stein.
    - ``group_sizes``: dict[str(group_label) -> int]; diagnostic.
    - ``min_group_size``: int; the threshold used.
    """
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _canonical_group_key, _james_stein_shrinkage_factor
    if groups is None:
        raise ValueError(
            "linear_residual_grouped requires a 1-D ``groups`` array of "
            "per-row group labels (configure ``group_column`` on the "
            "wrapper). For ungrouped fit, use ``linear_residual``."
        )
    groups = np.asarray(groups).reshape(-1)
    if len(groups) != len(y):
        raise ValueError(
            f"linear_residual_grouped: groups has {len(groups)} rows but "
            f"y has {len(y)} rows."
        )

    # Global OLS: same logic as legacy linear_residual.
    global_params = _linear_residual_fit(y, base, sample_weight=sample_weight)
    alpha_global = float(global_params["alpha"])
    beta_global = float(global_params["beta"])

    # Per-group OLS.
    per_group_alphas: dict[str, float] = {}
    per_group_betas: dict[str, float] = {}
    group_sizes: dict[str, int] = {}
    unique_groups, inverse_idx = np.unique(groups, return_inverse=True)
    # Sort rows once by group label so each group is a contiguous segment, instead
    # of re-scanning the whole length-n ``inverse_idx`` with a boolean mask per
    # group (the historic ``inverse_idx == i`` pattern is O(K*n) -- 13x slower at
    # K=2000/n=200k, see _benchmarks/bench_grouped_fit_segment.py). A STABLE argsort
    # preserves the original ascending row order within each group, so the per-group
    # ``y_g`` / ``base_g`` views are the SAME arrays (same values, same order) the
    # mask path produced -> the per-group OLS (lstsq) result is bit-identical.
    n_groups = unique_groups.size
    _order = np.argsort(inverse_idx, kind="stable")
    _y_sorted = np.asarray(y)[_order]
    _base_sorted = np.asarray(base)[_order]
    _sw_sorted = np.asarray(sample_weight)[_order] if sample_weight is not None else None
    _counts = np.bincount(inverse_idx, minlength=n_groups)
    _offsets = np.empty(n_groups + 1, dtype=np.int64)
    _offsets[0] = 0
    np.cumsum(_counts, out=_offsets[1:])

    # Cache residual squared sum across all groups to estimate σ² for
    # James-Stein. Computed against the per-group OLS predictions (the
    # estimator we're trying to shrink).
    total_resid_sq = 0.0
    total_n = 0

    alphas_for_shrink: list[float] = []
    sizes_for_shrink: list[float] = []
    # Per-group Var(base_g) for the eligible groups, aligned 1:1 with
    # alphas_for_shrink / sizes_for_shrink. The James-Stein noise proxy for
    # shrinking OLS *slopes* is Var(α_g) = σ²/(n_g·Var(base_g)); supplying
    # Var(base_g) makes the shrinkage factor scale-invariant (rescaling base
    # no longer changes which groups shrink). See _james_stein_shrinkage_factor.
    base_vars_for_shrink: list[float] = []
    # Per-group mean(base_g) for the eligible (own-OLS) groups. Needed to
    # re-centre beta_g when alpha_g is shrunk (see the shrinkage-apply
    # block below): the per-group OLS guarantees
    # mean(y_g) = a_g·mean(base_g) + b_g, so any change to a_g must be
    # absorbed by b_g to keep the per-group residual mean at 0.
    base_mean_for_shrink: dict[str, float] = {}

    for i, g in enumerate(unique_groups):
        n_g = int(_counts[i])
        # Canonical key so int<->float dtype drift between fit and predict cannot
        # silently miss every group and collapse to the global alpha/beta.
        g_key = _canonical_group_key(g)
        group_sizes[g_key] = n_g
        if n_g < min_group_size:
            # Skip per-group OLS; defer to global.
            per_group_alphas[g_key] = alpha_global
            per_group_betas[g_key] = beta_global
            continue
        _lo, _hi = int(_offsets[i]), int(_offsets[i + 1])
        y_g = _y_sorted[_lo:_hi]
        base_g = _base_sorted[_lo:_hi]
        sw_g = _sw_sorted[_lo:_hi] if _sw_sorted is not None else None
        try:
            params_g = _linear_residual_fit(y_g, base_g, sample_weight=sw_g)
            a_g = float(params_g["alpha"])
            b_g = float(params_g["beta"])
        except Exception:  # pragma: no cover - defensive
            a_g, b_g = alpha_global, beta_global
        per_group_alphas[g_key] = a_g
        per_group_betas[g_key] = b_g
        alphas_for_shrink.append(a_g)
        sizes_for_shrink.append(float(n_g))
        base_g64 = base_g.astype(np.float64)
        base_mean_for_shrink[g_key] = float(np.mean(base_g64))
        # Var(base_g) drives the scale-invariant JS slope-variance proxy.
        base_vars_for_shrink.append(float(np.var(base_g64)) if n_g > 1 else 0.0)
        # Accumulate residuals for σ² estimate.
        resid = y_g - a_g * base_g - b_g
        total_resid_sq += float(np.sum(resid * resid))
        total_n += n_g

    # James-Stein shrinkage of the eligible per-group alphas.
    if len(alphas_for_shrink) >= 4 and total_n > 0:
        sigma2 = total_resid_sq / max(total_n - 2 * len(alphas_for_shrink), 1)
        c = _james_stein_shrinkage_factor(
            np.asarray(alphas_for_shrink, dtype=np.float64),
            alpha_global,
            np.asarray(sizes_for_shrink, dtype=np.float64),
            sigma2,
            base_vars=np.asarray(base_vars_for_shrink, dtype=np.float64),
        )
    else:
        c = 0.0
    # Apply shrinkage to eligible groups (ones that ran their own OLS).
    # Re-centre beta_g in lockstep: shrinking alpha alone tilts the fitted
    # line about base=0, so the per-group residual mean drifts to
    # c·(a_g - alpha_global)·mean(base_g) (manufactured per-group bias).
    # Because the per-group OLS fixes mean(y_g) = a_g·mean(base_g) + b_g,
    # the bias-free intercept for the shrunk slope alpha_s is
    # b_g_new = mean(y_g) - alpha_s·mean(base_g)
    #         = b_g + (a_g - alpha_s)·mean(base_g)
    #         = b_g + c·(a_g - alpha_global)·mean(base_g),
    # which restores mean(y_g - alpha_s·base_g - b_g_new) == 0.
    if c > 0:
        for g_key, a_g in list(per_group_alphas.items()):
            n_g = group_sizes[g_key]
            if n_g < min_group_size:
                continue
            alpha_shrunk = (1.0 - c) * a_g + c * alpha_global
            base_mean_g = base_mean_for_shrink.get(g_key)
            if base_mean_g is not None:
                per_group_betas[g_key] = (
                    per_group_betas[g_key]
                    + (a_g - alpha_shrunk) * base_mean_g
                )
            per_group_alphas[g_key] = alpha_shrunk

    return {
        "alpha_global": alpha_global,
        "beta_global": beta_global,
        "per_group_alphas": per_group_alphas,
        "per_group_betas": per_group_betas,
        "shrinkage_factor": float(c),
        "group_sizes": group_sizes,
        "min_group_size": int(min_group_size),
    }
def _linear_residual_grouped_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _row_alpha_beta
    if groups is None:
        raise ValueError(
            "linear_residual_grouped.forward: groups kwarg is required."
        )
    groups = np.asarray(groups).reshape(-1)
    row_alpha, row_beta = _row_alpha_beta(groups, params)
    return y - row_alpha * base - row_beta
def _linear_residual_grouped_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
    groups: np.ndarray | None = None,
) -> np.ndarray:
    # Lazy import of parent-resident helpers: ``.predict`` re-imports
    # this sibling at its bottom, so a top-level ``from .predict
    # import ...`` would create a hard cycle the meta-test flags.
    from . import _row_alpha_beta
    if groups is None:
        raise ValueError(
            "linear_residual_grouped.inverse: groups kwarg is required."
        )
    groups = np.asarray(groups).reshape(-1)
    row_alpha, row_beta = _row_alpha_beta(groups, params)
    return t_hat + row_alpha * base + row_beta
def _linear_residual_grouped_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    return _linear_residual_domain(y, base)
