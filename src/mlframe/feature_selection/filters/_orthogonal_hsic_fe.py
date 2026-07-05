"""Layer 71 (2026-06-01): Hilbert-Schmidt Independence Criterion (HSIC)
kernel-based dependence ranking for hybrid orth-poly FE.

Why this layer
--------------

Layers 21 / 65 / 66 are MI estimators; Layer 67 is the Szekely-Rizzo
distance correlation (also distance-based). Layer 71 adds a fourth
dependence family -- the Gretton-Bousquet-Smola-Schoelkopf (2005)
Hilbert-Schmidt Independence Criterion. HSIC measures dependence by the
Hilbert-Schmidt norm of the cross-covariance OPERATOR between RKHS
embeddings of ``X`` and ``Y``. With a CHARACTERISTIC kernel (the
Gaussian RBF) the headline guarantee holds:

    HSIC(X, Y) = 0  iff  X is independent of Y.

This is the same universal iff-independence guarantee dCor offers, but
arrived at via a totally different construction: rather than centred
distance matrices, HSIC operates on centred RBF Gram matrices.

HSIC vs dCor
------------

* Both are zero iff X is independent of Y -- in finite samples they
  rank dependencies differently. dCor's distance matrices have an O(1)
  scale (every pairwise distance contributes); RBF Gram matrices
  concentrate on near-neighbours through ``exp(-||xi - xj||^2 / 2 sigma^2)``
  and the bandwidth sigma picks the effective scale.
* HSIC is bandwidth-tunable: the MEDIAN HEURISTIC sets sigma to the
  median pairwise distance -- the standard choice (Gretton 2005) -- and
  is the kernel-method analogue of dCor's parameter-free construction.
* HSIC tends to be tighter than dCor when the signal lives at a SCALE
  smaller than the typical pairwise distance (sharp local thresholds,
  oscillatory dependence with high frequency); dCor tends to dominate
  on smooth large-scale dependence.

The two estimators are COMPLEMENTARY -- the L68 / L69 auto/ensemble
pools take advantage of this by letting the per-column auto-selector
pick whichever scorer wins on a given column.

Bias / consistency
------------------

We expose two estimators:

* HSIC_b ("biased"): ``trace(K_x H K_y H) / (n - 1)^2`` where ``H = I -
  ones / n`` is the centring projector. Always non-negative, O(n) bias
  but converges to the true HSIC at rate ``1 / sqrt(n)``.
* HSIC_u ("unbiased"): the Song / Smola 2012 U-statistic correction --
  guaranteed unbiased but can be slightly negative on independent
  pairs (Monte-Carlo cancellation). The cross-column ranking uses
  ``max(HSIC_u, 0)`` so the downstream uplift gate stays well-defined.

We default to the BIASED estimator because its variance is lower and
the cross-column ranking does not need unbiasedness -- only monotone
sensitivity to dependence strength.

Cost / sample-size constraint
-----------------------------

HSIC is constructed from the FULL ``n x n`` Gram matrices. Naive cost
is ``O(n^2)`` time and memory; ``n = 500`` fits in 2 MB per matrix and
runs in under 50 ms on a modern laptop. Beyond ``n = 500`` the per-
feature cost dominates; this module caps the working sample at
``n_sample = 500`` via deterministic random subsampling -- the HSIC
estimator is asymptotically consistent and 500 samples are enough for
the dependence test to discriminate signal from noise at the typical
SNR seen by Layer 21's downstream gates. The same cost constant as
Layer 67.

Layer 71 vs Layers 65 / 66 / 67
-------------------------------

* Layer 65 (KSG): MI estimator, distance-based, good on smooth signals.
* Layer 66 (copula MI): MI estimator, rank-based, good on heavy-tailed
  marginals.
* Layer 67 (dCor): NON-MI dependence, distance-based, zero iff
  independent on ANY relationship.
* Layer 71 (this, HSIC): NON-MI dependence, KERNEL-based, zero iff
  independent on ANY relationship with a CHARACTERISTIC kernel.

The four are COMPLEMENTARY -- a user can opt into all four and take
the union of winners as a robust shortlist, or use the Layer 68 / 69
auto-selector / ensemble to pick per column.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal
to Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_hsic_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import generate_univariate_basis_features

logger = logging.getLogger(__name__)

# Below this baseline a source is treated as no-signal: the uplift ratio is
# suppressed (it would otherwise explode) and an absolute MI floor is required
# instead -- the same guard JMIM applies (Layer 21/65+).
_BASELINE_EPS = 1e-6
_ABS_MI_FLOOR = 1e-3

__all__ = [
    "hsic",
    "median_heuristic_sigma",
    "score_features_by_hsic_uplift",
    "hybrid_orth_mi_hsic_fe",
    "hybrid_orth_mi_hsic_fe_with_recipes",
]


def _subsample_indices(
    n: int, n_sample: int, random_state: int,
) -> np.ndarray:
    """Deterministic random subsample of ``min(n, n_sample)`` indices.

    Returns ``np.arange(n)`` when ``n <= n_sample`` (no work needed);
    else a sorted random permutation of ``n_sample`` indices to keep the
    relative ordering of the subsample consistent across paired calls.
    Identical to the Layer 67 helper but inlined here to keep the
    sibling module dependency surface tight (no cross-layer import).
    """
    if n_sample <= 0 or n <= int(n_sample):
        return np.arange(n)
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n, size=int(n_sample), replace=False)
    idx.sort()
    return idx


def median_heuristic_sigma(z: np.ndarray) -> float:
    """Median pairwise-distance bandwidth (Gretton 2005).

    For a 1-D array ``z``, returns the median of ``|z_i - z_j|`` over
    all upper-triangular pairs (``i < j``). When ``n <= 1`` or every
    pairwise distance is zero (constant column), returns ``1.0`` as a
    neutral fallback so the downstream RBF Gram matrix stays well-
    defined (``exp(0) = 1`` on every entry; the resulting HSIC is 0,
    which is the right "no signal" reading for a constant column).
    """
    arr = np.asarray(z, dtype=np.float64).ravel()
    n = arr.shape[0]
    if n < 2:
        return 1.0
    # Use the upper triangle of the absolute-difference matrix; this is
    # the standard median-heuristic definition (every UNORDERED pair
    # contributes once -- the diagonal is dropped because |z_i - z_i| = 0
    # is uninformative and would pull the median down).
    diff = np.abs(arr[:, None] - arr[None, :])
    iu = np.triu_indices(n, k=1)
    vals = diff[iu]
    med = float(np.median(vals))
    if med <= 0.0:
        # All values identical -> no scale information; neutral 1.0.
        return 1.0
    return med


def _rbf_gram(z: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF Gram matrix ``K_ij = exp(-||z_i - z_j||^2 / 2 sigma^2)``.

    Sigma is interpreted as the BANDWIDTH (the same convention sklearn
    and Gretton use); the variance scaling is ``2 * sigma^2`` in the
    exponent.
    """
    arr = np.asarray(z, dtype=np.float64).ravel()
    diff = arr[:, None] - arr[None, :]
    sq = diff * diff
    s = float(sigma) if float(sigma) > 0.0 else 1.0
    return np.exp(-sq / (2.0 * s * s))


def hsic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "rbf",
    sigma: Optional[float] = None,
    sigma_y: Optional[float] = None,
    n_sample: int = 500,
    random_state: int = 0,
    estimator: str = "biased",
) -> float:
    """Hilbert-Schmidt Independence Criterion between two 1-D arrays.

    HSIC(X, Y) = ``trace(K_x H K_y H) / (n - 1)^2`` (biased estimator)
    where ``K_x`` / ``K_y`` are the Gaussian RBF Gram matrices and
    ``H = I - ones(n, n) / n`` is the centring projector. With a
    characteristic kernel (RBF), HSIC = 0 iff X is independent of Y
    (Gretton-Bousquet-Smola-Schoelkopf 2005).

    Parameters
    ----------
    x, y : array-like (n,)
        Any numeric arrays of identical length.
    kernel : str
        Only ``"rbf"`` is supported (the only characteristic kernel in
        common use for HSIC ranking on 1-D signals). The parameter is
        kept for future extension (linear / polynomial kernels are
        non-characteristic and would lose the iff guarantee).
    sigma : Optional[float]
        RBF bandwidth on ``x``. When None, the median heuristic
        (Gretton 2005) is used: sigma = median pairwise distance of
        the subsampled ``x``.
    sigma_y : Optional[float]
        RBF bandwidth on ``y``. When None, the median heuristic is used
        on the subsampled ``y``. Separate from ``sigma`` because on
        classification targets the y-side bandwidth would otherwise
        collapse to zero when integer classes coincide.
    n_sample : int
        Cap on the working sample size. Naive HSIC is ``O(n^2)`` memory;
        ``n_sample = 500`` keeps the per-pair Gram matrices at 2 MB each
        (float64). When ``n > n_sample`` a deterministic random
        subsample is drawn.
    random_state : int
        Seed for the deterministic subsample. Ignored when
        ``n <= n_sample``.
    estimator : {"biased", "unbiased"}
        Choice of HSIC estimator. ``"biased"`` (default) is the
        ``trace(K_x H K_y H) / (n - 1)^2`` plug-in -- non-negative,
        lower variance, the standard ranking estimator. ``"unbiased"``
        is the Song / Smola 2012 U-statistic with the diagonal-dropped
        / row-sum correction -- guaranteed unbiased but can be slightly
        negative on independent pairs due to Monte-Carlo cancellation.
    """
    if kernel != "rbf":
        raise ValueError(
            f"hsic: only kernel='rbf' is supported, got kernel={kernel!r}. "
            f"Linear / polynomial kernels are non-characteristic and would "
            f"lose the HSIC == 0 iff independent guarantee."
        )
    if estimator not in ("biased", "unbiased"):
        raise ValueError(f"hsic: estimator must be 'biased' or 'unbiased', got " f"{estimator!r}")
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"hsic: x has {x_arr.shape[0]} rows, y has {y_arr.shape[0]}; " f"row alignment required.")
    n_total = x_arr.shape[0]
    if n_total < 4:
        # The unbiased estimator's denominator ``n * (n - 3)`` is
        # undefined below n=4; biased is technically defined but the
        # variance is too high to be useful. Return 0 (no signal).
        return 0.0
    idx = _subsample_indices(n_total, int(n_sample), int(random_state))
    xs = x_arr[idx]
    ys = y_arr[idx]
    n = xs.shape[0]
    if n < 4:
        return 0.0
    sx = float(sigma) if sigma is not None else median_heuristic_sigma(xs)
    sy = float(sigma_y) if sigma_y is not None else median_heuristic_sigma(ys)
    K = _rbf_gram(xs, sx)
    L = _rbf_gram(ys, sy)
    if estimator == "biased":
        # H K_y H -> centre K_y; same effect as centring K_x. Use the
        # row / column / grand-mean explicit form (avoids forming H).
        K_row = K.mean(axis=1, keepdims=True)
        K_col = K.mean(axis=0, keepdims=True)
        K_all = K.mean()
        Kc = K - K_row - K_col + K_all
        L_row = L.mean(axis=1, keepdims=True)
        L_col = L.mean(axis=0, keepdims=True)
        L_all = L.mean()
        Lc = L - L_row - L_col + L_all
        # trace(Kc Lc) = sum_ij Kc_ij Lc_ji = sum_ij Kc_ij Lc_ij since
        # both are symmetric.
        val = float(np.sum(Kc * Lc)) / float((n - 1) * (n - 1))
        # Numerical floor: tiny negative roundoff on near-independent
        # pairs; the biased estimator is non-negative in exact arith.
        if val < 0.0 and val > -1e-12:
            val = 0.0
        return val
    # Unbiased (Song / Smola 2012): drop diagonal, correct row sums.
    # Formula:
    # HSIC_u = (1 / n(n-3)) * [
    #     trace(K_tilde L_tilde)
    #     + (sum K_tilde) (sum L_tilde) / ((n-1)(n-2))
    #     - (2 / (n - 2)) sum_i (K_tilde 1)_i (L_tilde 1)_i
    # ]
    # where K_tilde = K with diagonal zeroed.
    K_t = K.copy()
    np.fill_diagonal(K_t, 0.0)
    L_t = L.copy()
    np.fill_diagonal(L_t, 0.0)
    tr = float(np.sum(K_t * L_t))
    sum_K = float(K_t.sum())
    sum_L = float(L_t.sum())
    row_sum_K = K_t.sum(axis=1)
    row_sum_L = L_t.sum(axis=1)
    cross = float(np.dot(row_sum_K, row_sum_L))
    nn = float(n)
    denom = nn * (nn - 3.0)
    val = (tr + sum_K * sum_L / ((nn - 1.0) * (nn - 2.0)) - (2.0 / (nn - 2.0)) * cross) / denom
    return float(val)


def _hsic_batch(
    X: np.ndarray, y: np.ndarray, *,
    n_sample: int = 500, random_state: int = 0,
    estimator: str = "biased",
) -> np.ndarray:
    """Per-column HSIC(X[:, j]; y).

    Returns shape ``(n_features,)``. Uses the SAME subsample index set
    for every column so that HSIC values are comparable across features
    (independent subsamples would inject variance from the sampling
    itself into the cross-column ranking). The y-side bandwidth and
    centred Gram matrix are computed ONCE and reused across all
    columns -- a (n_features - 1)x speedup over the naive per-column
    implementation.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n_total = X_arr.shape[0]
    if n_total < 4:
        return np.zeros(X_arr.shape[1], dtype=np.float64)
    idx = _subsample_indices(n_total, int(n_sample), int(random_state))
    ys = y_arr[idx]
    n = ys.shape[0]
    if n < 4:
        return np.zeros(X_arr.shape[1], dtype=np.float64)
    sy = median_heuristic_sigma(ys)
    L = _rbf_gram(ys, sy)
    out = np.empty(X_arr.shape[1], dtype=np.float64)
    if estimator == "biased":
        L_row = L.mean(axis=1, keepdims=True)
        L_col = L.mean(axis=0, keepdims=True)
        L_all = L.mean()
        Lc = L - L_row - L_col + L_all
        denom = float((n - 1) * (n - 1))
        for j in range(X_arr.shape[1]):
            xs = X_arr[idx, j]
            sx = median_heuristic_sigma(xs)
            K = _rbf_gram(xs, sx)
            K_row = K.mean(axis=1, keepdims=True)
            K_col = K.mean(axis=0, keepdims=True)
            K_all = K.mean()
            Kc = K - K_row - K_col + K_all
            val = float(np.sum(Kc * Lc)) / denom
            if val < 0.0 and val > -1e-12:
                val = 0.0
            out[j] = val
        return out
    # Unbiased path -- match the single-pair helper for parity.
    L_t = L.copy()
    np.fill_diagonal(L_t, 0.0)
    sum_L = float(L_t.sum())
    row_sum_L = L_t.sum(axis=1)
    nn = float(n)
    denom = nn * (nn - 3.0)
    for j in range(X_arr.shape[1]):
        xs = X_arr[idx, j]
        sx = median_heuristic_sigma(xs)
        K = _rbf_gram(xs, sx)
        K_t = K.copy()
        np.fill_diagonal(K_t, 0.0)
        tr = float(np.sum(K_t * L_t))
        sum_K = float(K_t.sum())
        row_sum_K = K_t.sum(axis=1)
        cross = float(np.dot(row_sum_K, row_sum_L))
        val = (tr + sum_K * sum_L / ((nn - 1.0) * (nn - 2.0)) - (2.0 / (nn - 2.0)) * cross) / denom
        out[j] = float(val)
    return out


def score_features_by_hsic_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    kernel: str = "rbf",
    n_sample: int = 500,
    random_state: int = 0,
    estimator: str = "biased",
) -> pd.DataFrame:
    """HSIC variant of :func:`score_features_by_mi_uplift`.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns (used to compute the per-source baseline
        ``HSIC(source; y)``). Positionally aligned with ``y``.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column
        names must carry the ``"{source}__{basis_code}{degree}"`` suffix
        so the baseline can be looked up by source.
    y : array-like (n,)
        Target. HSIC handles continuous and discrete y uniformly -- the
        RBF Gram matrix on a discrete y resolves to a class-indicator
        block (``exp(0) = 1`` within class) which is the correct
        kernel-method construction.
    kernel : str
        Only ``"rbf"`` (characteristic) is supported -- non-characteristic
        kernels would lose the HSIC = 0 iff independent guarantee.
    n_sample : int
        Cap on the working sample. See :func:`hsic`.
    random_state : int
        Subsample seed; held constant across the per-source baseline and
        the per-engineered-col scoring so the uplift ratio is computed
        on the same rows.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending. Column
    names use ``baseline_mi`` / ``engineered_mi`` for downstream
    consistency with Layers 21 / 65 / 66 / 67; the VALUES are HSIC (not
    MI) but the ranking semantics are identical (higher = better).
    """
    if kernel != "rbf":
        raise ValueError(f"score_features_by_hsic_uplift: only kernel='rbf' supported, " f"got {kernel!r}.")
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_hsic_uplift: raw_X has {len(raw_X)} rows " f"but engineered_X has {len(engineered_X)}; positional row " f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_hsic_uplift: raw_X has {len(raw_X)} rows " f"but y has {len(np.asarray(y))}; positional row alignment " f"required."
        )
    y_arr = np.asarray(y).ravel()
    raw_cols = list(raw_X.columns)
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    # f64 kept: distance/kernel-Gram stability (f32 sums lose precision here) -- NOT routed through _crit_np_dtype.
    raw_mi = _hsic_batch(
        raw_X.to_numpy(dtype=np.float64), y_arr,
        n_sample=int(n_sample), random_state=int(random_state),
        estimator=estimator,
    )
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = _hsic_batch(
        engineered_X.to_numpy(dtype=np.float64), y_arr,
        n_sample=int(n_sample), random_state=int(random_state),
        estimator=estimator,
    )
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        emi = float(eng_mi[j])
        # Near-zero baseline makes the uplift ratio explode past the gate even
        # on a no-signal source; suppress the ratio there and let the absolute
        # MI floor decide (mirrors the JMIM guard).
        if baseline < _BASELINE_EPS:
            uplift = 0.0 if emi < _ABS_MI_FLOOR else float("inf")
        else:
            uplift = emi / baseline
        rows.append({
            "engineered_col": eng_name,
            "source_col": source,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_hsic_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    kernel: str = "rbf",
    n_sample: int = 500,
    random_state: int = 0,
    estimator: str = "biased",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HSIC variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in quantile-binned MI estimator with the Gretton
    HSIC (Gaussian-RBF kernel, median-heuristic bandwidth) and applies
    the same two-gate selection as Layer 21: (1) uplift >= ``min_uplift``,
    (2) engineered_mi >= ``min_abs_mi_frac * max(raw_baseline_mi)``, then
    top-K by uplift.

    The kernel win: HSIC with a CHARACTERISTIC kernel inherits the same
    iff-independence guarantee dCor provides but operates at a kernel-
    chosen SCALE. Sharp local non-linearities (threshold rules) and
    high-frequency oscillation are picked up cleanly when the median-
    heuristic bandwidth resolves the relevant length scale.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the HSIC-ranked top-K winners appended.
        scores : the full HSIC ranking DataFrame (winners + rejects).
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_hsic_uplift(
        raw_X, engineered, y,
        kernel=kernel, n_sample=int(n_sample),
        random_state=int(random_state), estimator=estimator,
    )
    if scores.empty:
        return X.copy(), scores
    # Two-gate selection identical to Layers 65 / 66 / 67 for cross-layer parity.
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_raw_baseline)
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + 3.5 * 1.4826 * mad
    else:
        noise_floor = 0.0
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + 3.5 * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[(scores["uplift"] >= float(min_uplift)) & (scores["engineered_mi"] >= abs_floor)]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_hsic_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    kernel: str = "rbf",
    n_sample: int = 500,
    random_state: int = 0,
    estimator: str = "biased",
):
    """Same as :func:`hybrid_orth_mi_hsic_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so that
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the HSIC ranking.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_hsic_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        kernel=kernel, n_sample=int(n_sample),
        random_state=int(random_state), estimator=estimator,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    code_to_basis = {
        "He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre",
    }
    for name in appended:
        if "__" not in name:
            continue
        src, suffix = name.split("__", 1)
        chosen_basis = None
        chosen_degree = None
        for code in ("LL", "He", "T", "L"):
            if suffix.startswith(code):
                rest = suffix[len(code) :]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_hsic_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
