"""Layer 67 (2026-06-01): Distance correlation (dCor) ranking for hybrid
orth-poly FE.

Why this layer
--------------

Layer 21's plug-in MI estimator, Layer 65 (KSG k-NN), and Layer 66 (copula
MI) all share one weakness: they are MI estimators -- they all attempt to
estimate the same quantity (continuous mutual information), differ only in
HOW they estimate it. Distance correlation (Szekely-Rizzo 2007) is a
genuinely DIFFERENT dependence measure: it is constructed from the U-
centred distance matrices of ``x`` and ``y`` and equals zero if and only
if the two variables are INDEPENDENT -- the universal independence
guarantee that Pearson lacks (Pearson can be zero on non-monotone signals
like ``y = x ^ 2``).

For an orth-poly engineered column ``He_2(x) = x ^ 2 - 1``, dCor against
a target ``y = sign(x ^ 2 > 1)`` materialises a large dependence value
while Pearson collapses to zero (the engineered column is a symmetric
function of x; positive and negative tails contribute equally to ``y``).
The plug-in MI estimator on the same pair has the heavy-tail / binning
weakness Layer 66 was built to address. dCor is binning-FREE: every pair
of observations contributes to the centred distance matrices, so sub-bin
structure is preserved.

Cost / sample-size constraint
-----------------------------

dCor is constructed from the FULL ``n x n`` distance matrices. Naive
implementation is ``O(n^2)`` time and memory; ``n = 500`` fits in 2 MB
per matrix and runs in under 50 ms on a modern laptop. Beyond ``n = 500``
the per-feature cost starts to dominate; this module caps the working
sample at ``n_sample = 500`` via deterministic random subsampling -- the
dCor estimator is asymptotically consistent and 500 samples are enough
for the dependence test to discriminate signal from noise at the typical
SNR seen by Layer 21's downstream gates.

Layer 67 vs Layers 65 / 66
--------------------------

* Layer 65 (KSG): MI estimator, distance-based, good on smooth signals.
* Layer 66 (copula MI): MI estimator, rank-based, good on heavy-tailed
  marginals.
* Layer 67 (this, dCor): NON-MI dependence measure. Zero iff independent
  on ANY relationship (monotone, non-monotone, non-functional). Excels
  on non-monotone, non-functional, oscillatory dependencies where MI
  estimators converge slowly (sample-complexity gap).

The three are COMPLEMENTARY -- a user can opt into all three and take
the union of winners as a robust shortlist.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal to
Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_dcor_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import generate_univariate_basis_features

logger = logging.getLogger(__name__)

__all__ = [
    "distance_correlation",
    "score_features_by_dcor_uplift",
    "hybrid_orth_mi_dcor_fe",
    "hybrid_orth_mi_dcor_fe_with_recipes",
]


def _u_centered_distance_matrix(z: np.ndarray) -> np.ndarray:
    """Build the double-centred Euclidean distance matrix used by the
    Szekely-Rizzo dCor estimator.

    Given a 1-D array ``z`` of length ``n``, returns the ``n x n`` matrix
    ``A`` with entries ``a_ij - a_i. - a_.j + a_..`` where
    ``a_ij = |z_i - z_j|``. The double centring is what makes ``mean(A * B)``
    a valid covariance-like measure: independent ``z`` and ``w`` give
    ``mean(A * B) = 0``.
    """
    arr = np.asarray(z, dtype=np.float64).ravel()
    # Pairwise absolute differences -- the 1-D Euclidean distance.
    a = np.abs(arr[:, None] - arr[None, :])
    row_mean = a.mean(axis=1, keepdims=True)
    col_mean = a.mean(axis=0, keepdims=True)
    grand_mean = a.mean()
    return a - row_mean - col_mean + grand_mean


def _subsample_indices(
    n: int, n_sample: int, random_state: int,
) -> np.ndarray:
    """Deterministic random subsample of ``min(n, n_sample)`` indices.

    Returns ``np.arange(n)`` when ``n <= n_sample`` (no work needed); else
    a sorted random permutation of ``n_sample`` indices to keep the
    relative ordering of the subsample consistent across paired calls.
    """
    if n_sample <= 0 or n <= int(n_sample):
        return np.arange(n)
    rng = np.random.default_rng(int(random_state))
    idx = rng.choice(n, size=int(n_sample), replace=False)
    idx.sort()
    return idx


def distance_correlation(
    x: np.ndarray, y: np.ndarray, *,
    n_sample: int = 500, random_state: int = 0,
) -> float:
    """Szekely-Rizzo distance correlation between two 1-D arrays.

    ``dCor(X, Y) = sqrt( dCov^2(X, Y) / sqrt(dVar^2(X) * dVar^2(Y)) )``
    where ``dCov^2 = mean(A * B)``, ``dVar^2(X) = mean(A * A)``,
    ``dVar^2(Y) = mean(B * B)``, and ``A`` / ``B`` are the U-centred
    Euclidean distance matrices of ``x`` / ``y``.

    Key properties (Szekely-Rizzo 2007):

    * ``dCor(X, Y) == 0`` iff ``X`` and ``Y`` are INDEPENDENT (this is the
      headline property -- Pearson lacks this iff guarantee).
    * ``0 <= dCor(X, Y) <= 1``.
    * Symmetric: ``dCor(X, Y) == dCor(Y, X)``.
    * Detects ANY relationship (monotone, non-monotone, non-functional),
      not just linear / monotone like Pearson / Spearman.

    Parameters
    ----------
    x, y : array-like (n,)
        Any numeric arrays of identical length.
    n_sample : int
        Cap on the working sample size. Naive dCor is ``O(n^2)`` memory;
        ``n_sample = 500`` keeps the per-pair distance matrices at 2 MB
        each (float64). When ``n > n_sample`` a deterministic random
        subsample is drawn; the dCor estimator is asymptotically
        consistent and 500 samples are sufficient at the typical SNR
        seen by Layer 21's downstream gates.
    random_state : int
        Seed for the deterministic subsample. Ignored when
        ``n <= n_sample``.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"distance_correlation: x has {x_arr.shape[0]} rows, "
            f"y has {y_arr.shape[0]}; row alignment required."
        )
    n = x_arr.shape[0]
    if n < 2:
        return 0.0
    idx = _subsample_indices(n, int(n_sample), int(random_state))
    xs = x_arr[idx]
    ys = y_arr[idx]
    A = _u_centered_distance_matrix(xs)
    B = _u_centered_distance_matrix(ys)
    dcov2 = float(np.mean(A * B))
    dvar2_x = float(np.mean(A * A))
    dvar2_y = float(np.mean(B * B))
    denom = dvar2_x * dvar2_y
    # A non-finite denom (all-NaN/inf column -> NaN distance matrix) makes `denom <= 0` False, so a NaN dCor would otherwise escape into the ranking. Guard explicitly.
    if not np.isfinite(denom) or denom <= 0.0:
        # Constant / degenerate column -- no dispersion -> by convention dCor = 0.
        return 0.0
    if not np.isfinite(dcov2):
        return 0.0
    # Numerical floor: tiny negative dcov2 can arise from float roundoff
    # on very-near-independent pairs (the U-centring leaves a tail of
    # cancelling terms). Clamp before the sqrt to keep the output real.
    if dcov2 < 0.0:
        dcov2 = 0.0
    return float(np.sqrt(dcov2 / np.sqrt(denom)))


def _dcor_batch(
    X: np.ndarray, y: np.ndarray, *,
    n_sample: int = 500, random_state: int = 0,
) -> np.ndarray:
    """Per-column dCor(X[:, j]; y).

    Returns shape ``(n_features,)``. Uses the SAME subsample index set
    for every column so that ``dCor`` values are comparable across
    features (independent subsamples would inject variance from the
    sampling itself into the cross-column ranking).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    n = X_arr.shape[0]
    idx = _subsample_indices(n, int(n_sample), int(random_state))
    ys = y_arr[idx]
    B = _u_centered_distance_matrix(ys)
    dvar2_y = float(np.mean(B * B))
    out = np.empty(X_arr.shape[1], dtype=np.float64)
    for j in range(X_arr.shape[1]):
        xs = X_arr[idx, j]
        A = _u_centered_distance_matrix(xs)
        dvar2_x = float(np.mean(A * A))
        dcov2 = float(np.mean(A * B))
        denom = dvar2_x * dvar2_y
        # A non-finite denom (all-NaN/inf column) makes `denom <= 0` False, so a NaN score would otherwise escape and sort to the top; guard explicitly.
        if not np.isfinite(denom) or denom <= 0.0 or not np.isfinite(dcov2):
            out[j] = 0.0
            continue
        if dcov2 < 0.0:
            dcov2 = 0.0
        out[j] = float(np.sqrt(dcov2 / np.sqrt(denom)))
    return out


def score_features_by_dcor_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_sample: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    """dCor variant of :func:`score_features_by_mi_uplift`.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns (used to compute the per-source baseline
        ``dCor(source; y)``). Positionally aligned with ``y``.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column names
        must carry the ``"{source}__{basis_code}{degree}"`` suffix so the
        baseline can be looked up by source.
    y : array-like (n,)
        Target. dCor handles continuous and discrete y uniformly -- the
        distance matrix on a discrete y collapses to the class-indicator
        block structure (``|y_i - y_j| == 0`` within a class, ``> 0``
        across classes), which is the correct dependence-test
        construction.
    n_sample : int
        Cap on the working sample. See :func:`distance_correlation`.
    random_state : int
        Subsample seed; held constant across the per-source baseline and
        the per-engineered-col scoring so the uplift ratio is computed
        on the same rows.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending. Column names
    use ``baseline_mi`` / ``engineered_mi`` for downstream consistency
    with Layers 21 / 65 / 66; the VALUES are dCor (not MI) but the
    ranking semantics are identical (higher = better).
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_dcor_uplift: raw_X has {len(raw_X)} rows "
            f"but engineered_X has {len(engineered_X)}; positional row "
            f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_dcor_uplift: raw_X has {len(raw_X)} rows "
            f"but y has {len(np.asarray(y))}; positional row alignment "
            f"required."
        )
    y_arr = np.asarray(y).ravel()
    raw_cols = list(raw_X.columns)
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    raw_mi = _dcor_batch(
        raw_X.to_numpy(dtype=np.float64), y_arr,
        n_sample=int(n_sample), random_state=int(random_state),
    )
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = _dcor_batch(
        engineered_X.to_numpy(dtype=np.float64), y_arr,
        n_sample=int(n_sample), random_state=int(random_state),
    )
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
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


def hybrid_orth_mi_dcor_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_sample: int = 500,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """dCor variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in quantile-binned MI estimator with the Szekely-
    Rizzo distance correlation and applies the same two-gate selection
    as Layer 21: (1) uplift >= ``min_uplift``, (2) engineered_mi >=
    ``min_abs_mi_frac * max(raw_baseline_mi)``, then top-K by uplift.

    The non-monotone win: dCor scores a HE_2(x) = x^2 - 1 column against
    a ``y = sign(x^2 > 1)`` target as a strongly-dependent pair, while
    Pearson and even the plug-in MI on a coarse binning can miss it.
    Distance correlation's universal iff-independence guarantee is what
    makes this rank reliably.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the dCor-ranked top-K winners appended.
        scores : the full dCor ranking DataFrame (winners + rejects).
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
    scores = score_features_by_dcor_uplift(
        raw_X, engineered, y,
        n_sample=int(n_sample), random_state=int(random_state),
    )
    if scores.empty:
        return X.copy(), scores
    # Two-gate selection identical to Layers 65 / 66 for cross-layer parity.
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
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_dcor_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_sample: int = 500,
    random_state: int = 0,
):
    """Same as :func:`hybrid_orth_mi_dcor_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so that
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the dCor ranking.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_dcor_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        n_sample=int(n_sample), random_state=int(random_state),
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
                rest = suffix[len(code):]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_dcor_fe_with_recipes: cannot parse "
                "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
