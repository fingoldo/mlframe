"""Layer 66 (2026-06-01): Copula-based MI ranking for hybrid orth-poly FE.

Why this layer
--------------

Layer 21's ``score_features_by_mi_uplift`` ranks engineered columns via the
plug-in quantile-binned MI estimator. The binning is sensitive to the
MARGINAL distribution of each input: on heavy-tailed or strongly-skewed
signals a handful of tail observations dominate one bin and the
discriminating structure inside the bulk gets averaged into a single bin
edge, hiding genuine dependence. Layer 65 (KSG) helps on smooth signals
but is still distance-based and therefore scale-dependent.

This module ranks engineered columns by COPULA MI: each variable is rank-
transformed to a uniform on [0, 1] (Sklar's theorem -- the copula carries
the entire dependence structure independently of the marginals), then MI
is computed on the rank-uniformised pair. The estimate is INVARIANT under
any strictly-monotone transform of either variable (``MI(rank(x), rank(y))
== MI(rank(f(x)), rank(g(y)))`` for any strictly monotone ``f, g``), so
heavy-tail / skew artifacts that distort the plug-in MI cannot distort the
copula MI. Standard non-parametric dependence-measurement baseline in the
copula literature (Nelsen 2006; Poczos & Schneider 2012).

Layer 66 vs Layer 65
--------------------

* Layer 65 (KSG k-NN): same MI estimator family (distance-based), still
  sensitive to extreme-value influence on the k-NN graph. Wins on smooth
  signals that binning erases below bin resolution.
* Layer 66 (this, copula): rank-transform first, then bin. Wins on
  heavy-tailed / skewed signals where binning the raw values pools
  discriminating structure into a single tail bin.

The two are COMPLEMENTARY -- one user can opt into both, picking the
intersection of winners as a robust shortlist. Each runs independently;
neither requires the other.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal to
Layer 21; only the SCORING (and therefore the selection) changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_copula_enable=True``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ._orthogonal_univariate_fe import generate_univariate_basis_features

logger = logging.getLogger(__name__)

__all__ = [
    "copula_mi",
    "score_features_by_copula_mi_uplift",
    "hybrid_orth_mi_copula_fe",
    "hybrid_orth_mi_copula_fe_with_recipes",
]


def _rank_to_uniform(x: np.ndarray) -> np.ndarray:
    """Rank-transform x to ``(0, 1)`` uniforms.

    Uses ``scipy.stats.rankdata`` with the "average" tie-breaking rule
    (canonical for empirical copulas); normalises by ``n + 1`` so the
    output lies strictly inside ``(0, 1)`` -- no edge collapse on the
    ``[0, 1]`` boundary that downstream binning would round to bin 0 / n-1.
    The output is invariant under any strictly-monotone transform of ``x``
    (that's the whole point of the rank transform), so this is the engine
    that makes copula MI marginal-invariant.
    """
    arr = np.asarray(x, dtype=np.float64).ravel()
    n = arr.shape[0]
    if n == 0:
        return arr
    r = rankdata(arr, method="average")
    return np.asarray(r / float(n + 1))


def _bin_mi_uniform_pair(
    u: np.ndarray, v: np.ndarray, *, n_bins: int = 20,
) -> float:
    """Compute MI on a pair of uniformised variables via equal-width bins
    on ``[0, 1]``.

    Because both inputs are already uniform on ``(0, 1)`` (rank-
    transformed), equal-width bins are also equal-FREQUENCY bins by
    construction -- no qcut required, no degenerate-bin failure mode on
    skewed marginals (the marginal skew was rank-flattened away).

    Returns MI in nats. Uses the plug-in (Miller-Madow-corrected) estimator
    on the contingency table; the Miller-Madow correction reduces the
    well-known plug-in positive bias on small samples (Paninski 2003).
    """
    u_arr = np.asarray(u, dtype=np.float64).ravel()
    v_arr = np.asarray(v, dtype=np.float64).ravel()
    if u_arr.shape[0] != v_arr.shape[0]:
        raise ValueError(f"_bin_mi_uniform_pair: u has {u_arr.shape[0]} rows, " f"v has {v_arr.shape[0]}; row alignment required.")
    n = u_arr.shape[0]
    if n < 2:
        return 0.0
    nb = max(2, int(n_bins))
    # Map (0, 1) -> {0, .., nb-1}. Both inputs are bounded inside (0, 1)
    # via _rank_to_uniform's (n+1) normalisation, so clipping is a
    # safety belt against pathological float-edge rounding rather than a
    # routine path.
    ui = np.clip(np.floor(u_arr * nb).astype(np.int64), 0, nb - 1)
    vi = np.clip(np.floor(v_arr * nb).astype(np.int64), 0, nb - 1)
    # Joint histogram via single bincount on the flat index.
    flat = ui * nb + vi
    joint = np.bincount(flat, minlength=nb * nb).reshape(nb, nb)
    n_total = joint.sum()
    if n_total == 0:
        return 0.0
    pxy = joint.astype(np.float64) / float(n_total)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    # Plug-in MI in nats. The (px @ py)[mask] entries are strictly positive
    # because pxy[i, j] > 0 implies both px[i] > 0 and py[j] > 0; mask out
    # the rest BEFORE the log to avoid a divide-by-zero RuntimeWarning on
    # all-zero rows / columns.
    pxpy = (px @ py)[mask]
    pxy_m = pxy[mask]
    mi = float(np.sum(pxy_m * (np.log(pxy_m) - np.log(pxpy))))
    # Miller-Madow bias correction: + (Kxy - Kx - Ky + 1) / (2 * n_total).
    kxy = int(mask.sum())
    kx = int((px > 0).sum())
    ky = int((py > 0).sum())
    mi += (kxy - kx - ky + 1) / (2.0 * float(n_total))
    # Clip tiny negative residual that can arise from the Miller-Madow
    # correction overshooting on near-independent pairs.
    if mi < 0.0:
        mi = 0.0
    return mi


def copula_mi(
    x: np.ndarray, y: np.ndarray, *, n_bins: int = 20,
) -> float:
    """Copula MI: rank-transform both ``x`` and ``y`` to uniforms on
    ``(0, 1)``, then estimate MI on the uniform pair via equal-width binning
    with a Miller-Madow correction.

    Returned value is in nats and is invariant under any strictly-monotone
    transform of either input: ``copula_mi(x, y) == copula_mi(exp(x), y)``
    up to estimator noise; this is the headline property over the plug-in
    MI estimator on raw values.

    Parameters
    ----------
    x, y : array-like (n,)
        Any numeric arrays. Discrete y is handled correctly because the
        rank transform on a discrete target collapses each class to its
        average rank -- the resulting copula MI is the dependence between
        the class label and the rank of ``x``, which is the canonical
        rank-correlation interpretation (Kendall's tau / Spearman's rho
        are special cases).
    n_bins : int
        Number of equal-width bins per axis on the unit square. Defaults to
        20; with ``n_bins=20`` and the recommended ``n >= 200`` samples,
        each cell averages 0.5 samples on a random pair -- enough for the
        plug-in + Miller-Madow estimator to converge while keeping
        sensitivity to local structure inside the unit square.
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    # Mask non-finite BEFORE ranking. rankdata assigns NaN the largest rank,
    # so a NaN would map to a valid high uniform bin and masquerade as real
    # high-value signal -- corrupting the copula MI. Drop non-finite pairwise.
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite.all():
        x_arr = x_arr[finite]
        y_arr = y_arr[finite]
    if x_arr.shape[0] < 2:
        return 0.0
    u = _rank_to_uniform(x_arr)
    v = _rank_to_uniform(y_arr)
    return _bin_mi_uniform_pair(u, v, n_bins=int(n_bins))


def _copula_mi_batch(
    X: np.ndarray, y: np.ndarray, *, n_bins: int = 20, y_side: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-column copula MI(X[:, j]; y).

    Returns shape ``(n_features,)`` in nats. Pre-uniformises ``y`` ONCE
    (the rank transform is O(n log n) and constant across features) to
    amortise the cost across the batch.

    ``y_side`` (2026-07-12): an optional precomputed ``_rank_to_uniform(y_arr)`` (valid only when ``y`` is
    fully finite) -- threaded in by ``score_features_by_copula_mi_uplift`` so the raw-baseline and
    engineered-matrix batch calls (identical ``y``) share ONE rank-to-uniform transform instead of each
    recomputing it. Ignored (recomputed) whenever this call's own ``y`` is not fully finite, so a
    mismatched/stale ``y_side`` can never silently apply to the wrong y.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    y_finite = np.isfinite(y_arr)
    y_all_finite = bool(y_finite.all())
    # Hoist the constant y rank transform out of the per-column loop. When a
    # column has no extra NaNs (the common case), the masked y subset equals the
    # full y, so its uniformisation is identical across all such columns -- the
    # docstring's "uniformise y once" promise the loop previously broke by
    # re-ranking O(n log n) per column. Columns whose own NaNs force a different
    # mask still re-rank the subset (unavoidable, mask-dependent).
    if y_all_finite:
        v_full = y_side if y_side is not None else _rank_to_uniform(y_arr)
    else:
        v_full = None
    out = np.empty(X_arr.shape[1], dtype=np.float64)
    for j in range(X_arr.shape[1]):
        col = X_arr[:, j]
        # Mask non-finite pairwise BEFORE ranking: rankdata maps NaN to the
        # largest rank -> a valid high uniform bin that fakes high-value signal.
        finite = y_finite & np.isfinite(col)
        if finite.all():
            u = _rank_to_uniform(col)
            assert v_full is not None  # finite.all() implies y_finite.all(), so v_full was computed above
            v = v_full
        elif finite.sum() >= 2:
            u = _rank_to_uniform(col[finite])
            v = _rank_to_uniform(y_arr[finite])
        else:
            out[j] = 0.0
            continue
        out[j] = _bin_mi_uniform_pair(u, v, n_bins=int(n_bins))
    return out


def score_features_by_copula_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Copula-MI variant of :func:`score_features_by_mi_uplift`.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns (used to compute the per-source baseline
        copula-MI(source; y)). Indexing-position aligned to ``y``.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column names
        must carry the ``"{source}__{basis_code}{degree}"`` suffix so the
        baseline can be looked up by source.
    y : array-like (n,)
        Target. Discrete or continuous -- copula MI handles both via the
        rank transform (a discrete target is rank-collapsed to its average
        per-class rank).
    n_bins : int
        Equal-width bins per axis on the unit square. See :func:`copula_mi`.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending.

    Note: because both the engineered column ``E(x)`` and the source ``x``
    rank-transform to the SAME uniform (the rank transform is invariant
    under strictly-monotone transforms), the engineered_mi and baseline_mi
    will be EQUAL whenever the basis function is monotone in ``x`` over
    its domain (e.g. an odd-degree polynomial without sign flips). Uplift
    materialises on NON-monotone basis evaluations (``He_2(x) = x^2 - 1``,
    ``He_4``, ...), which is precisely where the orth-poly expansion is
    designed to help.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_copula_mi_uplift: raw_X has {len(raw_X)} "
            f"rows but engineered_X has {len(engineered_X)}; positional "
            f"row alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_copula_mi_uplift: raw_X has {len(raw_X)} " f"rows but y has {len(np.asarray(y))}; positional row " f"alignment required."
        )
    y_arr = np.asarray(y).ravel()
    raw_cols = list(raw_X.columns)
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    # f64 kept: distance/kernel-Gram stability (f32 sums lose precision here) -- NOT routed through _crit_np_dtype.
    # y-side dependence primitive (rank-to-uniform transform) depends only on y -- IDENTICAL for the
    # raw-baseline batch below and the engineered-matrix batch right after it. Build it ONCE (when y is
    # fully finite; None otherwise, matching _copula_mi_batch's own fallback) and thread it into both.
    _y_side = _rank_to_uniform(y_arr) if bool(np.isfinite(y_arr).all()) else None
    raw_mi = _copula_mi_batch(
        raw_X.to_numpy(dtype=np.float64), y_arr, n_bins=int(n_bins), y_side=_y_side,
    )
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = _copula_mi_batch(
        engineered_X.to_numpy(dtype=np.float64), y_arr, n_bins=int(n_bins), y_side=_y_side,
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


def hybrid_orth_mi_copula_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_bins: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copula-MI variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in quantile-binned MI estimator on RAW values with
    the copula MI estimator on rank-transformed values, then applies the
    same two-gate selection as Layer 21: (1) uplift >= ``min_uplift``,
    (2) engineered_mi >= ``min_abs_mi_frac * max(raw_baseline_mi)``,
    then top-K by uplift.

    The heavy-tail / skew win: on engineered columns whose discriminating
    structure is hidden behind extreme-value crowding in the marginal (the
    plug-in's qcut piles tail observations into a single bin), the rank
    transform redistributes the support uniformly and the copula MI scores
    the genuine dependence that the plug-in misses.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the copula-MI-ranked top-K winners appended.
        scores : the full copula-MI ranking DataFrame (winners + rejects).
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
    scores = score_features_by_copula_mi_uplift(
        raw_X, engineered, y, n_bins=int(n_bins),
    )
    if scores.empty:
        return X.copy(), scores
    # Same two-gate + MAD-noise floor as Layer 65 for parity. Copula MI is
    # less biased upward than the plug-in on raw values, so the absolute
    # floor is needed to reject noise columns whose uplift is high only
    # because their baseline copula-MI is tiny.
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


def hybrid_orth_mi_copula_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_bins: int = 20,
):
    """Same as :func:`hybrid_orth_mi_copula_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so that
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the copula MI ranking.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_copula_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        n_bins=int(n_bins),
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
                "hybrid_orth_mi_copula_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
