"""Layer 65 (2026-05-31): KSG / k-NN MI ranking for hybrid orth-poly FE.

Why this layer
--------------

Layer 21's ``score_features_by_mi_uplift`` ranks engineered columns via the
plug-in quantile-binned MI estimator (``_mi_classif_batch``): each feature
is digitised into ``nbins`` quantile bins, then the joint frequency table
gives the plug-in MI. The binning is fast (numba-jit, batched across
columns) and accurate enough for coarse signal -- but it has a known weak
spot: SMOOTH continuous structure that lives BELOW the bin resolution gets
averaged out. A subtle He_3-style cubic ripple inside a bin contributes 0
to the binned MI even though k-NN distance-based estimators see it
clearly.

This module ranks engineered columns via the Kraskov-Stoegbauer-Grassberger
k-NN MI estimator (Kraskov et al. 2004, Ross 2014 for the mixed
continuous-discrete variant used by ``sklearn.feature_selection.
mutual_info_classif``). It is the standard non-binning baseline in the
continuous-MI literature (Czyz et al. NeurIPS 2023) and is asymptotically
unbiased for continuous variables, recovering smooth signal that the
binned plug-in misses.

Layer 65 vs Layer 62 / Layer 63
-------------------------------

* Layer 62 (bootstrap): same MI estimator (plug-in), different SCORING
  AGGREGATION (LCB over B replicates).
* Layer 63 (three-gate + OOF): same MI estimator (plug-in), different
  SELECTION GATES (OOF + CMI).
* Layer 65 (this): different MI ESTIMATOR (KSG k-NN), same two-gate
  selection as Layer 21. Captures smooth-signal cases where binning
  hides the win.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layer 21 uses -- because the engineered VALUES are bit-equal to
Layer 21; only the SCORING (and therefore the selection) changes. Replay
reuses the existing ``_apply_orth_univariate`` path.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_ksg_enable=True``.
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
    "score_features_by_ksg_mi_uplift",
    "hybrid_orth_mi_ksg_fe",
    "hybrid_orth_mi_ksg_fe_with_recipes",
]


def _is_discrete_target(y: np.ndarray) -> bool:
    """Decide classif vs regression dispatch for KSG MI.

    Heuristic: integer-dtype OR floating-dtype with <= 32 unique values
    routes to ``mutual_info_classif`` (each class is a conditioning regime);
    higher-cardinality floats route to ``mutual_info_regression`` (KSG with
    continuous y). Mirrors the qcut-or-cast logic used downstream so
    Layer 65 stays consistent with the wider hybrid_orth wiring.
    """
    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.integer):
        return True
    if arr.dtype.kind == "b":
        return True
    if arr.dtype.kind in "fc":
        return int(np.unique(arr).size) <= 32
    return True


def _ksg_mi_batch(
    X: np.ndarray, y: np.ndarray, *, n_neighbors: int = 3,
    random_state: int = 0,
) -> np.ndarray:
    """Per-column KSG MI(X[:, j]; y) via sklearn's k-NN MI estimators.

    Dispatches to ``mutual_info_classif`` (Ross 2014 mixed-KSG variant)
    for discrete y, ``mutual_info_regression`` (classical Kraskov 2004
    KSG) for continuous y. Both are k-NN distance-based and
    asymptotically unbiased on continuous features -- the headline
    accuracy claim over the plug-in quantile-binned MI estimator.

    Returns shape ``(n_features,)`` in nats. We pass
    ``discrete_features=False`` because the orth-poly engineered columns
    are continuous-valued by construction.
    """
    from sklearn.feature_selection import (
        mutual_info_classif, mutual_info_regression,
    )

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y_arr = np.asarray(y).ravel()
    if _is_discrete_target(y_arr):
        if not np.issubdtype(y_arr.dtype, np.integer):
            y_arr = y_arr.astype(np.int64)
        mi = mutual_info_classif(
            X, y_arr,
            discrete_features=False,
            n_neighbors=int(n_neighbors),
            copy=True,
            random_state=int(random_state),
        )
    else:
        mi = mutual_info_regression(
            X, y_arr.astype(np.float64),
            discrete_features=False,
            n_neighbors=int(n_neighbors),
            copy=True,
            random_state=int(random_state),
        )
    return np.asarray(mi, dtype=np.float64)


def score_features_by_ksg_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_neighbors: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """KSG / k-NN-based variant of :func:`score_features_by_mi_uplift`.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns (used to compute the per-source baseline
        KSG-MI(source; y)). Indexing-position aligned to ``y``.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column
        names must carry the ``"{source}__{basis_code}{degree}"`` suffix
        so the baseline can be looked up by source.
    y : array-like (n,)
        Discrete target. KSG variant from Ross 2014 handles binary /
        multiclass directly via per-class k-NN distances; no qcut needed.
    n_neighbors : int
        ``k`` in the KSG estimator. Standard literature defaults are
        ``3..6``; sklearn defaults to ``3``. Larger k = more bias toward
        marginal independence (smaller MI) but lower variance. ``3`` is
        the asymptotically-unbiased recommendation from Kraskov 2004 and
        the Czyz 2023 NeurIPS benchmark sweet spot.
    random_state : int
        RNG seed for sklearn's internal tie-breaking jitter.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending. ``baseline_mi``
    and ``engineered_mi`` are KSG-estimated and may be SMALLER in absolute
    value than the plug-in equivalents on coarse-binned signal but LARGER
    on smooth signal that binning erases.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_ksg_mi_uplift: raw_X has {len(raw_X)} rows "
            f"but engineered_X has {len(engineered_X)}; positional row "
            f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_ksg_mi_uplift: raw_X has {len(raw_X)} rows "
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
    raw_mi = _ksg_mi_batch(
        raw_X.to_numpy(dtype=np.float64), y_arr,
        n_neighbors=int(n_neighbors), random_state=int(random_state),
    )
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = _ksg_mi_batch(
        engineered_X.to_numpy(dtype=np.float64), y_arr,
        n_neighbors=int(n_neighbors), random_state=int(random_state),
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


def hybrid_orth_mi_ksg_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_neighbors: int = 3,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KSG-MI variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in quantile-binned MI estimator with the KSG k-NN
    estimator (sklearn's ``mutual_info_classif``, n_neighbors=k).
    Selection rule mirrors Layer 21: (1) uplift >= ``min_uplift``,
    (2) engineered_mi >= ``min_abs_mi_frac * max(raw_baseline_mi)``,
    then top-K by uplift.

    The smooth-signal win: on engineered columns whose discriminating
    structure lives BELOW the plug-in's ``nbins=10`` quantile bin
    resolution (e.g. a He_3 cubic ripple modulating class probability
    inside a bin), the plug-in scores zero MI gain while the KSG k-NN
    estimator scores the genuine MI uplift and the column enters
    the top-K.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the KSG-ranked top-K winners appended.
        scores : the full KSG-MI ranking DataFrame (winners + rejects).
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
    scores = score_features_by_ksg_mi_uplift(
        raw_X, engineered, y,
        n_neighbors=int(n_neighbors), random_state=int(random_state),
    )
    if scores.empty:
        return X.copy(), scores
    # Two-gate selection on KSG-estimated MI:
    # 1. relative: uplift >= min_uplift
    # 2. absolute: engineered_mi >= max(legacy_floor, MAD-noise floor)
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_raw_baseline)
    # MAD-based noise floor on baseline distribution -- legitimate signal
    # is an extreme outlier above the noise band's median+sigma*MAD.
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


def hybrid_orth_mi_ksg_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    n_neighbors: int = 3,
    random_state: int = 0,
):
    """Same as :func:`hybrid_orth_mi_ksg_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so that
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the KSG MI ranking.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs. The recipe parser logic is reused unchanged.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_ksg_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        n_neighbors=int(n_neighbors), random_state=int(random_state),
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
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
                "hybrid_orth_mi_ksg_fe_with_recipes: cannot parse "
                "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
