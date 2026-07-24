"""Layer 57 (2026-05-31): ADAPTIVE PER-COLUMN DEGREE selection for the
orthogonal-polynomial univariate FE path.

Why this layer
--------------

The Layer 21 ``generate_univariate_basis_features`` emits EVERY degree in
``degrees`` for EVERY source column. With ``degrees=(1..6)`` and p=200
sources that is 1200 candidate columns dumped into the MI-uplift ranker.

In practice each source column has at most ONE optimal polynomial degree
(He_2 for a squared detector, He_4 for a quartic step, etc.); the OTHER
five degrees are nuisance candidates that pollute the candidate pool and
inflate the multiple-testing burden of the noise-aware abs floor in
:func:`hybrid_orth_mi_fe`.

Layer 57 picks the best degree PER source column up-front: compute MI for
each (col, degree) cell, retain only the argmax degree per column, and
SKIP the column entirely if even the best degree does not beat the raw
baseline by ``min_uplift``. Output:

* fewer candidate columns (1 per surviving source vs degree_count per col),
* the SAME ``orth_univariate`` recipe kind -- replay is unchanged, only
  the (basis, degree) value is the per-column argmax instead of a fixed
  sweep,
* a much smaller multiple-testing footprint at the abs-floor stage.

Recipe replay
-------------

Each appended column is backed by the existing
``build_orth_univariate_recipe(basis=<chosen>, degree=<chosen>)``. No
new recipe kind. ``MRMR.transform`` replays the column deterministically
from X alone, no y reference.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hermite_fe import _POLY_BASES
from ._orthogonal_univariate_fe import (
    _evaluate_basis_column,
    _mi_classif_batch,
    _BASIS_CODE,
    _dedup_collinear_source_cols,
    basis_route_by_signal,
    cached_raw_mi_baseline,
)

logger = logging.getLogger(__name__)

# Below this baseline MI a source is treated as carrying no signal, so the
# uplift ratio is suppressed (it would otherwise explode) and an absolute MI
# floor is required instead -- the same guard JMIM applies (Layer 21/65+).
_BASELINE_EPS = 1e-6
_ABS_MI_FLOOR = 1e-3

__all__ = [
    "generate_adaptive_degree_basis_features",
    "hybrid_orth_mi_adaptive_degree_fe",
    "hybrid_orth_mi_adaptive_degree_fe_with_recipes",
]


def _coerce_y_classif(y) -> np.ndarray:
    """Dense int64 class labels for ``_mi_classif_batch``.

    Integer dtypes pass straight through. Non-integer y (float / continuous /
    categorical) is DENSIFIED via ``np.unique(return_inverse=...)`` rather than
    truncated with ``.astype(int64)``: plain truncation merges distinct labels
    (1.2 and 1.8 -> 1) and destroys continuous-y signal entirely (every value
    in [0, 1) collapses to 0). The dense-rank mapping preserves every distinct
    value as its own class, which is the contract the MI estimator expects.
    """
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int64, copy=False)
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)


def generate_adaptive_degree_basis_features(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degree_range: Sequence[int] = (1, 2, 3, 4, 5, 6),
    basis: str = "auto",
    min_uplift: float = 1.05,
    nbins: int = 10,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
) -> tuple[pd.DataFrame, dict]:
    """For each source column, evaluate every degree in ``degree_range``,
    keep ONLY the single best-MI degree (if it beats raw by ``min_uplift``),
    and emit one column per surviving source.

    Parameters
    ----------
    X : DataFrame
        Source frame. Non-numeric columns are silently skipped.
    y : array-like (n,)
        Target. Must be discrete (binary or small-cardinality int codes);
        the caller is responsible for binning continuous targets if
        needed.
    cols : sequence of column names, optional
        Columns to scan. None = all numeric columns of X.
    degree_range : sequence of int
        Polynomial degrees to consider for each column. The ``MI(basis_d(c); y)``
        argmax over this sequence is the per-column chosen degree.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        ``auto`` routes per column via ``basis_route_by_moments``; an
        explicit string pins one basis for every column.
    min_uplift : float
        Skip a source entirely if its best-degree ``MI(basis_d(c); y)``
        does not reach ``min_uplift * MI(c; y)``. Defaults to 1.05 (5%
        gain over raw baseline) -- conservative; the column adds noise
        to the candidate pool otherwise.
    nbins : int
        Quantile bins for MI estimation. Forwarded to ``_mi_classif_batch``.
    dedup_collinear_sources : bool, default True
        Drop near-duplicate source columns BEFORE per-column scanning
        (mirrors :func:`generate_univariate_basis_features`).

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns, one per surviving source,
            named ``"{col}__{basis_code}{chosen_degree}"`` so the existing
            recipe parser recovers (basis, degree) from the name.
        meta : dict mapping each emitted column name to a dict carrying
            ``{"src": str, "basis": str, "degree": int, "uplift": float,
            "engineered_mi": float, "baseline_mi": float}`` for recipe
            replay and diagnostics. Source columns that did NOT clear the
            uplift gate are absent from both outputs.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if dedup_collinear_sources:
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    degree_range = tuple(int(d) for d in degree_range)
    if not cols or not degree_range:
        return pd.DataFrame(index=X.index), {}

    y_arr = _coerce_y_classif(np.asarray(y))

    # ---- Step 1: raw baselines for the chosen sources (one batch MI call)
    raw_X = X[cols]
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    # Fit-scoped memo: no-op passthrough outside an active orth_scoring_memo_scope() (byte-for-byte the
    # same _mi_classif_batch call); inside a scope, shares this raw MI(x; y) batch with sibling opt-in
    # layers (total-correlation / routing / cluster-basis / diff-basis / adaptive-arity).
    raw_mi_map = cached_raw_mi_baseline(cols, raw_X.to_numpy(dtype=_dt), y_arr, nbins=nbins)

    # ---- Step 2: build the (col, degree) candidate matrix in one shot
    # Generate every basis_d(c) value column, route per-col basis once.
    # Naming carries the SOURCE plus a degree marker so we can recover the
    # argmax per source after the batch MI call.
    code = _BASIS_CODE
    chosen_basis_per_col: dict[str, str] = {}
    cand_cols: list[str] = []
    cand_values: list[np.ndarray] = []
    cand_to_source: list[tuple[str, int, str]] = []  # (src, degree, basis)
    for col in cols:
        x = np.asarray(X[col].to_numpy(), dtype=_dt)  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
            x = np.where(finite_mask, x, fill)
        # 2026-06-03: signal-adaptive routing (route by which basis best linearises
        # y, max best-degree |corr|) -- mirrors the default Layer-21/58 routers;
        # beats moment routing on heavy-tailed/skewed x (bench: OOS-linear 0.92 vs
        # 0.77). Falls back to moment routing without a usable y.
        chosen_basis = basis_route_by_signal(x, y_arr, degrees=degree_range) if basis == "auto" else basis
        if chosen_basis not in _POLY_BASES:
            logger.warning(
                "generate_adaptive_degree_basis_features: unknown basis %r "
                "for col %r; skipping", chosen_basis, col,
            )
            continue
        chosen_basis_per_col[col] = chosen_basis
        for d in degree_range:
            try:
                vals = _evaluate_basis_column(x, chosen_basis, int(d))
            except Exception as exc:
                logger.warning(
                    "generate_adaptive_degree_basis_features: basis=%r "
                    "degree=%d on col=%r raised %r; skipping",
                    chosen_basis, d, col, exc,
                )
                continue
            # Reject constant / non-finite candidates outright -- they
            # carry zero MI and pollute downstream argmax tie-breaks.
            if not np.isfinite(vals).all():
                continue
            if float(np.std(vals)) <= 1e-12:
                continue
            cand_cols.append(f"{col}__{code.get(chosen_basis, chosen_basis)}{int(d)}")
            cand_values.append(vals)
            cand_to_source.append((col, int(d), chosen_basis))

    if not cand_cols:
        return pd.DataFrame(index=X.index), {}

    # ---- Step 3: ONE batch MI call across all (col, degree) candidates
    cand_mat = np.column_stack(cand_values).astype(np.float64, copy=False)
    eng_mi = _mi_classif_batch(cand_mat, y_arr, nbins=nbins)

    # ---- Step 4: per-source argmax + uplift gate
    # Group candidates by source, pick the MI argmax per source, then
    # gate by ``engineered_mi / baseline_mi >= min_uplift``.
    best_per_source: dict[str, dict] = {}
    for j, (src, deg, chosen_basis) in enumerate(cand_to_source):
        emi = float(eng_mi[j])
        if not np.isfinite(emi):
            continue
        cur = best_per_source.get(src)
        if cur is None or emi > cur["engineered_mi"]:
            best_per_source[src] = {
                "src": src,
                "basis": chosen_basis,
                "degree": deg,
                "engineered_mi": emi,
                "engineered_col": cand_cols[j],
                "values_idx": j,
            }

    out_cols: dict = {}
    meta: dict = {}
    min_uplift_f = float(min_uplift)
    for src, info in best_per_source.items():
        baseline = float(raw_mi_map.get(src, 0.0))
        emi = float(info["engineered_mi"])
        # Near-zero baseline makes the uplift ratio explode (a tiny noise
        # baseline divides a tiny emi into a huge ratio that passes the gate
        # despite no real signal). Mirror the JMIM guard: when the baseline is
        # below eps, suppress the ratio and require an absolute MI floor instead.
        if baseline < _BASELINE_EPS:
            uplift = 0.0
            if emi < _ABS_MI_FLOOR:
                continue
        else:
            uplift = emi / baseline
            if uplift < min_uplift_f:
                continue
        name = str(info["engineered_col"])
        vals = cand_values[int(info["values_idx"])]
        out_cols[name] = vals
        meta[name] = {
            "src": src,
            "basis": str(info["basis"]),
            "degree": int(info["degree"]),
            "uplift": float(uplift),
            "engineered_mi": emi,
            "baseline_mi": baseline,
        }

    return pd.DataFrame(out_cols, index=X.index), meta


def hybrid_orth_mi_adaptive_degree_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degree_range: Sequence[int] = (1, 2, 3, 4, 5, 6),
    basis: str = "auto",
    min_uplift: float = 1.05,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adaptive-degree hybrid: pick the best degree per source column,
    drop sources whose best degree fails the ``min_uplift`` gate, return
    the augmented frame plus a tidy scores DataFrame.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with surviving per-column best-degree columns
            appended in source-column iteration order. Index preserved.
        scores : DataFrame with columns
            ``[engineered_col, source_col, basis, degree, baseline_mi,
            engineered_mi, uplift]`` -- one row per surviving source,
            ordered by ``uplift`` descending.
    """
    engineered, meta = generate_adaptive_degree_basis_features(
        X, y,
        cols=cols, degree_range=degree_range, basis=basis,
        min_uplift=min_uplift, nbins=nbins,
    )
    if engineered.empty:
        scores_empty = pd.DataFrame(columns=[
            "engineered_col", "source_col", "basis", "degree",
            "baseline_mi", "engineered_mi", "uplift",
        ])
        return X, scores_empty
    rows = []
    for name, info in meta.items():
        rows.append({
            "engineered_col": name,
            "source_col": info["src"],
            "basis": info["basis"],
            "degree": info["degree"],
            "baseline_mi": info["baseline_mi"],
            "engineered_mi": info["engineered_mi"],
            "uplift": info["uplift"],
        })
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values("uplift", ascending=False).reset_index(drop=True)
    keep = list(engineered.columns)
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_adaptive_degree_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degree_range: Sequence[int] = (1, 2, 3, 4, 5, 6),
    basis: str = "auto",
    min_uplift: float = 1.05,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_adaptive_degree_fe` but additionally
    returns a list of ``EngineeredRecipe`` objects (kind
    ``orth_univariate``) -- one per appended column -- so ``MRMR.transform``
    can replay each engineered column on test data without re-running the
    per-column MI scan.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from .engineered_recipes import build_orth_univariate_recipe
    X_aug, scores = hybrid_orth_mi_adaptive_degree_fe(
        X, y,
        cols=cols, degree_range=degree_range, basis=basis,
        min_uplift=min_uplift, nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    if not appended:
        return X_aug, scores, []
    # The scores DataFrame already carries (basis, degree) per column;
    # build one recipe per appended column from the scores rows.
    name_to_row = {str(row["engineered_col"]): row for _, row in scores.iterrows()}
    recipes = []
    for name in appended:
        row = name_to_row.get(name)
        if row is None:
            logger.warning(
                "hybrid_orth_mi_adaptive_degree_fe_with_recipes: appended " "column %r missing from scores; skipping recipe.",
                name,
            )
            continue
        # freeze the fit-time basis-preprocess params (mirrors the
        # canonical Layer-21 hybrid_orth_mi_fe_with_recipes fix); recomputing on the FULL fit-time
        # source column is safe/exact -- it reproduces, not refits, the fit-time params.
        _pp = None
        try:
            _col_full = np.asarray(X[str(row["source_col"])].to_numpy(), dtype=np.float64)
            _, _pp = _evaluate_basis_column(_col_full, str(row["basis"]), int(row["degree"]), return_params=True)
        except Exception as exc:
            # ORTH_SCORING_A-3 fix: was a bare except with zero logging,
            # silently reverting this column to the pre-B-17 refit-at-replay behaviour on any
            # exception (including a genuine programming bug), with no diagnostic trace.
            logger.debug("failed to freeze fit-time basis preprocess_params (falling back to refit-at-replay): %r", exc)
            _pp = None
        recipes.append(build_orth_univariate_recipe(
            name=name,
            src_name=str(row["source_col"]),
            basis=str(row["basis"]),
            preprocess_params=_pp,
            degree=int(row["degree"]),
        ))
    return X_aug, scores, recipes
