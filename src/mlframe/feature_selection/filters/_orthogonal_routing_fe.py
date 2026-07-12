"""Layer 58 (2026-05-31): CONDITIONAL BASIS ROUTING for the orthogonal-
polynomial univariate FE path.

Why this layer
--------------

Layer 21 (``generate_univariate_basis_features``) picks ONE basis per
column via the moment fingerprint at :func:`basis_route_by_moments` and
sweeps every degree in ``degrees=(2,3)``. Layer 57
(``generate_adaptive_degree_basis_features``) extended that with a per-
column argmax over the degree axis -- still ONE (basis, transform) pair
per column, just with the best degree.

But the moment fingerprint mis-routes for two important regimes the
user flagged on 2026-05-31:

* **heavy-tail log-normal x driving y = f(log x)**: skew + kurtosis
  push the fingerprint toward Laguerre/Hermite-on-raw, but the actual
  signal is linear in ``log|x|`` -- Hermite-on-``log|x|`` is dominant.
* **uniform x driving y = T_3(x)**: the fingerprint sees a bounded
  domain and picks Chebyshev, which IS correct -- but only by luck;
  on the same column with the wrong target (y = He_3(x)) Hermite-on-raw
  would dominate. The fingerprint is a marginal-of-x heuristic; it
  cannot know what y looks like.

Layer 58 fixes this by ACTUALLY TRYING multiple (basis, pre_transform,
degree) combos per column and picking the MI-uplift winner. Pre-transforms
are sample-stateless functions of x alone (``raw``, ``log|x|+eps``,
``sqrt|x|``, ``tanh(x/std)``) so the recipe replay stays leakage-free.

Combinatorial budget
--------------------

Per column the candidate space is

    n_pre * n_basis * n_degrees
  = 4 * 4 * len(degrees)

with ``degrees=(2,3)`` that's 32 candidates / column. At p=200 sources
the candidate pool is 6400 columns -- larger than Layer 21's 1200 but
still cheap (one batch MI call). The ``min_uplift`` default is therefore
tightened to ``1.10`` (10% gain over raw baseline) to compensate for the
larger multiple-testing burden.

Recipe replay
-------------

Each appended column is backed by an ``orth_univariate`` recipe whose
``extra`` carries ``{basis, degree, pre_transform}``. The default
``pre_transform="raw"`` keeps Layer 21/57 recipes byte-identical; only
Layer 58 routing emits the new field.

Selection rule
--------------

* per-column argmax over the (pre_transform, basis, degree) cell;
* uplift gate: skip the column if its best engineered MI does not exceed
  ``min_uplift * baseline_MI`` where ``baseline_MI`` is the raw column MI;
* global top-K: across surviving columns keep the top-K by uplift.

NOT wired into MRMR.fit by default -- explicit opt-in via
``fe_hybrid_orth_conditional_routing_enable=True``.
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
    cached_raw_mi_baseline,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PRE_TRANSFORM_NAMES",
    "apply_pre_transform",
    "generate_conditional_basis_routing_features",
    "hybrid_orth_mi_conditional_routing_fe",
    "hybrid_orth_mi_conditional_routing_fe_with_recipes",
]

# Stable enumeration of supported pre-transforms. Mirrors
# :func:`engineered_recipes._apply_orth_pre_transform` so fit-time and
# replay-time agree on the value set. Public constant so callers can
# enumerate the routing space without re-importing the recipe helper.
PRE_TRANSFORM_NAMES: tuple[str, ...] = ("raw", "log_abs", "sqrt_abs", "tanh")

# Short tags embedded in engineered column names so the recipe parser can
# recover the pre-transform from the suffix. Two-char tags chosen so they
# can't collide with single-char basis codes (``He``, ``T``, ``L``, ``LL``).
_PRE_TRANSFORM_TAG = {
    "raw": "raw",
    "log_abs": "lga",
    "sqrt_abs": "sqa",
    "tanh": "tnh",
}
_TAG_TO_PRE_TRANSFORM = {v: k for k, v in _PRE_TRANSFORM_TAG.items()}


def apply_pre_transform(x: np.ndarray, pre_transform: str) -> np.ndarray:
    """Public wrapper around the canonical recipe-side implementation so
    test code and fit-time code use the SAME function (no risk of drift
    between fit-time and replay-time). Lazy import keeps the recipes
    module dependency-light at FE-module import time.
    """
    from .engineered_recipes import _apply_orth_pre_transform
    return _apply_orth_pre_transform(x, pre_transform)


def _routing_col_name(src: str, basis: str, degree: int, pre_transform: str) -> str:
    """Stable engineered column name: ``"{src}__{tag}_{basis_code}{degree}"``.

    The ``raw`` tag is included explicitly (rather than omitted) so the
    routing-FE column names are visually distinct from Layer 21/57
    columns even when the picked pre-transform happens to be identity --
    this avoids name collisions when both layers run on the same source.
    """
    code = _BASIS_CODE.get(basis, basis)
    tag = _PRE_TRANSFORM_TAG.get(pre_transform, pre_transform)
    return f"{src}__{tag}_{code}{int(degree)}"


def parse_routing_col_name(name: str) -> Optional[tuple[str, str, str, int]]:
    """Inverse of :func:`_routing_col_name`. Returns ``(src, pre_transform,
    basis, degree)`` or ``None`` if the name doesn't match the routing-FE
    naming convention.
    """
    if "__" not in name:
        return None
    src, suffix = name.split("__", 1)
    if "_" not in suffix:
        return None
    tag, rest = suffix.split("_", 1)
    pre_transform = _TAG_TO_PRE_TRANSFORM.get(tag)
    if pre_transform is None:
        return None
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
    for code in ("LL", "He", "T", "L"):
        if rest.startswith(code):
            deg_str = rest[len(code) :]
            if deg_str.isdigit():
                return (src, pre_transform, code_to_basis[code], int(deg_str))
    return None


def generate_conditional_basis_routing_features(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    candidate_bases: Optional[Sequence[str]] = None,
    transform_variants: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    top_k: int = 5,
    min_uplift: float = 1.10,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
    routing_criterion: str = "corr",
) -> tuple[pd.DataFrame, dict]:
    """For each source column, evaluate every (pre_transform, basis, degree)
    cell and keep ONLY the per-column argmax. Across all surviving columns
    keep the global top-K by uplift, dropping any whose engineered MI does
    not exceed ``min_uplift * baseline_MI``.

    Parameters
    ----------
    X : DataFrame
        Source frame. Non-numeric columns are silently skipped.
    y : array-like (n,)
        Target. Must be discrete (binary or small-cardinality int codes);
        the caller bins continuous targets if needed.
    cols : sequence of column names, optional
        Columns to scan. None = all numeric columns of X.
    candidate_bases : sequence of {'hermite', 'legendre', 'chebyshev',
        'laguerre'}, optional
        Bases to try per column. Defaults to all four polynomial families.
    transform_variants : sequence of pre-transform names, optional
        Defaults to :data:`PRE_TRANSFORM_NAMES`.
    degrees : sequence of int
        Polynomial degrees to consider for each (basis, transform) cell.
    top_k : int
        Global top-K winners by uplift. The per-column argmax produces at
        MOST ``len(cols)`` candidates; ``top_k`` further trims that pool.
    min_uplift : float
        Per-column gate: skip a source entirely if its best (basis, transform,
        degree) cell does not reach ``min_uplift * MI(c; y)``. Defaults to
        1.10 (10% gain), tighter than Layer 21's 1.05 because the
        combinatorial space is 4x larger so the noise tail is fatter.
    nbins : int
        Quantile bins for MI estimation. Forwarded to ``_mi_classif_batch``.
    dedup_collinear_sources : bool, default True
        Drop near-duplicate source columns BEFORE scanning (mirrors the
        polynomial univariate path).

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns, one per top-K winner,
            named via :func:`_routing_col_name`.
        meta : dict mapping each emitted column name to a dict carrying
            ``{"src": str, "basis": str, "degree": int, "pre_transform":
            str, "uplift": float, "engineered_mi": float, "baseline_mi":
            float}`` for recipe replay and diagnostics.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = [c for c in cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if dedup_collinear_sources:
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    degrees = tuple(int(d) for d in degrees)
    if candidate_bases is None:
        candidate_bases = ("hermite", "legendre", "chebyshev", "laguerre")
    candidate_bases = tuple(b for b in candidate_bases if b in _POLY_BASES)
    if transform_variants is None:
        transform_variants = PRE_TRANSFORM_NAMES
    transform_variants = tuple(t for t in transform_variants if t in PRE_TRANSFORM_NAMES)
    if not cols or not degrees or not candidate_bases or not transform_variants:
        return pd.DataFrame(index=X.index), {}

    y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)

    # ---- Step 1: raw baselines (one batch MI call across the chosen cols)
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    raw_X = X[list(cols)]
    # Fit-scoped memo: no-op passthrough outside an active orth_scoring_memo_scope() (byte-for-byte the
    # same _mi_classif_batch call); inside a scope, shares this raw MI(x; y) batch with sibling opt-in
    # layers (total-correlation / adaptive-degree / cluster-basis / diff-basis / adaptive-arity).
    raw_mi_map = cached_raw_mi_baseline(cols, raw_X.to_numpy(dtype=_dt), y_arr, nbins=nbins)

    # ---- Step 2: enumerate every (col, pre_transform, basis, degree) cell.
    # Pre-compute pre-transformed columns ONCE per (col, pre_transform) so
    # the basis evaluation doesn't re-run log/sqrt/tanh per basis.
    cand_cols: list[str] = []
    cand_values: list[np.ndarray] = []
    cand_meta: list[tuple[str, str, str, int]] = []  # (src, pre, basis, degree)

    for col in cols:
        # Value-construction site (feeds the engineered column, not just an MI score): always
        # float64, matching _apply_orth_univariate's hardcoded replay dtype -- a relaxed f32 cast
        # here reproduces the fit/replay drift bug fixed in _orthogonal_univariate_fe/__init__.py.
        x_raw = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x_raw)
        if not finite_mask.all():
            fill = float(np.nanmean(x_raw[finite_mask])) if finite_mask.any() else 0.0
            x_raw = np.where(finite_mask, x_raw, fill)
        # Pre-transform cache for this column.
        pt_cache: dict[str, np.ndarray] = {}
        for pt in transform_variants:
            try:
                xt = apply_pre_transform(x_raw, pt)
            except Exception as exc:
                logger.warning(
                    "generate_conditional_basis_routing_features: pre_transform=%r "
                    "on col=%r raised %r; skipping.", pt, col, exc,
                )
                continue
            # Guard non-finite outputs (e.g. log(0)).
            finite2 = np.isfinite(xt)
            if not finite2.all():
                fill2 = float(np.nanmean(xt[finite2])) if finite2.any() else 0.0
                xt = np.where(finite2, xt, fill2)
            if float(np.std(xt)) <= 1e-12:
                # Constant pre-transform output -- every basis_d(z) is zero
                # variance; skip the entire pre_transform * basis * degree
                # sub-tree for this column.
                continue
            pt_cache[pt] = xt
        for pt, xt in pt_cache.items():
            for basis_name in candidate_bases:
                for d in degrees:
                    try:
                        vals = _evaluate_basis_column(xt, basis_name, int(d))
                    except Exception as exc:
                        logger.warning(
                            "generate_conditional_basis_routing_features: "
                            "basis=%r pre=%r degree=%d on col=%r raised %r; "
                            "skipping.",
                            basis_name, pt, d, col, exc,
                        )
                        continue
                    if not np.isfinite(vals).all():
                        continue
                    if float(np.std(vals)) <= 1e-12:
                        continue
                    cand_cols.append(_routing_col_name(col, basis_name, int(d), pt))
                    cand_values.append(vals)
                    cand_meta.append((col, pt, basis_name, int(d)))

    if not cand_cols:
        return pd.DataFrame(index=X.index), {}

    # ---- Step 3: ONE batch MI call across every candidate
    cand_mat = np.column_stack(cand_values).astype(np.float64, copy=False)
    eng_mi = _mi_classif_batch(cand_mat, y_arr, nbins=nbins)

    # ---- Step 3b: routing score. The per-source ARGMAX (which (pre,basis,degree)
    # cell to keep) is a LINEARISATION decision -- pick the cell a shallow/linear
    # downstream can best use -- so route by |Pearson corr|, NOT by MI. A 30-class
    # OOS study over this exact (pre x basis x degree) space (2026-06-03) showed
    # corr-routing near-oracle (OOS-linear R^2 0.81 vs MI 0.52; MI picks an
    # informative-but-non-linear cell -- log|x|/tanh+Laguerre -- in 23/30 cases).
    # Mirrors ``basis_route_by_signal`` (the default Layer-21 router). The KEEP
    # gate below (uplift + noise floor) stays MI-based: relevance IS an MI
    # question. ``routing_criterion="mi"`` restores the legacy argmax.
    if str(routing_criterion).lower() == "corr":
        _yf = y_arr.astype(np.float64)
        _ystd = float(_yf.std())
        if _ystd > 1e-12:
            _Mz = (cand_mat - cand_mat.mean(axis=0)) / (cand_mat.std(axis=0) + 1e-12)
            route_score = np.abs(_Mz.T @ ((_yf - _yf.mean()) / _ystd) / _yf.size)
        else:
            route_score = eng_mi
    else:
        route_score = eng_mi

    # ---- Step 4: per-source argmax over (pre, basis, degree) by route_score;
    # the selected cell's MI is retained for the (MI-based) keep gate in Step 5.
    best_per_source: dict[str, dict] = {}
    for j, (src, pt, basis_name, deg) in enumerate(cand_meta):
        emi = float(eng_mi[j])
        rscore = float(route_score[j])
        if not np.isfinite(emi) or not np.isfinite(rscore):
            continue
        cur = best_per_source.get(src)
        if cur is None or rscore > cur["route_score"]:
            best_per_source[src] = {
                "src": src,
                "pre_transform": pt,
                "basis": basis_name,
                "degree": deg,
                "engineered_mi": emi,
                "route_score": rscore,
                "engineered_col": cand_cols[j],
                "values_idx": j,
            }

    # ---- Step 5: uplift gate + noise-aware absolute MI floor + top-K
    # The per-column argmax can still surface noise-only columns where raw
    # MI is itself tiny and the ratio amplifies pure tail sampling. Mirror
    # Layer 21's two-gate (relative uplift + noise-aware absolute floor)
    # but use the PER-SOURCE-BEST engineered MI distribution for the noise
    # floor instead of the full candidate distribution: per-source winners
    # are a much cleaner signal-vs-noise comparison (with 4 sources of
    # which 3 carry signal, the full candidate pool has too much signal
    # mass for MAD to identify the noise band, while the per-source-best
    # distribution has at most 1 row per source which preserves the
    # signal-vs-noise gap). The raw-baseline MAD floor is unchanged from
    # Layer 21 and remains the dominant safeguard on all-noise frames.
    min_uplift_f = float(min_uplift)
    raw_baselines = np.asarray(list(raw_mi_map.values()), dtype=np.float64)
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    # Bonferroni MAD floor only safe when there's enough columns for MAD
    # to identify the noise band. With p<16 a multi-signal frame can have
    # many baselines well above the noise mass; MAD then captures the
    # signal-vs-signal spread and the resulting threshold exceeds every
    # legitimate engineered MI. Gate on n_sources >= 16 so all-noise
    # frames at production p (typically >> 16) get the protection while
    # small benchmark frames still emit signal columns.
    if raw_baselines.size >= 16:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    # Per-source-best engineered MI distribution noise floor: only safe
    # when there are MANY sources (so MAD reliably tracks the noise band
    # and a few signals don't pull the median past every survivor). For
    # small p the per-source-best distribution has too few rows; one signal
    # source drags the median above the noise mass and even legitimate
    # signal columns at neighbouring sources can be filtered out. Gate
    # this floor on n_sources >= 16; otherwise rely on raw-baseline MAD +
    # legacy floor + the per-column relative-uplift gate (which still
    # screens out noise-MI tail amplifications).
    best_emis = np.asarray(
        [float(info["engineered_mi"]) for info in best_per_source.values()],
        dtype=np.float64,
    )
    if best_emis.size >= 16:
        med_e = float(np.median(best_emis))
        mad_e = float(np.median(np.abs(best_emis - med_e)))
        eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    survivors: list[dict] = []
    for src, info in best_per_source.items():
        baseline = float(raw_mi_map.get(src, 0.0))
        emi = float(info["engineered_mi"])
        uplift = emi / (baseline + 1e-12)
        if uplift < min_uplift_f:
            continue
        if emi < abs_floor:
            continue
        info_copy = dict(info)
        info_copy["uplift"] = float(uplift)
        info_copy["baseline_mi"] = float(baseline)
        survivors.append(info_copy)
    survivors.sort(key=lambda d: d["uplift"], reverse=True)
    winners = survivors[: int(top_k)]

    out_cols: dict = {}
    meta: dict = {}
    for info in winners:
        name = str(info["engineered_col"])
        vals = cand_values[int(info["values_idx"])]
        out_cols[name] = vals
        meta[name] = {
            "src": info["src"],
            "basis": str(info["basis"]),
            "degree": int(info["degree"]),
            "pre_transform": str(info["pre_transform"]),
            "uplift": float(info["uplift"]),
            "engineered_mi": float(info["engineered_mi"]),
            "baseline_mi": float(info["baseline_mi"]),
        }
    return pd.DataFrame(out_cols, index=X.index), meta


def hybrid_orth_mi_conditional_routing_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    candidate_bases: Optional[Sequence[str]] = None,
    transform_variants: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    top_k: int = 5,
    min_uplift: float = 1.10,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    routing_criterion: str = "corr",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conditional-basis routing hybrid: pick the best (basis, pre_transform,
    degree) cell per source column, drop sources whose best fails the
    ``min_uplift`` gate, return the augmented frame plus a tidy scores
    DataFrame.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the surviving top-K columns appended.
        scores : DataFrame with columns ``[engineered_col, source_col,
            basis, degree, pre_transform, baseline_mi, engineered_mi,
            uplift]`` ordered by ``uplift`` descending.
    """
    engineered, meta = generate_conditional_basis_routing_features(
        X, y,
        cols=cols,
        candidate_bases=candidate_bases,
        transform_variants=transform_variants,
        degrees=degrees,
        top_k=top_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        nbins=nbins,
        routing_criterion=routing_criterion,
    )
    if engineered.empty:
        scores_empty = pd.DataFrame(columns=[
            "engineered_col", "source_col", "basis", "degree", "pre_transform",
            "baseline_mi", "engineered_mi", "uplift",
        ])
        return X.copy(), scores_empty
    rows = []
    for name, info in meta.items():
        rows.append({
            "engineered_col": name,
            "source_col": info["src"],
            "basis": info["basis"],
            "degree": info["degree"],
            "pre_transform": info["pre_transform"],
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


def hybrid_orth_mi_conditional_routing_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    candidate_bases: Optional[Sequence[str]] = None,
    transform_variants: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    top_k: int = 5,
    min_uplift: float = 1.10,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    routing_criterion: str = "corr",
):
    """Same as :func:`hybrid_orth_mi_conditional_routing_fe` but additionally
    returns a list of ``EngineeredRecipe`` objects (kind ``orth_univariate``)
    -- one per appended column -- so ``MRMR.transform`` can replay each
    engineered column on test data without re-running the per-column MI scan.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from .engineered_recipes import build_orth_univariate_recipe
    X_aug, scores = hybrid_orth_mi_conditional_routing_fe(
        X, y,
        cols=cols,
        candidate_bases=candidate_bases,
        transform_variants=transform_variants,
        degrees=degrees,
        top_k=top_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        nbins=nbins,
        routing_criterion=routing_criterion,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    if not appended:
        return X_aug, scores, []
    name_to_row = {str(row["engineered_col"]): row for _, row in scores.iterrows()}
    recipes = []
    for name in appended:
        row = name_to_row.get(name)
        if row is None:
            logger.warning(
                "hybrid_orth_mi_conditional_routing_fe_with_recipes: appended " "column %r missing from scores; skipping recipe.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name,
            src_name=str(row["source_col"]),
            basis=str(row["basis"]),
            degree=int(row["degree"]),
            pre_transform=str(row["pre_transform"]),
        ))
    return X_aug, scores, recipes
