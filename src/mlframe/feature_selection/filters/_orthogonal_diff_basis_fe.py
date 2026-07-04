"""Layer 59 (2026-05-31): DIFF-BASIS FE for highly-correlated source pairs.

Why this layer
--------------

For tightly-correlated source pairs (e.g. ``price_today`` and ``price_yesterday``,
correlation 0.99) the marginal of either column is dominated by the shared
trend; what survives after subtracting the trend is the *residual* signal --
exactly what a basis expansion ``He_n(x_i - x_j)`` can lift out. Layer 21
(univariate) and Layer 27 (collinear-dedup) both DROP one of the two
collinear sources before any basis is evaluated, so the residual signal is
gone before MI ranking sees it. Pair-cross-basis (Layer 25's
``orth_pair_cross``) covers ``He_a(x_i) * He_b(x_j)`` -- multiplicative
interaction, NOT the additive residual.

Layer 59 fills that gap with a dedicated diff path: for every pair (i, j)
whose absolute Pearson correlation clears ``pair_corr_threshold``, compute
``d = X[i] - X[j]``, evaluate ``basis_d(preprocess(d))`` for each requested
degree, and feed the resulting columns through the same MI uplift + MAD
floor gates as Layer 21/58.

Combinatorial budget
--------------------

Auto-pair detection uses one bulk ``np.corrcoef`` call on the dense numeric
block (mirrors :func:`_dedup_collinear_source_cols` from
``_orthogonal_univariate_fe``), so the auto-pair step is O(p^2) Python-level
but one C call. At p=200 that's 200*199/2 = ~20k pairs scanned in ~50 ms;
typically only a handful clear the 0.7 threshold. Per surviving pair we emit
``len(degrees)`` candidates and run ONE batch MI call across the whole pool.

Recipe replay
-------------

Each emitted column is backed by an ``orth_diff_basis`` recipe whose
``extra`` carries ``{basis, degree, pre_transform}``. ``src_names`` is the
ordered tuple ``(col_a, col_b)``; the diff is ALWAYS computed as
``X[col_a] - X[col_b]`` so test-time replay reads the SAME orientation as
fit time. ``pre_transform`` defaults to ``"raw"`` so legacy code paths stay
byte-identical.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_diff_basis_enable=True``.
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
)

logger = logging.getLogger(__name__)

__all__ = [
    "detect_correlated_pairs",
    "generate_diff_basis_features",
    "hybrid_orth_mi_diff_basis_fe",
    "hybrid_orth_mi_diff_basis_fe_with_recipes",
    "parse_diff_basis_col_name",
]


def _diff_col_name(col_a: str, col_b: str, basis: str, degree: int) -> str:
    """Stable engineered column name for a diff-basis column. The prefix
    ``diff_`` is used so the recipe parser can detect the kind directly
    from the name and so the name visually distinguishes diff columns
    from Layer 21 univariate ones (``{col}__He2``).
    """
    code = _BASIS_CODE.get(basis, basis)
    return f"diff_{col_a}_{col_b}__{code}{int(degree)}"


def parse_diff_basis_col_name(name: str) -> Optional[tuple[str, str, str, int]]:
    """Inverse of :func:`_diff_col_name`. Returns ``(col_a, col_b, basis,
    degree)`` or ``None`` if ``name`` is not a diff-basis column name.

    The parser handles only the canonical naming convention; recipes
    carry the authoritative metadata in ``extra`` so callers that need
    bit-exact replay should NOT depend on name parsing.
    """
    if not name.startswith("diff_") or "__" not in name:
        return None
    body, suffix = name.split("__", 1)
    # body starts with "diff_"; strip prefix.
    body = body[len("diff_"):]
    # ``body`` is ``"{col_a}_{col_b}"`` but col names may themselves contain
    # underscores. We can't unambiguously split here, so the recipe
    # ``src_names`` tuple is the canonical source -- this helper is best-
    # effort and used only for diagnostics / lookup keys.
    if "_" not in body:
        return None
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
    for code in ("LL", "He", "T", "L"):
        if suffix.startswith(code):
            deg_str = suffix[len(code):]
            if deg_str.isdigit():
                # Split body on the LAST underscore -- if col names contain
                # underscores the split may be wrong; recipe metadata is the
                # authoritative source.
                col_a, col_b = body.rsplit("_", 1)
                return (col_a, col_b, code_to_basis[code], int(deg_str))
    return None


def detect_correlated_pairs(
    X: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    *,
    corr_threshold: float = 0.7,
    max_pairs: int = 200,
) -> list[tuple[str, str, float]]:
    """Identify all (col_a, col_b) pairs whose |Pearson corr| >= threshold.

    Bulk implementation: stacks the dense numeric block once and calls
    ``np.corrcoef`` ONCE; per-pair Python overhead is then O(K) where K is
    the number of surviving pairs. Constant / all-NaN columns are silently
    excluded (no correlation defined).

    Pairs are returned sorted by absolute correlation descending and
    truncated to ``max_pairs`` so an adversarially-collinear frame can't
    explode the downstream candidate pool. The pair ordering puts the
    column with the LARGER name (lexicographically) first only when the
    correlation tie cannot be broken; primary sort is on correlation
    magnitude so the strongest residual signals are tried first.

    Parameters
    ----------
    X : DataFrame
        Source frame.
    cols : sequence of column names, optional
        Restrict the scan. None = all numeric columns.
    corr_threshold : float, default 0.7
        Minimum absolute Pearson correlation. Below this the diff is
        typically dominated by independent noise; the basis expansion
        adds rows of garbage.
    max_pairs : int, default 200
        Cap on the returned pair count to bound the downstream MI batch
        size. Sorted by |corr| descending so the strongest pairs survive.

    Returns
    -------
    list of ``(col_a, col_b, abs_corr)`` tuples, sorted by ``abs_corr``
    descending.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = [
        c for c in cols
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    if len(cols) < 2:
        return []
    # Stack dense varying columns into a single matrix; skip constant /
    # all-NaN sources entirely (no meaningful correlation against anything).
    dense_arrays: list[np.ndarray] = []
    dense_names: list[str] = []
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
    for c in cols:
        arr = np.asarray(X[c].to_numpy(), dtype=_dt)
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        # Fill NaN with column mean before correlation (matches the rest of
        # the orth-FE pipeline).
        if not finite.all():
            arr = np.where(finite, arr, float(np.nanmean(arr[finite])))
        if float(arr.std()) <= 1e-12:
            continue
        dense_arrays.append(arr)
        dense_names.append(c)
    if len(dense_arrays) < 2:
        return []
    mat = np.vstack(dense_arrays)
    corr = np.corrcoef(mat)
    if corr.ndim == 0:
        return []
    abs_corr = np.abs(corr)
    out: list[tuple[str, str, float]] = []
    p = len(dense_names)
    for i in range(p):
        for j in range(i + 1, p):
            c_ij = float(abs_corr[i, j])
            if not np.isfinite(c_ij):
                continue
            if c_ij >= corr_threshold:
                # ``< 1 - 1e-12`` guards against literal duplicates (corr = 1.0)
                # where the diff is identically zero; the basis_d(0) value is
                # also a constant and would be dropped downstream, so save the
                # cycles up front.
                if c_ij < 1.0 - 1e-12:
                    out.append((dense_names[i], dense_names[j], c_ij))
    out.sort(key=lambda t: t[2], reverse=True)
    if len(out) > int(max_pairs):
        out = out[: int(max_pairs)]
    return out


def generate_diff_basis_features(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    pairs: Optional[Sequence[tuple[str, str]]] = None,
    cols: Optional[Sequence[str]] = None,
    basis: str = "hermite",
    degrees: Sequence[int] = (1, 2, 3),
    pair_corr_threshold: float = 0.7,
    max_pairs: int = 200,
    top_k: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """For each (col_a, col_b) pair, emit ``basis_d(preprocess(X[a] - X[b]))``
    columns for each degree, rank by MI uplift against the BETTER of the two
    source baselines, gate by ``min_uplift`` + a noise-aware absolute floor,
    and return the global top-K winners.

    Parameters
    ----------
    X : DataFrame
        Source frame. Non-numeric columns are silently ignored.
    y : array-like (n,)
        Discrete (binary or small-cardinality int codes) target. Continuous
        targets must be binned by the caller (the MRMR wiring uses qcut).
    pairs : sequence of (col_a, col_b) tuples, optional
        Explicit pair list. When None, pairs are auto-detected via
        :func:`detect_correlated_pairs` using ``pair_corr_threshold`` and
        the optional ``cols`` filter.
    cols : sequence of column names, optional
        Restrict the auto-pair scan. Ignored when ``pairs`` is non-None.
    basis : {'hermite', 'legendre', 'chebyshev', 'laguerre'}
        Polynomial family. ``hermite`` (default) is the right choice for
        residual signals -- ``He_1`` is the diff itself and ``He_2`` /
        ``He_3`` lift quadratic / cubic residuals.
    degrees : sequence of int
        Polynomial degrees per pair. Defaults to ``(1, 2, 3)``; ``He_1``
        is included because ``diff_a_b__He1`` IS the standardised diff
        (most common residual-signal case) and is far cheaper to compute
        than ``He_2``.
    pair_corr_threshold : float, default 0.7
        Forwarded to :func:`detect_correlated_pairs` when ``pairs`` is None.
    top_k : int, default 3
        Global top-K winners across all (pair, degree) cells.
    min_uplift : float, default 1.05
        Per-(pair, degree) gate: skip if engineered MI does not reach
        ``min_uplift * max(MI(col_a; y), MI(col_b; y))``. Comparing
        against the BEST source baseline is the right standard: we only
        want to keep the diff column when it beats BOTH original sources.
    min_abs_mi_frac : float, default 0.1
        Absolute MI floor as a fraction of the largest baseline MI.
        Mirrors Layer 21's two-gate noise control.
    nbins : int, default 10
        Quantile bins for MI estimation.

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of new columns (one per top-K winner)
            with names from :func:`_diff_col_name`.
        meta : dict mapping each emitted column name to a dict carrying
            ``{"col_a", "col_b", "basis", "degree", "pair_corr",
            "uplift", "engineered_mi", "baseline_mi"}`` for recipe
            replay and diagnostics.
    """
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    if basis not in _POLY_BASES:
        raise ValueError(
            f"generate_diff_basis_features: unknown basis {basis!r}; "
            f"expected one of {sorted(_POLY_BASES.keys())}."
        )
    degrees = tuple(int(d) for d in degrees)
    if not degrees:
        return pd.DataFrame(index=X.index), {}
    # ---- Step 1: resolve the pair list (explicit or auto-detected).
    if pairs is None:
        detected = detect_correlated_pairs(
            X, cols, corr_threshold=pair_corr_threshold, max_pairs=max_pairs,
        )
        pair_corr_map = {(a, b): c for (a, b, c) in detected}
        pairs_norm = [(a, b) for (a, b, _) in detected]
    else:
        pairs_norm = []
        pair_corr_map = {}
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(
                    f"generate_diff_basis_features: every entry in ``pairs`` "
                    f"must be a 2-tuple; got {pair!r}."
                )
            a, b = pair
            if a not in X.columns or b not in X.columns:
                logger.warning(
                    "generate_diff_basis_features: pair (%r, %r) skipped; "
                    "column missing from X.", a, b,
                )
                continue
            if not (pd.api.types.is_numeric_dtype(X[a])
                    and pd.api.types.is_numeric_dtype(X[b])):
                logger.warning(
                    "generate_diff_basis_features: pair (%r, %r) skipped; "
                    "non-numeric dtype.", a, b,
                )
                continue
            pairs_norm.append((a, b))
            # Compute correlation for diagnostics; not gating an explicit pair.
            from ._fe_usability_signal import _crit_np_dtype
            _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
            arr_a = np.asarray(X[a].to_numpy(), dtype=_dt)
            arr_b = np.asarray(X[b].to_numpy(), dtype=_dt)
            mask = np.isfinite(arr_a) & np.isfinite(arr_b)
            if mask.sum() >= 8 and float(arr_a[mask].std()) > 1e-12 and float(arr_b[mask].std()) > 1e-12:
                pair_corr_map[(a, b)] = float(abs(np.corrcoef(arr_a[mask], arr_b[mask])[0, 1]))
            else:
                pair_corr_map[(a, b)] = 0.0
    if not pairs_norm:
        return pd.DataFrame(index=X.index), {}

    y_arr = (
        np.asarray(y).astype(np.int64)
        if not np.issubdtype(np.asarray(y).dtype, np.integer)
        else np.asarray(y, dtype=np.int64)
    )
    # ---- Step 2: raw baselines for every column touched by a pair.
    touched = sorted({c for pair in pairs_norm for c in pair})
    raw_mat = X[touched].to_numpy(dtype=np.float64, copy=False)
    # NaN-safe fill mirrors fit-side basis evaluation.
    finite = np.isfinite(raw_mat)
    if not finite.all():
        col_means = np.where(
            finite.any(axis=0),
            np.where(finite, raw_mat, 0.0).sum(axis=0) / np.maximum(finite.sum(axis=0), 1),
            0.0,
        )
        raw_mat = np.where(finite, raw_mat, col_means[None, :])
    raw_mi = _mi_classif_batch(raw_mat, y_arr, nbins=nbins)
    raw_mi_map = dict(zip(touched, raw_mi.tolist()))

    # ---- Step 3: enumerate (pair, degree) cells.
    cand_cols: list[str] = []
    cand_values: list[np.ndarray] = []
    cand_meta: list[tuple[str, str, int]] = []  # (col_a, col_b, degree)

    for col_a, col_b in pairs_norm:
        x_a = np.asarray(X[col_a].to_numpy(), dtype=_dt)
        x_b = np.asarray(X[col_b].to_numpy(), dtype=_dt)
        finite_ab = np.isfinite(x_a) & np.isfinite(x_b)
        if not finite_ab.all():
            # Per-column mean-fill keeps the diff well-defined elementwise.
            if finite_ab.any():
                a_fill = float(np.nanmean(x_a[finite_ab])) if finite_ab.any() else 0.0
                b_fill = float(np.nanmean(x_b[finite_ab])) if finite_ab.any() else 0.0
            else:
                a_fill = 0.0
                b_fill = 0.0
            x_a = np.where(np.isfinite(x_a), x_a, a_fill)
            x_b = np.where(np.isfinite(x_b), x_b, b_fill)
        diff = x_a - x_b
        if not np.isfinite(diff).all():
            finite_d = np.isfinite(diff)
            fill_d = float(np.nanmean(diff[finite_d])) if finite_d.any() else 0.0
            diff = np.where(finite_d, diff, fill_d)
        if float(np.std(diff)) <= 1e-12:
            # Constant diff (identical sources after fill) -- no signal.
            continue
        for d in degrees:
            try:
                vals = _evaluate_basis_column(diff, basis, int(d))
            except Exception as exc:
                logger.warning(
                    "generate_diff_basis_features: basis=%r degree=%d on "
                    "pair (%r, %r) raised %r; skipping cell.",
                    basis, d, col_a, col_b, exc,
                )
                continue
            if not np.isfinite(vals).all():
                continue
            if float(np.std(vals)) <= 1e-12:
                continue
            cand_cols.append(_diff_col_name(col_a, col_b, basis, int(d)))
            cand_values.append(vals)
            cand_meta.append((col_a, col_b, int(d)))

    if not cand_cols:
        return pd.DataFrame(index=X.index), {}

    # ---- Step 4: ONE batch MI call across every candidate.
    cand_mat = np.column_stack(cand_values).astype(np.float64, copy=False)
    eng_mi = _mi_classif_batch(cand_mat, y_arr, nbins=nbins)

    # ---- Step 5: uplift gate (vs BEST source baseline) + noise-aware
    # absolute floor + global top-K.
    raw_baselines = np.asarray(list(raw_mi_map.values()), dtype=np.float64)
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    n_baselines = int(raw_baselines.size)
    # Mirror Layer 21's adaptive sigma so an all-noise frame trips the MAD
    # floor while a small-p signal frame still emits winners.
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * max(n_baselines, 1)))) + 1.5),
    )
    if n_baselines >= 16:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor)

    survivors: list[dict] = []
    for j, (col_a, col_b, deg) in enumerate(cand_meta):
        emi = float(eng_mi[j])
        if not np.isfinite(emi):
            continue
        baseline_a = float(raw_mi_map.get(col_a, 0.0))
        baseline_b = float(raw_mi_map.get(col_b, 0.0))
        baseline = max(baseline_a, baseline_b)
        uplift = emi / (baseline + 1e-12)
        if uplift < float(min_uplift):
            continue
        if emi < abs_floor:
            continue
        survivors.append({
            "engineered_col": cand_cols[j],
            "values_idx": j,
            "col_a": col_a,
            "col_b": col_b,
            "basis": str(basis),
            "degree": int(deg),
            "engineered_mi": emi,
            "baseline_mi": baseline,
            "uplift": float(uplift),
            "pair_corr": float(pair_corr_map.get((col_a, col_b), 0.0)),
        })

    survivors.sort(key=lambda d: d["uplift"], reverse=True)
    winners = survivors[: int(top_k)]

    out_cols: dict = {}
    meta: dict = {}
    for info in winners:
        name = str(info["engineered_col"])
        vals = cand_values[int(info["values_idx"])]
        out_cols[name] = vals
        meta[name] = {
            "col_a": info["col_a"],
            "col_b": info["col_b"],
            "basis": info["basis"],
            "degree": int(info["degree"]),
            "pair_corr": float(info["pair_corr"]),
            "engineered_mi": float(info["engineered_mi"]),
            "baseline_mi": float(info["baseline_mi"]),
            "uplift": float(info["uplift"]),
        }
    return pd.DataFrame(out_cols, index=X.index), meta


def hybrid_orth_mi_diff_basis_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    pairs: Optional[Sequence[tuple[str, str]]] = None,
    cols: Optional[Sequence[str]] = None,
    basis: str = "hermite",
    degrees: Sequence[int] = (1, 2, 3),
    pair_corr_threshold: float = 0.7,
    max_pairs: int = 200,
    top_k: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end diff-basis hybrid: auto-detect pairs (when ``pairs`` is
    None), evaluate basis expansions of the residual, rank by MI uplift,
    return augmented frame + tidy scores DataFrame.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with surviving top-K columns appended.
        scores : DataFrame with columns ``[engineered_col, col_a, col_b,
            basis, degree, pair_corr, baseline_mi, engineered_mi, uplift]``
            ordered by ``uplift`` descending.
    """
    engineered, meta = generate_diff_basis_features(
        X, y,
        pairs=pairs,
        cols=cols,
        basis=basis,
        degrees=degrees,
        pair_corr_threshold=pair_corr_threshold,
        max_pairs=max_pairs,
        top_k=top_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        nbins=nbins,
    )
    if engineered.empty:
        scores_empty = pd.DataFrame(columns=[
            "engineered_col", "col_a", "col_b", "basis", "degree", "pair_corr",
            "baseline_mi", "engineered_mi", "uplift",
        ])
        return X.copy(), scores_empty
    rows = []
    for name, info in meta.items():
        rows.append({
            "engineered_col": name,
            "col_a": info["col_a"],
            "col_b": info["col_b"],
            "basis": info["basis"],
            "degree": info["degree"],
            "pair_corr": info["pair_corr"],
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


def hybrid_orth_mi_diff_basis_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    pairs: Optional[Sequence[tuple[str, str]]] = None,
    cols: Optional[Sequence[str]] = None,
    basis: str = "hermite",
    degrees: Sequence[int] = (1, 2, 3),
    pair_corr_threshold: float = 0.7,
    max_pairs: int = 200,
    top_k: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_diff_basis_fe` but additionally returns
    a list of ``EngineeredRecipe`` objects (kind ``"orth_diff_basis"``) so
    ``MRMR.transform`` can replay each appended column on test data without
    re-running the MI scan.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from ._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); hoisted so _dt is bound on every branch
    from .engineered_recipes import build_orth_diff_basis_recipe
    X_aug, scores = hybrid_orth_mi_diff_basis_fe(
        X, y,
        pairs=pairs,
        cols=cols,
        basis=basis,
        degrees=degrees,
        pair_corr_threshold=pair_corr_threshold,
        max_pairs=max_pairs,
        top_k=top_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    if not appended:
        return X_aug, scores, []
    name_to_row = {
        str(row["engineered_col"]): row for _, row in scores.iterrows()
    }
    recipes = []
    for name in appended:
        row = name_to_row.get(name)
        if row is None:
            logger.warning(
                "hybrid_orth_mi_diff_basis_fe_with_recipes: appended column "
                "%r missing from scores; skipping recipe.", name,
            )
            continue
        _ca, _cb = str(row["col_a"]), str(row["col_b"])
        _basis_d, _degree_d = str(row["basis"]), int(row["degree"])
        # REPLAY-FIDELITY FIX (2026-06-13): freeze the diff's fit-time basis-preprocess params so
        # transform() replays the axis byte-exactly (no slice-vs-full refit drift). Mirror the apply
        # NaN-fill so the frozen params match. Guarded -> params stay None (legacy refit) on any failure.
        _pp_d = None
        try:
            from ._fe_usability_signal import _crit_np_dtype
            _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); MI binning is scale-robust
            _va = np.asarray(X[_ca], dtype=_dt)
            _vb = np.asarray(X[_cb], dtype=_dt)
            _fa = np.isfinite(_va)
            if not _fa.all():
                _va = np.where(_fa, _va, float(np.nanmean(_va[_fa])) if _fa.any() else 0.0)
            _fb = np.isfinite(_vb)
            if not _fb.all():
                _vb = np.where(_fb, _vb, float(np.nanmean(_vb[_fb])) if _fb.any() else 0.0)
            _, _pp_d = _evaluate_basis_column(_va - _vb, _basis_d, _degree_d, return_params=True)
        except Exception:
            pass
        recipes.append(build_orth_diff_basis_recipe(
            name=name,
            col_a=_ca,
            col_b=_cb,
            basis=_basis_d,
            degree=_degree_d,
            preprocess_params=_pp_d,
        ))
    return X_aug, scores, recipes


def _apply_orth_diff_basis(recipe, X) -> np.ndarray:
    """Replay a diff-basis column. Lazy-imported by the recipes dispatcher
    so the recipes module stays under its size budget.

    Stateless given (col_a, col_b, basis, degree, pre_transform); no y
    reference is captured at fit time. The diff is ALWAYS ``X[col_a] -
    X[col_b]`` so train/test orientation parity holds.
    """
    from .engineered_recipes import _extract_column, _eval_orth_basis_column
    if len(recipe.src_names) != 2:
        raise ValueError(
            f"orth_diff_basis recipe '{recipe.name}' must have exactly 2 "
            f"src_names; got {len(recipe.src_names)}"
        )
    for key in ("basis", "degree"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_diff_basis recipe '{recipe.name}' missing '{key}' "
                f"in extra. Re-fit MRMR to regenerate."
            )
    col_a, col_b = recipe.src_names
    basis = str(recipe.extra["basis"])
    degree = int(recipe.extra["degree"])
    pre_transform = str(recipe.extra.get("pre_transform", "raw"))
    vals_a = np.asarray(_extract_column(X, col_a), dtype=np.float64)
    vals_b = np.asarray(_extract_column(X, col_b), dtype=np.float64)
    # NaN-safe mean-fill mirrors the fit-time pipeline so train/test parity
    # holds row-by-row even when the test frame has different NaN positions.
    finite_a = np.isfinite(vals_a)
    if not finite_a.all():
        fill_a = float(np.nanmean(vals_a[finite_a])) if finite_a.any() else 0.0
        vals_a = np.where(finite_a, vals_a, fill_a)
    finite_b = np.isfinite(vals_b)
    if not finite_b.all():
        fill_b = float(np.nanmean(vals_b[finite_b])) if finite_b.any() else 0.0
        vals_b = np.where(finite_b, vals_b, fill_b)
    diff = vals_a - vals_b
    # REPLAY-FIDELITY FIX (2026-06-13): frozen fit-time basis-preprocess params (mirrors the pair/
    # triplet/quad fix); without them _eval_orth_basis_column refits the z-score/min-max of the diff
    # from apply-time rows -> slice-replay corruption. None (legacy pickles) -> refit path, byte-identical.
    pp = recipe.extra.get("preprocess_params")
    return _eval_orth_basis_column(diff, basis, degree, pre_transform=pre_transform, preprocess_params=pp)
