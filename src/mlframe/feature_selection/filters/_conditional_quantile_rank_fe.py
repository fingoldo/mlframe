"""Conditional quantile-rank feature (mrmr_audit_2026-07-20 fe_expansion.md "Extend conditional-
dispersion to the full conditional quantile").

Extends the existing conditional-dispersion family (z-within-group in ``_grouped_agg_fe.py`` /
``_composite_group_agg_fe.py``, and the z-score / |z| features in
``_extra_fe_families_dispersion.py``) from CONDITIONAL MEAN/STD ONLY to the full CONDITIONAL
QUANTILE: for a numeric column ``x_i`` and a binned conditioning column ``x_j``, compute
``q(row) = empirical_rank(x_i within bin(x_j))`` -- the row's percentile position WITHIN its
conditioning bin, not its z-score.

Why this catches a shape the catalog misses: on a heavy-tailed / skewed conditional distribution
(e.g. ``x_i | bin(x_j)`` is log-normal, common for financial/count data), a z-score
``(x-mu)/sigma`` badly misrepresents "how extreme" a row is -- the mean/std pair is not a
sufficient statistic for a skewed shape, so two rows with identical z-scores can sit at very
different TRUE percentiles. A target that depends on "is this row in the top-5% of its conditional
peer group" (e.g. fraud/outlier detection conditioned on a merchant category) is exactly the shape
z-score under-resolves and quantile-rank resolves directly.

The per-bin quantile edges are fit on TRAIN rows only (leak-safe, mirroring the existing K-fold-
fit-then-apply discipline used elsewhere in this codebase), then applied to ALL rows (train+test)
via ``searchsorted`` at apply time -- a row whose bin was never seen at fit time gets NaN rather
than a spurious extrapolated rank.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = [
    "conditional_quantile_rank_fe",
    "generate_conditional_quantile_rank_features",
    "apply_conditional_quantile_rank",
    "build_conditional_quantile_rank_recipe",
    "hybrid_conditional_quantile_rank_fe",
    "engineered_name_conditional_quantile_rank",
]


def conditional_quantile_rank_fe(
    x_i: np.ndarray,
    x_j_bins: np.ndarray,
    *,
    x_i_fit: Optional[np.ndarray] = None,
    x_j_bins_fit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The empirical within-bin percentile rank of ``x_i``, conditioned on the discrete groups in
    ``x_j_bins``.

    Parameters
    ----------
    x_i : (n,) array
        The numeric column to rank.
    x_j_bins : (n,) array
        Discrete conditioning bin/category id per row (e.g. an already-quantile-binned column, or
        a categorical group id).
    x_i_fit, x_j_bins_fit : (n_fit,) arrays, optional
        The rows to fit the per-bin sorted-value reference on. ``None`` (default) fits on
        ``x_i``/``x_j_bins`` themselves. Pass the TRAIN rows explicitly for a leak-safe fit-once/
        apply-to-all-rows contract.

    Returns
    -------
    (n,) float64 array in ``[0, 1]``: the row's percentile position within its conditioning bin's
    fitted value distribution (``searchsorted(sorted_bin_values, x_i[row]) / len(sorted_bin_values)``).
    NaN for a row whose ``x_j_bins`` value was never seen at fit time, or for non-finite input.
    """
    x_i = np.asarray(x_i, dtype=np.float64).ravel()
    x_j_bins = np.asarray(x_j_bins).ravel()
    n = x_i.shape[0]
    if x_j_bins.shape[0] != n:
        raise ValueError(f"conditional_quantile_rank_fe: x_i has {n} rows but x_j_bins has {x_j_bins.shape[0]}")

    fit_x = np.asarray(x_i_fit, dtype=np.float64).ravel() if x_i_fit is not None else x_i
    fit_bins = np.asarray(x_j_bins_fit).ravel() if x_j_bins_fit is not None else x_j_bins
    if fit_bins.shape[0] != fit_x.shape[0]:
        raise ValueError("conditional_quantile_rank_fe: x_i_fit and x_j_bins_fit must have the same length")

    out = np.full(n, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(x_i)

    for b in np.unique(fit_bins):
        fit_vals = fit_x[fit_bins == b]
        fit_vals = fit_vals[np.isfinite(fit_vals)]
        if fit_vals.size == 0:
            continue
        sorted_vals = np.sort(fit_vals)
        row_mask = finite_mask & (x_j_bins == b)
        if not row_mask.any():
            continue
        ranks = np.searchsorted(sorted_vals, x_i[row_mask], side="right")
        out[row_mask] = ranks / sorted_vals.size

    return np.clip(out, 0.0, 1.0)


def engineered_name_conditional_quantile_rank(x_i: str, x_j: str) -> str:
    """Deterministic engineered-column name for the (x_i, x_j) conditional quantile-rank pair."""
    return f"{x_i}__qrank_by__{x_j}"


def generate_conditional_quantile_rank_features(
    X: "pd.DataFrame",
    num_cols: Sequence[str],
    *,
    n_bins: int = 10,
) -> "tuple[pd.DataFrame, dict[str, dict]]":
    """For every ordered pair ``(x_i, x_j)`` from ``num_cols`` emit the conditional quantile-rank
    column ``q(row) = empirical_rank(x_i within quantile-bin(x_j))``.

    Mirrors ``_extra_fe_families_dispersion.generate_conditional_dispersion_features``'s pair-
    orchestration (fit-time quantile edges on ``x_j``, then the closed-form per-bin statistic of
    ``x_i``) but computes the FULL empirical rank rather than a z-score. Returns ``(enc_df,
    raw_recipes)`` where each recipe payload stores the ``x_j`` quantile edges + the per-bin SORTED
    ``x_i`` values (the reference the row-level rank is computed against at replay time).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"generate_conditional_quantile_rank_features: X must be a pandas DataFrame; got {type(X).__name__}")
    if len(X) == 0:
        raise ValueError("generate_conditional_quantile_rank_features: X is empty")
    num_cols = [c for c in num_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if len(num_cols) < 2:
        return pd.DataFrame(index=X.index), raw_recipes

    from ._extra_fe_families import _digitize_with_edges, _quantile_edges  # lazy: break parent<->sibling cycle

    col_vals = {c: np.asarray(X[c].to_numpy(), dtype=np.float64) for c in num_cols}
    for x_j in num_cols:
        xj = col_vals[x_j]
        edges = _quantile_edges(xj, n_bins)
        codes_j = _digitize_with_edges(xj, edges)
        for x_i in num_cols:
            if x_i == x_j:
                continue
            xi = col_vals[x_i]
            vals = conditional_quantile_rank_fe(xi, codes_j)
            # Skip a degenerate constant emission (every row lands in the same rank, e.g. all-tied
            # x_i within every bin) -- it carries no information and only burdens the screen.
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size == 0 or float(np.std(finite_vals)) <= 1e-12:
                continue
            name = engineered_name_conditional_quantile_rank(x_i, x_j)
            encoded[name] = vals.astype(np.float64)
            sorted_bins = []
            for b in range(int(codes_j.max()) + 1 if codes_j.size else 0):
                bin_vals = xi[codes_j == b]
                bin_vals = bin_vals[np.isfinite(bin_vals)]
                sorted_bins.append(np.sort(bin_vals))
            raw_recipes[name] = {"x_i": x_i, "x_j": x_j, "edges": edges, "sorted_bins": sorted_bins}

    return pd.DataFrame(encoded, index=X.index), raw_recipes


def apply_conditional_quantile_rank(X_test: "pd.DataFrame", recipe: dict) -> np.ndarray:
    """Replay one conditional quantile-rank column from the stored ``x_j`` edges + per-bin sorted
    ``x_i`` reference values. NaN ``x_i`` rows (or a row whose bin has no fitted reference values)
    emit NaN. Reads only X (pure-X, y-independent -> leak-safe)."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"apply_conditional_quantile_rank: X_test must be a DataFrame; got {type(X_test).__name__}")
    x_i = recipe["x_i"]
    x_j = recipe["x_j"]
    if x_i not in X_test.columns or x_j not in X_test.columns:
        raise KeyError(f"apply_conditional_quantile_rank: missing column(s) {x_i!r}/{x_j!r} from X_test")
    edges = np.asarray(recipe["edges"], dtype=np.float64)
    sorted_bins = recipe["sorted_bins"]

    from ._extra_fe_families import _digitize_with_edges  # lazy: break parent<->sibling cycle

    xi = np.asarray(X_test[x_i].to_numpy(), dtype=np.float64)
    xj = np.asarray(X_test[x_j].to_numpy(), dtype=np.float64)
    codes_j = _digitize_with_edges(xj, edges)

    out = np.full(xi.shape[0], np.nan, dtype=np.float64)
    finite_mask = np.isfinite(xi)
    for b, sorted_vals in enumerate(sorted_bins):
        if sorted_vals.size == 0:
            continue
        row_mask = finite_mask & (codes_j == b)
        if not row_mask.any():
            continue
        ranks = np.searchsorted(sorted_vals, xi[row_mask], side="right")
        out[row_mask] = ranks / sorted_vals.size
    return np.clip(out, 0.0, 1.0)


def build_conditional_quantile_rank_recipe(
    *, name: str, x_i: str, x_j: str, edges: np.ndarray, sorted_bins: list,
) -> "EngineeredRecipe":
    """Frozen recipe for one NUM x NUM conditional quantile-rank column. Stores the ``x_j``
    quantile edges + the per-bin sorted ``x_i`` reference values; replay reads only X (no y), so
    transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="conditional_quantile_rank",
        src_names=(str(x_i), str(x_j)),
        extra={
            "x_i": str(x_i),
            "x_j": str(x_j),
            "edges": np.asarray(edges, dtype=np.float64).copy(),
            "sorted_bins": [np.asarray(b, dtype=np.float64).copy() for b in sorted_bins],
        },
    )


def _apply_conditional_quantile_rank_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    x_i = str(recipe.extra["x_i"])
    x_j = str(recipe.extra["x_j"])
    if isinstance(X, pd.DataFrame):
        X_view = X
    else:
        try:
            import polars as _pl

            if isinstance(X, _pl.DataFrame):
                X_view = pd.DataFrame({x_i: X[x_i].to_numpy(), x_j: X[x_j].to_numpy()})
            else:
                X_view = pd.DataFrame({x_i: np.asarray(X[x_i]), x_j: np.asarray(X[x_j])})
        except ImportError:
            X_view = pd.DataFrame({x_i: np.asarray(X[x_i]), x_j: np.asarray(X[x_j])})
    return apply_conditional_quantile_rank(X_view, recipe.extra)


def hybrid_conditional_quantile_rank_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    top_k: int = 10,
    max_pair_cols: int = 6,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """End-to-end conditional-quantile-rank FE: bound the column set by top raw-MI (cardinality
    bound on the O(p^2) pair pool), materialise the quantile-rank columns, MI-gate against the
    raw-baseline floor, keep top ``top_k``.

    On a homoscedastic, non-skewed conditional distribution, quantile-rank is a near-monotone
    reparametrization of the raw column / z-score and clears no uplift over either -- it is
    self-limiting the same way the conditional-dispersion family is. Only a genuinely skewed
    conditional distribution (where quantile-rank resolves "how extreme" more accurately than a
    z-score) survives the MI gate. Returns ``(X_aug, appended, recipes, scores)``. ``y`` is
    consumed only by the column-ranking + MI gate; recipes carry no ``y`` -> leak-safe replay.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_conditional_quantile_rank_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [c for c in num_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) < 2:
        return X.copy(), [], [], pd.DataFrame()

    if y is not None:
        from ._extra_fe_families import _top_mi_num_cols  # lazy: break parent<->sibling cycle

        num_cols = _top_mi_num_cols(X, num_cols, y, max_pair_cols)
    else:
        num_cols = list(num_cols)[: int(max_pair_cols)]
    if len(num_cols) < 2:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_conditional_quantile_rank_features(X, num_cols, n_bins=n_bins)
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        _gate_top_k = int(mi_gate_top_k) if mi_gate_top_k else int(top_k)
        winners = local_mi_gate(enc_df, y, raw_X=X, top_k=_gate_top_k, reject_sink=reject_sink)
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_conditional_quantile_rank_recipe(
            name=name,
            x_i=raw_recipes[name]["x_i"],
            x_j=raw_recipes[name]["x_j"],
            edges=raw_recipes[name]["edges"],
            sorted_bins=raw_recipes[name]["sorted_bins"],
        )
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df
