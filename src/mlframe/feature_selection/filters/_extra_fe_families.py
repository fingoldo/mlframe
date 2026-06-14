"""Layer 104 (2026-06-01): THREE recipe-based FE families filling genuine gaps
in the L21-L103 chain.

FAMILY A -- Rare-category indicator + frequency-band encoding
-------------------------------------------------------------
A category being RARE is itself predictive (a rare merchant id is a fraud
signal; a rare device fingerprint a bot signal). Extends Layer 34 count /
frequency encoding (raw count) with the SHAPE of rarity:

* ``is_rare_{col}``   -- 1.0 when the category's fit-time frequency
  (count / n) is below ``rare_threshold`` (default 1%), else 0.0.
* ``freq_band_{col}`` -- ordinal frequency band of the category:
  ``0=very_rare`` (< rare_threshold), ``1=rare`` (< 4x threshold),
  ``2=common`` (< the 90th-percentile of per-category frequency),
  ``3=dominant`` (>= that). Captures the rare/common/dominant tier without
  the raw count's heavy-tail scale.

The fit-time per-category frequency lookup is the only state; replay maps each
row's category through it (unseen categories -> frequency 0 -> very_rare /
is_rare=1, which is the natural prior: a category never seen in training is, if
anything, rarer than the rarest training category). No ``y`` reference.

FAMILY B -- Cross-feature conditional residual (NUM x NUM)
----------------------------------------------------------
Extends Layer 34's cat-num residual (``x - E[x | cat]``) to the NUM x NUM case:
for an ordered pair ``(x_i, x_j)`` bin ``x_j`` into ``n_bins`` quantile bins and
emit ``x_i - E[x_i | bin(x_j)]`` -- how far ``x_i`` deviates from its conditional
expectation given ``x_j``'s bin. Captures conditional anomalies a marginal
feature cannot express: an income that is high FOR a given age bracket, a
latency that is high FOR a given payload size. The fit-time state is the
``x_j`` bin EDGES + the per-bin mean of ``x_i``; replay digitises ``x_j`` with
the stored edges and subtracts the stored per-bin mean (no ``y``, leak-safe).

FAMILY C -- Quantile-transform / rank-Gaussianisation (RankGauss)
-----------------------------------------------------------------
Map each numeric column to its rank-based Gaussian quantile (the classic Kaggle
"RankGauss"): rank -> normalised rank in (0, 1) -> ``norm.ppf``. Produces an
exactly-Gaussian marginal, which materially helps downstream LINEAR / NN models
that assume Gaussian inputs, and is a monotone-invariant representation.

DPI CONTRACT (CRITICAL -- the Layer 90 lesson): RankGauss is a strictly
MONOTONE transform of the raw column, so by the data-processing inequality it
CANNOT add mutual information about ``y`` -- ``MI(rankgauss(x); y) ==
MI(x; y)`` up to binning noise. Its value is DOWNSTREAM (linear / NN), not in
MI. Family C is therefore NOT MI-gated (an MI floor would be a no-op at best
and wrongly drop a genuinely useful Gaussianisation at worst). Pin a downstream
linear-model lift test, NEVER an MI-gain test. The pool is bounded by ranking
candidate columns by their RAW marginal MI and keeping the top ``top_k``.

The fit-time state is the SORTED unique fit values (+ their normalised-rank
Gaussian targets); replay interpolates each test value's rank against the
stored sorted values (``np.searchsorted``), so an unseen test value maps to the
Gaussian quantile of its interpolated rank -- leak-safe, reads only X.

NOT wired ON by default -- opt-in via ``fe_rare_category_enable`` /
``fe_conditional_residual_enable`` / ``fe_rankgauss_enable``.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    # Family A
    "engineered_name_is_rare",
    "engineered_name_freq_band",
    "generate_rare_category_features",
    "apply_rare_category",
    "build_rare_category_recipe",
    "hybrid_rare_category_fe",
    # Family B
    "engineered_name_conditional_residual",
    "generate_conditional_residual_features",
    "apply_conditional_residual",
    "build_conditional_residual_recipe",
    "hybrid_conditional_residual_fe",
    # Family C
    "engineered_name_rankgauss",
    "generate_rankgauss_features",
    "apply_rankgauss",
    "build_rankgauss_recipe",
    "hybrid_rankgauss_fe",
    # Family D (conditional dispersion / 2nd-moment) -- re-exported from
    # the sibling module ``_extra_fe_families_dispersion`` (kept in its own
    # file under the module-size limit; the public import path stays here).
    "engineered_name_conditional_dispersion",
    "generate_conditional_dispersion_features",
    "apply_conditional_dispersion",
    "build_conditional_dispersion_recipe",
    "_apply_conditional_dispersion_recipe",
    "hybrid_conditional_dispersion_fe",
]


def _column_to_str(col) -> np.ndarray:
    """Canonicalise a (categorical / object / numeric) column to a 1-D str
    ndarray, mapping NaN to the sentinel ``"__nan__"`` so it is a stable lookup
    key (mirrors Layer 33/34's ``_column_to_str`` contract). Integral int/float
    values collapse to the same token (``1`` and ``1.0`` -> ``'1'``) so a
    fit-int / predict-float dtype drift still resolves the per-category entry
    instead of the global fallback."""
    # Delegate to the canonical per-unique factorize-gather implementation in ``_target_encoding_fe`` (identical ``"__nan__"`` sentinel +
    # ``canonical_group_token`` contract). That copy canonicalises each DISTINCT value once and gathers per-row via the factorize codes
    # (Python-level ``canonical_group_token`` runs per-unique instead of per-row -- 8-65x at 10M rows / few-hundred uniques) and carries the
    # bool/0/1 collision gate that falls back to the exact per-row loop on bool-in-object columns. Sharing it retires the duplicate per-row map.
    from ._target_encoding_fe import _column_to_str as _canonical_column_to_str

    return _canonical_column_to_str(pd.Series(col))


# ===========================================================================
# FAMILY A -- rare-category indicator + frequency-band encoding
# ===========================================================================

# Frequency-band cut points (in units of ``rare_threshold``) for the two low
# tiers; the ``common`` / ``dominant`` split is data-driven (90th percentile of
# the per-category frequency distribution) so it adapts to the cardinality.
_RARE_BAND_MULT = 4.0
_FREQ_BANDS = {"very_rare": 0, "rare": 1, "common": 2, "dominant": 3}


def engineered_name_is_rare(col: str) -> str:
    return f"is_rare__{col}"


def engineered_name_freq_band(col: str) -> str:
    return f"freq_band__{col}"


def _freq_band_codes(
    freqs: np.ndarray, rare_threshold: float, dominant_cut: float,
) -> np.ndarray:
    """Vectorised band assignment over a per-row frequency array. Cut points are
    ``[rare_threshold, 4*rare_threshold, dominant_cut]`` -> bands 0..3
    (very_rare / rare / common / dominant) via a single ``np.searchsorted``."""
    cuts = np.array(
        [float(rare_threshold), _RARE_BAND_MULT * float(rare_threshold),
         float(dominant_cut)],
        dtype=np.float64,
    )
    cuts = np.maximum.accumulate(cuts)  # guard monotonicity for searchsorted
    return np.searchsorted(cuts, np.asarray(freqs, dtype=np.float64), side="right").astype(np.float64)


def generate_rare_category_features(
    X: pd.DataFrame,
    cat_cols: Sequence[str],
    *,
    rare_threshold: float = 0.01,
):
    """Per categorical column, emit ``is_rare_{col}`` + ``freq_band_{col}``.

    Returns ``(enc_df, raw_recipes)``. Each ``raw_recipes[name]`` payload stores
    the per-category frequency lookup + the band cut points; replay reads only X
    (no ``y`` reference), so transform() is leakage-free.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"generate_rare_category_features: X must be a pandas DataFrame; "
            f"got {type(X).__name__}"
        )
    if len(X) == 0:
        raise ValueError("generate_rare_category_features: X is empty")
    cat_cols = [c for c in cat_cols if c in X.columns]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not cat_cols:
        return pd.DataFrame(index=X.index), raw_recipes

    n = len(X)
    for col in cat_cols:
        cats = _column_to_str(X[col])
        unique_cats, inverse, counts = np.unique(
            cats, return_inverse=True, return_counts=True,
        )
        freqs = counts.astype(np.float64) / float(n)
        freq_lookup = {
            str(unique_cats[c]): float(freqs[c]) for c in range(unique_cats.shape[0])
        }
        # Data-driven common/dominant split: 90th percentile of the per-category
        # frequency distribution (frequency-weighted is wrong here -- we want the
        # split over the CATEGORIES, so a few dominant categories don't drag it).
        dominant_cut = float(np.quantile(freqs, 0.90)) if freqs.size else 1.0
        # Guard: if all categories share the same frequency the 90th pctile
        # equals the max, so "dominant" never fires; nudge the cut just above so
        # the most-frequent tier is reachable.
        if dominant_cut <= _RARE_BAND_MULT * float(rare_threshold):
            dominant_cut = _RARE_BAND_MULT * float(rare_threshold) + 1e-12

        row_freq = freqs[inverse]
        is_rare = (row_freq < float(rare_threshold)).astype(np.float64)
        band = _freq_band_codes(row_freq, float(rare_threshold), dominant_cut)

        common = {
            "src_col": col,
            "freq_lookup": freq_lookup,
            "rare_threshold": float(rare_threshold),
            "dominant_cut": float(dominant_cut),
        }
        is_rare_name = engineered_name_is_rare(col)
        encoded[is_rare_name] = is_rare
        raw_recipes[is_rare_name] = {**common, "kind": "is_rare"}

        band_name = engineered_name_freq_band(col)
        encoded[band_name] = band
        raw_recipes[band_name] = {**common, "kind": "freq_band"}

    return pd.DataFrame(encoded, index=X.index), raw_recipes


def apply_rare_category(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay one rare-category column. ``kind`` selects is_rare / freq_band.
    Unseen categories map to frequency 0 (=> very_rare / is_rare=1), the natural
    prior for a category absent from training. Reads only X_test."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_rare_category: X_test must be a DataFrame; got "
            f"{type(X_test).__name__}"
        )
    src_col = recipe["src_col"]
    if src_col not in X_test.columns:
        raise KeyError(
            f"apply_rare_category: missing source column {src_col!r} from X_test"
        )
    kind = recipe.get("kind", "is_rare")
    freq_lookup = dict(recipe["freq_lookup"])
    rare_threshold = float(recipe["rare_threshold"])
    dominant_cut = float(recipe["dominant_cut"])
    cats = _column_to_str(X_test[src_col])
    # Resolve once per UNIQUE key, broadcast back (hot-path friendly).
    uniq, inverse = np.unique(cats, return_inverse=True)
    uniq_freq = np.array(
        [freq_lookup.get(str(k), 0.0) for k in uniq], dtype=np.float64
    )
    row_freq = uniq_freq[inverse]
    if kind == "is_rare":
        return (row_freq < rare_threshold).astype(np.float64)
    if kind == "freq_band":
        return _freq_band_codes(row_freq, rare_threshold, dominant_cut)
    raise ValueError(f"apply_rare_category: unknown kind {kind!r}")


def build_rare_category_recipe(
    *, name: str, src_col: str, kind: str, freq_lookup: dict,
    rare_threshold: float, dominant_cut: float,
):
    """Frozen recipe for one rare-category column (``is_rare`` or
    ``freq_band``). Stores the per-category frequency lookup + band cut points;
    replay reads only X (no y), so transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    if kind not in ("is_rare", "freq_band"):
        raise ValueError(
            f"rare_category kind must be 'is_rare' or 'freq_band'; got {kind!r}"
        )
    return EngineeredRecipe(
        name=name,
        kind="rare_category",
        src_names=(str(src_col),),
        extra={
            "src_col": str(src_col),
            "rare_kind": str(kind),
            "freq_lookup": {str(k): float(v) for k, v in freq_lookup.items()},
            "rare_threshold": float(rare_threshold),
            "dominant_cut": float(dominant_cut),
        },
    )


def _apply_rare_category_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    src_col = str(recipe.extra["src_col"])
    if isinstance(X, pd.DataFrame):
        X_view = X
    else:
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                X_view = pd.DataFrame({src_col: X[src_col].to_numpy()})
            else:
                raise TypeError
        except (ImportError, TypeError):
            if isinstance(X, np.ndarray) and X.dtype.names is not None:
                X_view = pd.DataFrame({src_col: X[src_col]})
            else:
                raise TypeError(
                    f"rare_category recipe '{recipe.name}': cannot extract "
                    f"{src_col!r} from X of type {type(X).__name__}."
                )
    return apply_rare_category(
        X_view,
        {
            "src_col": src_col,
            "kind": str(recipe.extra.get("rare_kind", "is_rare")),
            "freq_lookup": dict(recipe.extra.get("freq_lookup", {})),
            "rare_threshold": float(recipe.extra.get("rare_threshold", 0.01)),
            "dominant_cut": float(recipe.extra.get("dominant_cut", 1.0)),
        },
    )


def _auto_detect_cat_cols(X: pd.DataFrame, max_cols: int = 8) -> list[str]:
    """Reuse Layer 33's categorical auto-detector when available; fall back to a
    low-cardinality scan."""
    try:
        from ._target_encoding_fe import auto_detect_te_cols
        return list(auto_detect_te_cols(X, min_card=2, max_card=10000))[:max_cols]
    except Exception:
        out: list[str] = []
        n = len(X)
        for c in X.columns:
            col = X[c]
            if pd.api.types.is_float_dtype(col):
                continue
            nun = int(col.nunique(dropna=True))
            if 2 <= nun <= max(2, n // 2):
                out.append(str(c))
        return out[:max_cols]


def hybrid_rare_category_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    rare_threshold: float = 0.01,
    top_k: int = 10,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """End-to-end rare-category FE: materialise is_rare / freq_band columns,
    MI-gate against the raw-baseline noise floor (Layer 91), keep top ``top_k``.

    Returns ``(X_aug, appended, recipes, scores)``. ``y`` is consumed only by
    the MI gate; recipes carry no ``y`` reference -> leak-safe replay.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"hybrid_rare_category_fe: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if cat_cols is None or len(cat_cols) == 0:
        cat_cols = _auto_detect_cat_cols(X)
    else:
        cat_cols = [c for c in cat_cols if c in X.columns]
    if not cat_cols:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_rare_category_features(
        X, cat_cols, rare_threshold=rare_threshold,
    )
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate
        winners = local_mi_gate(
            enc_df, y, raw_X=X,
            top_k=int(mi_gate_top_k) if mi_gate_top_k else int(top_k),
            reject_sink=reject_sink,
        )
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_rare_category_recipe(
            name=name,
            src_col=raw_recipes[name]["src_col"],
            kind=raw_recipes[name]["kind"],
            freq_lookup=raw_recipes[name]["freq_lookup"],
            rare_threshold=raw_recipes[name]["rare_threshold"],
            dominant_cut=raw_recipes[name]["dominant_cut"],
        )
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df


# ===========================================================================
# FAMILY B -- cross-feature conditional residual (NUM x NUM)
# ===========================================================================


def engineered_name_conditional_residual(x_i: str, x_j: str) -> str:
    """``x_i - E[x_i | bin(x_j)]`` reads as "deviation of x_i from its
    x_j-conditional expectation"."""
    return f"{x_i}__cond_resid_by__{x_j}"


def _quantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Fit-time quantile bin EDGES for the conditioning column (finite values
    only). Deduped; degenerate columns collapse to a single bin."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    q = np.quantile(finite, np.linspace(0.0, 1.0, int(n_bins) + 1))
    q = np.unique(q)
    if q.size < 2:
        lo = float(finite.min())
        q = np.array([lo, lo + 1.0], dtype=np.float64)
    return q.astype(np.float64)


def _digitize_with_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each value to a bin in ``[0, len(edges)-2]`` using the interior
    edges (``edges[1:-1]``). NaN -> bin 0 (matches the fit-time fill)."""
    x = np.asarray(x, dtype=np.float64)
    codes = np.searchsorted(edges[1:-1], x, side="right")
    codes = np.clip(codes, 0, max(0, edges.size - 2))
    codes[~np.isfinite(x)] = 0
    return codes.astype(np.int64)


def generate_conditional_residual_features(
    X: pd.DataFrame,
    num_cols: Sequence[str],
    *,
    n_bins: int = 10,
):
    """For every ordered pair ``(x_i, x_j)`` from ``num_cols`` emit
    ``x_i - E[x_i | bin(x_j)]``.

    Returns ``(enc_df, raw_recipes)``. Each payload stores the ``x_j`` quantile
    edges + the per-bin mean of ``x_i``; replay digitises ``x_j`` with the
    stored edges and subtracts the stored per-bin mean (no ``y``, leak-safe).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"generate_conditional_residual_features: X must be a pandas "
            f"DataFrame; got {type(X).__name__}"
        )
    if len(X) == 0:
        raise ValueError("generate_conditional_residual_features: X is empty")
    num_cols = [
        c for c in num_cols
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if len(num_cols) < 2:
        return pd.DataFrame(index=X.index), raw_recipes

    col_vals = {c: np.asarray(X[c].to_numpy(), dtype=np.float64) for c in num_cols}
    # Per-x_i invariants (finiteness mask, global mean, masked values) depend only on x_i, so hoist them out of the inner x_j loop -- recomputing them per (x_i, x_j) pair walked the full 10M array C-1 extra times each.
    finite_of: dict[str, np.ndarray] = {}
    global_mean_of: dict[str, float] = {}
    for x_i in num_cols:
        xi = col_vals[x_i]
        fin = np.isfinite(xi)
        finite_of[x_i] = fin
        global_mean_of[x_i] = float(xi[fin].mean()) if fin.any() else 0.0
    for x_j in num_cols:
        xj = col_vals[x_j]
        edges = _quantile_edges(xj, n_bins)
        codes_j = _digitize_with_edges(xj, edges)
        n_bins_eff = edges.size - 1
        for x_i in num_cols:
            if x_i == x_j:
                continue
            xi = col_vals[x_i]
            finite_i = finite_of[x_i]
            global_mean = global_mean_of[x_i]
            codes_jf = codes_j[finite_i]
            # np.bincount accumulates per bin in element order exactly as np.add.at does -- bit-identical sum/count, but a single C pass instead of the unbuffered scatter.
            bin_sum = np.bincount(codes_jf, weights=xi[finite_i], minlength=n_bins_eff).astype(np.float64)
            bin_cnt = np.bincount(codes_jf, minlength=n_bins_eff).astype(np.float64)
            bin_mean = np.where(
                bin_cnt > 0.0, bin_sum / np.maximum(bin_cnt, 1.0), global_mean,
            )
            per_row_mean = bin_mean[codes_j]
            residual = np.where(finite_i, xi - per_row_mean, 0.0).astype(np.float64)
            name = engineered_name_conditional_residual(x_i, x_j)
            encoded[name] = residual
            raw_recipes[name] = {
                "x_i": x_i,
                "x_j": x_j,
                "edges": edges,
                "bin_mean": bin_mean,
                "global_mean": global_mean,
            }

    return pd.DataFrame(encoded, index=X.index), raw_recipes


def apply_conditional_residual(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay ``x_i - E[x_i | bin(x_j)]`` from the stored x_j edges + per-bin
    mean. NaN ``x_i`` rows emit 0.0 (no deviation information). Reads only X."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_conditional_residual: X_test must be a DataFrame; got "
            f"{type(X_test).__name__}"
        )
    x_i = recipe["x_i"]
    x_j = recipe["x_j"]
    if x_i not in X_test.columns or x_j not in X_test.columns:
        raise KeyError(
            f"apply_conditional_residual: missing column(s) {x_i!r}/{x_j!r} "
            f"from X_test"
        )
    edges = np.asarray(recipe["edges"], dtype=np.float64)
    bin_mean = np.asarray(recipe["bin_mean"], dtype=np.float64)
    xi = np.asarray(X_test[x_i].to_numpy(), dtype=np.float64)
    xj = np.asarray(X_test[x_j].to_numpy(), dtype=np.float64)
    codes_j = _digitize_with_edges(xj, edges)
    codes_j = np.clip(codes_j, 0, bin_mean.size - 1)
    per_row_mean = bin_mean[codes_j]
    finite_i = np.isfinite(xi)
    return np.where(finite_i, xi - per_row_mean, 0.0).astype(np.float64)


def build_conditional_residual_recipe(
    *, name: str, x_i: str, x_j: str, edges: np.ndarray, bin_mean: np.ndarray,
    global_mean: float,
):
    """Frozen recipe for one NUM x NUM conditional-residual column. Stores the
    ``x_j`` quantile edges + the per-bin mean of ``x_i``; replay reads only X
    (no y), so transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="conditional_residual",
        src_names=(str(x_i), str(x_j)),
        extra={
            "x_i": str(x_i),
            "x_j": str(x_j),
            "edges": np.asarray(edges, dtype=np.float64).copy(),
            "bin_mean": np.asarray(bin_mean, dtype=np.float64).copy(),
            "global_mean": float(global_mean),
        },
    )


def _apply_conditional_residual_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    x_i = str(recipe.extra["x_i"])
    x_j = str(recipe.extra["x_j"])
    if isinstance(X, pd.DataFrame):
        X_view = X
    else:
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                X_view = pd.DataFrame(
                    {x_i: X[x_i].to_numpy(), x_j: X[x_j].to_numpy()}
                )
            else:
                raise TypeError
        except (ImportError, TypeError):
            if isinstance(X, np.ndarray) and X.dtype.names is not None:
                X_view = pd.DataFrame({x_i: X[x_i], x_j: X[x_j]})
            else:
                raise TypeError(
                    f"conditional_residual recipe '{recipe.name}': cannot "
                    f"extract {x_i!r}/{x_j!r} from X of type {type(X).__name__}."
                )
    return apply_conditional_residual(
        X_view,
        {
            "x_i": x_i,
            "x_j": x_j,
            "edges": np.asarray(recipe.extra["edges"], dtype=np.float64),
            "bin_mean": np.asarray(recipe.extra["bin_mean"], dtype=np.float64),
            "global_mean": float(recipe.extra.get("global_mean", 0.0)),
        },
    )


def _top_mi_num_cols(
    X: pd.DataFrame, num_cols: Sequence[str], y: np.ndarray, max_cols: int,
) -> list[str]:
    """Rank numeric columns by RAW marginal ``MI(col; y)`` and keep the top
    ``max_cols`` -- the cardinality bound that keeps the O(p^2) conditional-
    residual pair pool tractable."""
    num_cols = [c for c in num_cols if c in X.columns]
    if len(num_cols) <= max_cols:
        return list(num_cols)
    from ._orthogonal_univariate_fe import _mi_classif_batch
    from ._unified_fe_gate import _coerce_y_classes

    y_bin = _coerce_y_classes(y)
    arr = X[num_cols].to_numpy(dtype=np.float64)
    mi = np.asarray(_mi_classif_batch(arr, y_bin, nbins=10), dtype=np.float64)
    order = np.argsort(-mi)
    return [num_cols[i] for i in order[: int(max_cols)]]


def hybrid_conditional_residual_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    top_k: int = 10,
    max_pair_cols: int = 6,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    reject_sink: Optional[Callable[..., None]] = None,
):
    """End-to-end conditional-residual FE: bound the column set by top raw-MI
    (cardinality bound on the O(p^2) pair pool), materialise residuals, MI-gate
    against the raw-baseline floor (Layer 91), keep top ``top_k``.

    Returns ``(X_aug, appended, recipes, scores)``. ``y`` is consumed only by
    the column-ranking + MI gate; recipes carry no ``y`` reference -> leak-safe.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"hybrid_conditional_residual_fe: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if num_cols is None or len(num_cols) == 0:
        num_cols = [
            c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
        ]
    else:
        num_cols = [
            c for c in num_cols
            if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
        ]
    if len(num_cols) < 2:
        return X.copy(), [], [], pd.DataFrame()

    if y is not None:
        num_cols = _top_mi_num_cols(X, num_cols, y, max_pair_cols)
    else:
        num_cols = list(num_cols)[: int(max_pair_cols)]
    if len(num_cols) < 2:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, raw_recipes = generate_conditional_residual_features(
        X, num_cols, n_bins=n_bins,
    )
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate
        winners = local_mi_gate(
            enc_df, y, raw_X=X,
            top_k=int(mi_gate_top_k) if mi_gate_top_k else int(top_k),
            reject_sink=reject_sink,
        )
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_conditional_residual_recipe(name=name, **raw_recipes[name])
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df


# ===========================================================================
# FAMILY C -- RankGauss (quantile / rank-Gaussianisation)
# ===========================================================================
#
# DPI CONTRACT: RankGauss is a strictly MONOTONE map of the raw column, so
# MI(rankgauss(x); y) == MI(x; y) up to binning noise (data-processing
# inequality). NOT MI-gated -- the value is DOWNSTREAM (linear / NN), pinned by
# a linear-model lift test, never an MI-gain test.
#
# bench-rejected (2026-06-10): an ISOTONIC (monotone-constrained, free-shape)
# Family-D sibling to RankGauss was proposed (backlog #14) and benchmarked. The
# decisive held-out Ridge R^2 subsumption test (y=sigmoid(3x)+noise, the monotone
# link, 24 seeds, feature set [x, T(x)]) shows isotonic LOSES to the SHIPPED
# 5-inner-knot (8-basis) quantile cubic B-spline in ALL 6 (n,noise) cells
# [n in {500,1500,4000} x noise in {0.15,0.35}], by -0.0014 .. -0.0110 R^2 (0/6
# isotonic wins). The "monotone prior cuts small-n variance" claim is FALSIFIED
# vs the production spline -- the spline wins by the LARGEST margin at the
# smallest-n/highest-noise cell. Isotonic only beats a 16-knot over-parameterised
# spline that mlframe does NOT ship (and explicitly chose against -- see the
# supervised-knots bench-reject in _orth_extra_basis_fe._fit_spline_for_col). The
# cubic spline's own regularisation already subsumes the monotone prior, exactly
# as it subsumes RankGauss. Complementarity control passes (isotonic loses on
# non-monotone y=x^2); noise control passes (admits 0). So NO Family-D isotonic
# kind / fe_isotonic_enable was added. Don't re-attempt as a default-on feature.
# Numbers: D:/Temp/isotonic_results.md.


def engineered_name_rankgauss(col: str) -> str:
    return f"rankgauss__{col}"


def _rank_to_gauss(ranks: np.ndarray, n: int) -> np.ndarray:
    """Map 0-based ranks in ``[0, n-1]`` to Gaussian quantiles via
    ``norm.ppf((rank + 0.5) / n)`` (the (r + 0.5)/n plotting position keeps the
    transform finite at the extremes -- no +/-inf at rank 0 / n-1)."""
    from scipy.special import ndtri
    u = (np.asarray(ranks, dtype=np.float64) + 0.5) / float(max(n, 1))
    u = np.clip(u, 1e-6, 1.0 - 1e-6)
    # ndtri == norm.ppf for the standard normal (bit-identical), ~2.4x faster -- skips the rv_continuous broadcast/validation wrapper.
    return ndtri(u).astype(np.float64)


def _avg_tie_rank(fit_sorted: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Average (mid) rank of each ``vals`` element among ``fit_sorted`` == ``(lo + hi - 1)/2``.

    ``lo``/``hi`` are the left/right ``searchsorted`` insertion points; they differ only where ``vals`` exactly equals
    a fit value (a tie). For continuous data (the common rank-Gauss case) there are no exact ties, so ``hi == lo`` and
    the result collapses to ``lo - 0.5`` -- skipping the entire second ``searchsorted`` sweep (~half the cost at 10M).
    A cheap ``fit_sorted[lo] == vals`` probe detects whether any tie exists; the exact two-pass path runs only then,
    keeping the output bit-identical on tied / discrete inputs.
    """
    lo = np.searchsorted(fit_sorted, vals, side="left")
    n = fit_sorted.size
    in_range = lo < n
    if in_range.any():
        idx = np.nonzero(in_range)[0]
        if (fit_sorted[lo[idx]] == vals[idx]).any():
            hi = np.searchsorted(fit_sorted, vals, side="right")
            return (lo + hi - 1) / 2.0
    return lo - 0.5


def generate_rankgauss_features(
    X: pd.DataFrame,
    num_cols: Sequence[str],
):
    """Map each numeric column to its rank-based Gaussian quantile (RankGauss).

    Returns ``(enc_df, raw_recipes)``. Each payload stores the SORTED unique fit
    values + the count ``n``; replay interpolates each test value's rank against
    the stored sorted values (``np.searchsorted``) and maps it to a Gaussian
    quantile -- leak-safe, reads only X.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"generate_rankgauss_features: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if len(X) == 0:
        raise ValueError("generate_rankgauss_features: X is empty")
    num_cols = [
        c for c in num_cols
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    encoded: dict[str, np.ndarray] = {}
    raw_recipes: dict[str, dict] = {}
    if not num_cols:
        return pd.DataFrame(index=X.index), raw_recipes

    for col in num_cols:
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite = np.isfinite(x)
        fit_sorted = np.sort(x[finite]) if finite.any() else np.array([0.0])
        n_fit = int(fit_sorted.size)
        # Fit-time output: rank each value among the finite sorted fit values
        # (average rank for ties via searchsorted left/right midpoint), then
        # map to Gaussian. NaN rows -> 0.0 (the Gaussian centre).
        out = np.zeros_like(x)
        if n_fit > 0 and finite.any():
            avg_rank = _avg_tie_rank(fit_sorted, x[finite])
            out[finite] = _rank_to_gauss(avg_rank, n_fit)
        name = engineered_name_rankgauss(col)
        encoded[name] = out
        raw_recipes[name] = {
            "src_col": col,
            "fit_sorted": fit_sorted,
            "n_fit": n_fit,
        }

    return pd.DataFrame(encoded, index=X.index), raw_recipes


def apply_rankgauss(X_test: pd.DataFrame, recipe: dict) -> np.ndarray:
    """Replay RankGauss: interpolate each test value's rank against the stored
    sorted fit values, then map to a Gaussian quantile. Unseen / out-of-range
    test values rank at the appropriate extreme (the searchsorted position).
    NaN rows -> 0.0. Reads only X."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            f"apply_rankgauss: X_test must be a DataFrame; got "
            f"{type(X_test).__name__}"
        )
    src_col = recipe["src_col"]
    if src_col not in X_test.columns:
        raise KeyError(
            f"apply_rankgauss: missing source column {src_col!r} from X_test"
        )
    fit_sorted = np.asarray(recipe["fit_sorted"], dtype=np.float64)
    n_fit = int(recipe["n_fit"])
    x = np.asarray(X_test[src_col].to_numpy(), dtype=np.float64)
    out = np.zeros_like(x)
    finite = np.isfinite(x)
    if n_fit > 0 and finite.any():
        avg_rank = _avg_tie_rank(fit_sorted, x[finite])
        # Clip the averaged rank into [0, n_fit-1] so a test value below the
        # smallest / above the largest fit value lands at the extreme rank.
        avg_rank = np.clip(avg_rank, 0.0, float(max(n_fit - 1, 0)))
        out[finite] = _rank_to_gauss(avg_rank, n_fit)
    return out


def build_rankgauss_recipe(
    *, name: str, src_col: str, fit_sorted: np.ndarray, n_fit: int,
):
    """Frozen recipe for one RankGauss column. Stores the sorted unique fit
    values + count; replay interpolates each test value's rank against them and
    maps to a Gaussian quantile. Reads only X (no y), so transform() is
    leakage-free. (RankGauss is monotone -> MI-invariant by the DPI; the value
    is downstream for linear / NN models, never an MI gain.)"""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="rankgauss",
        src_names=(str(src_col),),
        extra={
            "src_col": str(src_col),
            "fit_sorted": np.asarray(fit_sorted, dtype=np.float64).copy(),
            "n_fit": int(n_fit),
        },
    )


def _apply_rankgauss_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    src_col = str(recipe.extra["src_col"])
    if isinstance(X, pd.DataFrame):
        X_view = X
    else:
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                X_view = pd.DataFrame({src_col: X[src_col].to_numpy()})
            else:
                raise TypeError
        except (ImportError, TypeError):
            if isinstance(X, np.ndarray) and X.dtype.names is not None:
                X_view = pd.DataFrame({src_col: X[src_col]})
            else:
                raise TypeError(
                    f"rankgauss recipe '{recipe.name}': cannot extract "
                    f"{src_col!r} from X of type {type(X).__name__}."
                )
    return apply_rankgauss(
        X_view,
        {
            "src_col": src_col,
            "fit_sorted": np.asarray(recipe.extra["fit_sorted"], dtype=np.float64),
            "n_fit": int(recipe.extra.get("n_fit", 0)),
        },
    )


def hybrid_rankgauss_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    top_k: int = 10,
):
    """End-to-end RankGauss FE. Bounds the pool by ranking candidate columns on
    their RAW marginal ``MI(col; y)`` and keeping the top ``top_k`` (NOT an
    MI-gain gate on the engineered column -- RankGauss is monotone, so by the
    DPI it cannot add MI; the value is downstream for linear / NN models).

    Returns ``(X_aug, appended, recipes, scores)``. ``y`` is consumed only to
    RANK which raw columns are worth Gaussianising; recipes carry no ``y``
    reference -> leak-safe replay.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"hybrid_rankgauss_fe: X must be a pandas DataFrame; got "
            f"{type(X).__name__}"
        )
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [
            c for c in num_cols
            if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
        ]
    if not num_cols:
        return X.copy(), [], [], pd.DataFrame()

    # Pool bound by raw marginal MI (DPI: don't gate on engineered MI gain).
    if y is not None and len(num_cols) > int(top_k):
        num_cols = _top_mi_num_cols(X, num_cols, y, int(top_k))
    else:
        num_cols = list(num_cols)[: int(top_k)]

    enc_df, raw_recipes = generate_rankgauss_features(X, num_cols)
    if enc_df.empty:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [
        build_rankgauss_recipe(name=name, **raw_recipes[name])
        for name in winners
    ]
    return X_aug, winners, recipes, enc_df


# ===========================================================================
# FAMILY D -- cross-feature conditional DISPERSION / 2nd-moment (NUM x NUM)
# ===========================================================================
# Family D extends Family B from conditional LOCATION to conditional SCALE
# (volatility / dispersion regimes). It lives in the sibling module
# ``_extra_fe_families_dispersion`` to keep this file under the module-size
# limit; the public names are re-exported here so the established import path
# ``from .._extra_fe_families import _apply_conditional_dispersion_recipe`` (used
# by the recipe dispatcher, mirroring Families A-C) stays stable. The import is
# at the bottom AFTER the Family-B helpers Family D reuses (``_quantile_edges`` /
# ``_digitize_with_edges`` / ``_top_mi_num_cols``) are defined, so the
# parent->sibling->parent cycle resolves cleanly.
from ._extra_fe_families_dispersion import (  # noqa: E402
    engineered_name_conditional_dispersion,
    generate_conditional_dispersion_features,
    apply_conditional_dispersion,
    build_conditional_dispersion_recipe,
    _apply_conditional_dispersion_recipe,
    hybrid_conditional_dispersion_fe,
)
