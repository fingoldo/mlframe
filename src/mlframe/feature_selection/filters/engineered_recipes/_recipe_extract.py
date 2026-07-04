"""Column extraction + int-coercion helpers shared by the recipe replay paths.

``_extract_column`` pulls one column from a pandas / polars / structured-ndarray
``X`` as a 1-D ndarray without a full-frame copy; ``_coerce_to_int_with_nan_handling``
maps test-time values to int64 for factorize / target-encoding lookups, honouring
the recipe's ``unknown_strategy``. Both are pure (numpy + optional pandas/polars),
so this submodule imports nothing from a sibling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]


def _extract_column(X: Any, name: str) -> np.ndarray:
    """Pull a single column from X by name as a 1-D ndarray, no full-frame copy. Supports pandas / polars DataFrame and numpy structured arrays."""
    if pd is not None and isinstance(X, pd.DataFrame):
        # ``.values`` is zero-copy for numeric dtypes; categorical/object materialises only the single column.
        return X[name].to_numpy() if hasattr(X[name], "to_numpy") else X[name].values
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[name].to_numpy()
    if isinstance(X, np.ndarray):
        if X.dtype.names is not None:
            return X[name]
        raise KeyError(
            f"Cannot resolve column '{name}' on a plain 2-D ndarray. "
            "Pass a pandas / polars frame or a structured array."
        )
    raise TypeError(f"Unsupported X type for engineered-recipe replay: {type(X)!r}")


_NAN_CODE_KEY = "__MLFRAME_CAT_NAN__"


def build_category_code_map(values: Any, block_has_nan: bool | None = None) -> dict:
    """Build the fit-time ``raw_value -> integer_code`` mapping for a categorical
    source column so a ``factorize`` / ``target_encoding`` recipe can reproduce
    the SAME codes at transform time.

    Reproduces ``categorize_dataset`` / ``_multi_col_factorize_native`` EXACTLY,
    including the NaN +1 shift:

    * base codes: Categorical -> ``.cat.codes`` (category-dictionary order);
      object / string / bool -> ``pd.factorize(use_na_sentinel=True)`` (first-
      appearance order on the TRAINING data). NaN gets the sentinel ``-1``.
    * NaN shift: when the training column had ANY NaN, ``categorize_dataset``
      shifts every code by ``+1`` so the ``-1`` NaN sentinel becomes ``0`` and
      real categories become ``1..K``. The map mirrors this: with NaN present,
      real-category codes are ``base + 1`` and the NaN bin maps to ``0`` (stored
      under ``_NAN_CODE_KEY`` so the replay can route NaN cells there); without
      NaN, codes are the un-shifted base (``0..K-1``).

    ``block_has_nan``: ``categorize_dataset`` factorises ALL categorical columns
    as ONE 2-D block and applies the ``+1`` shift to the WHOLE block when ANY
    column in that block has a NaN (``_discretization_dataset.py`` :
    ``_has_nan = (new_vals < 0).any()`` over the full block ->
    ``new_vals = new_vals + 1``). So a NaN-FREE categorical paired with a
    NaN-bearing categorical ALSO gets its codes shifted ``+1`` at fit time. A
    per-column ``has_nan`` would say "NaN-free -> unshifted" and produce
    off-by-one codes at transform -- a silent train/serve skew for the
    ``factorize(cat_nanfree__cat_withnan)`` recipe. Pass ``block_has_nan=True``
    (computed once over the full categorical block at the stamping site) to apply
    the shift even for a NaN-free column: its real categories become ``base + 1``,
    but NO ``_NAN_CODE_KEY`` is added because that column owns no NaN cell and
    never receives code ``0``. ``block_has_nan=False`` forces the unshifted base.
    ``None`` (default) keeps the legacy per-column behaviour, which is correct
    only for a single-column block.

    Returns a plain ``{str(value): int(code)}`` dict (JSON / pickle friendly),
    optionally with the ``_NAN_CODE_KEY`` entry. Returns ``{}`` for numeric
    columns (already integer-coded; no map needed)."""
    if pd is None:
        return {}
    ser = values if isinstance(values, pd.Series) else pd.Series(values)
    if isinstance(ser.dtype, pd.CategoricalDtype):
        cats = ser.cat.categories
        base = {str(c): int(i) for i, c in enumerate(cats)}
        col_has_nan = bool(ser.isna().any())
    elif ser.dtype.kind in ("O", "U", "S", "b") or str(ser.dtype) in ("string", "boolean"):
        codes, uniques = pd.factorize(ser, use_na_sentinel=True)
        base = {str(u): int(i) for i, u in enumerate(uniques)}
        col_has_nan = bool((np.asarray(codes) < 0).any())
    else:
        return {}
    # Whether the +1 shift applies is a BLOCK property in ``categorize_dataset``
    # (shifts the whole block if ANY categorical column has a NaN), not a
    # per-column one. When the caller knows the block-level answer, honour it;
    # otherwise fall back to this column's own NaN presence (single-column block).
    apply_shift = col_has_nan if block_has_nan is None else bool(block_has_nan)
    if apply_shift:
        # ``categorize_dataset`` shifts -1 (NaN) -> 0 and real categories +1 when any NaN present in the block.
        shifted = {k: v + 1 for k, v in base.items()}
        if col_has_nan:
            # Only THIS column's NaN cells map to 0; a NaN-free column in a shifted block owns no 0-code cell.
            shifted[_NAN_CODE_KEY] = 0
        return shifted
    return base


def _coerce_to_int_with_nan_handling(
    vals: np.ndarray, n_bins: int, recipe_name: str, col_name: str,
    unknown_strategy: str, cat_code_map: dict | None = None,
    bin_edges: np.ndarray | None = None,
) -> np.ndarray:
    """Coerce test-time values to int64 for factorize lookup, handling NaN/non-integer per ``unknown_strategy`` (clip -> max bin, sentinel -> new bin,
    raise -> error). Float non-NaN rounds to NEAREST int (recovers a fit-time integer code that round-tripped through float, e.g. 2.9999 -> 3, not 2); object/categorical via the stored fit-time ``cat_code_map`` (``raw_value -> code``)
    when supplied, else ``astype(int64)``.

    ``bin_edges``: when supplied (``include_numeric`` quantile-binned numeric source), raw values are binned via
    ``np.searchsorted(bin_edges, value, side="right")`` -- the EXACT fit-time convention (``_quantile_bin_with_edges``)
    -- reproducing identical codes with no train/serve skew. Takes precedence over the int-cast / ``cat_code_map``
    paths. Non-finite values resolve via ``unknown_strategy`` (``raise`` -> error; else clip to the top bin).

    ``cat_code_map``: the fit-time ``str(value) -> code`` table built by
    ``build_category_code_map``. When the source column is categorical / string,
    raw values are looked up here so transform reproduces the SAME integer codes
    the fit-time discretiser assigned. Without it, string sources silently
    coerced to all-zero codes (every cell -> the same wrong lookup cell), which
    destroyed cat-interaction features at serving time. Unseen values resolve via
    ``unknown_strategy`` (sentinel bin ``n_bins - 1`` for clip/sentinel)."""
    # Numeric source binned at fit via stored quantile edges (include_numeric): reproduce the SAME codes.
    if bin_edges is not None and len(bin_edges) > 0:
        v = np.asarray(vals, dtype=np.float64)
        nan_mask = ~np.isfinite(v)
        if nan_mask.any() and unknown_strategy == "raise":
            raise ValueError(
                f"Recipe '{recipe_name}': numeric column '{col_name}' has "
                f"{int(nan_mask.sum())} non-finite value(s) at transform time. Set "
                f"unknown_strategy='clip' or 'sentinel' to handle silently."
            )
        codes = np.searchsorted(np.asarray(bin_edges, dtype=np.float64), v, side="right").astype(np.int64)
        if nan_mask.any():
            codes[nan_mask] = n_bins - 1  # clip non-finite to the top bin
        return np.clip(codes, 0, n_bins - 1)
    # Categorical / string source: replay via the stored fit-time code map so
    # transform reproduces the EXACT codes (category-order for Categorical,
    # first-appearance order for object via pd.factorize). Numeric columns skip
    # this branch (cat_code_map is empty / None for them).
    if cat_code_map and not (
        np.issubdtype(vals.dtype, np.floating) or np.issubdtype(vals.dtype, np.integer)
    ):
        # NaN cells route to the dedicated NaN code (``categorize_dataset`` shifts NaN -> 0 when the
        # training column had any NaN); absent that key (training column was NaN-free) a transform-time
        # NaN is genuinely unseen and resolves via ``unknown_strategy``.
        # bench-attempt-rejected (2026-07-05): a pandas ``.map``-vectorised replacement of this loop measured only
        # 1.11x @300k (bit-identical, 1200-case A/B) -- str(_v) + dict.get on OBJECT arrays stays Python-per-element
        # even through pandas, so vectorising it buys almost nothing while complicating this correctness-critical
        # (serving-time recipe replay) path. Kept as the clear loop.
        _nan_code = cat_code_map.get(_NAN_CODE_KEY)
        out = np.empty(len(vals), dtype=np.int64)
        for _i, _v in enumerate(vals):
            if _v is None or (isinstance(_v, float) and _v != _v):
                if _nan_code is not None:
                    out[_i] = _nan_code
                    continue
                _code = None  # NaN unseen at fit -> unknown_strategy below
            else:
                _code = cat_code_map.get(str(_v))
            if _code is None:
                if unknown_strategy == "raise":
                    raise ValueError(
                        f"Recipe '{recipe_name}': column '{col_name}' has value "
                        f"{_v!r} unseen at fit time. Set unknown_strategy='clip' "
                        f"or 'sentinel' to handle unseen categories silently."
                    )
                out[_i] = n_bins - 1  # sentinel bin (clip/sentinel both resolve via lookup)
            else:
                out[_i] = _code
        return out
    if np.issubdtype(vals.dtype, np.floating):
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            if unknown_strategy == "raise":
                n_nan = int(nan_mask.sum())
                raise ValueError(
                    f"Recipe '{recipe_name}': column '{col_name}' has "
                    f"{n_nan} NaN value(s) at transform time. Set "
                    f"unknown_strategy='clip' or 'sentinel' to handle "
                    f"silently."
                )
            # 'clip'/'sentinel' both leave NaN unhandled at the float level; replace with n_bins-1 sentinel so the lookup (which encoded the strategy at
            # fit time) resolves it. The clip path in ``_apply_factorize`` handles the rest.
            vals = vals.copy()
            vals[nan_mask] = n_bins - 1
        # Round to nearest before the int cast: a raw-integer ordinal source can arrive as float at replay (int->float
        # round-trip -- a NaN elsewhere promoted the column, a Parquet reload), and a bare astype(int64) truncates
        # toward zero (a fit-time code 3 that round-trips to 2.9999 would become 2), mismatching the fit assignment.
        return np.rint(vals).astype(np.int64, copy=False)
    if np.issubdtype(vals.dtype, np.integer):
        return vals.astype(np.int64, copy=False)
    # Object / categorical / string -- try int conversion
    try:
        return vals.astype(np.int64, copy=False)
    except (ValueError, TypeError) as e:
        if unknown_strategy == "raise":
            raise ValueError(
                f"Recipe '{recipe_name}': column '{col_name}' has "
                f"non-integer dtype {vals.dtype!r} that cannot be "
                f"coerced. Pass ordinal-encoded ints or set "
                f"unknown_strategy='clip'."
            ) from e
        # Clip-equivalent fallback: all to 0
        return np.zeros(len(vals), dtype=np.int64)
