"""Polars categorical fixes applied once before model training.

1. Null-fill with ``__MISSING__`` sentinel (avoids CatBoost 1.2.x fused-cpdef TypeError on null-bearing cats).
2. Dict alignment via pl.Enum(union) (avoids XGB silent process kill when val has unseen categories with different physical codes).
3. Utf8 cast to Categorical (so pandas conversion produces ``category`` dtype, not ``object``, for XGB/LGB).

Caller hook: ``apply_polars_categorical_fixes`` accepts an optional
``precomputed_category_union`` mapping. When the suite computes
per-cat-feature unions at frame-load time (BEFORE global outlier detection
filters rows from train), it can thread that mapping in so rare categories
that were OD-filtered out of train are still in the Enum domain - otherwise
val rows carrying those categories silently cast to null.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional

import polars as pl


class PolarsCategoricalFixesResult(NamedTuple):
    """Return shape for ``apply_polars_categorical_fixes`` (H-CORE-20).

    NamedTuple stays iterable + indexable so positional tuple-unpack callers keep working; new code
    can read ``.train_df_pd`` etc. and future field additions won't silently shift positions.
    """
    train_df_polars: Any
    val_df_polars: Any
    test_df_polars: Any
    train_df_pd: Any
    val_df_pd: Any
    test_df_pd: Any
    filtered_train_df: Any
    filtered_val_df: Any

logger = logging.getLogger(__name__)

_DICT_ALIGN_SKIP_CARD = 50_000


def _cast_utf8_cats_to_categorical(
    df_,
    cat_features: list[str],
    enum_domains: Optional[Dict[str, List[str]]] = None,
):
    """Cast Utf8/String cat_features to pl.Enum when an enum domain is supplied for that column;
    fall back to pl.Categorical only when no enum domain is known.

    pl.Categorical participates in polars' process-wide global string cache (the cache grows
    monotonically across calls; pl.disable_string_cache() is a no-op in polars 1.x). Each new
    pl.Categorical Series widens that cache. pl.Enum is per-Series and does not touch the global
    cache. Callers that have a precomputed train+val union should always supply enum_domains so the
    cast lands on Enum, not Categorical (memory rule: reference_polars_global_string_cache).
    """
    if not isinstance(df_, pl.DataFrame):
        return df_
    exprs = []
    enum_domains = enum_domains or {}
    for c in cat_features:
        if c in df_.columns and df_.schema[c] in (pl.Utf8, pl.String):
            _dom = enum_domains.get(c)
            if _dom:
                # Build per-Series Enum, no global string cache impact.
                exprs.append(pl.col(c).cast(pl.Enum(list(_dom)), strict=False))
            else:
                # Last-resort fallback when caller did not thread an enum domain in (e.g. legacy
                # caller of _cast_utf8_cats_to_categorical without precomputed_category_union).
                exprs.append(pl.col(c).cast(pl.Categorical))
    return df_.with_columns(exprs) if exprs else df_


def apply_polars_categorical_fixes(
    *,
    train_df_polars: pl.DataFrame | None,
    val_df_polars: pl.DataFrame | None,
    test_df_polars: pl.DataFrame | None,
    train_df_pd: Any,
    val_df_pd: Any,
    test_df_pd: Any,
    filtered_train_df: Any,
    filtered_val_df: Any,
    cat_features: list[str] | None,
    align_polars_categorical_dicts: bool,
    defer_pandas_conv: bool,
    was_polars_input: bool,
    verbose: bool,
    precomputed_category_union: Optional[Dict[str, List[str]]] = None,
) -> PolarsCategoricalFixesResult:
    """Apply null-fill + dict alignment + utf8 cast to Polars cat features.

    ``precomputed_category_union`` (optional): per-column list of category
    values computed PRE-outlier-detection (e.g. at frame-load time). When
    supplied for a column, it wins over the post-OD train+val recomputation.
    Mitigates the silent val->null cast that happens when OD filtered out
    the only row in train carrying a rare category level.

    ``defer_pandas_conv`` contract: this flag controls WHICH pandas aliases
    get re-pointed to the post-fill+align polars frames at the end of the
    function. When ``True``, ``train_df_pd`` / ``val_df_pd`` / ``test_df_pd``
    / ``filtered_train_df`` / ``filtered_val_df`` were aliased to the ORIGINAL
    (pre-fill, pre-Enum-cast) polars frames at caller time and we re-point
    them HERE so downstream consumers see the cleaned frames. When ``False``,
    those pandas aliases were already converted from polars to pandas earlier
    in the suite and we DO NOT re-point them (re-pointing would silently
    swap a pandas DataFrame for a polars DataFrame in the same ctx slot, and
    every downstream isinstance() check would flip behaviour).

    A caller that toggles ``defer_pandas_conv`` between calls of this function
    on the same NamedTuple result will see ``*_pd`` slots backed by different
    Python types (polars vs pandas) per branch -- that asymmetry is intentional
    but easy to miss when reading the function in isolation.
    """
    # Track which cat columns got null-filled with __MISSING__ so phase-2 Enum union below can include
    # the sentinel in the union -- without that, ``pl.col(c).cast(Enum(union))`` silently casts every
    # __MISSING__ back to null for any split that didn't contribute __MISSING__ to the union, which
    # re-introduces the CatBoost crash this phase is supposed to prevent.
    _filled_with_missing_sentinel: set[str] = set()

    # 1. Null-fill. Null values in Polars Categorical cat_features trip CatBoost 1.2.10's fused-cpdef dispatcher.
    if train_df_polars is not None:
        from mlframe.training.trainer import (
            _polars_nullable_categorical_cols,
            _polars_fill_null_in_categorical,
        )
        # Union across train/val/test: val-only nulls are common on time-ordered splits and must also trigger the fill.
        train_null_cats = set(_polars_nullable_categorical_cols(
            train_df_polars, cat_features=cat_features,
        ))
        val_null_cats = set(_polars_nullable_categorical_cols(
            val_df_polars, cat_features=cat_features,
        )) if val_df_polars is not None else set()
        test_null_cats = set(_polars_nullable_categorical_cols(
            test_df_polars, cat_features=cat_features,
        )) if test_df_polars is not None else set()
        nullable_cats = sorted(train_null_cats | val_null_cats | test_null_cats)
        _filled_with_missing_sentinel = set(nullable_cats)
        if nullable_cats:
            val_only = sorted((val_null_cats | test_null_cats) - train_null_cats)
            if verbose:
                logger.info(
                    "  Pre-fit fill_null('__MISSING__') on %d nullable Polars "
                    "Categorical cat_feature(s) [union train/val/test]: %s. "
                    "Keeps CB 1.2.x's Polars fastpath alive (avoids the "
                    "~15-min pandas-path detour) and gives XGB/HGB the "
                    "same pre-filled frame.",
                    len(nullable_cats), nullable_cats,
                )
                if val_only:
                    logger.warning(
                        "  val/test introduced nulls in %d cat_feature(s) that "
                        "train never had: %s. Without pre-fill these would "
                        "slip into the model's val DMatrix as raw nulls and "
                        "crash CB/XGB native layer.",
                        len(val_only), val_only,
                    )
            train_df_polars = _polars_fill_null_in_categorical(train_df_polars, nullable_cats)
            if val_df_polars is not None:
                val_df_polars = _polars_fill_null_in_categorical(val_df_polars, nullable_cats)
            if test_df_polars is not None:
                test_df_polars = _polars_fill_null_in_categorical(test_df_polars, nullable_cats)

    # 2. Align dicts across train/val/test via pl.Enum.
    # pl.Categorical assigns physical codes per-Series (order-of-first-occurrence) and XGB's native layer treats val's physical codes as indices into train's bin structure without re-reading the Arrow dict. pl.Enum(list) enforces a shared dict by construction. Opt out via ``align_polars_categorical_dicts=False``.
    # Tracks the per-column train+val union built below; Step 4 below reuses these domains when
    # casting raw Utf8 columns to Enum so the cast does not poison the global string cache (P0-B2).
    _enum_domains_built: Dict[str, List[str]] = {}
    if (train_df_polars is not None and cat_features
            and align_polars_categorical_dicts):
        aligned_cols: list = []
        skipped_cols: list = []
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        # First pass: filter to cat-like columns present in train, and split into those whose union
        # is supplied via ``precomputed_category_union`` (no scan needed) vs those that need to be
        # collected from train + val. Batching the collects into one ``.lazy().select(...).collect()``
        # call drops N per-col sync collects to a single materialization (~3x faster on a wide
        # 30-cat-col frame; measured 187ms -> 56ms for the union-compute step alone).
        _eligible_cols: list[str] = []
        _need_scan_cols: list[str] = []
        for col in cat_features:
            if col not in train_df_polars.columns:
                continue
            dt = train_df_polars.schema[col]
            is_cat_like = (
                dt == pl.Categorical
                or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))
                or dt in _string_types
            )
            if not is_cat_like:
                continue
            _eligible_cols.append(col)
            if not (precomputed_category_union and col in precomputed_category_union):
                _need_scan_cols.append(col)
        # Batched union collect: ONE collect on train + ONE collect on val instead of 2*N.
        # implode() collapses each column's unique values into a single list-cell so positional
        # unpack via [0] works the same as the per-column path that this replaces.
        _tr_unique_lists = None
        _val_unique_lists = None
        if _need_scan_cols:
            try:
                _tr_unique_lists = train_df_polars.lazy().select([
                    pl.col(c).drop_nulls().unique().implode().alias(c) for c in _need_scan_cols
                ]).collect()
            except Exception as _e:
                logger.warning(
                    "  Batched train-side unique collect failed (%s); falling back to per-col scan.", _e,
                )
            if val_df_polars is not None:
                try:
                    _val_unique_lists = val_df_polars.lazy().select([
                        pl.col(c).drop_nulls().unique().implode().alias(c) for c in _need_scan_cols
                    ]).collect()
                except Exception as _e:
                    logger.warning(
                        "  Batched val-side unique collect failed (%s); falling back to per-col scan.", _e,
                    )
        # Accumulate the per-frame cast expressions so all aligned casts on each frame land in one
        # ``with_columns`` call instead of N (saves N-1 DataFrame rewrites; each rewrite is O(ncols)
        # at the Arrow buffer level).
        _train_cast_exprs: list = []
        _val_cast_exprs: list = []
        _test_cast_exprs: list = []
        # Pre-collected test null counts for the OOV diagnostic; ONE batched collect instead of
        # 2*N per-col sync calls. Built only when ``verbose`` so non-verbose runs pay nothing.
        _test_nulls_pre: dict[str, int] = {}
        if verbose and test_df_polars is not None and _eligible_cols:
            try:
                _pre_df = test_df_polars.lazy().select([
                    pl.col(c).null_count().alias(c) for c in _eligible_cols if c in test_df_polars.columns
                ]).collect()
                _test_nulls_pre = {c: int(_pre_df[c][0]) for c in _pre_df.columns}
            except Exception:
                _test_nulls_pre = {}
        for col in _eligible_cols:
            try:
                # ONLY train + val contribute to the Enum vocabulary - held-out test must remain "truly unseen". Test-only categories cast with ``strict=False`` so OOV values land as nulls.
                # If a pre-OD union was supplied for this column (computed at
                # frame-load time before global outlier detection ran), use it
                # verbatim - prevents rare categories filtered out of train by OD
                # from being lost in the Enum and silently casting val to null.
                if precomputed_category_union and col in precomputed_category_union:
                    union = set(precomputed_category_union[col])
                else:
                    if _tr_unique_lists is not None and col in _tr_unique_lists.columns:
                        union = set(_tr_unique_lists[col][0].to_list())
                    else:
                        tr_u = train_df_polars.select(pl.col(col).drop_nulls().unique())[col]
                        union = set(tr_u.to_list())
                    if val_df_polars is not None:
                        if _val_unique_lists is not None and col in _val_unique_lists.columns:
                            union |= set(_val_unique_lists[col][0].to_list())
                        else:
                            v_u = val_df_polars.select(pl.col(col).drop_nulls().unique())[col]
                            union |= set(v_u.to_list())
                # If phase-1 null-filled this column with __MISSING__, the sentinel MUST be in the Enum
                # union; otherwise ``cast(Enum(union))`` silently casts every __MISSING__ row back to null
                # and re-introduces the CatBoost crash phase-1 just fixed. Also covers test-only nulls
                # filled by phase-1 even when train+val had no nulls.
                if col in _filled_with_missing_sentinel:
                    union.add("__MISSING__")
                if len(union) > _DICT_ALIGN_SKIP_CARD:
                    skipped_cols.append((col, len(union)))
                    continue
                # precomputed_category_union may contain ints from an unstringified Categorical -- mixing str("__MISSING__")
                # with int raises TypeError on sort, which the broad except below would swallow and the cat-alignment would
                # be skipped, then XGB/CB would crash later on val DMatrix with a misleading unseen-category error. Coerce
                # via str-key so the alignment works on heterogeneous input.
                # bench-attempt-rejected (2026-05-24): guarding ``sorted(key=str)`` behind a homogeneous-str isinstance() check on the union saved only
                # ~4% wall on 50k-cardinality unions (28.5ms -> 27.3ms) -- not worth the extra branch; keep the unconditional str-key form.
                union_sorted = sorted(union, key=str)
                enum_dt = pl.Enum(union_sorted)
                _enum_domains_built[col] = union_sorted
                _train_cast_exprs.append(pl.col(col).cast(enum_dt))
                if val_df_polars is not None:
                    _val_cast_exprs.append(pl.col(col).cast(enum_dt))
                if test_df_polars is not None and col in test_df_polars.columns:
                    _test_cast_exprs.append((col, pl.col(col).cast(enum_dt, strict=False)))
                aligned_cols.append((col, len(union_sorted)))
            except Exception as _e:
                logger.warning(
                    "  Failed to align category dict for %s: %s. "
                    "XGB/CB may crash on val-DMatrix if val has unseen categories.",
                    col, _e,
                )
        # Apply all aligned casts in one with_columns per frame.
        if _train_cast_exprs:
            train_df_polars = train_df_polars.with_columns(_train_cast_exprs)
        if val_df_polars is not None and _val_cast_exprs:
            val_df_polars = val_df_polars.with_columns(_val_cast_exprs)
        if test_df_polars is not None and _test_cast_exprs:
            test_df_polars = test_df_polars.with_columns([_e for _, _e in _test_cast_exprs])
            if verbose:
                try:
                    _post_df = test_df_polars.lazy().select([
                        pl.col(c).null_count().alias(c) for c, _ in _test_cast_exprs
                    ]).collect()
                    for c, _ in _test_cast_exprs:
                        _post = int(_post_df[c][0])
                        _pre = _test_nulls_pre.get(c, 0)
                        if _post > _pre:
                            logger.info(
                                "[cat-alignment] test col=%s: %d row(s) cast-failed to null (OOV vs train+val Enum domain)",
                                c, _post - _pre,
                            )
                except Exception:
                    pass
        if verbose and aligned_cols:
            aligned_summary = ", ".join(f"{c}:{n}" for c, n in aligned_cols)
            logger.info(
                "  Aligned Categorical dicts across train/val/test for %d "
                "cat_feature(s) via pl.Enum(union): %s. Prevents XGB/CB "
                "native crash on val-DMatrix construction when val has "
                "categories absent from train.",
                len(aligned_cols), aligned_summary,
            )
        if verbose and skipped_cols:
            skipped_summary = ", ".join(f"{c}:{n}" for c, n in skipped_cols)
            logger.warning(
                "  Skipped dict alignment for %d high-cardinality "
                "cat_feature(s) (union > %d): %s. These columns are "
                "still at risk of XGB/CB val-DMatrix crash.",
                len(skipped_cols), _DICT_ALIGN_SKIP_CARD, skipped_summary,
            )

    # 3. Re-point pandas aliases to filled+aligned polars frames. With ``defer_pandas_conv=True``, train_df_pd / filtered_train_df were aliased to the ORIGINAL polars frames before fill_null / Enum-alignment.
    if defer_pandas_conv and train_df_polars is not None:
        train_df_pd = train_df_polars
        filtered_train_df = train_df_polars
        if val_df_polars is not None:
            val_df_pd = val_df_polars
            filtered_val_df = val_df_polars
        if test_df_polars is not None:
            test_df_pd = test_df_polars

    # 4. Cast remaining Utf8/String cat_features so pandas conversion produces ``category`` dtype
    # (XGBClassifier's sklearn wrapper rejects ``object``). Prefer pl.Enum keyed off the Step 2
    # train+val union when available so we don't widen the global string cache (P0-B2 fix); fall
    # back to pl.Categorical only when no domain was computed for the column (e.g. align_polars_
    # categorical_dicts=False).
    if was_polars_input and cat_features:
        # Also expose any precomputed (pre-OD) unions: those rows reflect the broadest known
        # category set for the column and are preferable when present.
        _pre_unions = dict(precomputed_category_union or {})
        _domains = dict(_pre_unions)
        for _k, _v in _enum_domains_built.items():
            _domains.setdefault(_k, _v)
        train_df_polars = _cast_utf8_cats_to_categorical(train_df_polars, cat_features, _domains)
        val_df_polars = _cast_utf8_cats_to_categorical(val_df_polars, cat_features, _domains)
        test_df_polars = _cast_utf8_cats_to_categorical(test_df_polars, cat_features, _domains)
        if defer_pandas_conv:
            train_df_pd = train_df_polars if train_df_polars is not None else train_df_pd
            filtered_train_df = train_df_polars if train_df_polars is not None else filtered_train_df
            if val_df_polars is not None:
                val_df_pd = val_df_polars
                filtered_val_df = val_df_polars
            if test_df_polars is not None:
                test_df_pd = test_df_polars

    return PolarsCategoricalFixesResult(
        train_df_polars=train_df_polars,
        val_df_polars=val_df_polars,
        test_df_polars=test_df_polars,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        test_df_pd=test_df_pd,
        filtered_train_df=filtered_train_df,
        filtered_val_df=filtered_val_df,
    )
