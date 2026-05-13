"""
Polars categorical fixes applied once before model training.

1. Null-fill: Polars Categorical columns with nulls get ``__MISSING__`` sentinel
   to prevent CatBoost 1.2.x fused-cpdef ``TypeError`` on null-bearing cats.
2. Dict alignment: pl.Enum(union) across train/val/test prevents XGB silent
   process kill when val has unseen categories (different physical codes).
3. Utf8 cast: String cat_features cast to Categorical so pandas conversion
   produces ``category`` dtype (not ``object``) for XGB/LGB.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)

_DICT_ALIGN_SKIP_CARD = 50_000


def _cast_utf8_cats_to_categorical(df_, cat_features: List[str]):
    """Cast Utf8/String cat_features to pl.Categorical for pandas compat."""
    if not isinstance(df_, pl.DataFrame):
        return df_
    exprs = []
    for c in cat_features:
        if c in df_.columns and df_.schema[c] in (pl.Utf8, pl.String):
            exprs.append(pl.col(c).cast(pl.Categorical))
    return df_.with_columns(exprs) if exprs else df_


def apply_polars_categorical_fixes(
    *,
    train_df_polars,
    val_df_polars,
    test_df_polars,
    train_df_pd,
    val_df_pd,
    test_df_pd,
    filtered_train_df,
    filtered_val_df,
    cat_features: Optional[List[str]],
    align_polars_categorical_dicts: bool,
    can_skip_pandas_conv: bool,
    was_polars_input: bool,
    verbose: bool,
):
    """Apply null-fill + dict alignment + utf8 cast to Polars cat features.

    Returns updated frames:
    (train_df_polars, val_df_polars, test_df_polars, train_df_pd, val_df_pd, test_df_pd,
     filtered_train_df, filtered_val_df)
    """
    # -----------------------------------------------------------------
    # 1. Null-fill: Polars Categorical cat_features with null values trip
    #    CatBoost 1.2.10's fused-cpdef dispatcher. Fill once here so every
    #    polars-native strategy gets the same pre-filled frame.
    #    Single-pass null detection via ``df.null_count()``; fill expr list
    #    reused across train/val/test for consistent sentinel codes.
    if train_df_polars is not None:
        from mlframe.training.trainer import (
            _polars_nullable_categorical_cols,
            _polars_fill_null_in_categorical,
        )
        # Union train+val+test nullable cats. Previously only inspecting train
        # missed the bug where val had nulls but train didn't (common on
        # time-ordered splits) -- union ensures fill if ANY split has nulls.
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

    # -----------------------------------------------------------------
    # 2. Align Polars Categorical dicts across train/val/test via pl.Enum.
    #    Empirically prevents silent process kill on Windows when XGB 3.x
    #    constructs val IterativeDMatrix with ref=train on large frames.
    #    pl.Categorical assigns physical codes per-Series (order-of-first-
    #    occurrence); XGB's native layer treats val's physical codes as
    #    indices into train's bin structure without re-reading the Arrow
    #    dict. pl.Enum(list) enforces a shared dict by construction.
    #    Opt out via ``align_polars_categorical_dicts=False``.
    if (train_df_polars is not None and cat_features
            and align_polars_categorical_dicts):
        aligned_cols: list = []
        skipped_cols: list = []
        for col in cat_features:
            if col not in train_df_polars.columns:
                continue
            dt = train_df_polars.schema[col]
            is_cat_like = (
                dt == pl.Categorical
                or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))
            )
            if not is_cat_like:
                continue
            try:
                # ONLY train + val contribute to the Enum vocabulary. test
                # categories MUST NOT leak back -- the held-out test must
                # represent "truly unseen" data. Test-only categories are
                # cast with ``strict=False`` -> OOV values land as nulls.
                tr_u = train_df_polars.select(pl.col(col).drop_nulls().unique())[col]
                v_u = val_df_polars.select(pl.col(col).drop_nulls().unique())[col] if val_df_polars is not None else None
                union = set(tr_u.to_list())
                if v_u is not None:
                    union |= set(v_u.to_list())
                if len(union) > _DICT_ALIGN_SKIP_CARD:
                    skipped_cols.append((col, len(union)))
                    continue
                union_sorted = sorted(union)
                enum_dt = pl.Enum(union_sorted)
                train_df_polars = train_df_polars.with_columns(pl.col(col).cast(enum_dt))
                if val_df_polars is not None:
                    val_df_polars = val_df_polars.with_columns(pl.col(col).cast(enum_dt))
                if test_df_polars is not None:
                    test_df_polars = test_df_polars.with_columns(
                        pl.col(col).cast(enum_dt, strict=False)
                    )
                aligned_cols.append((col, len(union_sorted)))
            except Exception as _e:
                logger.warning(
                    "  Failed to align category dict for %s: %s. "
                    "XGB/CB may crash on val-DMatrix if val has unseen categories.",
                    col, _e,
                )
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

    # -----------------------------------------------------------------
    # 3. Re-point pandas aliases to filled+aligned polars frames.
    #    When ``can_skip_pandas_conv=True``, train_df_pd / filtered_train_df
    #    were aliased to the ORIGINAL polars frames BEFORE fill_null /
    #    Enum-alignment. Re-point now so downstream consumers see the
    #    sentinel-filled values.
    if can_skip_pandas_conv and train_df_polars is not None:
        train_df_pd = train_df_polars
        filtered_train_df = train_df_polars
        if val_df_polars is not None:
            val_df_pd = val_df_polars
            filtered_val_df = val_df_polars
        if test_df_polars is not None:
            test_df_pd = test_df_polars

    # -----------------------------------------------------------------
    # 4. Cast remaining Utf8/String cat_features to pl.Categorical.
    #    When input is polars_utf8 and alignment didn't touch the column,
    #    cat_features stay as pl.Utf8 -> pandas ``object`` (not ``category``).
    #    XGBClassifier's sklearn wrapper raises ``ValueError: Invalid
    #    columns: object``. Cast after fill+align so the pandas conversion
    #    produces ``category`` dtype. Gate on ``was_polars_input``.
    if was_polars_input and cat_features:
        train_df_polars = _cast_utf8_cats_to_categorical(train_df_polars, cat_features)
        val_df_polars = _cast_utf8_cats_to_categorical(val_df_polars, cat_features)
        test_df_polars = _cast_utf8_cats_to_categorical(test_df_polars, cat_features)
        if can_skip_pandas_conv:
            train_df_pd = train_df_polars if train_df_polars is not None else train_df_pd
            filtered_train_df = train_df_polars if train_df_polars is not None else filtered_train_df
            if val_df_polars is not None:
                val_df_pd = val_df_polars
                filtered_val_df = val_df_polars
            if test_df_polars is not None:
                test_df_pd = test_df_polars

    return (
        train_df_polars, val_df_polars, test_df_polars,
        train_df_pd, val_df_pd, test_df_pd,
        filtered_train_df, filtered_val_df,
    )
