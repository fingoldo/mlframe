"""``HGBStrategy`` -- the HistGradientBoosting model pipeline strategy."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from .base import ModelPipelineStrategy
from ._cat_levels_shared import build_polars_enum_map as _build_polars_enum_map

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import polars as pl


class HGBStrategy(ModelPipelineStrategy):
    """
    Strategy for HistGradientBoosting models.

    These models:
    - Handle NaN values natively
    - Don't require feature scaling
    - Support Polars DataFrames natively (numeric + pl.Categorical)
    - Require category encoding only on the pandas fallback path
    - Hard limit: categorical cardinality must be <= 255 (max_bins constraint)
    """

    cache_key = "hgb"
    requires_scaling = False
    requires_encoding = True  # pandas fallback path still needs encoding
    requires_imputation = False
    supports_polars = True
    # sklearn HistGradientBoostingClassifier auto-detects multiclass from y dtype
    # (no library kwarg needed). No native multilabel; uses MultiOutputClassifier.
    supports_native_multiclass = True

    # HGB max_bins is capped at 255 in sklearn
    _MAX_CATEGORICAL_CARDINALITY = 255

    def prepare_polars_dataframe(
        self,
        df: "pl.DataFrame",
        cat_features: List[str],
        category_map: Optional[Dict[str, "pl.Enum"]] = None,
    ) -> "pl.DataFrame":
        """Cast categorical columns for HGB compatibility, using leak-free
        ``pl.Enum`` (not ``pl.Categorical``) for the same reason XGB does:
        polars 1.x's default global string cache makes every
        ``pl.Categorical`` Series in the process share one growing
        dictionary, so the column's physical codes drift across runs.
        sklearn HGB reads the underlying integer codes directly when
        the dtype reports as categorical, so cross-run code drift is a
        latent pickle-reload hazard.

        - Cardinality <= 255: cast to ``pl.Enum`` (HGB auto-detects via from_dtype)
        - Cardinality > 255: ordinal-encode to ``pl.UInt32`` (treated as continuous)

        ``category_map`` (preferred): a {col -> pl.Enum} dict the caller
        builds from the union of train+val unique values via
        ``build_polars_enum_map``. When supplied, train/val/test cast to
        the SAME Enum so codes are consistent across splits.
        """
        import polars as pl

        # Lazy: strategies/__init__.py imports HGBStrategy from this module at its own top level,
        # so a top-level `from . import get_polars_cat_columns` here would be a circular import.
        from . import get_polars_cat_columns
        from ..utils import filter_existing

        schema_cats = set(get_polars_cat_columns(df))
        all_cats = schema_cats | set(cat_features or [])
        existing = filter_existing(df, all_cats)
        if not existing:
            return df

        casts = []
        # track which cols use strict=False (test-side
        # OOV-tolerant cast) so we can quantify cast-failure rate post-with_columns.
        _strict_false_cols: list[str] = []
        for col in existing:
            n_unique = df[col].n_unique()
            high_card = n_unique > self._MAX_CATEGORICAL_CARDINALITY
            if category_map is not None and col in category_map:
                enum_dt = category_map[col]
                # category_map is built from train+val UNION (test EXCLUDED, leak-free).
                # Test rows therefore can carry values absent from the Enum's domain.
                # Use strict=False so OOV values fall through to null rather than
                # crash the lazy collect, matching the dict-alignment routine at
                # core.py:2992 which also passes strict=False on the test split.
                _strict_false_cols.append(col)
                if high_card:
                    casts.append(pl.col(col).cast(pl.String).cast(enum_dt, strict=False).to_physical().cast(pl.UInt32).alias(col))
                else:
                    casts.append(pl.col(col).cast(pl.String).cast(enum_dt, strict=False).alias(col))
                continue
            # No supplied map: build a per-DF Enum from this frame's own values.
            try:
                vals = df[col].drop_nulls().unique().cast(pl.String).to_list()
            except Exception as e:
                logger.debug("could not resolve unique values for column %s: %s", col, e)
                vals = []
            local_enum = pl.Enum(sorted(set(vals))) if vals else None
            if local_enum is None:
                continue
            if high_card:
                casts.append(pl.col(col).cast(pl.String).cast(local_enum).to_physical().cast(pl.UInt32).alias(col))
            else:
                casts.append(pl.col(col).cast(pl.String).cast(local_enum).alias(col))

        if casts:
            # pre-cast null counts for strict=False columns;
            # post-cast delta surfaces silent OOV-nulling.
            _null_pre = {c: int(df[c].null_count()) for c in _strict_false_cols if c in df.columns}
            df = df.with_columns(casts)
            if _null_pre:
                _null_deltas = {c: int(df[c].null_count()) - _null_pre[c] for c in _null_pre}
                _nonzero = {c: d for c, d in _null_deltas.items() if d > 0}
                if _nonzero:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "[hgb cat-cast] %d col(s) had OOV nulls cast-failed: %s",
                        len(_nonzero), _nonzero,
                    )
        return df

    build_polars_enum_map = _build_polars_enum_map
