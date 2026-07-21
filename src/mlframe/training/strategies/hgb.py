"""``HGBStrategy`` -- the HistGradientBoosting model pipeline strategy."""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from .base import ModelPipelineStrategy

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
        # Wave 72 (2026-05-21): track which cols use strict=False (test-side
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
            except Exception:
                vals = []
            local_enum = pl.Enum(sorted(set(vals))) if vals else None
            if local_enum is None:
                continue
            if high_card:
                casts.append(pl.col(col).cast(pl.String).cast(local_enum).to_physical().cast(pl.UInt32).alias(col))
            else:
                casts.append(pl.col(col).cast(pl.String).cast(local_enum).alias(col))

        if casts:
            # Wave 72 (2026-05-21): pre-cast null counts for strict=False columns;
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

    def build_polars_enum_map(
        self,
        train_df: "pl.DataFrame",
        val_df: "Optional[pl.DataFrame]",
        cat_features: List[str],
    ) -> "Dict[str, pl.Enum]":
        """Mirror of ``XGBoostStrategy.build_polars_enum_map``: leak-free
        per-column Enum from train+val UNION (test EXCLUDED). HGB's
        cardinality split into Enum vs UInt32 happens at
        ``prepare_polars_dataframe`` time using the same map - the map
        always carries the FULL value set, the cardinality decision is
        applied per frame.
        """
        import polars as pl

        cat_features = cat_features or []
        candidate_cols = [
            name
            for name, dtype in train_df.schema.items()
            if dtype in (pl.Utf8, pl.String) or dtype == pl.Categorical or isinstance(dtype, pl.Enum) or name in cat_features
        ]
        candidate_cols = [c for c in candidate_cols if c in train_df.columns]

        # 2026-05-08 perf: batch per-column unique extraction into one
        # collect() per frame (train + val). The previous loop did
        # ``df[col].unique()`` per cat col -- on c0031 (15 cat cols x
        # 2 frames = 30 collects per build) that cost ~300ms across
        # the suite via PyLazyFrame.collect. Batched via implode() it's
        # 2 collects total per call. Same pattern as session 1 win #2
        # (get_trainset_features_stats_polars). Falls back to per-col
        # loop on any error so one bad cast doesn't poison the frame.
        out: Dict[str, pl.Enum] = {}

        def _batched_unique(df: "pl.DataFrame") -> "Dict[str, list]":
            """Collect distinct string values for every candidate column in one lazy ``.select`` + ``.collect()`` instead of one collect per column; falls back to a per-column loop on any error so a single bad cast doesn't poison the whole batch."""
            cols_present = [c for c in candidate_cols if c in df.columns]
            if not cols_present:
                return {}
            try:
                lf = df.lazy().select([pl.col(c).cast(pl.String).drop_nulls().unique().implode().alias(c) for c in cols_present])
                row = lf.collect()
                return {c: row[c][0].to_list() for c in cols_present}
            except Exception:
                d: Dict[str, list] = {}
                for c in cols_present:
                    try:
                        d[c] = df[c].drop_nulls().unique().cast(pl.String).to_list()
                    except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                        d[c] = []
                return d

        train_levels = _batched_unique(train_df)
        val_levels = _batched_unique(val_df) if val_df is not None else {}
        for col in candidate_cols:
            levels: set = set()
            levels.update(train_levels.get(col, []))
            levels.update(val_levels.get(col, []))
            out[col] = pl.Enum(sorted(levels))
        return out
