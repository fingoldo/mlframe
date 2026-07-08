"""``XGBoostStrategy`` -- the XGBoost model pipeline strategy."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

# Parent package re-exports XGBoostStrategy AFTER TreeModelStrategy is bound, so this partial-package import resolves.
from . import TreeModelStrategy

logger = logging.getLogger(__name__)


class XGBoostStrategy(TreeModelStrategy):
    """
    Strategy for XGBoost models (>= 3.1).

    Inherits tree model behavior and additionally supports native Polars DataFrames.
    XGBoost auto-detects pl.Categorical columns when enable_categorical=True,
    but pl.String columns must be cast to pl.Categorical first.
    No cardinality limit (unlike HGB).
    """

    supports_polars = True
    # XGB has native multiclass via objective='multi:softprob'+num_class.
    # XGB 3.x has experimental multi_strategy='multi_output_tree' for
    # multilabel -- opt-in via MultilabelDispatchConfig.force_native_xgb_multilabel
    # (default False, uses MultiOutputClassifier wrapper). Marked WIP by
    # upstream until v3.1 stable; opting in earlier accepts the upstream
    # stability risk for vector-output trees (smaller model, integrated
    # GPU/SHAP support, faster inference).
    supports_native_multiclass = True
    supports_native_ranking = True
    # 2026-05-08 QR: XGBoost >=2.0 supports single-fit multi-quantile via
    # ``objective="reg:quantileerror", quantile_alpha=[0.1,0.5,0.9]``;
    # predict() returns (N, K).
    supports_native_quantile = True
    # F-34 (2026-05-31): XGBoost >=2.0 supports native multi-target
    # regression via ``multi_strategy="multi_output_tree"`` paired with
    # ``tree_method="hist"``. Single ensemble outputs (N, K).
    supports_native_multi_target = True
    # supports_native_multilabel: declared False at class level (matches the
    # ABC default + tells callers "wrapper by default"). The actual native-
    # multilabel decision is dynamic -- see wrap_multilabel + get_classif_
    # objective_kwargs overrides below, which BOTH consult the runtime
    # MultilabelDispatchConfig.force_native_xgb_multilabel flag.
    # Inherits cache_key = "tree" from TreeModelStrategy.

    def get_multi_target_objective_kwargs(self) -> dict:
        """XGBoost >=2.0 native multi-output trees.

        ``multi_strategy="multi_output_tree"`` trains a single tree that
        emits (N, K) per leaf; ``tree_method="hist"`` is required.
        """
        return {
            "multi_strategy": "multi_output_tree",
            "tree_method": "hist",
        }

    def get_quantile_objective_kwargs(self, qr_config) -> dict:
        """XGBoost native multi-quantile via ``reg:quantileerror`` +
        ``quantile_alpha=[a1, a2, ...]`` (XGBoost >= 2.0).

        predict() then returns (N, K) -- one column per alpha.
        """
        return {
            "objective": "reg:quantileerror",
            "quantile_alpha": list(qr_config.alphas),
        }

    def get_ranker_objective_kwargs(self, ranking_config=None, y_max=None):
        """XGBRanker objective + auto-fallback for graded labels.

        Default ``rank:ndcg`` (works on graded relevance). When user
        pinned ``rank:map`` but ``y_max > 1`` is detected (graded
        labels), auto-fall-back to ``rank:ndcg`` with WARN -- XGBoost's
        C++ ``is_binary`` check would otherwise crash with a cryptic
        message.

        ``rank:pairwise`` accepted for both binary and graded.
        """
        objective = "rank:ndcg"
        if ranking_config is not None:
            objective = getattr(ranking_config, "xgb_objective", None) or objective

        if (
            objective == "rank:map"
            and y_max is not None
            and y_max > 1
            and ranking_config is not None
            and getattr(ranking_config, "autodetect_label_format", True)
        ):
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "XGB rank:map requires binary labels (y in {0,1}); detected "
                "y_max=%s > 1 (graded relevance). Auto-falling back to "
                "rank:ndcg. Set ranking_config.autodetect_label_format=False "
                "to disable this safety check.",
                y_max,
            )
            objective = "rank:ndcg"

        return {
            "objective": objective,
            # XGB's eval_metric for ranker also defaults to ndcg; expose
            # explicitly so log/early-stop reads the right thing.
            "eval_metric": "ndcg",
        }

    def get_classif_objective_kwargs(self, target_type, n_classes: int, multilabel_config=None) -> dict:
        """Override base to support opt-in native XGB multilabel.

        When ``target_type == MULTILABEL_CLASSIFICATION`` AND
        ``multilabel_config.force_native_xgb_multilabel`` is True, return
        the native vector-output-tree kwargs:
          {'objective': 'binary:logistic',
           'multi_strategy': 'multi_output_tree',
           'tree_method': 'hist'}

        Otherwise falls back to the base dispatcher (which returns ``{}``
        for multilabel -- wrapper path takes over).
        """
        from ..configs import TargetTypes
        if (
            target_type == TargetTypes.MULTILABEL_CLASSIFICATION
            and multilabel_config is not None
            and getattr(multilabel_config, "force_native_xgb_multilabel", False)
        ):
            # XGB 3.x experimental native multilabel. tree_method='hist'
            # is required; binary:logistic per-output objective.
            return {
                "objective": "binary:logistic",
                "multi_strategy": "multi_output_tree",
                "tree_method": "hist",
            }
        return super().get_classif_objective_kwargs(target_type, n_classes)

    def wrap_multilabel(self, estimator, target_type, multilabel_config=None, n_labels: Optional[int] = None):
        """Override base to opt into native XGB multilabel when configured.

        When ``force_native_xgb_multilabel=True``, return ``estimator``
        unchanged (no MultiOutputClassifier wrapper) -- the kwargs from
        ``get_classif_objective_kwargs`` already configured the native
        multi-output tree path.
        """
        from ..configs import TargetTypes
        if (
            target_type == TargetTypes.MULTILABEL_CLASSIFICATION
            and multilabel_config is not None
            and getattr(multilabel_config, "force_native_xgb_multilabel", False)
        ):
            return estimator
        # Fall back to base (MultiOutputClassifier / chain / etc.)
        return super().wrap_multilabel(
            estimator, target_type,
            multilabel_config=multilabel_config, n_labels=n_labels,
        )

    def prepare_polars_dataframe(
        self,
        df: "pl.DataFrame",
        cat_features: List[str],
        category_map: Optional[Dict[str, "pl.Enum"]] = None,
    ) -> "pl.DataFrame":
        """Cast string columns to a categorical dtype for XGBoost auto-detection.

        XGBoost detects polars categorical/Enum dtypes natively when
        ``enable_categorical=True`` but does not handle raw ``pl.String``.

        Why ``pl.Enum`` over ``pl.Categorical`` (2026-04-28):
        Polars 1.x has the global string cache enabled by default and
        ``pl.disable_string_cache()`` is a no-op. Every ``pl.Categorical``
        Series in the process therefore shares one monotonically growing
        dictionary. Across pytest runs this lets categories from earlier
        tests bleed into a later test's column dtype: train_df might be
        fitted when the global dict is small, val_df at predict time has
        the same column with a larger dtype.cats list, and XGBoost's
        ``cat_container`` raises "Found a category not in the training
        set" for the ghost levels even when the actual values column is
        clean. ``pl.Enum`` is per-Series (no shared cache), so the dtype
        is fully determined by the levels we pass in.

        ``category_map`` (preferred): a {col -> pl.Enum} dict the caller
        builds from the union of train+val unique values (test must be
        excluded to avoid label leakage). When supplied, all string /
        Categorical / existing-Enum columns named in the map are cast
        to that exact Enum, giving train/val/test (and predict-time
        frames) identical, leak-free dtypes.

        Fallback (no map): cast each column to a per-DF Enum built from
        its own unique values. This still avoids the global-cache leak,
        but train/val/test produced from independent calls will have
        different Enum dtypes -- fine when the caller only ever passes
        one frame, but at predict time XGBoost will reject any val/test
        category not present in the train Enum. Use the explicit
        ``category_map`` route for predict-time correctness.
        """
        import polars as pl

        # When a leak-free pl.Enum map is supplied, cast every named
        # column (cat or already-categorical) to that exact Enum so
        # train/val/test dtypes match across calls.
        if category_map is not None:
            exprs: List[Any] = []
            _logged_cols: List[str] = []
            for c, enum_dtype in category_map.items():
                if c in df.columns:
                    # Map is built train+val UNION (test excluded for leak-freeness),
                    # so test rows may carry OOV values. strict=False nulls them out
                    # rather than crashing -- consistent with core.py:2992 behaviour.
                    exprs.append(pl.col(c).cast(pl.String).cast(enum_dtype, strict=False).alias(c))
                    _logged_cols.append(c)
            if exprs:
                # Wave 72 (2026-05-21): quantify OOV-nulling so silent test-cat loss
                # becomes visible.
                _null_pre = {c: int(df[c].null_count()) for c in _logged_cols}
                df = df.with_columns(exprs)
                _null_deltas = {c: int(df[c].null_count()) - _null_pre[c] for c in _logged_cols}
                _nonzero = {c: d for c, d in _null_deltas.items() if d > 0}
                if _nonzero:
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "[xgb cat-cast] %d col(s) had OOV nulls cast-failed: %s",
                        len(_nonzero), _nonzero,
                    )
            # Still need to cover any pl.String / pl.Utf8 not present in
            # the map (e.g. brand-new columns surfacing only after the
            # map was built) -- fall back to per-DF Enum.
            remaining = {name for name, dtype in df.schema.items() if (dtype in (pl.Utf8, pl.String)) and name not in category_map}
            if remaining:
                fallback_exprs = []
                for c in remaining:
                    vals = df[c].drop_nulls().unique().cast(pl.String).to_list()
                    enum_dtype = pl.Enum(sorted(set(vals)))
                    fallback_exprs.append(pl.col(c).cast(pl.String).cast(enum_dtype).alias(c))
                df = df.with_columns(fallback_exprs)
            return df

        # No map supplied: legacy behaviour, but using pl.Enum (not
        # pl.Categorical) so the cast doesn't pollute polars 1.x's
        # default global string cache. ``pl.String`` / ``pl.Utf8``
        # columns get an Enum built from their own unique values.
        schema_cats = {name for name, dtype in df.schema.items() if dtype in (pl.Utf8, pl.String)}
        cols_to_cast = schema_cats | {c for c in (cat_features or []) if c in df.columns and df[c].dtype in (pl.Utf8, pl.String)}
        if not cols_to_cast:
            return df
        exprs = []
        for c in cols_to_cast:
            vals = df[c].drop_nulls().unique().cast(pl.String).to_list()
            enum_dtype = pl.Enum(sorted(set(vals)))
            exprs.append(pl.col(c).cast(pl.String).cast(enum_dtype).alias(c))
        return df.with_columns(exprs)

    def build_polars_enum_map(
        self,
        train_df: "pl.DataFrame",
        val_df: "Optional[pl.DataFrame]",
        cat_features: List[str],
    ) -> "Dict[str, pl.Enum]":
        """Build per-column ``pl.Enum`` dtypes from the union of train+val
        unique values. Test data is intentionally excluded -- letting test
        levels widen the Enum would leak label-time information back into
        the model's accepted-category set.

        Returns ``{col_name: pl.Enum([...])}`` for every string /
        Categorical / Enum column present in ``train_df``. Columns absent
        from ``val_df`` contribute only their train levels (still safe).
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
                    except Exception:
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
