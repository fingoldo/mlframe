"""``ModelPipelineStrategy`` -- the abstract base class for all model strategies.

The class implements the Strategy pattern to handle model-specific
preprocessing pipelines. Each model type may require different
preprocessing (scaling, encoding, imputation).
"""

from __future__ import annotations


import importlib.util
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict, FrozenSet, Tuple, TYPE_CHECKING
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _cast_numeric_to_float32(X):
    """Down-cast a fully-numeric transformer output to float32.

    SimpleImputer / StandardScaler upcast float32 input to float64 on older
    sklearn (the mean/var accumulation runs in float64), doubling the memory
    of the cached transformed frame (a 4.1M x 470 frame is ~7.7 GB in
    float32, ~15 GB in float64). Appended after the scaler on the numeric
    (linear) path so the persisted/cached output is float32 regardless of
    sklearn version. Preserves the pandas/polars container when set_output
    is active. Idempotent; safe only where every column is numeric (the
    requires_scaling path, after encoding + imputation)."""
    import numpy as _np
    if hasattr(X, "astype"):
        try:
            return X.astype(_np.float32)
        except (TypeError, ValueError):
            return _np.asarray(X, dtype=_np.float32)
    return _np.asarray(X, dtype=_np.float32)


from sklearn.base import BaseEstimator, TransformerMixin


class _Float32CastTransformer(TransformerMixin, BaseEstimator):
    """Minimal sklearn-compatible transformer that casts numeric input to
    float32. We used to wrap ``_cast_numeric_to_float32`` in a
    ``FunctionTransformer(feature_names_out="one-to-one")``, but sklearn
    1.8 resolves ``feature_names_out`` at TRANSFORM time on the test split
    and raises ``TypeError: iteration over a 0-d array`` when the input is
    a numpy array (no ``.columns`` attribute). Going direct sidesteps the
    feature-names-out machinery entirely while keeping pickle / clone /
    sklearn-tags semantics intact via BaseEstimator+TransformerMixin."""

    def fit(self, X, y=None):  # noqa: ARG002 -- sklearn signature
        # Stamp the standard fitted attributes so ``check_is_fitted`` succeeds
        # AND sklearn's pipeline name-tracker (which calls
        # ``get_feature_names_out(input_features=None)`` on each step in turn,
        # passing the prior step's output names back in) gets back a non-empty
        # name vector that matches the data width.
        import numpy as _np
        if hasattr(X, "columns"):
            self.feature_names_in_ = _np.asarray(list(X.columns))
        _shape = getattr(X, "shape", None)
        if _shape is not None and len(_shape) >= 2:
            self.n_features_in_ = int(_shape[1])
        return self

    def transform(self, X):
        return _cast_numeric_to_float32(X)

    def fit_transform(self, X, y=None):  # noqa: ARG002
        self.fit(X)
        return _cast_numeric_to_float32(X)

    def get_feature_names_out(self, input_features=None):
        import numpy as _np
        if input_features is not None:
            return _np.asarray(input_features)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        n = getattr(self, "n_features_in_", 0)
        return _np.asarray([f"x{i}" for i in range(n)])

class _InfToNaNTransformer(TransformerMixin, BaseEstimator):
    """Replace +/-inf with NaN so the downstream SimpleImputer fills them.

    SimpleImputer handles NaN but NOT infinity; StandardScaler / linear /
    MLP then raise ``Input X contains infinity or a value too large for
    dtype`` the moment an inf cell survives. mlframe's global
    ``fix_infinities`` preprocessing neutralises inf for the shared frame,
    but a caller can disable it (``fix_infinities=False``) for the GBDT
    backends that tolerate inf -- which leaves the inf-intolerant linear /
    MLP per-model pipeline exposed. This step makes that pipeline
    self-sufficient: inf -> NaN -> imputed, independent of the global flag.
    Inserted only when an imputer follows it (so the introduced NaN is
    always filled). Elementwise + column-count-preserving, so it does not
    perturb the pipeline's ``n_features_in_`` input-width contract.
    Surfaced by fuzz (inject_inf_nan=True + fix_infinities=False + linear/mlp).
    """

    def fit(self, X, y=None):  # noqa: ARG002 -- sklearn signature
        import numpy as _np
        if hasattr(X, "columns"):
            self.feature_names_in_ = _np.asarray(list(X.columns))
        _shape = getattr(X, "shape", None)
        if _shape is not None and len(_shape) >= 2:
            self.n_features_in_ = int(_shape[1])
        return self

    @staticmethod
    def _replace(X):
        import numpy as _np
        if hasattr(X, "replace") and hasattr(X, "columns"):  # pandas DataFrame
            return X.replace([_np.inf, -_np.inf], _np.nan)
        arr = _np.asarray(X)
        if arr.dtype.kind == "f":  # float -> safe to test finiteness
            return _np.where(_np.isfinite(arr), arr, _np.nan)
        # Non-float (int/bool): no inf possible; pass through unchanged.
        return X

    def transform(self, X):
        return self._replace(X)

    def fit_transform(self, X, y=None):  # noqa: ARG002
        self.fit(X)
        return self._replace(X)

    def get_feature_names_out(self, input_features=None):
        import numpy as _np
        if input_features is not None:
            return _np.asarray(input_features)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        n = getattr(self, "n_features_in_", 0)
        return _np.asarray([f"x{i}" for i in range(n)])


class _NumericOnlyTransformer(TransformerMixin, BaseEstimator):
    """Apply an inner transformer (imputer / scaler) to all columns EXCEPT the named categoricals, passing cats through unchanged in place.

    Used by ``build_pipeline`` when ``requires_encoding`` is False (learnable cat embeddings active): the raw categorical columns must reach the
    MLP estimator un-scaled / un-imputed so its fit-boundary factorizer + ``nn.Embedding`` can index them, while the numeric block still gets
    the strategy's imputation + scaling. We avoid sklearn's ``ColumnTransformer`` because it reorders (transformed first, passthrough last) and
    drops the original column NAMES the estimator's name-based cat reorder relies on; this wrapper preserves both the original column order and
    names. Operates on pandas frames only (the build_pipeline output is pinned to pandas via set_output); on a non-frame input it falls back to
    transforming the whole array (cats can't be identified by name).
    """

    def __init__(self, inner, cat_features):
        self.inner = inner
        self.cat_features = list(cat_features or [])

    def _num_cols(self, X):
        import pandas as _pd
        named = set(self.cat_features)
        # Route ONLY numeric columns to the inner imputer/scaler. Categoricals -- whether explicitly named OR raw object/category/string columns
        # that arrive when the suite didn't thread cat_features (requires_encoding off) -- must pass THROUGH untouched so the estimator's own
        # factorizer/embedding handles them; feeding a string column to StandardScaler raises "could not convert string to float".
        return [c for c in X.columns
                if c not in named and getattr(X[c], "ndim", 1) == 1 and _pd.api.types.is_numeric_dtype(X[c])]

    def fit(self, X, y=None):
        import numpy as _np
        if hasattr(X, "columns"):
            self.feature_names_in_ = _np.asarray(list(X.columns))
            num_cols = self._num_cols(X)
            self._num_cols_ = num_cols
            if num_cols:
                self.inner.fit(X[num_cols], y)
        else:
            self._num_cols_ = None
            self.inner.fit(X, y)
        _shape = getattr(X, "shape", None)
        if _shape is not None and len(_shape) >= 2:
            self.n_features_in_ = int(_shape[1])
        return self

    def transform(self, X):
        if getattr(self, "_num_cols_", None) is None or not hasattr(X, "columns"):
            return self.inner.transform(X)
        num_cols = self._num_cols_
        if not num_cols:
            return X
        transformed = self.inner.transform(X[num_cols])
        if not hasattr(transformed, "columns"):
            import pandas as _pd
            # The inner transformer is expected to preserve the numeric block width (imputers configured with keep_empty_features=True, scalers, inf->NaN / float32 casts all do). If it nonetheless changed the column count -- e.g. a SimpleImputer left at its default keep_empty_features=False that dropped an all-NaN column -- recover the surviving column labels from get_feature_names_out so the reassembly maps each output column back to the right name instead of crashing the DataFrame constructor with a width mismatch.
            _n_out = transformed.shape[1] if getattr(transformed, "ndim", 1) >= 2 else 1
            if _n_out == len(num_cols):
                _out_cols = num_cols
            else:
                _out_cols = None
                try:
                    _names = list(self.inner.get_feature_names_out(num_cols))
                    if len(_names) == _n_out:
                        _out_cols = _names
                except (AttributeError, ValueError, NotImplementedError, TypeError):
                    pass
                if _out_cols is None:
                    _out_cols = list(num_cols[:_n_out]) if _n_out <= len(num_cols) else [f"num_ext_{i}" for i in range(_n_out)]
            transformed = _pd.DataFrame(transformed, columns=_out_cols, index=X.index)
        # Reassemble in the ORIGINAL column order so the cat columns keep their positions + names (the estimator reorders cats leading by name,
        # but downstream feature-name continuity + the input-width contract still expect a stable layout here). Only map back columns the inner
        # actually produced; a column the inner dropped stays at its pre-transform value (the keep_empty_features fix above makes this the rare path).
        out = X.copy()
        _present = set(transformed.columns) if hasattr(transformed, "columns") else set(num_cols)
        for c in num_cols:
            if c not in _present:
                continue
            out[c] = transformed[c].to_numpy() if hasattr(transformed[c], "to_numpy") else transformed[c]
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        import numpy as _np
        if input_features is not None:
            return _np.asarray(input_features)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        n = getattr(self, "n_features_in_", 0)
        return _np.asarray([f"x{i}" for i in range(n)])


# Pre-compiled slug pattern (MEMORY.md: pre-compile regex at module level).
# Only allow alnum, dash, underscore; everything else collapses to a single "_".
_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")

if TYPE_CHECKING:
    import polars as pl

# =============================================================================
# Unified categorical type constants
# =============================================================================
# Used across pipeline.py, trainer.py, utils.py, core.py to detect categoricals.
# Import these instead of hardcoding type lists.

# 2026-05-21: include "str" so the categorical detector also matches the
# pandas-3.0 / `future.infer_string=True` "str" dtype that auto-converts
# object-of-strings during pd.DataFrame construction. Without this entry,
# tests/training/test_fit_pipeline_*_skip.py and test_ranker_object_cat_*
# failed on envs where the future flag is enabled: cat_low column landed
# as ``<StringDtype(na_value=nan)>`` (dtype.name == "str"), the set lookup
# missed it, and downstream LGB / cat encoders saw raw strings.
PANDAS_CATEGORICAL_DTYPES: FrozenSet[str] = frozenset({
    "category", "object", "string", "string[pyarrow]", "large_string[pyarrow]", "str",
})

# pandas>=2.3 ``select_dtypes`` rejects the literal ``"str"`` with
# ``TypeError: numpy string dtypes are not allowed, use 'str' or 'object'
# instead`` — pandas parses ``"str"`` as a numpy "<U..." dtype, not as the
# pandas StringDtype(na_value=nan) the comment above refers to. Keep ``"str"``
# in the membership-check set (``dtype.name`` lookups in pipeline.py work fine
# with it) but use this filtered tuple for ``select_dtypes(include=...)`` calls
# in _nan_processing.py / _eval_helpers.py / etc.; the StringDtype is already
# covered by the ``"string"`` entry on its own.
PANDAS_CATEGORICAL_SELECT_DTYPES: tuple = tuple(
    sorted(d for d in PANDAS_CATEGORICAL_DTYPES if d != "str")
)


class ModelPipelineStrategy(ABC):
    """
    Abstract base class for model-specific pipeline strategies.

    Different model types have different requirements:
    - Tree models (CB, LGB, XGB): Handle NaN natively, no scaling needed
    - HGB: Needs category encoding but no scaling
    - Neural nets (MLP, NGBoost): Need full preprocessing (encoding + imputation + scaling)
    - Linear models: Need full preprocessing
    """

    @property
    @abstractmethod
    def cache_key(self) -> str:
        """Unique key for caching transformed DataFrames."""
        ...

    @property
    @abstractmethod
    def requires_scaling(self) -> bool:
        """Whether this model type requires feature scaling."""
        ...

    @property
    @abstractmethod
    def requires_encoding(self) -> bool:
        """Whether this model type requires category encoding."""
        ...

    @property
    @abstractmethod
    def requires_imputation(self) -> bool:
        """Whether this model type requires missing value imputation."""
        ...

    @property
    def supports_polars(self) -> bool:
        """Whether this model type can accept Polars DataFrames directly for training."""
        return False

    @property
    def supports_text_features(self) -> bool:
        """Whether this model supports text features (free-text string columns)."""
        return False

    @property
    def supports_embedding_features(self) -> bool:
        """Whether this model supports embedding features (list-of-float vector columns)."""
        return False

    # ---- Multi-output (multiclass + multilabel) capability flags ----
    # Strategies override these to opt INTO native dispatch. Default False
    # means the dispatcher falls back to wrapper-based handling
    # (MultiOutputClassifier for multilabel, default sklearn for
    # multiclass -- most libraries already support multiclass natively
    # via library-specific objective kwargs in helpers._classif_objective_kwargs).

    @property
    def supports_native_multiclass(self) -> bool:
        """Whether this strategy supports K>2 single-label classification
        natively (via library objective kwargs).

        True for CB/XGB/LGB/HGB/Linear (all 5 mlframe strategies have
        native paths). NeuralNet / Recurrent default False (their multi-
        output handling is its own track).
        """
        return False

    @property
    def supports_native_multilabel(self) -> bool:
        """Whether this strategy supports K binary independent labels
        natively (single fitted model returns (N, K) probabilities,
        no MultiOutputClassifier wrapper needed).

        Today only CatBoost (loss_function='MultiLogloss'). Override to
        True in CatBoostStrategy.
        """
        return False

    @property
    def supports_native_ranking(self) -> bool:
        """Whether this strategy supports learning-to-rank natively.

        True for CatBoostStrategy / XGBoostStrategy / LightGBMStrategy
        (all three ship native rankers: ``CatBoostRanker``, ``XGBRanker``,
        ``LGBMRanker``). False for HGB / Linear / Neural / Recurrent --
        no ranker exists in those backends, and the suite skips them
        with NotImplementedError when ``target_type.is_ranking``.
        """
        return False

    @property
    def supports_native_quantile(self) -> bool:
        """Whether this strategy supports single-fit multi-quantile
        regression natively.

        True for CatBoost (``loss_function=MultiQuantile:alpha=...``)
        and XGBoost (``objective=reg:quantileerror, quantile_alpha=[...]``).
        False for everyone else -- LGB, HGB, Linear, MLP, Recurrent
        all need K independent fits stacked via
        ``_QuantileMultiOutputWrapper``.
        """
        return False

    def get_quantile_objective_kwargs(self, qr_config) -> dict:
        """Per-strategy kwargs for quantile-regression objective.

        Returns the dict to merge into the regressor constructor kwargs
        when ``target_type.is_quantile`` is in scope. Default returns
        ``{}`` (subclasses without native quantile support route through
        the wrapper instead -- see ``wrap_quantile``).
        """
        return {}

    @property
    def supports_native_multi_target(self) -> bool:
        """F-34 (2026-05-31): whether this strategy supports K
        independent continuous targets ``y of shape (N, K>=2) float``
        natively -- a single fitted model returns (N, K) predictions.

        Native paths:
          * CatBoost: ``loss_function="MultiRMSE"``
          * XGBoost (>=2.0): ``multi_strategy="multi_output_tree"``
          * sklearn linear / RandomForest: native by handing (N, K) y to fit()
          * mlframe MLP: ``PytorchLightningRegressor`` auto-detects (N, K)
            (F-24 commit 2d300944)

        Non-native paths fall back to ``sklearn.multioutput.MultiOutputRegressor``
        (LightGBM / HistGradientBoosting / NGBoost). The suite uses
        ``wrap_multi_target`` to wrap the base regressor at build time.

        Override per-strategy. Default False is safe — un-overridden
        strategies route through the wrapper.
        """
        return False

    def get_multi_target_objective_kwargs(self) -> dict:
        """F-34 (2026-05-31): per-strategy kwargs for multi-target
        regression objective.

        Returns the dict to merge into the regressor constructor when
        ``target_type.is_multi_target_regression`` is in scope. Default
        returns ``{}`` — subclasses with native multi-target route
        through this override (e.g. CatBoost returns
        ``{"loss_function": "MultiRMSE"}``); subclasses without native
        support are wrapped via ``wrap_multi_target`` instead.
        """
        return {}

    def wrap_multi_target(self, estimator):
        """F-34 (2026-05-31): wrap a single-target regressor in
        sklearn.multioutput.MultiOutputRegressor when this strategy does
        NOT natively support multi-target.

        - Native strategies (``supports_native_multi_target=True``):
          return estimator unchanged.
        - Non-native: wrap with ``MultiOutputRegressor`` (per sklearn
          convention — K independent fits, no joint training across
          target columns).

        Per-target ``sample_weight`` is NOT supported by
        ``MultiOutputRegressor`` (sklearn limitation); a single
        ``(N,)`` sample_weight applies to all K targets uniformly.
        """
        if self.supports_native_multi_target:
            return estimator
        from sklearn.multioutput import MultiOutputRegressor
        return MultiOutputRegressor(estimator, n_jobs=1)

    def wrap_quantile(self, estimator, qr_config):
        """Wrap a base regressor for quantile-regression dispatch.

        - Strategies with ``supports_native_quantile=True`` (CB / XGB)
          return ``estimator`` unchanged -- the native objective kwargs
          (injected via ``get_quantile_objective_kwargs``) make the
          single fit produce (N, K) predictions.
        - Strategies with ``supports_native_quantile=False`` wrap the
          estimator in ``_QuantileMultiOutputWrapper(base, alphas)`` so
          K independent fits are stacked into an (N, K) prediction.
        """
        if self.supports_native_quantile:
            return estimator
        from ..quantile_wrapper import _QuantileMultiOutputWrapper

        return _QuantileMultiOutputWrapper(
            base_estimator=estimator,
            alphas=qr_config.alphas,
            crossing_fix=qr_config.crossing_fix,
            n_jobs=qr_config.wrapper_n_jobs,
        )

    def get_ranker_objective_kwargs(
        self,
        ranking_config=None,
        y_max: Optional[float] = None,
    ) -> dict:
        """Per-strategy ranker kwargs (loss_function / objective + auxiliaries).

        Returns the dict to merge into the ranker constructor kwargs.
        Default implementation returns ``{}`` (subclasses without native
        ranker should never reach this -- ``supports_native_ranking=False``
        gates it). Override in CB/XGB/LGB strategies.

        Parameters
        ----------
        ranking_config : LearningToRankConfig, optional
            User-pinned ranking objectives + ensemble method. Defaults
            applied when None.
        y_max : float, optional
            Maximum value of ``y_train`` -- used by XGB to auto-fall-back
            from ``rank:map`` (binary-only) to ``rank:ndcg`` when graded
            relevance is detected (``y_max > 1``).
        """
        return {}

    def get_classif_objective_kwargs(self, target_type, n_classes: int) -> dict:
        """Per-strategy classifier kwargs for the target type.

        Default implementation delegates to the freestanding
        ``helpers._classif_objective_kwargs`` dispatcher. Strategies can
        override to customise (e.g. force a specific eval_metric on
        multilabel).
        """
        from ..helpers import _classif_objective_kwargs

        flavor_map = {
            "CatBoostStrategy": "catboost",
            "XGBoostStrategy": "xgboost",
            "TreeModelStrategy": "lightgbm",  # default tree model is LGB
            "HGBStrategy": "hgb",
            "LinearModelStrategy": "linear",
        }
        flavor = flavor_map.get(type(self).__name__, "")
        return _classif_objective_kwargs(flavor, target_type, n_classes)

    def wrap_multilabel(self, estimator, target_type, multilabel_config=None,
                       n_labels: Optional[int] = None):
        """Multilabel dispatch: native vs wrapper vs chain ensemble.

        Default delegates to ``helpers._maybe_wrap_multilabel`` with the
        strategy's ``supports_native_multilabel`` flag. CatBoostStrategy
        overrides this only if it needs CB-specific behaviour beyond the
        flag check (currently doesn't).
        """
        from ..helpers import _maybe_wrap_multilabel

        return _maybe_wrap_multilabel(
            estimator,
            target_type,
            multilabel_config=multilabel_config,
            strategy_supports_native_multilabel=self.supports_native_multilabel,
            n_labels=n_labels,
        )

    def feature_tier(self) -> tuple:
        """Hashable key for grouping models by feature support level.

        Models with the same tier can share trimmed DataFrames (text/embedding
        columns dropped once per tier). Higher tiers support more feature types
        and should train first.
        """
        return (self.supports_text_features, self.supports_embedding_features)

    def prepare_polars_dataframe(self, df: "pl.DataFrame", cat_features: List[str]) -> "pl.DataFrame":
        """Prepare a Polars DataFrame for models that support native Polars input.

        Called in the Polars fastpath before training. Override in subclasses
        to apply model-specific transformations (e.g., casting string columns
        to pl.Categorical for HGB).

        Default implementation returns the DataFrame unchanged (suitable for
        CatBoost which handles all dtypes natively).
        """
        return df

    def _extra_pre_encoding_steps(self, embedding_features, text_features) -> list:
        """Extra pipeline steps inserted after feature selection and BEFORE category encoding / imputation / scaling.

        Base: none. Strategies that consume embedding-vector (``List``) or free-text columns but feed a numeric-only
        model (e.g. neural) override this to expand / encode those columns to numeric here, so the downstream numeric
        steps and the model see a pure-numeric frame.
        """
        return []

    def build_pipeline(
        self,
        base_pipeline: Optional[Pipeline],
        cat_features: List[str],
        category_encoder: Optional[Any] = None,
        imputer: Optional[Any] = None,
        scaler: Optional[Any] = None,
        embedding_features: Optional[List[str]] = None,
        text_features: Optional[List[str]] = None,
    ) -> Optional[Pipeline]:
        """
        Build the preprocessing pipeline for this model type.

        Feature selectors (MRMR, RFECV, SelectorMixin) run FIRST (before preprocessing); custom
        transformers (PCA, etc.) run LAST (after preprocessing).

        Per audit FE-L-1: set_output is hard-wired to ``"pandas"`` (the only format any caller ever
        used). Polars-native consumers take the polars fastpath upstream of this builder; the sklearn
        pipeline always emits pandas.
        """
        from sklearn.feature_selection import SelectorMixin
        from mlframe.feature_selection.filters import MRMR

        steps = []

        # Determine if base_pipeline is a feature selector (runs FIRST) or transformer (runs LAST)
        is_feature_selector = False
        if base_pipeline is not None:
            is_feature_selector = (
                isinstance(base_pipeline, SelectorMixin)
                or isinstance(base_pipeline, MRMR)
                or isinstance(base_pipeline, Pipeline)  # nested sklearn Pipeline is treated as pre-step
                or hasattr(base_pipeline, 'get_support')  # RFECV and similar
            )

        # Whether the cat encoder must run BEFORE the feature selector: a selector whose internal estimator is numeric
        # (RFECV with a linear estimator, MRMR's numeric MI path) crashes with "could not convert string to float: 'C'"
        # on raw string cats. When the strategy itself requires encoding, place the encoder ahead of the selector so the
        # selector operates on encoded numeric features. Strategies with native cat handling (requires_encoding=False)
        # keep the selector-first order so their estimator sees the raw cats.
        _encode_before_selector = bool(self.requires_encoding and cat_features and category_encoder is not None)

        if _encode_before_selector:
            steps.append(("ce", category_encoder))

        # Feature selectors go FIRST (before preprocessing, but AFTER cat-encoding when the strategy requires it).
        if base_pipeline is not None and is_feature_selector:
            steps.append(("pre", base_pipeline))

        # Embedding-vector + text columns -> numeric, BEFORE cat-encoding / imputation / scaling, so every downstream
        # numeric step and the model see a pure-numeric frame. No-op unless the strategy overrides the hook.
        steps.extend(self._extra_pre_encoding_steps(embedding_features, text_features))

        # Add category encoding if required and categorical features exist (unless already placed before the selector).
        # Observability guard (2026-04-19 round-9 probe): if the strategy
        # declares ``requires_encoding=True`` AND there are cat_features
        # in the data BUT the caller passed ``category_encoder=None``,
        # silently skipping the step meant unbounded categorical string
        # values fed to a model that expected numeric -- sklearn then
        # raised opaquely inside ``LinearRegression.fit`` / etc. Now: WARN
        # so operators see the missing dependency at the source. We don't
        # raise because some tests/callers legitimately pre-encode cats
        # upstream and pass encoder=None; the WARN is enough signal.
        if self.requires_encoding and cat_features and not _encode_before_selector:
            if category_encoder is not None:
                steps.append(("ce", category_encoder))
            else:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "%s.build_pipeline: requires_encoding=True and %d "
                    "categorical feature(s) present, but category_encoder "
                    "is None. Encoding step skipped -- downstream model.fit "
                    "may raise on raw string categoricals. Supply a "
                    "category_encoder (e.g. sklearn.preprocessing."
                    "OrdinalEncoder) or pre-encode cats upstream.",
                    type(self).__name__, len(cat_features),
                )

        # ``_cats_passthrough_raw``: when the strategy declares requires_encoding=False (learnable cat embeddings active) AND cat columns are
        # present, the cats must survive imputation + scaling UNCHANGED so the downstream MLP estimator can factorize them and index its
        # nn.Embedding. The numeric block still gets imputation + scaling via the _NumericOnlyTransformer wrapper. When True (the encoder path)
        # the cats are already numeric by the time the imputer runs, so the wrapper is unnecessary.
        _cats_passthrough_raw = (not self.requires_encoding) and bool(cat_features)

        # Add imputation if required.
        # WARN when requires_imputation=True but caller passed imputer=None: silently skipping the step sent raw NaN into LinearRegression.fit (prod log 2026-05-14 4M-row regression suite).
        # Mirrors the requires_encoding WARN above. Root-cause was in caller (ctx.imputer not propagated from _get_pipeline_components); see ef123ff + regression suite in test_strategy_imputer_propagation.py.
        if self.requires_imputation:
            if imputer is not None:
                # inf -> NaN BEFORE the imputer so the imputer fills inf too
                # (SimpleImputer only handles NaN). Guards the inf-intolerant
                # scaler / linear / MLP when the global fix_infinities flag is
                # off. Paired with the imputer so the introduced NaN is always
                # filled; no-op on finite data. Restricted to numeric columns
                # when raw cats must pass through (inf-replace on a string cat
                # column is a no-op anyway, but the imputer would choke on it).
                if _cats_passthrough_raw:
                    steps.append(("inf_to_nan", _NumericOnlyTransformer(_InfToNaNTransformer(), cat_features)))
                    steps.append(("imp", _NumericOnlyTransformer(imputer, cat_features)))
                else:
                    steps.append(("inf_to_nan", _InfToNaNTransformer()))
                    steps.append(("imp", imputer))
            else:
                logger.warning(
                    "%s.build_pipeline: requires_imputation=True but imputer is None. Imputation step skipped - downstream model.fit may raise ValueError on NaN input. "
                    "Supply a sklearn.impute.SimpleImputer or pre-impute upstream.",
                    type(self).__name__,
                )

        # Add scaling if required.
        # Same defence-in-depth: WARN on silent skip when requires_scaling=True but scaler=None (LinearRegression doesn't crash without scaling but regularised variants converge slower; surface the misconfiguration).
        if self.requires_scaling:
            if scaler is not None:
                # Scale numerics only when raw cats must pass through; scaling integer cat codes would destroy them as embedding indices.
                if _cats_passthrough_raw:
                    steps.append(("scaler", _NumericOnlyTransformer(scaler, cat_features)))
                else:
                    steps.append(("scaler", scaler))
                # Guarantee float32 output: SimpleImputer/StandardScaler
                # upcast float32 -> float64 on older sklearn, doubling the
                # cached transformed-frame memory (the prod 4.1M x 470 linear
                # path sat at 15 GB float64 / RAM 111->128 GB). The cast keeps
                # the numeric output (and PipelineCache entry) float32.
                # Uses a tiny custom transformer (not FunctionTransformer)
                # because sklearn 1.8's feature_names_out="one-to-one"
                # validator crashes on numpy 0-d at transform-time on the
                # test split (TypeError: iteration over a 0-d array). When raw
                # cats pass through they may still be string/category dtype
                # here (the estimator factorizes them later), so the float32
                # cast must skip them or it raises on the string column.
                if _cats_passthrough_raw:
                    steps.append(("to_float32", _NumericOnlyTransformer(_Float32CastTransformer(), cat_features)))
                else:
                    steps.append(("to_float32", _Float32CastTransformer()))
            else:
                logger.warning(
                    "%s.build_pipeline: requires_scaling=True but scaler is None. Scaling step skipped - Ridge/Lasso/ElasticNet regularisation will be feature-magnitude-dependent. Supply a sklearn.preprocessing.StandardScaler.",
                    type(self).__name__,
                )

        # Custom transformers go LAST (after preprocessing)
        if base_pipeline is not None and not is_feature_selector:
            steps.append(("transform", base_pipeline))

        if not steps:
            return base_pipeline

        # Avoid wrapping base_pipeline in redundant Pipeline when it's the only step
        if len(steps) == 1 and steps[0][0] == "pre":
            return base_pipeline
        if len(steps) == 1 and steps[0][0] == "transform":
            return base_pipeline

        pipeline = Pipeline(steps=steps)
        # Ensure DataFrame dtypes (pd.Categorical, object, pl.Enum) survive the chain.
        # sklearn's default returns numpy, which destroys categoricals -- LGB/CB/XGB
        # then receive numpy with string values and crash on Dataset construction
        # (e.g. "could not convert string to float: 'HOURLY'"). set_output keeps
        # the frame as pandas so downstream isinstance(X, pd.DataFrame) branches
        # take the native fastpath. Best-effort: some nested transformers
        # (custom, third-party) don't declare get_feature_names_out and sklearn
        # refuses to configure; swallow and continue.
        try:
            pipeline = pipeline.set_output(transform="pandas")
        except Exception:
            pass
        return pipeline
