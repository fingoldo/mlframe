"""Stages of ``build_pipeline``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.training.strategies.base import _Float32CastTransformer
from mlframe.training.strategies.base import _InfToNaNTransformer
from mlframe.training.strategies.base import _NumericOnlyTransformer
from mlframe.training.strategies.base import logger
# --- end imports ---


def _impute_and_scale_steps(self, imputer, _cats_passthrough_raw, steps, cat_features, scaler):
    """Stage of ``build_pipeline``."""
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


def _set_pandas_output(pipeline):
    """Stage of ``build_pipeline``."""
    try:
        pipeline = pipeline.set_output(transform="pandas")
    except Exception as e:
        logger.debug("swallowed exception in base.py: %s", e)
        pass
    return pipeline
