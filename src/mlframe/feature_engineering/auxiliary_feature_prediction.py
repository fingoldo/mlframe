"""``compute_auxiliary_feature_prediction_features``: predict a named important feature from the others,
expose (predicted value, residual) as new engineered features.

Source: Home Credit Default Risk 3rd place -- a submodel predicting `ext_source` features (an important,
often-missing external credit-score feature) from the rest, with the model's predicted value AND its
residual/diff from the true `ext_source` value both used as new features (e.g. "this row's ext_source_1 is
much lower than what our other features would predict it to be").

Distinct from :func:`transformer.cross_feature_reconstruction.compute_cross_feature_reconstruction_features`
(search-for-reuse found this precedent): that function reconstructs EVERY input feature and only emits 5
AGGREGATE outlierness z-stats over ALL per-feature residuals pooled together (unsupervised, no notion of
which feature matters). This function instead targets a caller-specified subset of NAMED IMPORTANT features
(e.g. just ``["ext_source_1", "ext_source_2"]`` out of hundreds of columns -- far cheaper than reconstructing
every column when only a few are known to matter) and exposes each one's own RAW (predicted value, signed
residual) pair as separate, individually named output columns -- not a pooled aggregate.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def _default_model_factory() -> Any:
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=50, max_depth=3)


def compute_auxiliary_feature_prediction_features(
    X_train: pd.DataFrame,
    target_features: Sequence[str],
    X_query: Optional[pd.DataFrame] = None,
    splitter: Optional[Any] = None,
    *,
    seed: int,
    model_factory: Optional[Callable[[], Any]] = None,
    column_prefix: str = "auxfeat",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """For each column in ``target_features``, OOF-fit a model predicting it from every OTHER column, and
    emit that feature's own (predicted value, signed residual) pair as two named output columns.

    Parameters
    ----------
    X_train
        Feature frame with named columns, including every name in ``target_features``.
    target_features
        The specific "important" columns to predict-and-residualize (NOT every column -- pass only the ones
        known/suspected to matter, e.g. an external score feature).
    X_query, splitter
        Same two-mode contract as the sibling transformer-FE functions: ``X_query=None`` (default) computes
        OOF features for ``X_train`` itself via ``splitter`` (required in that case); ``X_query`` given
        computes features for a genuinely held-out query frame using a model fit on the full ``X_train``.
    model_factory
        Zero-arg callable returning a fresh unfitted regressor; defaults to a small
        ``GradientBoostingRegressor``.

    Returns
    -------
    pl.DataFrame
        Two columns per target feature: ``{column_prefix}_{feature}_pred`` and
        ``{column_prefix}_{feature}_resid`` (``actual - predicted``).
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("compute_auxiliary_feature_prediction_features: X_train must be a pandas DataFrame.")
    missing = set(target_features) - set(X_train.columns)
    if missing:
        raise ValueError(f"compute_auxiliary_feature_prediction_features: target_features not in X_train.columns: {sorted(missing)}")
    factory = model_factory or _default_model_factory

    def _fit_predict_one(Xt: pd.DataFrame, Xq: pd.DataFrame, feature: str, fold_seed: int) -> tuple[np.ndarray, np.ndarray]:
        other_cols = [c for c in Xt.columns if c != feature]
        model = factory()
        try:
            model.set_params(random_state=fold_seed)
        except (ValueError, TypeError):
            pass
        model.fit(Xt[other_cols], Xt[feature])
        pred_q = np.asarray(model.predict(Xq[other_cols]), dtype=np.float64)
        resid_q = Xq[feature].to_numpy(dtype=np.float64) - pred_q
        return pred_q, resid_q

    def _make_df(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {name: arr.astype(dtype, copy=False) for name, arr in cols.items()}

    if X_query is not None:
        cols: dict[str, np.ndarray] = {}
        for feature in target_features:
            pred, resid = _fit_predict_one(X_train, X_query, feature, seed)
            cols[f"{column_prefix}_{feature}_pred"] = pred
            cols[f"{column_prefix}_{feature}_resid"] = resid
        return pl.DataFrame(_make_df(cols))

    if splitter is None:
        raise ValueError("compute_auxiliary_feature_prediction_features: Mode A (X_query=None) requires a splitter.")
    n_train = X_train.shape[0]
    out_cols = {f"{column_prefix}_{f}_pred": np.zeros(n_train, dtype=dtype) for f in target_features}
    out_cols.update({f"{column_prefix}_{f}_resid": np.zeros(n_train, dtype=dtype) for f in target_features})
    splits = list(splitter.split(X_train))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        Xt = X_train.iloc[train_idx].reset_index(drop=True)
        Xq = X_train.iloc[val_idx].reset_index(drop=True)
        for feature in target_features:
            pred, resid = _fit_predict_one(Xt, Xq, feature, int(seed) + fold_idx * 23)
            out_cols[f"{column_prefix}_{feature}_pred"][val_idx] = pred.astype(dtype, copy=False)
            out_cols[f"{column_prefix}_{feature}_resid"][val_idx] = resid.astype(dtype, copy=False)
        logger.info("auxiliary_feature_prediction: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(out_cols)


__all__ = ["compute_auxiliary_feature_prediction_features"]
