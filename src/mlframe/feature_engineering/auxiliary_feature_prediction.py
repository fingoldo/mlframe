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
    """Build the default unfitted regressor used when no custom model_factory is supplied."""
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
    n_uncertainty_repeats: int = 1,
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
    n_uncertainty_repeats
        Opt-in (default 1, matching the historical single-fit behavior bit-for-bit). When >1, fits that many
        differently-seeded models per feature/fold (or per feature in Mode B), averages their predictions into
        ``{column_prefix}_{feature}_pred``/``_resid`` as before, and additionally emits
        ``{column_prefix}_{feature}_uncertainty`` -- the across-repeat prediction std -- so callers can down-weight
        an auxiliary feature on rows where the reconstruction is unstable (disagreeing models -> unreliable
        estimate). Costs ``n_uncertainty_repeats`` extra model fits per feature per fold; only pay it when the
        uncertainty signal is actually consumed.

    Returns
    -------
    pl.DataFrame
        Two columns per target feature: ``{column_prefix}_{feature}_pred`` and
        ``{column_prefix}_{feature}_resid`` (``actual - predicted``); plus, when ``n_uncertainty_repeats > 1``,
        ``{column_prefix}_{feature}_uncertainty``.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("compute_auxiliary_feature_prediction_features: X_train must be a pandas DataFrame.")
    missing = set(target_features) - set(X_train.columns)
    if missing:
        raise ValueError(f"compute_auxiliary_feature_prediction_features: target_features not in X_train.columns: {sorted(missing)}")
    if n_uncertainty_repeats < 1:
        raise ValueError(f"compute_auxiliary_feature_prediction_features: n_uncertainty_repeats must be >= 1, got {n_uncertainty_repeats}.")
    factory = model_factory or _default_model_factory
    want_uncertainty = n_uncertainty_repeats > 1

    def _fit_one_model(Xt: pd.DataFrame, other_cols: list[str], feature: str, model_seed: int) -> Any:
        """Fit a fresh model instance predicting feature from other_cols on Xt."""
        model = factory()
        try:
            model.set_params(random_state=model_seed)
        except (ValueError, TypeError):
            pass
        model.fit(Xt[other_cols], Xt[feature])
        return model

    def _fit_predict_one(Xt: pd.DataFrame, Xq: pd.DataFrame, feature: str, fold_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Fit a single model on Xt and return its predictions and residuals for feature on Xq."""
        other_cols = [c for c in Xt.columns if c != feature]
        model = _fit_one_model(Xt, other_cols, feature, fold_seed)
        pred_q = np.asarray(model.predict(Xq[other_cols]), dtype=np.float64)
        resid_q = Xq[feature].to_numpy(dtype=np.float64) - pred_q
        return pred_q, resid_q

    def _fit_predict_ensemble(Xt: pd.DataFrame, Xq: pd.DataFrame, feature: str, fold_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit an ensemble of bootstrap-resampled models and return mean prediction, residual, and prediction std."""
        # Bootstrap-resample the training rows per repeat (not just vary the model's own random_state):
        # many regressors (e.g. GradientBoostingRegressor with its default subsample=1.0, max_features=None)
        # are otherwise deterministic given the same training rows, which would make an across-repeat std
        # collapse to ~0 and carry no real disagreement signal.
        other_cols = [c for c in Xt.columns if c != feature]
        n_fit = Xt.shape[0]
        preds = np.empty((n_uncertainty_repeats, Xq.shape[0]), dtype=np.float64)
        for repeat_idx in range(n_uncertainty_repeats):
            repeat_seed = fold_seed + repeat_idx * 97
            boot_idx = np.random.default_rng(repeat_seed).integers(0, n_fit, size=n_fit)
            Xt_boot = Xt.iloc[boot_idx]
            model = _fit_one_model(Xt_boot, other_cols, feature, repeat_seed)
            preds[repeat_idx] = np.asarray(model.predict(Xq[other_cols]), dtype=np.float64)
        pred_mean = preds.mean(axis=0)
        pred_std = preds.std(axis=0)
        resid_q = Xq[feature].to_numpy(dtype=np.float64) - pred_mean
        return pred_mean, resid_q, pred_std

    def _make_df(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Cast every column array in cols to the configured output dtype."""
        return {name: arr.astype(dtype, copy=False) for name, arr in cols.items()}

    if X_query is not None:
        cols: dict[str, np.ndarray] = {}
        for feature in target_features:
            if want_uncertainty:
                pred, resid, unc = _fit_predict_ensemble(X_train, X_query, feature, seed)
                cols[f"{column_prefix}_{feature}_uncertainty"] = unc
            else:
                pred, resid = _fit_predict_one(X_train, X_query, feature, seed)
            cols[f"{column_prefix}_{feature}_pred"] = pred
            cols[f"{column_prefix}_{feature}_resid"] = resid
        return pl.DataFrame(_make_df(cols))

    if splitter is None:
        raise ValueError("compute_auxiliary_feature_prediction_features: Mode A (X_query=None) requires a splitter.")
    n_train = X_train.shape[0]
    out_cols = {f"{column_prefix}_{f}_pred": np.zeros(n_train, dtype=dtype) for f in target_features}
    out_cols.update({f"{column_prefix}_{f}_resid": np.zeros(n_train, dtype=dtype) for f in target_features})
    if want_uncertainty:
        out_cols.update({f"{column_prefix}_{f}_uncertainty": np.zeros(n_train, dtype=dtype) for f in target_features})
    splits = list(splitter.split(X_train))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        Xt = X_train.iloc[train_idx].reset_index(drop=True)
        Xq = X_train.iloc[val_idx].reset_index(drop=True)
        for feature in target_features:
            fold_seed = int(seed) + fold_idx * 23
            if want_uncertainty:
                pred, resid, unc = _fit_predict_ensemble(Xt, Xq, feature, fold_seed)
                out_cols[f"{column_prefix}_{feature}_uncertainty"][val_idx] = unc.astype(dtype, copy=False)
            else:
                pred, resid = _fit_predict_one(Xt, Xq, feature, fold_seed)
            out_cols[f"{column_prefix}_{feature}_pred"][val_idx] = pred.astype(dtype, copy=False)
            out_cols[f"{column_prefix}_{feature}_resid"][val_idx] = resid.astype(dtype, copy=False)
        logger.info("auxiliary_feature_prediction: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(out_cols)


__all__ = ["compute_auxiliary_feature_prediction_features"]
