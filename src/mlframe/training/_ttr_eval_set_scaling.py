"""Module-level ``TransformedTargetRegressor`` subclass that scales ``eval_set``.

Pre-fix this class was defined as a LOCAL class inside ``_configure_neural_params`` in ``trainer.py``. Local classes carry a hidden closure reference to the enclosing function's namespace, and ``dill.dump`` cannot serialise the ABC-metaclass-derived ``_abc._abc_data`` slot through that closure -- production saves of the MLP model failed with ``cannot pickle '_abc._abc_data' object``. Promoting the class to module level fixes that: same pickle path that works for any sklearn ``TransformedTargetRegressor`` instance now also works for this subclass.

Same idea as Pack F's ``PrePipelinePredictShim`` -- module-level classes are friends of dill / pickle.

Mirror the historical local class semantics exactly:
- Override ``fit`` to additionally scale any ``eval_set`` argument's y component through ``self.transformer_`` BEFORE the parent ``TransformedTargetRegressor.fit`` runs. Stock sklearn TTR scales only the ``y`` arg of fit; eval_set is left raw so PyTorch-Lightning / LightGBM that consume eval_set see RAW y_val while the model predicts scaled. The override closes that gap.
- ``eval_set`` is either ``(X_val, y_val)`` (MLP / LGB convention) or ``[(X_val, y_val), ...]`` (XGB / CB convention).
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.base import clone as _clone
from sklearn.compose import TransformedTargetRegressor

logger = logging.getLogger(__name__)


class _TTRWithEvalSetScaling(TransformedTargetRegressor):
    """``TransformedTargetRegressor`` extension that ALSO standardises any ``eval_set`` / ``X_val`` + ``y_val`` arrays in fit_params.

    Required for inner estimators (PyTorch-Lightning MLP, LightGBM, etc.) that consume eval_set for their own val-loss / early-stopping. Without this, train sees standardised y and val sees raw y, making the early-stop metric nonsensical.
    """

    def predict(self, X, **predict_params):
        # 2026-05-22 diagnostic: log the pre-inverse and post-inverse stats
        # of the FIRST predict call so a scale mismatch (chart predictions
        # in 650-1150 range while raw target is ~11500) immediately
        # surfaces the broken transformer_ instead of looking like an MLP
        # training bug. Cheap once-per-model log; bypasses if the inner
        # estimator's predict is recursive.
        if getattr(self, "_ttr_predict_diag_done", False):
            return super().predict(X, **predict_params)
        self._ttr_predict_diag_done = True
        try:
            raw_pred = self.regressor_.predict(X, **predict_params)
            raw_pred_arr = np.asarray(raw_pred, dtype=np.float64).ravel()
            pre_stats = (
                float(raw_pred_arr.mean()),
                float(raw_pred_arr.std()),
                float(raw_pred_arr.min()),
                float(raw_pred_arr.max()),
            )
            t = self.transformer_
            t_mean = getattr(t, "mean_", None)
            t_scale = getattr(t, "scale_", None)
            post_pred = super().predict(X, **predict_params)
            post_arr = np.asarray(post_pred, dtype=np.float64).ravel()
            post_stats = (
                float(post_arr.mean()),
                float(post_arr.std()),
                float(post_arr.min()),
                float(post_arr.max()),
            )
            logger.warning(
                "[_TTRWithEvalSetScaling-predict-diag] inner regressor_ pre-inverse: "
                "mean=%.4f std=%.4f range=[%.4f, %.4f]. transformer_=%s "
                "mean_=%s scale_=%s. After inverse_transform: mean=%.4f std=%.4f "
                "range=[%.4f, %.4f].",
                *pre_stats, type(t).__name__,
                t_mean.tolist() if hasattr(t_mean, "tolist") else t_mean,
                t_scale.tolist() if hasattr(t_scale, "tolist") else t_scale,
                *post_stats,
            )
            return post_pred
        except Exception as _diag_err:
            logger.warning("_TTRWithEvalSetScaling predict-diag failed: %s; falling back to plain super().predict.", _diag_err)
            return super().predict(X, **predict_params)

    def fit(self, X, y, **fit_params):
        # Fit the transformer FIRST on y so we have the same scale to apply to eval_set's y_val. Mirrors what ``TransformedTargetRegressor.fit`` does internally.
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr_2d = y_arr.reshape(-1, 1)
        else:
            y_arr_2d = y_arr
        self.transformer_ = _clone(self.transformer) if self.transformer is not None else None
        if self.transformer_ is not None:
            self.transformer_.fit(y_arr_2d)
            # Intercept + transform eval_set's y_val before delegating.
            if "eval_set" in fit_params and fit_params["eval_set"] is not None:
                es = fit_params["eval_set"]
                if isinstance(es, tuple) and len(es) == 2:
                    X_val, y_val = es
                    y_val_arr = np.asarray(y_val, dtype=np.float64)
                    y_val_2d = y_val_arr.reshape(-1, 1) if y_val_arr.ndim == 1 else y_val_arr
                    y_val_scaled = self.transformer_.transform(y_val_2d).reshape(y_val_arr.shape)
                    fit_params["eval_set"] = (X_val, y_val_scaled)
                elif isinstance(es, list):
                    new_es = []
                    for entry in es:
                        if isinstance(entry, tuple) and len(entry) == 2:
                            X_v, y_v = entry
                            y_v_arr = np.asarray(y_v, dtype=np.float64)
                            y_v_2d = y_v_arr.reshape(-1, 1) if y_v_arr.ndim == 1 else y_v_arr
                            y_v_scaled = self.transformer_.transform(y_v_2d).reshape(y_v_arr.shape)
                            new_es.append((X_v, y_v_scaled))
                        else:
                            new_es.append(entry)
                    fit_params["eval_set"] = new_es
        # Defer the actual fit to the parent (which will refit transformer + call regressor.fit).
        return super().fit(X, y, **fit_params)
