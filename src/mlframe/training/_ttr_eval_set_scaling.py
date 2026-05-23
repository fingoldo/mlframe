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

    def predict(self, X, **predict_params):
        """Predict + diagnostic check on the scaled-space prediction range.

        The parent ``TransformedTargetRegressor.predict`` runs the inner
        regressor to get T_hat (scaled space) and then applies
        ``transformer_.inverse_transform`` to recover y_hat. For a
        StandardScaler-wrapped target, ``inverse_transform`` is the
        linear map ``T_hat * scale_ + mean_`` -- no clipping, no sanity
        check. If the inner regressor is an Identity-MLP / unbounded
        linear model and the test rows are OOD, T_hat can land at e.g.
        -17 sigma and inverse_transform faithfully maps that to a
        completely wrong y_hat (prod TVT 2026-05-22 incident).

        Detect the -17 sigma signal IN SCALED SPACE before
        ``inverse_transform`` obscures it. WARN-log on |T_hat|.max() > 10
        so operators see the scaled-space outlier directly.
        """
        if self.transformer_ is None:
            return super().predict(X, **predict_params)
        # iter191 (2026-05-23): inline the parent TransformedTargetRegressor.predict
        # path so we predict ONCE (not twice). The previous form called
        # self.regressor_.predict(X) for the sensor probe then super().predict(X)
        # for the actual return -- both invoke the inner regressor on the full X,
        # doubling predict wall time. c0115 profile attributed 1.222s to TTR
        # predict over 2 calls (611ms each on 200k rows). Saving the second
        # invocation drops to ~611ms across both, ~600ms saved.
        try:
            t_hat = self.regressor_.predict(X, **predict_params)
        except TypeError:
            # Some regressors (sklearn linear models) reject **predict_params.
            t_hat = self.regressor_.predict(X)
        try:
            t_hat_arr = np.asarray(t_hat, dtype=np.float64).reshape(-1)
            if t_hat_arr.size:
                _abs_max = float(np.max(np.abs(t_hat_arr)))
                if _abs_max > 10.0:
                    logger.warning(
                        "[ttr-scaled-extrapolation-sensor] inner regressor "
                        "%s emitted |T_hat|.max()=%.2f sigma in scaled "
                        "target space (>10 sigma is unphysical for "
                        "z-scored y). inverse_transform will faithfully "
                        "map this to a far-OOD y_hat. Likely cause: "
                        "unbounded-output downstream model (Identity-MLP, "
                        "plain LinearRegression) extrapolating on test "
                        "rows whose feature distribution differs from "
                        "train (prod TVT 2026-05-22).",
                        type(self.regressor_).__name__, _abs_max,
                    )
        except Exception as _sensor_err:
            logger.debug(
                "ttr-scaled-extrapolation-sensor probe failed (non-fatal): %s",
                _sensor_err,
            )
        # Inline the parent's inverse_transform path so we don't re-predict.
        # Mirrors sklearn.compose.TransformedTargetRegressor.predict bit-for-bit.
        t_hat_np = np.asarray(t_hat)
        if t_hat_np.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(t_hat_np.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(t_hat_np)
        if pred_trans.ndim == 2 and pred_trans.shape[1] == 1:
            pred_trans = pred_trans.squeeze(axis=1)
        return pred_trans
