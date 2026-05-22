"""Module-level ``TransformedTargetRegressor`` subclass that scales ``eval_set``.

Pre-fix this class was defined as a LOCAL class inside ``_configure_neural_params`` in ``trainer.py``. Local classes carry a hidden closure reference to the enclosing function's namespace, and ``dill.dump`` cannot serialise the ABC-metaclass-derived ``_abc._abc_data`` slot through that closure -- production saves of the MLP model failed with ``cannot pickle '_abc._abc_data' object``. Promoting the class to module level fixes that: same pickle path that works for any sklearn ``TransformedTargetRegressor`` instance now also works for this subclass.

Same idea as Pack F's ``PrePipelinePredictShim`` -- module-level classes are friends of dill / pickle.

Mirror the historical local class semantics exactly:
- Override ``fit`` to additionally scale any ``eval_set`` argument's y component through ``self.transformer_`` BEFORE the parent ``TransformedTargetRegressor.fit`` runs. Stock sklearn TTR scales only the ``y`` arg of fit; eval_set is left raw so PyTorch-Lightning / LightGBM that consume eval_set see RAW y_val while the model predicts scaled. The override closes that gap.
- ``eval_set`` is either ``(X_val, y_val)`` (MLP / LGB convention) or ``[(X_val, y_val), ...]`` (XGB / CB convention).
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone as _clone
from sklearn.compose import TransformedTargetRegressor


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
