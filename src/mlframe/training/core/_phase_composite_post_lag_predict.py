"""``_LagPredictDeployableModel`` wrapper for the cross-target ensemble's lag-predict component.

Carved out of ``_phase_composite_post.py`` to keep the parent below the 1k-line monolith threshold. The parent re-exports the class so existing imports continue to resolve.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class _LagPredictDeployableModel:
    """Wraps the ``lag_predict`` dummy baseline as a deployable model.

    Production TVT 2026-05-23: lag_predict beat every trained component
    on TEST RMSE (11.58 vs ensemble's 12.73). The dummy was visible in
    the dummy-baselines table but invisible to CT_ENSEMBLE's component
    pool -- final delivery used the worse stacker output. This wrapper
    presents lag_predict via the same ``predict(X) -> ndarray`` shape
    every other CT_ENSEMBLE component exposes, so the existing
    honest-OOF gate naturally selects it when it dominates.

    Prediction rule: ``y_hat[i] = X[lag_column].iloc[i]`` -- zero
    trainable parameters, returns the lag-target value verbatim per
    row. Inference cost is one column access; never extrapolates.

    Implements ``get_params``/``set_params``/``fit`` so ``sklearn.clone``
    accepts it during honest-OOF refit (CompositeCrossTargetEnsemble path,
    2026-05-23 prod incident: clone failed -> component dropped -> NNLS
    weights missed lag_predict and ensemble landed at RMSE 13.30 vs
    lag_predict's 11.58 floor).
    """

    def __init__(self, lag_column: str) -> None:
        self.lag_column = str(lag_column)

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002 - sklearn API
        return {"lag_column": self.lag_column}

    def set_params(self, **params: Any) -> "_LagPredictDeployableModel":
        for k, v in params.items():
            if k != "lag_column":
                raise ValueError(
                    f"_LagPredictDeployableModel has no parameter {k!r}; "
                    f"valid: ['lag_column']"
                )
            self.lag_column = str(v)
        return self

    def fit(self, X: Any, y: Any = None, **fit_params: Any) -> "_LagPredictDeployableModel":  # noqa: ARG002
        return self

    def predict(self, X: Any) -> np.ndarray:
        # polars get_column is 1-D zero-copy for numerics; sidesteps the prior select().to_numpy().reshape that built a (N,1) frame first.
        if hasattr(X, "get_column"):
            try:
                col = X.get_column(self.lag_column).to_numpy()
                return col.astype(np.float64, copy=False).reshape(-1)
            except Exception:
                pass
        if hasattr(X, "loc") or hasattr(X, "__getitem__"):
            try:
                col = X[self.lag_column]
                if hasattr(col, "to_numpy"):
                    return col.to_numpy().astype(np.float64, copy=False).reshape(-1)
                return np.asarray(col, dtype=np.float64).reshape(-1)
            except (KeyError, TypeError):
                pass
        raise KeyError(f"_LagPredictDeployableModel: column {self.lag_column!r} not found on X (type={type(X).__name__})")

    def __repr__(self) -> str:
        return f"_LagPredictDeployableModel(lag_column={self.lag_column!r})"
