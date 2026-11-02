"""The oracle baseline of diagnose_error_accumulation advances the other per-step features exactly like the recursive run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.inference.recursive_forecast import diagnose_error_accumulation


class _CounterModel:
    """Predicts the 'age' counter, so a frozen counter gives a wrong oracle."""

    def predict(self, X):
        """Return the age column."""
        return X["age"].to_numpy(dtype=float)


def _advance(features, _pred, _step):
    """Increment the age counter, as a caller's update hook would."""
    out = features.copy(deep=False)
    out["age"] = out["age"] + 1.0
    return out


def test_oracle_error_is_zero_when_the_model_depends_only_on_the_updated_counter():
    """Targets equal the advancing counter (1, 2, 3); a model reading it is exact, so the oracle MSE must be 0 at every step."""
    X = pd.DataFrame({"lag": [0.0, 0.0], "age": [1.0, 1.0]})
    targets = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    res = diagnose_error_accumulation(_CounterModel(), X, 3, "lag", _advance, targets, oracle_lag_values=np.zeros((3, 2)))
    np.testing.assert_allclose(res["oracle_mse"], 0.0, atol=1e-12)
