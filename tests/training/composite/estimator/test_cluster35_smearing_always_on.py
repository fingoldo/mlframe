"""Audit 2026-11-20 finding 35-F53: a smeared transform always stores its residual quantiles.

The gate that could have turned smearing off read an attribute no constructor sets, so it was always on; the code now says so.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator


def test_log_transform_fit_stores_smearing_quantiles():
    """A log_y fit keeps non-empty residual quantiles in fitted_params_['smearing_quantiles']."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"f": rng.normal(size=n), "b": rng.uniform(1.0, 2.0, size=n)})
    y = np.exp(0.5 * X["f"].to_numpy() + 0.1 * rng.normal(size=n)) + 0.1
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="log_y", base_column="b")
    est.fit(X, y)
    q = est.fitted_params_["smearing_quantiles"]
    assert q is not None and len(q) > 0
