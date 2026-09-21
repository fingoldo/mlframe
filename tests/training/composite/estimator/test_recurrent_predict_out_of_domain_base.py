"""A recurrent inverse must not let one out-of-domain base row move the rows around it.

``predict`` inverts over the whole base sequence for a recurrent transform, so whatever stands in for a NaN base is carried
through the recurrence into every LATER row. Those rows are returned as real predictions (only the flagged row itself is
replaced by the fallback), so a placeholder unrelated to the data silently biases them. The contract here is the one fit already
keeps for its own dropped rows: carry the last valid base forward, which makes the surviving rows equal to what a caller gets by
gap-filling the base themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY

_RECURRENT = sorted(n for n, t in TRANSFORMS_REGISTRY.items() if getattr(t, "recurrent", False) and t.requires_base and not t.requires_groups)


def _frame(n: int = 240) -> tuple[pd.DataFrame, np.ndarray]:
    """A smooth, strictly positive base far from 1.0 plus one informative feature, so a 1.0 stand-in is visibly wrong."""
    rng = np.random.default_rng(3)
    t = np.arange(n, dtype=np.float64)
    base = 1000.0 + 40.0 * np.sin(t / 11.0) + rng.normal(scale=2.0, size=n)
    f1 = rng.normal(size=n)
    y = 0.85 * base + 3.0 * f1 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"base": base, "f1": f1}), y


@pytest.mark.parametrize("name", _RECURRENT)
def test_one_nan_base_row_leaves_the_other_rows_where_carry_forward_puts_them(name: str) -> None:
    """Blanking one base row must move only that row: the rest must match a carry-forward-filled base exactly."""
    X, y = _frame()
    # An inner that takes NaN natively: the contract under test is the recurrence, not how the inner imputes a hole.
    est = CompositeTargetEstimator(base_estimator=HistGradientBoostingRegressor(max_iter=30, random_state=0), transform_name=name, base_column="base")
    est.fit(X, y)

    hole = 120
    holed = X.copy()
    holed.loc[hole, "base"] = np.nan
    filled = X.copy()
    filled.loc[hole, "base"] = X["base"].iloc[hole - 1]  # what carry-forward substitutes

    got = np.asarray(est.predict(holed), dtype=np.float64)
    oracle = np.asarray(est.predict(filled), dtype=np.float64)

    others = np.ones(len(X), dtype=bool)
    others[hole] = False
    np.testing.assert_allclose(
        got[others],
        oracle[others],
        rtol=1e-9,
        atol=1e-9,
        err_msg=f"{name}: rows other than the blanked one moved, so the placeholder base leaked into the recurrence",
    )
