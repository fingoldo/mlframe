"""Audit 2026-11-20 findings 26-F30 and 26-F7.

26-F30: a diverged fold (NaN) does not make its round the aggregate best. The NaN-aware argmax skips it, and an all-NaN curve raises.
26-F7: the fractional-digit probe refuses max_fract_digits below 2 with a ValueError, not an UnboundLocalError.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.preprocessing.cleaning import is_variable_truly_continuous
from mlframe.training._aggregate_cv_early_stopping import select_best_iteration_by_aggregate_cv


def test_diverged_fold_does_not_win_the_best_round():
    """Fold 2 is NaN at round 2; the best aggregate round stays at round 1, not the NaN round."""
    curves = np.array([[0.5, 0.6, 0.7], [0.5, 0.6, np.nan]])
    out = select_best_iteration_by_aggregate_cv(curves, maximize=True)
    assert out["best_round"] == 1


def test_all_nan_curve_raises():
    """A curve with no finite round has no best round, and says so."""
    with pytest.raises(ValueError, match="NaN"):
        select_best_iteration_by_aggregate_cv(np.full((2, 3), np.nan), maximize=True)


def test_fraction_probe_refuses_too_few_digits():
    """max_fract_digits=1 gives a ValueError naming the minimum, not an UnboundLocalError."""
    df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 50)})
    with pytest.raises(ValueError, match="at least 2"):
        is_variable_truly_continuous(df=df, variable_name="x", max_fract_digits=1)
