"""robust_float_ensemble ignores a NaN member instead of returning 0.0, and the default AUC metric refuses a non-binary target with a 1-D blend."""
import numpy as np
import pytest

from mlframe.models.ensembling.float_aggregation import robust_float_ensemble
from mlframe.models.ensembling.selection import _score_blend


def test_nan_member_does_not_zero_the_prediction():
    """[1, 2, nan]: the column median went NaN, nothing was kept, and the masked sum 0.0 / K came back as 0.0."""
    out = robust_float_ensemble(np.array([[1.0], [2.0], [np.nan]]))
    assert np.allclose(out, 1.5)


def test_all_nan_column_stays_nan():
    """No finite member: NaN, not a made-up number."""
    out = robust_float_ensemble(np.array([[np.nan], [np.nan], [np.nan]]))
    assert np.isnan(out).all()


def test_default_metric_rejects_a_continuous_target_with_a_one_dimensional_blend():
    """A regression target without an explicit metric used to be cast to int and scored by a 0/1 AUC kernel."""
    y = np.random.default_rng(0).standard_normal(50)
    with pytest.raises(ValueError, match="binary"):
        _score_blend(np.random.default_rng(1).random(50), y, None)
