"""transform before fit raises sklearn's NotFittedError, which pipelines and fallbacks catch (tracker 2026-11-02 wave 14, SKL14-4, SKL14-5, SKL14-10)."""

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector
from mlframe.feature_selection.filters._stability_fe import StabilityFESelector
from mlframe.feature_selection.hybrid_selector import HybridSelector


@pytest.mark.parametrize("make", [lambda: HybridSelector(), lambda: CorrelatedFeaturesSelector(estimator=object()), lambda: StabilityFESelector()])
def test_transform_before_fit(make):
    """AttributeError and RuntimeError used to escape; the catchable sklearn error is required."""
    with pytest.raises(NotFittedError):
        make().transform(pd.DataFrame({"a": [1.0, 2.0]}))
