"""The screen's FDR null floor defaults to the 0.998 quantile.

At the former 0.975 default a pure-noise column (uniform inputs, binary target, MDLP binning) cleared the floor on 2000-row draws
(family-wise p about 0.002 under the honest null). The stricter default removes that selection without moving recall on the
weak-signal and null-FDR signal fixtures.
"""

import inspect

from mlframe.feature_selection.filters._screen_predictors import screen_predictors


def test_default_screen_fdr_null_quantile_is_the_strict_level():
    """Calling ``screen_predictors`` without the quantile argument uses the 0.998 null quantile."""
    default = inspect.signature(screen_predictors).parameters["screen_fdr_null_quantile"].default
    assert default == 0.998
