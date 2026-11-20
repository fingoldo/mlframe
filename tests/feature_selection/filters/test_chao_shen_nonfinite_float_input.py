"""Float input holding NaN/inf must raise instead of silently returning a wrong 0.0."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._chao_shen import chao_shen_entropy, chao_shen_mi


def test_entropy_rejects_nan_float_input() -> None:
    """A NaN in a float sample makes every quantile edge NaN, so the call must raise."""
    x = np.array([0.1, 0.5, np.nan, 0.9, 0.3])
    with pytest.raises(ValueError, match="NaN/inf"):
        chao_shen_entropy(x)


def test_mi_rejects_nan_in_either_argument() -> None:
    """Both the binned feature and the target are checked."""
    good = np.array([0, 1, 0, 1, 1, 0])
    bad = np.array([0.0, 1.0, np.inf, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="x_binned"):
        chao_shen_mi(bad, good)
    with pytest.raises(ValueError, match="y"):
        chao_shen_mi(good, bad)


def test_finite_float_input_still_works() -> None:
    """Finite floats are quantile-binned as before."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert chao_shen_entropy(x) > 0.0
