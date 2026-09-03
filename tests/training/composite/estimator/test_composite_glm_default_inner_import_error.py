"""Regression test for TRAINING_COMPOSITE_CORE_B-8: ``_default_inner``'s lightgbm import guard.

A broad ``except Exception`` around ``import lightgbm`` relabeled ANY failure (e.g. a broken
native DLL or a version-incompatible binary) as "lightgbm not installed", masking the real
error. The guard must catch only ``ImportError``.
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from mlframe.training.composite import glm as glm_module


def test_default_inner_non_import_failure_not_relabeled_as_missing_lightgbm() -> None:
    """A non-ImportError raised while importing lightgbm must propagate as-is."""
    real_import = builtins.__import__

    def _broken_import(name, *args, **kwargs):
        """Raise a non-ImportError for 'lightgbm', delegate everything else to the real import."""
        if name == "lightgbm":
            raise RuntimeError("broken native DLL")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_broken_import):
        with pytest.raises(RuntimeError, match="broken native DLL"):
            glm_module._default_inner(family="poisson", tweedie_power=1.5)


def test_default_inner_missing_lightgbm_raises_clear_import_error() -> None:
    """A genuine ImportError still raises the clear, actionable ImportError message."""
    real_import = builtins.__import__

    def _missing_import(name, *args, **kwargs):
        """Raise a genuine ImportError for 'lightgbm', delegate everything else to the real import."""
        if name == "lightgbm":
            raise ImportError("No module named 'lightgbm'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_missing_import):
        with pytest.raises(ImportError, match="CompositeGLMEstimator default inner requires lightgbm"):
            glm_module._default_inner(family="poisson", tweedie_power=1.5)


def test_default_inner_pins_random_state() -> None:
    """TRAINING_COMPOSITE_CORE_B-6: _default_inner must pin random_state=0, matching the sibling
    default-builder highlevel.py's _default_inner_estimator -- otherwise reproducibility is an implicit,
    undocumented invariant of LightGBM's own unset default rather than an explicit contract."""
    pytest.importorskip("lightgbm")
    model = glm_module._default_inner(family="poisson", tweedie_power=1.5)
    assert model.get_params()["random_state"] == 0
