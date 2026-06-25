"""Regression: an MRMR pickle produced BEFORE a constructor param existed must re-FIT
without AttributeError. ``__setstate__`` injected defaults only for the hand-maintained
legacy roster (a subset of the ~460 ctor params); the rest were never injected, and the
fit path reads many via bare ``self.<param>`` (e.g. ``self.dtype`` at _fit_impl_core), so a
resurrected old pickle raised ``AttributeError`` on the next fit. The fix overlays every
ctor default the roster did not cover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

# Bare-``self.<param>`` reads on the fit path (not guarded by getattr); absent from the
# legacy roster, so a pickle predating them would crash the re-fit pre-fix.
_PARAMS_PREDATING_ROSTER = ("dtype", "extra_x_shuffling", "fe_max_steps", "min_occupancy", "baseline_npermutations")


def _tiny_xy(n=400):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = ((X["a"].to_numpy() + rng.normal(0, 0.1, n)) > 0).astype(int)
    return X, y


def test_legacy_pickle_missing_ctor_params_reinjected():
    """``__setstate__`` injects ctor defaults for params absent from the legacy roster."""
    state = dict(MRMR(max_runtime_mins=1.0).__dict__)
    for k in _PARAMS_PREDATING_ROSTER:
        state.pop(k, None)
    m = MRMR.__new__(MRMR)
    m.__setstate__(state)
    for k in _PARAMS_PREDATING_ROSTER:
        assert hasattr(m, k), f"__setstate__ did not re-inject ctor default for {k!r}"


def test_legacy_pickle_refits_without_attributeerror():
    """A resurrected pre-param pickle fits cleanly (pre-fix: AttributeError: 'dtype')."""
    X, y = _tiny_xy()
    state = dict(MRMR(max_runtime_mins=1.0).__dict__)
    for k in _PARAMS_PREDATING_ROSTER:
        state.pop(k, None)
    m = MRMR.__new__(MRMR)
    m.__setstate__(state)
    m.fit(X, y)
    assert hasattr(m, "support_")


def test_legacy_override_keys_keep_legacy_value():
    """The fix must NOT clobber the intentional legacy-pickle overrides (e.g. the identity-cache
    y-corr gate stays at the legacy 0.0 for an attribute-less pickle, not the live ctor 0.5)."""
    state = dict(MRMR().__dict__)
    state.pop("mrmr_identity_cache_ycorr_threshold", None)
    m = MRMR.__new__(MRMR)
    m.__setstate__(state)
    assert m.mrmr_identity_cache_ycorr_threshold == 0.0
