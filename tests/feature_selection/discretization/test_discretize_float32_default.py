"""Regression: ``categorize_dataset``'s dense discretisation-input copy defaults to float32.

Pre-fix, ``_discretize_input_dtype()`` defaulted to float64 and float32 was opt-in only via
``MLFRAME_DISCRETIZE_FLOAT32=1``. This is the dominant term of the large-n FE memory peak (a 1M-row
fit projects to ~21GB, OOMing a 16GB box -- see ``tests/feature_selection/MRMR_FE_PERF_NOTES.md``
2026-06-17 entry, which measured the ~21GB->~10GB cut with selection IDENTICAL float32-vs-float64 on
the canonical 60k fit). Per the "enable corrective mechanisms by default" convention, an
already-validated corrective mechanism must default ON. This module pins: (a) the new default is
float32, (b) ``MLFRAME_DISCRETIZE_FLOAT32=0`` still forces the legacy float64 path, and (c) MRMR
selection is IDENTICAL between the float64-forced and float32-default paths on a synthetic dataset
with real signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization._discretization_dataset import (
    _discretize_input_dtype,
)
from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(autouse=True)
def _clear_discretize_env(monkeypatch):
    """Clear discretize env."""
    monkeypatch.delenv("MLFRAME_DISCRETIZE_FLOAT32", raising=False)


def test_default_dtype_is_float32():
    """Default dtype is float32."""
    assert _discretize_input_dtype() is np.float32


def test_env_zero_forces_float64():
    """Env zero forces float64."""
    import os

    os.environ["MLFRAME_DISCRETIZE_FLOAT32"] = "0"
    try:
        assert _discretize_input_dtype() is np.float64
    finally:
        del os.environ["MLFRAME_DISCRETIZE_FLOAT32"]


@pytest.mark.parametrize("env_value", ["false", "False"])
def test_env_false_variants_force_float64(env_value):
    """Env false variants force float64."""
    import os

    os.environ["MLFRAME_DISCRETIZE_FLOAT32"] = env_value
    try:
        assert _discretize_input_dtype() is np.float64
    finally:
        del os.environ["MLFRAME_DISCRETIZE_FLOAT32"]


def test_env_one_still_selects_float32():
    """The legacy opt-in spelling must keep working (now a no-op since it's already the default)."""
    import os

    os.environ["MLFRAME_DISCRETIZE_FLOAT32"] = "1"
    try:
        assert _discretize_input_dtype() is np.float32
    finally:
        del os.environ["MLFRAME_DISCRETIZE_FLOAT32"]


def _make_signal_dataset(n: int = 4000, seed: int = 123):
    """Make signal dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "informative_1": rng.normal(size=n),
            "informative_2": rng.normal(size=n),
            "informative_3": rng.normal(size=n),
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
        }
    )
    y = 2.0 * X["informative_1"] - 1.5 * X["informative_2"] + 0.8 * X["informative_3"] + rng.normal(scale=0.05, size=n)
    return X, y


def test_float32_default_selection_matches_float64_forced():
    """Fit MRMR twice with the identical seed/config: once forcing float64 via the env override
    (the pre-fix default), once at the new float32 default. ``support_`` must be identical."""
    import os

    X, y = _make_signal_dataset()

    os.environ["MLFRAME_DISCRETIZE_FLOAT32"] = "0"
    try:
        m64 = MRMR(n_jobs=1, n_workers=1, random_seed=13, verbose=0, fe_max_steps=0)
        m64.fit(X, y)
    finally:
        del os.environ["MLFRAME_DISCRETIZE_FLOAT32"]

    # Default path (float32) -- env left unset by the autouse fixture.
    m32 = MRMR(n_jobs=1, n_workers=1, random_seed=13, verbose=0, fe_max_steps=0)
    m32.fit(X, y)

    assert list(m64.support_) == list(
        m32.support_
    ), f"selection diverged between float64-forced and float32-default: {np.asarray(X.columns)[m64.support_]} vs {np.asarray(X.columns)[m32.support_]}"
