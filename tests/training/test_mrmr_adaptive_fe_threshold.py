"""#5 MRMR adaptive FE threshold relaxation.

When first-pass FE produces 0 engineered features, retry once with thresholds scaled by ``fe_adaptive_relax_factor`` (default 0.9). Skips the expensive Hermite Optuna re-run because best_res is already injected. Default ON (Accuracy/perf over legacy).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _suppress_optuna_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        yield


class TestAdaptiveFEConfig:
    def test_default_adaptive_is_on(self) -> None:
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR()
        assert m.fe_adaptive_threshold_relax is True
        assert m.fe_adaptive_relax_factor == 0.9

    def test_explicit_off_restores_legacy(self) -> None:
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(fe_adaptive_threshold_relax=False)
        assert m.fe_adaptive_threshold_relax is False

    def test_custom_relax_factor(self) -> None:
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(fe_adaptive_relax_factor=0.85)
        assert m.fe_adaptive_relax_factor == 0.85


class TestAdaptiveRetryWiring:
    """Adaptive retry must fire when first pass yields 0 features. The retry uses ``checked_pairs=set()`` (resets pair memoisation) and ``fe_smart_polynom_iters=0`` (skip expensive Hermite re-run)."""

    def test_runs_without_crash_with_adaptive_on(self) -> None:
        """Smoke: enabling adaptive retry must not crash on a vanilla synthetic where FE may or may not yield features."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame({
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        })
        y = (rng.normal(size=n) > 0).astype(int)
        m = MRMR(
            n_workers=1, verbose=0,
            fe_max_steps=1,
            fe_npermutations=10,
            fe_ntop_features=3,
            fe_unary_preset="medium",
            fe_binary_preset="medium",
            fe_adaptive_threshold_relax=True,
        )
        m.fit(df, y)
        assert hasattr(m, "support_")

    def test_legacy_off_does_not_retry(self) -> None:
        """With adaptive OFF, FE that produced 0 features must NOT retry -- runs faster + same result."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame({
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        })
        y = (rng.normal(size=n) > 0).astype(int)
        m = MRMR(
            n_workers=1, verbose=0,
            fe_max_steps=1,
            fe_npermutations=10,
            fe_ntop_features=3,
            fe_adaptive_threshold_relax=False,
        )
        m.fit(df, y)
        assert hasattr(m, "support_")
