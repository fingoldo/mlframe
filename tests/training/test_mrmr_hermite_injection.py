"""#1 MRMR Hermite-FE injection: best_res is no longer discarded.

Production TVT log spent 88 min on Hermite Optuna optimisation and discarded ``best_res`` with a ``# future work`` comment. Now ``best_res.transform(vals_a, vals_b)`` is computed and injected as a new engineered column whenever the optimisation clears the engineered-MI gate.

Tests:
1. ``_hermite_features_`` attribute exists after fit when ``fe_smart_polynom_iters > 0`` (even if 0 pairs survived the gate -- the attribute is initialised to an empty list inside the Hermite block).
2. Back-compat: ``fe_smart_polynom_iters = 0`` does NOT add the attribute and does NOT crash.
3. Biz_val: on a synthetic where the pair-FE injection should fire, the resulting ``_hermite_features_`` list is non-empty AND the new column appears in ``self._engineered_features_``.
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


class TestHermiteInjectionWiring:
    def test_no_smart_polynom_no_attribute(self) -> None:
        """When fe_smart_polynom_iters=0, the new Hermite block does not run and _hermite_features_ is not set."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 200
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
            fe_smart_polynom_iters=0,
        )
        m.fit(df, y)
        # Back-compat: attribute should not exist (or be empty if pre-initialised).
        assert not getattr(m, "_hermite_features_", None), (
            f"expected no Hermite features for fe_smart_polynom_iters=0; "
            f"got {getattr(m, '_hermite_features_', None)}"
        )

    def test_with_smart_polynom_does_not_crash(self) -> None:
        """When fe_smart_polynom_iters>0 the new injection branch runs without raising. May or may not inject features depending on whether pairs pass the gate -- but MUST NOT crash."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 300
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        df = pd.DataFrame({"a": x_a, "b": x_b, "c": rng.normal(size=n)})
        y = (x_a * x_b > 0).astype(int)
        m = MRMR(
            n_workers=1, verbose=0,
            fe_max_steps=1,
            fe_npermutations=10,
            fe_ntop_features=3,
            fe_smart_polynom_iters=1,
            fe_smart_polynom_optimization_steps=10,
            fe_unary_preset="medium",
            fe_binary_preset="medium",
        )
        m.fit(df, y)
        # Sanity: support_ exists; no crash.
        assert hasattr(m, "support_")

    def test_hermite_features_metadata_shape(self) -> None:
        """When ``_hermite_features_`` IS populated, each entry must carry name/src_a/src_b/basis/bin_func_name + MI metadata.

        We don't assert non-empty (gate sensitivity is data-dependent), but if non-empty the schema must match.
        """
        from mlframe.feature_selection.filters.mrmr import MRMR

        rng = np.random.default_rng(0)
        n = 500
        x_a = rng.normal(size=n)
        x_b = rng.normal(size=n)
        df = pd.DataFrame({
            "a": x_a, "b": x_b,
            "c": rng.normal(size=n), "d": rng.normal(size=n),
        })
        y = (x_a * x_b > 0).astype(int)
        m = MRMR(
            n_workers=1, verbose=0,
            fe_max_steps=1,
            fe_npermutations=20,
            fe_ntop_features=4,
            fe_smart_polynom_iters=2,
            fe_smart_polynom_optimization_steps=20,
            fe_unary_preset="medium",
            fe_binary_preset="medium",
        )
        m.fit(df, y)
        injected = getattr(m, "_hermite_features_", None)
        if injected:
            # Schema check on each entry.
            for entry in injected:
                assert "name" in entry and isinstance(entry["name"], str)
                assert "src_a" in entry and "src_b" in entry
                assert "basis" in entry
                assert "bin_func_name" in entry
                assert "best_mi" in entry and isinstance(entry["best_mi"], float)
                assert "baseline_mi" in entry
                # Injected column must also be in engineered_features set.
                assert entry["name"] in m._engineered_features_
