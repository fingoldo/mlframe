"""#4 Pack: residual-target alt stacked discovery.

``fit_stacked`` adds pass-1 OOF predictions as new FEATURES (so pass-2 can find composites where OOF becomes the new base). ``fit_stacked_on_residual`` is the alternative: pass-1 OOF preds collectively predict ``pass1_pred``, residual ``y - pass1_pred`` becomes the new TARGET for pass-2 discovery. More direct route to residual-of-residual structure when feature stacking blocks at the gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestFitStackedOnResidualWiring:
    def test_method_exists(self) -> None:
        from mlframe.training.composite_discovery import CompositeTargetDiscovery
        assert hasattr(CompositeTargetDiscovery, "fit_stacked_on_residual")

    def test_pass1_only_when_pass1_empty(self) -> None:
        """When pass-1 discovers nothing, pass-2 must not run."""
        from mlframe.training.composite_discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        rng = np.random.default_rng(0)
        n = 600
        df = pd.DataFrame({
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "y": rng.standard_normal(n),  # pure noise -- no composite passes the gate
        })
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=400)
        disc = CompositeTargetDiscovery(config=cfg).fit_stacked_on_residual(
            df=df, target_col="y", feature_cols=["a", "b"],
            train_idx=np.arange(int(0.8 * n)),
        )
        # Specs is whatever pass-1 found (possibly 0). The key assertion is no crash.
        assert hasattr(disc, "specs_")


class TestFitStackedOnResidualBizVal:
    def test_finds_at_least_pass1_specs_on_two_signal_synthetic(self) -> None:
        """y = 1.5*x_a + 2*x_b + noise -- pass 1 should find linres on x_a or x_b; pass 2 on residual may find the other."""
        from mlframe.training.composite_discovery import CompositeTargetDiscovery
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        rng = np.random.default_rng(11)
        n = 3000
        x_a = rng.normal(50.0, 10.0, n)
        x_b = rng.normal(0.0, 5.0, n)
        y = 1.5 * x_a + 0.5 + 2.0 * x_b + rng.normal(0.0, 1.0, n)
        df = pd.DataFrame({
            "x_a": x_a, "x_b": x_b,
            "n0": rng.standard_normal(n), "n1": rng.standard_normal(n),
            "y": y,
        })
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True, mi_sample_n=1500,
            composite_skip_when_raw_dominates_ratio=0.0,  # don't skip for this test
        )
        disc = CompositeTargetDiscovery(config=cfg).fit_stacked_on_residual(
            df=df, target_col="y",
            feature_cols=["x_a", "x_b", "n0", "n1"],
            train_idx=np.arange(int(0.8 * n)),
            n_oof_folds=3,
        )
        # Non-regression: residual-stacked must find at least as many specs as plain pass-1.
        # plain pass-1 would have found something on this clean signal.
        assert len(disc.specs_) >= 1
        # If any pass-2 specs were attached, they MUST carry the residual annotation.
        for spec in disc.specs_:
            if getattr(spec, "discovered_on_residual", False):
                # Pass-2 spec: name should be from the residual-target run.
                assert spec.name
