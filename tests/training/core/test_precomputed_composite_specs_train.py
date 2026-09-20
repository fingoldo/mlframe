"""Feeding a prior run's composite specs back in must train those targets, not just record them.

The documented reuse path (``bundle.composite_target_specs = prior_run_metadata["composite_target_specs"]``) skipped the
discovery phase and seeded ``metadata`` only. Nothing then built the composite target columns, so ``target_by_type``
never gained an entry and the run trained no composite target at all: the fast path silently discarded the feature it
exists to speed up. The specs are now replayed through discovery, the same way a cache hit is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig, TargetTypes
from mlframe.training.core._phase_composite_discovery import run_composite_target_discovery


def _frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame whose target is an additive residual on a real base column."""
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=50.0, scale=5.0, size=n)
    f1 = rng.normal(size=n)
    y = base + 2.0 * f1 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"base": base, "f1": f1, "y": y}), y


def _spec_export(name: str = "y__linear_residual__base") -> dict:
    """A minimal exported spec of the shape a prior run's metadata carries."""
    return {
        "name": name,
        "target_col": "y",
        "transform_name": "linear_residual",
        "base_column": "base",
        "fitted_params": {"alpha": 1.0, "beta": 0.0},
        "mi_gain": 0.5,
        "mi_y": 0.2,
        "mi_t": 0.7,
        "valid_domain_frac": 1.0,
        "n_train_rows": 400,
    }


def _run(precomputed_specs, *, enabled: bool):
    """Drive the discovery phase with (or without) caller-supplied specs and return its target_by_type."""
    df, y = _frame()
    target_by_type = {TargetTypes.REGRESSION: {"y": y}}
    metadata: dict = {}
    idx = np.arange(len(df))
    out, _meta = run_composite_target_discovery(
        composite_target_discovery_config=CompositeTargetDiscoveryConfig(enabled=enabled, base_candidates=["base"]),
        target_by_type=target_by_type, mlframe_models=["lightgbm"], metadata=metadata,
        filtered_train_df=df, filtered_train_idx=idx,
        train_df_pd=df, val_df_pd=None, test_df_pd=None,
        train_idx=idx, val_idx=None, test_idx=None,
        baseline_diagnostics_config=None, cat_features=None, verbose=False,
        precomputed_specs=precomputed_specs,
    )
    return out, _meta


def test_caller_supplied_specs_add_the_composite_target():
    """The replayed spec becomes a real entry in ``target_by_type``, which is what the trainer iterates."""
    specs = {str(TargetTypes.REGRESSION): {"y": [_spec_export()]}}
    out, meta = _run(specs, enabled=True)
    names = list(out[TargetTypes.REGRESSION])
    assert any(n != "y" for n in names), f"the replayed spec must add a composite target; got {names}"
    assert meta["composite_target_specs"][str(TargetTypes.REGRESSION)]["y"], "the specs must stay recorded in metadata"


def test_specs_are_replayed_even_when_discovery_itself_is_off():
    """Reuse means skipping the search, not the training: the specs run with ``enabled=False``."""
    specs = {str(TargetTypes.REGRESSION): {"y": [_spec_export()]}}
    out, _meta = _run(specs, enabled=False)
    assert any(n != "y" for n in list(out[TargetTypes.REGRESSION])), "caller-supplied specs must train with discovery off"


def test_without_specs_a_disabled_discovery_still_does_nothing():
    """The disabled path is unchanged when no specs are supplied."""
    out, _meta = _run(None, enabled=False)
    assert list(out[TargetTypes.REGRESSION]) == ["y"]


def test_the_values_of_the_added_target_are_the_transform_applied_to_y():
    """The added column is the spec's T, so the trainer fits the residual the spec describes."""
    specs = {str(TargetTypes.REGRESSION): {"y": [_spec_export()]}}
    out, _meta = _run(specs, enabled=True)
    added = [n for n in out[TargetTypes.REGRESSION] if n != "y"]
    assert added, "no composite target was added"
    values = np.asarray(out[TargetTypes.REGRESSION][added[0]], dtype=np.float64)
    df, y = _frame()
    expected = y - (1.0 * df["base"].to_numpy() + 0.0)  # alpha=1, beta=0
    finite = np.isfinite(values) & np.isfinite(expected)
    assert finite.sum() > 0
    assert values[finite] == pytest.approx(expected[finite], rel=1e-6, abs=1e-6)
