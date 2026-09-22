"""Every registry transform listed alone in ``transforms`` is accepted, isolated and honoured by the discovery phase.

Listing a grouped transform made discovery raise and trained no composite for the target; ``transforms=["linear_residual"]``
still built a chain spec; and kept specs could name an engineered base that existed only in discovery's private frame.
Each transform here runs through ``run_composite_target_discovery`` alone, with a group column set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY
from mlframe.training.configs import CompositeTargetDiscoveryConfig, TargetTypes

# One transform per family runs every time; the rest of the registry runs outside fast mode under ``slow``.
_REPRESENTATIVES = {
    "linear_residual", "diff", "ratio", "log_y", "quantile_residual", "monotonic_residual", "linear_residual_grouped",
    "target_encoding_residual", "ewma_residual", "chain_linres_cbrt",
}
# Snapshot at collection: other tests register auto-chain transforms at runtime, so the live registry can grow afterwards.
_NAMES = sorted(TRANSFORMS_REGISTRY)
_PARAMS = [n if n in _REPRESENTATIVES else pytest.param(n, marks=pytest.mark.slow) for n in _NAMES]


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A positive base with a group-level shift, a noise feature, a group column and a target."""
    rng = np.random.default_rng(seed)
    g = np.arange(n) % 12
    base = rng.uniform(1.0, 10.0, n) + 0.2 * g
    return pd.DataFrame({"b": base, "x": rng.normal(size=n), "g": g, "y": 2.0 * base + rng.normal(0.0, 0.5, n)})


def _run(transform: str):
    """The discovery phase on the fixture with ``transforms=[transform]``: ``(metadata, frame columns)``."""
    from mlframe.training.core._phase_composite_discovery import run_composite_target_discovery

    df = _frame()
    feats = df.drop(columns=["y"])
    idx = np.arange(len(df))
    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, base_candidates=["b"], transforms=[transform], group_column="g",
                                         screening="mi", eps_mi_gain=-10.0, min_honest_gain_to_train=None)
    _tbt, metadata = run_composite_target_discovery(
        composite_target_discovery_config=cfg, target_by_type={TargetTypes.REGRESSION: {"y": df["y"].to_numpy()}}, mlframe_models=None,
        metadata={}, filtered_train_df=feats, filtered_train_idx=idx, train_df_pd=feats, val_df_pd=feats, test_df_pd=feats,
        train_idx=idx, val_idx=idx, test_idx=idx, baseline_diagnostics_config=None, cat_features=None, verbose=False,
    )
    return metadata, set(df.columns)


def test_every_registry_transform_is_covered():
    """The parametrisation spans the whole registry, and every representative is a registered name."""
    assert len(_PARAMS) == len(_NAMES) and set(_NAMES) <= set(TRANSFORMS_REGISTRY)
    assert _REPRESENTATIVES <= set(_NAMES)


@pytest.mark.parametrize("transform", _PARAMS)
def test_a_transform_listed_alone_is_accepted_isolated_and_honoured(transform: str):
    """(a) discovery completes without a phase failure, (b) only the listed family is exported, (c) every base column exists."""
    metadata, columns = _run(transform)
    failures = [f for by_t in metadata.get("composite_target_failures", {}).values() for fs in by_t.values() for f in fs]
    # A per-candidate problem is a candidate rejection; a failure recorded against the raw target itself means discovery aborted.
    assert not [f for f in failures if f.get("name") == "y" or "discovery fit raised" in str(f.get("reason", ""))], failures
    specs = [s for by_t in metadata.get("composite_target_specs", {}).values() for sl in by_t.values() for s in sl]
    wrong = [s["transform_name"] for s in specs if not str(s["transform_name"]).startswith(transform)]
    assert not wrong, f"transforms=[{transform!r}] exported {wrong}"
    missing = [(s["name"], c) for s in specs for c in [s.get("base_column"), *(s.get("extra_base_columns") or ())] if c and c not in columns]
    assert not missing, f"specs name base columns the frame does not have: {missing}"
