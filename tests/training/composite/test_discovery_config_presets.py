"""Unit: ``CompositeTargetDiscoveryConfig.preset`` bundles + composition with ``suggest_discovery_config`` (G7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.autoconfig import suggest_discovery_config
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def test_preset_fast_bundle():
    """``preset("fast")`` sets its documented speed knobs while keeping every corrective gate on."""
    cfg = CompositeTargetDiscoveryConfig.preset("fast")
    assert cfg.enabled is True
    assert cfg.screening == "mi"
    assert cfg.mi_sample_n == 20_000
    assert cfg.max_base_candidates == 2 and cfg.auto_base_top_k == 2
    assert cfg.tiny_model_n_seed_repeats == 1
    assert cfg.multi_base_enabled is False
    assert cfg.auto_chain_discovery_enabled is False and cfg.interaction_base_discovery_enabled is False
    # Speed presets never trade safety: corrective gates stay ON.
    assert cfg.honest_rmse_gate_enabled is True
    assert cfg.yscale_holdout_gate_enabled is True
    assert cfg.structural_fragility_gate_enabled is True
    assert cfg.reject_on_alpha_drift is True


def test_preset_thorough_bundle():
    """``preset("thorough")`` sets its documented quality knobs (wider screen, seed repeats, bootstrap CI)."""
    cfg = CompositeTargetDiscoveryConfig.preset("thorough")
    assert cfg.screening == "hybrid"
    assert cfg.mi_sample_n == 200_000
    assert cfg.tiny_model_n_seed_repeats == 5
    assert cfg.mi_gain_bootstrap_n == 50 and cfg.use_wilcoxon_gate is True
    assert cfg.auto_base_null_perms == 30
    assert cfg.multi_base_enabled is True and cfg.auto_chain_top_k == 3


def test_preset_overrides_win_and_bad_name_raises():
    """Explicit kwargs override the preset bundle; an unknown preset name raises ValueError."""
    cfg = CompositeTargetDiscoveryConfig.preset("fast", mi_sample_n=555, random_state=7)
    assert cfg.mi_sample_n == 555 and cfg.random_state == 7 and cfg.screening == "mi"
    with pytest.raises(ValueError, match="preset must be one of"):
        CompositeTargetDiscoveryConfig.preset("blazing")


def test_suggest_discovery_config_composes_with_preset():
    """``suggest_discovery_config(preset=...)`` overlays the preset bundle on top of data-derived suggestions."""
    rng = np.random.default_rng(0)
    n = 3000
    t = np.arange(n, dtype=np.float64)  # monotone -> detected time column
    df = pd.DataFrame({"t": t, "x0": rng.normal(size=n), "y": rng.normal(size=n)})
    cfg, rationale = suggest_discovery_config(df, "y", ["t", "x0"], preset="fast")
    # Preset pins its bundle over the data-derived mi_sample_n (n<50k would have suggested None).
    assert cfg.mi_sample_n == 20_000 and cfg.screening == "mi"
    # Data-derived fields the preset does not pin survive (time column detection).
    assert cfg.time_column == "t"
    assert "preset" in rationale
    # Caller overrides beat both.
    cfg2, _ = suggest_discovery_config(df, "y", ["t", "x0"], preset="fast", mi_sample_n=777)
    assert cfg2.mi_sample_n == 777
