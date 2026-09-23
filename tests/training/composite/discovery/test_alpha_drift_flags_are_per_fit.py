"""The alpha-drift flags belong to one fit, and a duck-typed config gets the documented reject default (DSC-25).

The drift gate cleared ``_alpha_drift_flags`` only when a ``linear_residual`` spec survived, so a second fit of the same
instance - a stability replicate, the stacked second pass, per-group reuse - that kept none left the previous fit's flags
in place, and a same-named survivor was warned about for drift measured on other data.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.discovery._eval_stats import apply_alpha_drift_gate
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _dataset(n: int = 400, drift: bool = False) -> pd.DataFrame:
    """A frame whose base-to-y slope either holds throughout or quadruples halfway through."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=n) * 3.0 + 10.0
    alpha = np.where(np.arange(n) < n // 2, 1.0, 4.0) if drift else np.ones(n)
    return pd.DataFrame({"b": base, "t": np.arange(n, dtype=float), "y": alpha * base + rng.normal(scale=0.2, size=n)})


def _discovery() -> CompositeTargetDiscovery:
    """Discovery restricted to ``linear_residual`` on the ``b`` base."""
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True, base_candidates=["b"], transforms=["linear_residual"], mi_sample_n=200, eps_mi_gain=-1.0,
    )
    return CompositeTargetDiscovery(config=cfg)


def test_a_refit_does_not_inherit_the_previous_fits_drift_flags():
    """A second fit that keeps no ``linear_residual`` must not carry the first fit's flags into its own warnings."""
    disc = _discovery()
    rows = np.arange(400)
    disc.fit(_dataset(drift=True), "y", ["b", "t"], rows)
    disc._alpha_drift_flags = {"stale_spec": {"z_score": 99.0}}  # stand-in for whatever the first fit left behind

    disc.fit(_dataset(drift=False), "y", ["b", "t"], rows)
    assert "stale_spec" not in disc._alpha_drift_flags, "the previous fit's drift flags survived into this one"


def test_a_duck_typed_config_rejects_a_drifting_spec_like_the_pydantic_default():
    """A config object without ``reject_on_alpha_drift`` behaves as the documented default does: the drifting spec goes."""
    assert CompositeTargetDiscoveryConfig(enabled=True).reject_on_alpha_drift is True

    df = _dataset(drift=True)
    disc = _discovery()
    disc._auto_base_pool = {}
    disc._target_col = "y"
    disc.config = SimpleNamespace(detect_linear_residual_alpha_drift=True, alpha_drift_z_threshold=3.0,
                                  alpha_drift_min_effect_size=0.01)  # no reject_on_alpha_drift attribute at all
    spec = SimpleNamespace(name="lr__b", transform_name="linear_residual", base_column="b")
    kept = apply_alpha_drift_gate(
        disc, [spec], df=df, train_idx=np.arange(len(df)), y_full=df["y"].to_numpy(),
        extract_column_array=lambda frame, col: frame[col].to_numpy(),
    )
    assert kept == [], "a duck-typed config kept a drifting spec the pydantic default would have rejected"
    assert disc._alpha_drift_flags["lr__b"]["z_score"] > 3.0
