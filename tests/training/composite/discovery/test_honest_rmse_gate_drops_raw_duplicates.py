"""A composite whose reconstruction IS the raw model's prediction must not ship.

The honest-holdout RMSE gate keeps a spec within 5% of the raw baseline, so that a spec trading a little accuracy for a
different view of the data survives. A spec whose reconstruction reproduces raw's predictions offers neither accuracy
nor a different view: it ships a second trained model for nothing. The canonical case is a unary transform on data with
none of the structure it models - ``seasonal_residual`` on a non-periodic frame subtracts near-constant phase means, the
tiny model learns the same function, and the inverse puts the shift back (measured corr 1.000000, y-RMSE 39.665 against
raw 39.665).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _non_periodic_frame(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """A plain wide-range linear DGP: nothing periodic for a seasonal transform to find."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1000.0, n)
    y = 2.5 * base + 10.0 + rng.normal(0.0, 5.0, n)
    return pd.DataFrame({"base": base, "x0": rng.normal(size=n), "noise0": rng.normal(size=n), "y": y})


def _config(**overrides) -> CompositeTargetDiscoveryConfig:
    """A fast discovery config for this gate."""
    kw = dict(
        enabled=True, random_state=0, tiny_model_n_estimators=40,
        base_candidates=["base"], transforms=["seasonal_residual"],
    )
    kw.update(overrides)
    return CompositeTargetDiscoveryConfig(**kw)


def _fit(**overrides) -> CompositeTargetDiscovery:
    """Discover on the non-periodic frame."""
    df = _non_periodic_frame()
    disc = CompositeTargetDiscovery(_config(**overrides))
    disc.fit(df, "y", ["base", "x0", "noise0"], np.arange(len(df)))
    return disc


def test_a_spec_that_reproduces_the_raw_model_is_dropped():
    """No-lift-no-diversity means no ship, even though its RMSE is comfortably inside the 5% tolerance."""
    disc = _fit()
    assert not disc.specs_, f"a spec duplicating raw survived: {[s.name for s in disc.specs_]}"


def test_the_drop_is_recorded_against_the_honest_rmse_gate():
    """The rejection is attributable: the ledger names the stage and says the reconstruction duplicates raw."""
    disc = _fit()
    rows = [row for row in disc.rejection_ledger if row["stage"] == "honest_rmse"]
    assert rows, "the drop must be recorded under the honest_rmse stage"
    assert any("duplicates the raw model" in str(row.get("reason", "")) for row in rows), rows


def test_a_spec_that_genuinely_differs_still_survives_inside_the_tolerance():
    """The gate must not become a no-lift filter: a real residual transform on a base-driven DGP still ships."""
    df = _non_periodic_frame()
    disc = CompositeTargetDiscovery(_config(transforms=["linear_residual", "diff"]))
    disc.fit(df, "y", ["base", "x0", "noise0"], np.arange(len(df)))
    assert disc.specs_, "a genuinely different reconstruction must still survive the gate"
