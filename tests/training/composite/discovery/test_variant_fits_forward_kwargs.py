"""The stacked and stability-check variants of ``fit`` hand ``fit`` the same time order and val frame a plain fit gets.

They called ``self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)`` and dropped ``time_ordering``,
``val_df`` and ``val_y``, so on temporal data the screen fell back to shuffled folds and the y-scale gate never saw the
unseen-group val split, silently, for exactly the callers that asked for the more careful variant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _frame(n: int = 400):
    """A base, a feature and a target."""
    rng = np.random.default_rng(0)
    b = rng.uniform(1.0, 10.0, n)
    return pd.DataFrame({"b": b, "x": rng.normal(size=n), "y": 2.0 * b + rng.normal(0.0, 0.5, n)})


@pytest.mark.parametrize("variant, extra", [("fit_stacked", {}), ("fit_stacked_on_residual", {}), ("fit_with_stability_check", {"n_bootstrap_runs": 1})])
def test_a_variant_forwards_time_order_and_the_val_frame_to_fit(monkeypatch, variant, extra):
    """The first ``fit`` call a variant makes receives the caller's ``time_ordering``, ``val_df`` and ``val_y``."""
    df = _frame()
    seen = []

    def spy(self, *args, **kwargs):
        """Record the forwarded keywords, then stop: only the forwarding is under test."""
        seen.append({k: kwargs.get(k) for k in ("time_ordering", "val_df", "val_y")})
        self.specs_ = []
        return self

    monkeypatch.setattr(CompositeTargetDiscovery, "fit", spy)
    disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True, random_state=0, base_candidates=["b"]))
    order, val_df, val_y = np.arange(len(df)), df.drop(columns=["y"]), df["y"].to_numpy()
    getattr(disc, variant)(df, "y", ["b", "x"], np.arange(len(df)), time_ordering=order, val_df=val_df, val_y=val_y, **extra)
    assert seen, f"{variant} never called fit"
    first = seen[0]
    assert first["time_ordering"] is order and first["val_df"] is val_df and first["val_y"] is val_y, f"{variant} dropped {first}"
