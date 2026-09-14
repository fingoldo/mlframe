"""Target-aware group binning must discretise its target through the shared encoder (mrmr_audit_2026-09-14 IMPL-4).

``generate_target_aware_group_bins`` sits outside the MRMR cluster the RO-4 audit agent covered, but carried the same
truncating ``<= 32 -> astype(np.int64)`` discretisation: a half-step target reached its per-group supervised binning as
two classes instead of four. It now calls ``encode_y_for_classif_mi``; this observes that call on the real function.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _grouped_quantile_fe as GQ
from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi


def _frame_and_half_step_target(n=240, seed=0):
    """Four groups, one numeric column, and a half-step target the old cast would have merged."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"g": rng.integers(0, 4, size=n), "v": rng.normal(size=n)})
    y = np.tile(np.array([0.0, 0.5, 1.0, 1.5]), n // 4)
    return X, y


def test_the_target_is_routed_through_encode_y_for_classif_mi(monkeypatch):
    """The encoder must be called with the caller's target, and what it returns must keep all four labels."""
    X, y = _frame_and_half_step_target()
    seen = []
    real = encode_y_for_classif_mi

    def _spy(target):
        """Record the target handed to the shared encoder, then encode it for real."""
        seen.append(np.asarray(target).copy())
        return real(target)

    # raising=False: before the fix the module never imported the encoder, so there was no attribute to replace. Setting
    # it anyway keeps the failure honest -- the function simply never calls the spy -- instead of an AttributeError.
    monkeypatch.setattr(GQ, "encode_y_for_classif_mi", _spy, raising=False)
    GQ.generate_target_aware_group_bins(X, y, ["g"], ["v"])

    assert seen, "generate_target_aware_group_bins no longer routes its target through the shared encoder"
    assert np.array_equal(seen[0], y)
    assert len(np.unique(real(seen[0]))) == 4
