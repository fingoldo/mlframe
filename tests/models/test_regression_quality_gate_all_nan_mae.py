"""Regression: the relative member-quality gate must not silently keep every member when all per-member MAE are NaN.

Pre-fix ``np.nanmedian`` of an all-NaN per-member-MAE array returned NaN, making every ``tot_mae > rel_mae_threshold``
comparison False -> the relative gate became a silent no-op. Post-fix the all-NaN case explicitly disables the relative
gate (threshold 0.0) instead of leaking a NaN threshold.
"""

import numpy as np

from mlframe.models.ensembling.quality_gate import compute_member_quality_gate


def test_relative_gate_all_nan_mae_threshold_is_zero_not_nan():
    """Relative gate all nan mae threshold is zero not nan."""
    n_rows = 50
    # Three members, every prediction NaN -> per-member MAE/STD are all NaN.
    preds_list = [np.full(n_rows, np.nan) for _ in range(3)]
    _kept, _excluded, stats = compute_member_quality_gate(
        preds_list,
        max_mae=0.0,
        max_std=0.0,
        max_mae_relative=2.5,
        max_std_relative=2.5,
    )
    # Pre-fix: rel_mae_threshold was NaN. Post-fix: explicitly 0.0 (relative gate disabled).
    assert stats["rel_mae_threshold"] == 0.0
    assert stats["rel_std_threshold"] == 0.0
    assert not np.isnan(stats["rel_mae_threshold"])
    assert not np.isnan(stats["rel_std_threshold"])
