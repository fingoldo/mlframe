"""Cluster 08 regression: the per-fold float32 cast of the training target is computed once, not per head (audit 2026-11-20, finding 08-F9).

The source casts ``y_tr`` to float32 once before the head loop and reuses the result in every head's kernel call. The values are unchanged.
"""

from pathlib import Path

from mlframe.feature_engineering.transformer import _oof


def test_target_cast_is_outside_the_head_loop():
    """The float32 cast of y_tr is written once per fold, and the loop uses the hoisted name (08-F9)."""
    src = Path(_oof.__file__).resolve().read_text(encoding="utf-8")
    assert src.count("y_tr.astype(np.float32, copy=False)") == 1
    assert "y_tr_f32" in src
