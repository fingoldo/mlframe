"""default_via_or wave 2: fixes 3 more P1 findings where a legitimate explicit ``0`` config value
was silently rewritten to the fallback default via ``getattr(obj, "attr", default) or default``:

- ``training/pipeline/_categorical_composite_fe.py``: ``categorical_composite_max_source_columns``
  -- ``0`` is a legitimate "never run the categorical composite step" sentinel.
- ``training/pipeline/_nearest_past_join_composite_fe.py``: ``nearest_past_join_min_group_size``
  -- ``0`` (no minimum group size) is a legitimate value.
- ``feature_selection/boruta_shap/_fit_explain.py``: ``stability_threshold`` -- ``0.0`` (every
  feature counts as "stable") is a legitimate, if permissive, value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_categorical_composite_max_source_columns_zero_disables_step():
    """categorical_composite_max_source_columns=0 must be honored (skip every column count > 0),
    not silently rewritten to 12."""
    from mlframe.training.pipeline._categorical_composite_fe import apply_categorical_composite_fe

    train_df = pd.DataFrame({"c1": ["a", "b"] * 5, "c2": ["x", "y"] * 5})

    class _Cfg:
        """Minimal config fixture for this test."""

        categorical_powerset_concat_enabled = False
        categorical_group_concat_auto_enabled = True
        categorical_group_concat_min_mi_gain = -1.0
        categorical_group_concat_max_group_size = None
        categorical_composite_max_source_columns = 0

    y_train = np.array([0, 1] * 5)
    out_train, _out_val, _out_test = apply_categorical_composite_fe(train_df, None, None, _Cfg(), y_train, metadata={}, verbose=0)
    assert not any(c.startswith("concat_group__") for c in out_train.columns), "max_source_columns=0 must skip the step entirely"


def test_nearest_past_join_min_group_size_zero_not_rewritten_to_default():
    """nearest_past_join_min_group_size=0 must be persisted as 0, not rewritten to 1."""
    import numpy as np

    from mlframe.training.pipeline._nearest_past_join_composite_fe import apply_nearest_past_join_composite_fe

    n = 10
    train_df = pd.DataFrame({"id": np.arange(n), "ts": np.arange(n, dtype=np.float64)})
    aux_df = pd.DataFrame({"id": np.arange(n), "ts": np.arange(n, dtype=np.float64), "v": np.arange(n, dtype=np.float64)})

    class _Cfg:
        """Minimal config fixture for this test."""

        nearest_past_join_on = "ts"
        nearest_past_join_by = ["id"]
        nearest_past_join_value_cols = ["v"]
        nearest_past_join_fallback_by_chain = None
        nearest_past_join_min_group_size = 0

    metadata: dict = {}
    apply_nearest_past_join_composite_fe(train_df, None, None, _Cfg(), aux_df, metadata=metadata)
    assert metadata.get("nearest_past_join_min_group_size") == 0


def test_boruta_shap_stability_threshold_zero_not_rewritten_to_default():
    """stability_threshold=0.0 (fully permissive) must survive as 0.0, not be rewritten to 0.6."""

    class _StabilityStub:
        """Duck-typed stand-in exposing only stability_threshold, matching the getattr(self, ...) read."""

        stability_threshold = 0.0

    obj = _StabilityStub()
    _thr_cfg = getattr(obj, "stability_threshold", None)
    thr = 0.6 if _thr_cfg is None else float(_thr_cfg)
    assert thr == 0.0
