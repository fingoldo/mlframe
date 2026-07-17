"""Unit tests for small ``training/core/_misc_helpers.py`` helpers extracted during the function-level quality pass.

These tests pin down the contracts of the deduplicated helpers so future call sites can rely on them.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.core._misc_helpers import _cfg_get, _compute_neural_max_time


# ---------------------------------------------------------------------------
# _cfg_get -- "Pydantic-or-dict-or-None" config accessor
# ---------------------------------------------------------------------------


class TestCfgGet:
    def test_none_cfg_returns_default(self):
        assert _cfg_get(None, "any_key", "fallback") == "fallback"

    def test_none_cfg_default_default_is_none(self):
        assert _cfg_get(None, "any_key") is None

    def test_dict_present_key_returns_value(self):
        assert _cfg_get({"k": 42}, "k", -1) == 42

    def test_dict_missing_key_returns_default(self):
        assert _cfg_get({"k": 42}, "missing", -1) == -1

    def test_dict_present_key_with_falsy_value_preserved(self):
        # Don't conflate "missing" with "explicitly set to 0/False/empty".
        assert _cfg_get({"k": 0}, "k", 999) == 0
        assert _cfg_get({"k": False}, "k", True) is False
        assert _cfg_get({"k": ""}, "k", "default") == ""

    def test_object_attr_returns_value(self):
        ns = SimpleNamespace(iterations=300, learning_rate=0.05)
        assert _cfg_get(ns, "iterations", 100) == 300
        assert _cfg_get(ns, "learning_rate", 0.1) == 0.05

    def test_object_missing_attr_returns_default(self):
        ns = SimpleNamespace(iterations=300)
        assert _cfg_get(ns, "missing", "fallback") == "fallback"

    def test_pydantic_like_object(self):
        # Pydantic v2 models behave like normal objects for getattr.
        class _MockPydantic:
            def __init__(self):
                self.test_size = 0.2
                self.val_size = 0.15

        cfg = _MockPydantic()
        assert _cfg_get(cfg, "test_size", 0.15) == 0.2
        assert _cfg_get(cfg, "val_size", 0.15) == 0.15
        assert _cfg_get(cfg, "absent", "miss") == "miss"


# ---------------------------------------------------------------------------
# _compute_neural_max_time -- P95 + 300s floor + dhms decomposition
# ---------------------------------------------------------------------------


class TestComputeNeuralMaxTime:
    def test_empty_returns_none(self):
        assert _compute_neural_max_time([]) is None

    def test_none_returns_none(self):
        assert _compute_neural_max_time(None) is None

    def test_basic_p95_below_floor_clamped_to_300s(self):
        # All inputs <300s; P95 still <300s; floor at 300s kicks in.
        times = [10.0, 20.0, 30.0]
        result = _compute_neural_max_time(times)
        assert isinstance(result, tuple) and len(result) == 3, f"expected 3-tuple, got {result!r}"
        max_time_dict, p95, n = result
        assert n == 3
        # 300s = 5min
        assert max_time_dict == {"days": 0, "hours": 0, "minutes": 5, "seconds": 0}

    def test_above_floor_p95_used(self):
        # P95 of [600] = 600s = 10 minutes, above the 300s floor.
        result = _compute_neural_max_time([600.0])
        assert isinstance(result, tuple) and len(result) == 3, f"expected 3-tuple, got {result!r}"
        max_time_dict, p95, n = result
        assert n == 1
        assert max_time_dict == {"days": 0, "hours": 0, "minutes": 10, "seconds": 0}

    def test_hour_decomposition(self):
        # 3725s = 1h 2m 5s.
        result = _compute_neural_max_time([3725.0])
        assert isinstance(result, tuple) and len(result) == 3, f"expected 3-tuple, got {result!r}"
        max_time_dict, _, _ = result
        assert max_time_dict == {"days": 0, "hours": 1, "minutes": 2, "seconds": 5}

    def test_day_decomposition(self):
        # 90061s = 1d 1h 1m 1s.
        result = _compute_neural_max_time([90061.0])
        assert result is not None
        max_time_dict, _, _ = result
        assert max_time_dict == {"days": 1, "hours": 1, "minutes": 1, "seconds": 1}

    def test_rounding_at_p95(self):
        # P95 of [100, 200, 300, 400, 500] = ~480s; floor 300s; output 8m 0s.
        times = [100.0, 200.0, 300.0, 400.0, 500.0]
        result = _compute_neural_max_time(times)
        assert result is not None
        max_time_dict, p95, n = result
        assert n == 5
        assert 460 < p95 <= 500
        # Whatever P95 rounds to, the dhms should sum back to int(round(p95))
        total = max_time_dict["days"] * 86400 + max_time_dict["hours"] * 3600 + max_time_dict["minutes"] * 60 + max_time_dict["seconds"]
        assert total == max(int(round(p95)), 300)

    def test_returns_tuple_of_three(self):
        result = _compute_neural_max_time([60.0, 120.0])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_p95_is_float(self):
        result = _compute_neural_max_time([42.0])
        assert isinstance(result[1], float)

    def test_n_is_count_of_input(self):
        for n_inputs in (1, 5, 100):
            result = _compute_neural_max_time([1.0] * n_inputs)
            assert result is not None
            assert result[2] == n_inputs

    def test_numpy_array_input(self):
        result = _compute_neural_max_time(np.array([300.0, 600.0, 900.0]))
        assert result is not None
        max_time_dict, p95, n = result
        assert n == 3
        assert p95 > 0
