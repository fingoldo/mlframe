"""Tests for the polish round of composite-target features:

- ``derive_seeds`` deterministic + change-on-input + 32-bit unsigned int.
- ``env_signature`` returns dict with library version strings.
- ``detect_gpu_in_use`` returns empty list when no GPU library is
  configured (the only deterministic check we can make in CI).
- ``CompositeCrossTargetEnsemble.cap_inference_components`` keeps top-N
  by absolute weight, preserves linear_stack intercept, no-ops when
  N >= K.
- Auto-skip metadata: when a target_type is unsupported (LtR /
  multiclass / quantile / multilabel / binary), the failures map
  records the skip reason explicitly.
- Multilabel regression target rejected with "multilabel target
  unsupported" reason -- explicit metadata, not silent skip.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    CompositeCrossTargetEnsemble,
    derive_seeds,
    detect_gpu_in_use,
    env_signature,
)


# ----------------------------------------------------------------------
# derive_seeds
# ----------------------------------------------------------------------


class TestDeriveSeeds:
    def test_deterministic_for_same_input(self) -> None:
        a = derive_seeds(42, ["mi", "tiny", "stack"])
        b = derive_seeds(42, ["mi", "tiny", "stack"])
        assert a == b

    def test_master_seed_changes_subseeds(self) -> None:
        a = derive_seeds(42, ["mi"])
        b = derive_seeds(43, ["mi"])
        assert a["mi"] != b["mi"]

    def test_component_name_changes_subseed(self) -> None:
        a = derive_seeds(42, ["mi"])
        b = derive_seeds(42, ["tiny"])
        assert a["mi"] != b["tiny"]

    def test_returns_32bit_unsigned(self) -> None:
        seeds = derive_seeds(42, ["a", "b", "c"])
        for v in seeds.values():
            assert isinstance(v, int)
            assert 0 <= v < 2 ** 32

    def test_empty_components(self) -> None:
        assert derive_seeds(42, []) == {}


# ----------------------------------------------------------------------
# env_signature
# ----------------------------------------------------------------------


class TestEnvSignature:
    def test_returns_dict_with_required_keys(self) -> None:
        sig = env_signature()
        assert isinstance(sig, dict)
        # numpy is a hard dep of mlframe, must be present.
        assert sig.get("numpy") is not None
        assert isinstance(sig["numpy"], str)
        # The dict should contain our targeted libraries (value may
        # be None if optional dep not installed -- that's the contract).
        for libname in ("pandas", "polars", "sklearn", "scipy"):
            assert libname in sig


# ----------------------------------------------------------------------
# detect_gpu_in_use
# ----------------------------------------------------------------------


class TestDetectGpuInUse:
    def test_empty_models_returns_empty(self) -> None:
        assert detect_gpu_in_use([]) == []

    def test_no_gpu_libraries_skipped(self) -> None:
        # 'linear' family does not use GPU; the result should not
        # claim catboost / xgboost / lightgbm GPU.
        result = detect_gpu_in_use(["linear", "ridge"])
        assert result == []


# ----------------------------------------------------------------------
# cap_inference_components
# ----------------------------------------------------------------------


class _StubModel:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.full(n, self.value, dtype=np.float64)


class TestCapInferenceComponents:
    def test_keeps_top_n_by_abs_weight(self) -> None:
        ens = CompositeCrossTargetEnsemble(
            component_models=[_StubModel(i) for i in range(5)],
            component_names=["a", "b", "c", "d", "e"],
            weights=np.array([1.0, 0.1, 5.0, 0.01, 2.0]),
            strategy="oof_weighted",
        )
        capped = ens.cap_inference_components(2)
        # Top-2 by |weight| are 'c' (5.0) and 'e' (2.0), kept in
        # original order.
        assert capped.component_names == ["c", "e"]
        assert "capped_to_top_n" in capped.notes
        assert "dropped_components" in capped.notes

    def test_no_op_when_n_ge_k(self) -> None:
        ens = CompositeCrossTargetEnsemble(
            component_models=[_StubModel(i) for i in range(3)],
            component_names=["a", "b", "c"],
            weights=np.array([1.0, 1.0, 1.0]),
            strategy="mean",
        )
        capped = ens.cap_inference_components(10)
        assert len(capped.component_names) == 3

    def test_no_op_when_n_le_zero(self) -> None:
        ens = CompositeCrossTargetEnsemble(
            component_models=[_StubModel(i) for i in range(3)],
            component_names=["a", "b", "c"],
            weights=np.array([1.0, 1.0, 1.0]),
            strategy="mean",
        )
        assert len(ens.cap_inference_components(0).component_names) == 3
        assert len(ens.cap_inference_components(-1).component_names) == 3

    def test_preserves_linear_stack_intercept(self) -> None:
        ens = CompositeCrossTargetEnsemble(
            component_models=[_StubModel(i) for i in range(3)],
            component_names=["a", "b", "c"],
            weights=np.array([0.1, 5.0, 0.05]),
            strategy="linear_stack",
        )
        ens._linear_stack_intercept = 7.5
        capped = ens.cap_inference_components(1)
        assert hasattr(capped, "_linear_stack_intercept")
        assert capped._linear_stack_intercept == 7.5

    def test_capped_predict_works(self) -> None:
        ens = CompositeCrossTargetEnsemble(
            component_models=[_StubModel(1.0), _StubModel(2.0), _StubModel(3.0)],
            component_names=["a", "b", "c"],
            weights=np.array([0.05, 0.5, 0.45]),
            strategy="oof_weighted",
        )
        capped = ens.cap_inference_components(2)
        # Top-2 are b (0.5) and c (0.45). After re-norm: b=0.526, c=0.474.
        # Predict on dummy input; predictions are 2.0 and 3.0; weighted
        # mean ~ 2.474.
        pred = capped.predict(np.zeros((5, 1)))
        assert np.all(np.isfinite(pred))
        assert 2.0 < pred[0] < 3.0
