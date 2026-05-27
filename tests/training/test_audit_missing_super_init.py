"""Wave 56 (2026-05-20): missing super().__init__() in subclasses.

Audit class: subclass __init__ that omits super().__init__(), leaving parent's
state uninitialised. Symptom is silent today because the parent's __init__ is
empty (torch Dataset / Sampler / Lightning Callback) or attribute-only with
the subclass manually re-assigning every known attr (sklearn TransformedTargetRegressor).
Future-rot risk: any parent attribute added in a minor upstream release silently
drops from get_params/clone introspection.

5 P2 cosmetic fixes applied:

  1. estimators/custom.py:50 (ESTransformedTargetRegressor)
     Forward 5 known attrs via super().__init__(); any new sklearn attrs added
     in minor releases get populated automatically.

  2. training/neural/base.py:784 (AggregatingValidationCallback)
     super().__init__() before store_params_in_object so future Lightning
     Callback state is honoured.

  3. training/neural/data.py:62 (TorchDataset)
     super().__init__() for forward-compat with torch.utils.data.Dataset.

  4. training/neural/ranker.py:187 (GroupBatchSampler)
     super().__init__(data_source=None) for forward-compat with torch Sampler.

  5. training/neural/ranker.py:240 (_RankerDataset)
     Same as #3.

Per the audit, NONE of these cause runtime AttributeError today because each
parent's __init__ is currently empty/lazy; the fixes are pure forward-compat /
convention hardening.
"""
from __future__ import annotations

from pathlib import Path

import pytest


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def test_es_transformed_target_regressor_calls_super_init() -> None:
    src = _read("estimators/custom.py")
    helper_idx = src.find("class ESTransformedTargetRegressor")
    assert helper_idx != -1
    # Read the next 50 lines after the class declaration.
    snippet = src[helper_idx : helper_idx + 1500]
    assert "super().__init__(" in snippet
    assert "regressor=regressor," in snippet
    assert "transformer=transformer," in snippet


def test_aggregating_validation_callback_calls_super_init() -> None:
    # ``AggregatingValidationCallback`` was carved out of ``training/neural/base.py``
    # into ``_base_callbacks.py``; concat parent + sibling so the source-grep
    # guard survives the split.
    src = _read("training/neural/base.py")
    sib = MLFRAME_ROOT / "training" / "neural" / "_base_callbacks.py"
    if sib.exists():
        src += "\n" + sib.read_text(encoding="utf-8")
    helper_idx = src.find("class AggregatingValidationCallback")
    assert helper_idx != -1
    snippet = src[helper_idx : helper_idx + 800]
    assert "super().__init__()" in snippet


def test_torch_dataset_calls_super_init() -> None:
    src = _read("training/neural/data.py")
    helper_idx = src.find("class TorchDataset")
    assert helper_idx != -1
    # Read enough lines to reach the __init__ body.
    snippet = src[helper_idx : helper_idx + 4000]
    assert "super().__init__()" in snippet


def test_group_batch_sampler_calls_super_init() -> None:
    src = _read("training/neural/ranker.py")
    helper_idx = src.find("class GroupBatchSampler")
    assert helper_idx != -1
    snippet = src[helper_idx : helper_idx + 1200]
    assert "super().__init__(data_source=None)" in snippet


def test_ranker_dataset_calls_super_init() -> None:
    src = _read("training/neural/ranker.py")
    helper_idx = src.find("class _RankerDataset")
    assert helper_idx != -1
    snippet = src[helper_idx : helper_idx + 600]
    assert "super().__init__()" in snippet
