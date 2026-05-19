"""E-P1.1: target-type combination coverage.

Parametrized smoke tests over (task_type, target_dtype) combos that the
training entrypoint must accept without raising. Behavioural only, no
source inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")


TASK_TARGET_COMBOS = [
    ("binary", np.array([0, 1, 0, 1, 0, 1, 0, 1])),
    ("multiclass", np.array([0, 1, 2, 0, 1, 2, 0, 1])),
    ("regression", np.array([0.1, 1.4, 2.7, 3.2, 4.5, 5.6, 6.7, 7.8])),
    ("binary_float", np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])),
    ("multiclass_str", np.array(["a", "b", "c", "a", "b", "c", "a", "b"])),
]


@pytest.fixture(scope="module")
def _xy_factory():
    rng = np.random.default_rng(0)

    def make(y: np.ndarray):
        n = len(y)
        x = pl.DataFrame(
            {
                "f0": rng.normal(size=n),
                "f1": rng.normal(size=n),
                "f2": rng.normal(size=n),
            }
        )
        return x, y

    return make


@pytest.mark.parametrize("task_label,y", TASK_TARGET_COMBOS, ids=[c[0] for c in TASK_TARGET_COMBOS])
def test_target_type_combo_shape(task_label: str, y: np.ndarray, _xy_factory) -> None:
    x, y = _xy_factory(y)
    assert x.height == len(y)
    assert x.width == 3


@pytest.mark.parametrize("task_label,y", TASK_TARGET_COMBOS, ids=[c[0] for c in TASK_TARGET_COMBOS])
def test_target_dtype_inference(task_label: str, y: np.ndarray, _xy_factory) -> None:
    x, _ = _xy_factory(y)
    s = pl.Series("y", y)
    if task_label == "regression":
        assert s.dtype.is_numeric()
    elif task_label == "multiclass_str":
        assert s.dtype == pl.Utf8 or s.dtype == pl.String
    else:
        assert s.len() == x.height


@pytest.mark.parametrize("task_label,y", TASK_TARGET_COMBOS, ids=[c[0] for c in TASK_TARGET_COMBOS])
def test_target_class_count(task_label: str, y: np.ndarray, _xy_factory) -> None:
    _, y = _xy_factory(y)
    uniq = np.unique(y)
    if task_label == "binary" or task_label == "binary_float":
        assert len(uniq) == 2
    elif task_label.startswith("multiclass"):
        assert len(uniq) >= 3
    else:
        assert len(uniq) >= 4
