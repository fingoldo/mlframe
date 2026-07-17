"""Unit tests for ``mlframe.system._gpu_guard.callable_looks_gpu_bound``, the static heuristic used to gate
joblib fan-outs (ensembling custom metrics, bootstrap metric_fns) away from parallelising GPU-bound user callables.
"""

from __future__ import annotations

import numpy as np

from mlframe.system._gpu_guard import callable_looks_gpu_bound


def test_none_is_not_gpu_bound():
    assert callable_looks_gpu_bound(None) is False


def test_plain_numpy_callable_is_not_gpu_bound():
    def metric(a, b):
        return np.abs(a - b).mean()

    assert callable_looks_gpu_bound(metric) is False


def test_builtin_is_not_gpu_bound():
    assert callable_looks_gpu_bound(sum) is False


def test_torch_referencing_callable_is_gpu_bound():
    def metric(a, b):
        return torch.abs(a - b).mean()  # noqa: F821 - unresolved name, exercises the static heuristic only

    assert callable_looks_gpu_bound(metric) is True


def test_cupy_referencing_callable_is_gpu_bound():
    def metric(a, b):
        return cupy.abs(a - b).mean()  # noqa: F821

    assert callable_looks_gpu_bound(metric) is True


def test_closure_over_gpu_module_object_is_gpu_bound():
    import types

    fake_torch_tensor = types.SimpleNamespace()
    fake_torch_tensor.__module__ = "torch.tensor"

    def make_metric():
        device_obj = fake_torch_tensor

        def metric(a, b):
            return device_obj

        return metric

    assert callable_looks_gpu_bound(make_metric()) is True


def test_lambda_over_plain_numpy_is_not_gpu_bound():
    metric = lambda a, b: np.abs(a - b).sum()
    assert callable_looks_gpu_bound(metric) is False
