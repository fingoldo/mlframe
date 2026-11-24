"""Cluster 39 regression: to_numpy_safe moves to CPU by default (audit 2026-11-20, finding 39-F64).

The docstring promises a move to CPU "if needed", but the default was ``cpu=False``, so a CUDA tensor raised TypeError. The default is now
``cpu=True``. The test checks the signature and a CPU round-trip, since CUDA is not assumed on the test machine.
"""

import inspect

import numpy as np
import torch

from mlframe.training.neural._base_tensor_helpers import to_numpy_safe


def test_cpu_default_is_true():
    """The default for ``cpu`` is True, matching the docstring (39-F64)."""
    assert inspect.signature(to_numpy_safe).parameters["cpu"].default is True


def test_cpu_tensor_round_trips():
    """Control: a CPU float tensor converts to the same values (39-F64)."""
    out = to_numpy_safe(torch.tensor([1.0, 2.5]))
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [1.0, 2.5])
