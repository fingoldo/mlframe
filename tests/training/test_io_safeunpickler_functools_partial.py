"""Regression: a model carrying a ``functools.partial`` must survive the
save -> safe-load round-trip instead of silently loading as None.

Fitted model state routinely holds a ``functools.partial``:
  * neural config ``weights_init_fcn=partial(kaiming_normal_, nonlinearity=...)``
  * ``_PartialFitEarlyStoppingWrapper.metric`` parametrized-metric slot.

Before the fix ``functools``/``_functools`` were absent from
``_SAFE_MODULE_PREFIXES`` so ``load_mlframe_model(safe=True)`` raised
"Unsafe class blocked ... functools.partial" and returned None -- no model,
no exception. Allowing functools is safe: a partial only stores its func +
args; a wrapped dangerous func (e.g. os.system) is independently re-resolved
through ``find_class`` on unpickle and stays blocked.
"""
from __future__ import annotations

import functools
import io
import os
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.io import (
    _SafeUnpickler,
    load_mlframe_model,
    save_mlframe_model,
    _load_model_cache_clear,
)


def test_partial_survives_save_load_roundtrip(tmp_path):
    _load_model_cache_clear()
    # Wrapped func is ``numpy.clip`` (an allowlisted module) -- mirrors the
    # production case where the partial wraps an mlframe / torch.nn.init func.
    # This isolates the test to the ``functools.partial`` allowlist entry.
    model = SimpleNamespace(
        name="m",
        metric=functools.partial(np.clip, a_min=0.0, a_max=1.0),
        weight=1.0,
    )
    f = os.path.join(tmp_path, "m.dump")
    assert save_mlframe_model(model, f, verbose=0) is True
    loaded = load_mlframe_model(f, safe=True)
    # Pre-fix: loaded is None (allowlist blocked functools.partial).
    assert loaded is not None, "safe-load silently returned None for a partial-carrying model"
    assert isinstance(loaded.metric, functools.partial)
    # The partial is intact and callable with its bound kwargs.
    assert float(loaded.metric(2.0)) == 1.0
    assert loaded.metric.keywords == {"a_min": 0.0, "a_max": 1.0}


def test_partial_wrapping_dangerous_func_still_blocked():
    """Allowing functools.partial must NOT open an RCE hole: a partial of
    ``os.system`` is blocked because the inner func fails find_class."""
    payload = pickle.dumps(
        functools.partial(os.system, "echo pwned"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    with pytest.raises(Exception) as ei:
        _SafeUnpickler(io.BytesIO(payload)).load()
    # Block happens on the inner os/nt.system, never on partial itself.
    assert "system" in str(ei.value)
