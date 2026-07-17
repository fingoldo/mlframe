"""A model carrying a functools.partial (parametrized metric / neural weights_init_fcn)
must survive a safe save/load round-trip; a partial wrapping a dangerous func stays blocked."""

from __future__ import annotations

import functools
import os
from types import SimpleNamespace

import numpy as np

from mlframe.training.io import save_mlframe_model, load_mlframe_model


def test_partial_survives_save_load_roundtrip(tmp_path):
    # Inner func is numpy (allowlisted); the partial itself needs functools allowed,
    # otherwise safe-load returns None SILENTLY (no model, no exception).
    model = SimpleNamespace(metric=functools.partial(np.clip, a_min=0.0, a_max=1.0))
    bundle = tmp_path / "m.zst"
    assert save_mlframe_model(model, str(bundle), verbose=0) is True
    loaded = load_mlframe_model(str(bundle), safe=True)
    assert loaded is not None
    assert isinstance(loaded.metric, functools.partial)
    assert loaded.metric(np.array([-1.0, 0.5, 2.0])).tolist() == [0.0, 0.5, 1.0]


def test_partial_wrapping_dangerous_func_still_blocked(tmp_path):
    # Allowing functools.partial must not open an RCE: the wrapped os.system is
    # independently re-resolved through find_class on unpickle and stays blocked.
    model = SimpleNamespace(evil=functools.partial(os.system, "echo pwned"))
    bundle = tmp_path / "evil.zst"
    save_mlframe_model(model, str(bundle), verbose=0)
    loaded = load_mlframe_model(str(bundle), safe=True)
    # Either the load is blocked (returns None) or the dangerous inner func did not survive.
    assert loaded is None or not isinstance(getattr(loaded, "evil", None), functools.partial)
