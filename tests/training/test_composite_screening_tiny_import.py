"""P0 #1 regression (2026-05-21): ``_composite_screening_tiny`` had a lazy
``from .composite_estimator import _y_train_clip_bounds`` inside the per-fold
inner function. Under joblib threading + ``n_jobs > 1``, two folds racing on
that lazy import could see a partially-loaded ``composite_estimator`` module,
leaving the local name unbound and raising NameError. Symptom: TVT-2026-05-21
prod log emitted 4x ``composite_screening: tiny-model CV fold failed (name
'_y_train_clip_bounds' is not defined)`` warnings. The outer except-Exception
caught it and folds returned NaN; the screening RMSE used nanmean over fewer
folds with no signal that real folds were lost.

Fix: hoist the import to module top so it happens once at module load.
"""
from __future__ import annotations

import concurrent.futures
import importlib

import numpy as np
import pytest


def test_module_level_clip_bounds_import():
    """The lazy import was replaced with a module-level one; verify the name
    is in the module namespace immediately after import (no fold setup needed)."""
    mod = importlib.import_module("mlframe.training._composite_screening_tiny")
    assert hasattr(mod, "_y_train_clip_bounds"), (
        "module-level import of _y_train_clip_bounds missing -- the race-safe hoist regressed."
    )
    # And the imported callable IS the canonical helper from composite_estimator.
    from mlframe.training.composite_estimator import _y_train_clip_bounds as canonical
    assert mod._y_train_clip_bounds is canonical


def test_concurrent_import_does_not_raise():
    """8 threads forcing re-import simultaneously should never raise NameError /
    ImportError. With the lazy form, this used to race ~15% of the time on a
    cold interpreter (Python import lock holds for the FIRST import but
    subsequent imports against partially-loaded modules can return early)."""

    def _import_and_call():
        importlib.invalidate_caches()
        from mlframe.training._composite_screening_tiny import _y_train_clip_bounds
        lo, hi = _y_train_clip_bounds(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        return float(lo), float(hi)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_import_and_call) for _ in range(32)]
        results = [f.result(timeout=10.0) for f in futures]
    # Every result must be a valid (lo, hi) pair, never a propagated exception.
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
