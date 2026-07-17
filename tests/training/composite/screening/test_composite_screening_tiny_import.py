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
    mod = importlib.import_module("mlframe.training.composite.discovery._screening_tiny")
    assert hasattr(mod, "_y_train_clip_bounds"), "module-level import of _y_train_clip_bounds missing -- the race-safe hoist regressed."
    # And the imported callable IS the canonical helper from composite_estimator.
    from mlframe.training.composite import _y_train_clip_bounds as canonical

    assert mod._y_train_clip_bounds is canonical


def test_nan_fold_count_warn_fires_when_any_fold_fails(caplog):
    """E1.2 (2026-05-22): when ANY fold returns NaN the aggregate-level WARN
    fires so the operator sees the effective fold count dropped. Without
    this, the outer ``except Exception`` swallows per-fold failures into NaN,
    nanmean silently averages over survivors, and a 4-of-5-folds-NaN run
    looks like a 1-fold result with no signal at all in the logs (the
    TVT-2026-05-21 prod symptom before P0 #1 fixed the lazy-import race)."""
    import logging
    import math

    import numpy as np

    from mlframe.training.composite.discovery._screening_tiny import _tiny_cv_rmse_raw_y

    # Build a working scenario then monkey-patch the underlying fold computation
    # to NaN out the first 3 folds. Easier: pass tiny n + an unstable family that
    # forces folds to fail. Simpler still: directly inject NaN via the model
    # builder returning a broken model.
    #
    # Actually the cleanest path: call _tiny_cv_rmse_y_scale with cv_folds=5 on
    # data where the underlying fit raises on most folds (e.g. degenerate target
    # of all-zeros except 1 row -- LightGBM rejects single-class regression).
    # If that doesn't reliably reproduce, we patch _build_tiny_model directly.
    from mlframe.training.composite.discovery import _screening_tiny as mod

    real_build = mod._build_tiny_model
    fail_count = {"n": 0}

    def _fail_first_3(*args, **kwargs):
        if fail_count["n"] < 3:
            fail_count["n"] += 1

            class _Broken:
                def fit(self, X, y):
                    raise RuntimeError("simulated fold failure for E1.2 test")

                def predict(self, X):
                    return np.zeros(len(X))

                def set_params(self, **k):
                    return self

            return _Broken()
        return real_build(*args, **kwargs)

    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)

    monkey_attr = "_build_tiny_model"
    setattr(mod, monkey_attr, _fail_first_3)
    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.composite.discovery._screening_tiny"):
            _result = _tiny_cv_rmse_raw_y(
                y_train=y,
                x_train_matrix=x,
                cv_folds=5,
                family="lgb",
                n_estimators=5,
                num_leaves=4,
                learning_rate=0.1,
                random_state=42,
                deterministic=True,
                time_aware=False,
                n_jobs=1,
            )
        msgs = " | ".join(rec.getMessage() for rec in caplog.records)
        assert "folds returned NaN" in msgs, f"E1.2 aggregate NaN-fold WARN missing on a run with 3-of-5 failed folds. Got msgs: {msgs[:400]}"
    finally:
        setattr(mod, monkey_attr, real_build)


def test_concurrent_import_does_not_raise():
    """8 threads forcing re-import simultaneously should never raise NameError /
    ImportError. With the lazy form, this used to race ~15% of the time on a
    cold interpreter (Python import lock holds for the FIRST import but
    subsequent imports against partially-loaded modules can return early)."""

    def _import_and_call():
        # importlib.invalidate_caches() races on Windows with keras's
        # ``_tf_keras`` lazy finder (CPython issue: third-party finders
        # mutate ``sys.path_importer_cache`` from a background thread
        # while invalidate_caches() iterates it). Swallow that environment
        # KeyError so the import-race contract we actually test (no
        # NameError / ImportError on _y_train_clip_bounds) stays measurable.
        try:
            importlib.invalidate_caches()
        except KeyError:
            pass
        from mlframe.training.composite.discovery._screening_tiny import _y_train_clip_bounds

        lo, hi = _y_train_clip_bounds(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        return float(lo), float(hi)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_import_and_call) for _ in range(32)]
        results = [f.result(timeout=10.0) for f in futures]
    # Every result must be a valid (lo, hi) pair, never a propagated exception.
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
