"""Audit 2026-11-20 finding 35-F37: a WAIC scoring failure is logged at WARNING, naming the spec, not swallowed at DEBUG.

The spec then keeps its RMSE position and the tie-break silently skips it, so the failure must be visible.
"""

import logging
from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery import _tiny_rerank_waic as waic


def test_failing_forward_is_logged_with_the_spec_name(monkeypatch, caplog):
    """When the transform forward raises, a WARNING names the spec and the exception."""
    n = 200
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    x_mat = rng.normal(size=(n, 3))
    specs = [SimpleNamespace(name="spec_fails", base_column="b", transform_name="diff", fitted_params={}),
             SimpleNamespace(name="spec_b", base_column="b", transform_name="diff", fitted_params={})]

    def boom(*args, **kwargs):
        """Stand in for a transform forward that raises."""
        raise RuntimeError("forward broke")

    monkeypatch.setattr(waic, "call_transform", boom)
    self_ = SimpleNamespace(config=SimpleNamespace(transform_waic_n_folds=4, random_state=0))

    class _Cache:
        """A per-base cache returning the base column and the matrix."""

        def get(self, key):
            """Return the cached arrays for any base."""
            return base, x_mat

    y = rng.normal(size=n)
    with caplog.at_level(logging.WARNING):
        waic._apply_waic_tiebreak(self_, np.array([0, 1]), specs, np.array([0.1, 0.1005]), ["spec_fails", "spec_b"], y_screen=y, per_base_cache=_Cache())
    assert any(r.levelno == logging.WARNING and "spec_fails" in r.getMessage() and "RuntimeError" in r.getMessage() for r in caplog.records)
