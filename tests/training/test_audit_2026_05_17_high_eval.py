"""Regression tests for the 2026-05-17 evaluation/diagnostics audit.

H-EVA-01: ``_compute_quantile_baselines`` had two definitions in
``dummy_baselines.py`` — one imported from ``_dummy_baseline_compute`` and a
duplicate further down the file shadowing it (F811-suppressed). The shadow
copy was removed; this test pins the resolved symbol to the canonical
``_dummy_baseline_compute`` module so the duplicate cannot silently return.
"""
from __future__ import annotations

import inspect

import pytest


def test_compute_quantile_baselines_resolves_to_canonical_module():
    """The name resolves to the imported version, not an in-file shadow."""
    from mlframe.training import dummy_baselines

    fn = dummy_baselines._compute_quantile_baselines
    assert fn.__module__ == "mlframe.training._dummy_baseline_compute", (
        f"_compute_quantile_baselines must come from _dummy_baseline_compute, "
        f"got {fn.__module__}"
    )


def test_compute_quantile_baselines_defined_only_in_compute_module():
    """The shadow definition in dummy_baselines.py is gone (no F811 noqa left)."""
    from mlframe.training import dummy_baselines

    src = inspect.getsource(dummy_baselines)
    # Only the `from ._dummy_baseline_compute import (... _compute_quantile_baselines ...)`
    # mention should remain; no `def _compute_quantile_baselines(` line.
    assert "def _compute_quantile_baselines(" not in src, (
        "dummy_baselines.py must not redefine _compute_quantile_baselines"
    )


def test_compute_quantile_baselines_callable_via_dummy_baselines():
    """Smoke: the re-exported symbol is callable and returns the expected shape."""
    import numpy as np

    from mlframe.training.dummy_baselines import _compute_quantile_baselines

    rng = np.random.default_rng(0)
    train_y = rng.normal(size=200)
    val_y = rng.normal(size=50)
    test_y = rng.normal(size=50)
    alphas = (0.1, 0.5, 0.9)

    val_preds, test_preds, extras = _compute_quantile_baselines(
        target_name="y",
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        alphas=alphas,
        config=None,
    )
    # multi_quantile_empirical should be (N, K) shaped
    assert "multi_quantile_empirical" in val_preds
    assert val_preds["multi_quantile_empirical"].shape == (50, 3)
    assert test_preds["multi_quantile_empirical"].shape == (50, 3)
    # median_for_all is always emitted
    assert "median_for_all" in val_preds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
