"""Regression tests for the 2026-05-17 evaluation/diagnostics audit.

H-EVA-01: ``_compute_quantile_baselines`` had two definitions in
``dummy_baselines.py`` -- one imported from ``_dummy_baseline_compute`` and a
duplicate further down the file shadowing it (F811-suppressed). The shadow
copy was removed; this test pins the resolved symbol to the canonical
module so the duplicate cannot silently return.

Wave 92 (2026-05-21): the canonical home moved from
``_dummy_baseline_compute`` to the sibling file
``_dummy_baseline_quantile`` (the original module re-exports). The
identity check below still guards against shadowing because both
modules expose the same underlying function object.
"""

from __future__ import annotations

import pytest


def test_compute_quantile_baselines_resolves_to_canonical_module():
    """The name resolves to the imported version, not an in-file shadow."""
    from mlframe.training.baselines import dummy as dummy_baselines

    fn = dummy_baselines._compute_quantile_baselines
    # Wave 92: canonical module is now the sibling split file; the original
    # _dummy_baseline_compute re-exports for backward compat. Either path is
    # acceptable but a local shadow def (mlframe.training.dummy_baselines)
    # would be the bug we're guarding against.
    assert fn.__module__ in (
        "mlframe.training.baselines._dummy_baseline_quantile",
        "mlframe.training.baselines._dummy_baseline_compute",
    ), f"_compute_quantile_baselines must come from the compute / quantile split module, got {fn.__module__}"


def test_compute_quantile_baselines_defined_only_in_compute_module():
    """Behavioural: the canonical definition lives in _dummy_baseline_compute
    (and post-wave-92, in the _dummy_baseline_quantile sibling that the
    compute module re-exports). The dummy_baselines module's symbol must be
    the SAME function object as the one re-exported from
    _dummy_baseline_compute -- not a wrapper, not a shadow.
    """
    from mlframe.training.baselines import dummy as dummy_baselines
    from mlframe.training.baselines import _dummy_baseline_compute

    # Identity: dummy_baselines._compute_quantile_baselines IS the function
    # from the compute module - not a wrapper, not a shadow.
    assert dummy_baselines._compute_quantile_baselines is _dummy_baseline_compute._compute_quantile_baselines, (
        "dummy_baselines._compute_quantile_baselines was shadowed by a local def"
    )


def test_compute_quantile_baselines_callable_via_dummy_baselines():
    """Smoke: the re-exported symbol is callable and returns the expected shape."""
    import numpy as np

    from mlframe.training.baselines.dummy import _compute_quantile_baselines

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
