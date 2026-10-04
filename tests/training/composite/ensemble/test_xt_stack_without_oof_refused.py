"""A stacking strategy without honest OOF predictions must not fit weights on in-sample predictions."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
from tests.training.composite.ensemble.test_composite_ensemble_failed_component import _run_builder


def test_nnls_stack_without_oof_falls_back_to_uniform_mean_and_says_so():
    """With the honest OOF block disabled, nnls_stack yields a uniform convex mean and stamps ``xt_ensemble_stack_source``."""
    tt, models, metadata = _run_builder("nnls_stack")
    ens = models[tt.REGRESSION]["_CT_ENSEMBLE__y"][0].model
    assert isinstance(ens, CompositeCrossTargetEnsemble)
    weights = np.asarray(ens.weights, dtype=np.float64)
    np.testing.assert_allclose(weights, np.full(weights.size, 1.0 / weights.size))
    flags = metadata["xt_ensemble_stack_source"]
    assert any("uniform_mean_no_oof" in v and "nnls_stack" in v for v in flags.values())


def test_linear_stack_without_oof_also_refused():
    """The linear stack takes the same refusal path as nnls_stack."""
    tt, models, metadata = _run_builder("linear_stack")
    ens = models[tt.REGRESSION]["_CT_ENSEMBLE__y"][0].model
    weights = np.asarray(ens.weights, dtype=np.float64)
    np.testing.assert_allclose(weights, np.full(weights.size, 1.0 / weights.size))
    assert "xt_ensemble_stack_source" in metadata
