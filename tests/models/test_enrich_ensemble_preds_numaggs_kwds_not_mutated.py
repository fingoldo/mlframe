"""MODELS-8 regression test: enrich_ensemble_preds_with_numaggs must not mutate the caller's numaggs_kwds dict.

The bug (fixed): the function called ``numaggs_kwds.update(...)`` directly on the caller-supplied dict --
a caller reusing that dict for a second call (or reading it afterward) would silently see
directional_only/return_hurst/return_entropy keys injected by this function.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.base import enrich_ensemble_preds_with_numaggs

pytestmark = pytest.mark.fast


def test_caller_numaggs_kwds_not_mutated_wide_predictions():
    """A caller's numaggs_kwds dict must be unchanged after a call with >=10 prediction columns."""
    predictions = np.random.default_rng(0).uniform(0, 1, size=(5, 12)).astype(np.float32)
    kwds = {"whiten_means": False}
    kwds_before = dict(kwds)

    enrich_ensemble_preds_with_numaggs(predictions, numaggs_kwds=kwds)

    assert kwds == kwds_before, f"caller's numaggs_kwds should be unchanged, got {kwds}"


def test_caller_numaggs_kwds_not_mutated_narrow_predictions():
    """A caller's numaggs_kwds dict must be unchanged after a call with <10 prediction columns."""
    predictions = np.random.default_rng(1).uniform(0, 1, size=(5, 3)).astype(np.float32)
    kwds = {"whiten_means": True}
    kwds_before = dict(kwds)

    enrich_ensemble_preds_with_numaggs(predictions, numaggs_kwds=kwds)

    assert kwds == kwds_before, f"caller's numaggs_kwds should be unchanged, got {kwds}"


def test_same_kwds_dict_reused_across_two_calls_of_different_width():
    """Reusing the same dict across two calls with different prediction widths must not leak state
    between calls (each call must independently derive its own directional_only/return_hurst/... flags)."""
    kwds = {"whiten_means": False}
    wide = np.random.default_rng(2).uniform(0, 1, size=(4, 12)).astype(np.float32)
    narrow = np.random.default_rng(3).uniform(0, 1, size=(4, 3)).astype(np.float32)

    out_wide = enrich_ensemble_preds_with_numaggs(wide, numaggs_kwds=kwds)
    out_narrow = enrich_ensemble_preds_with_numaggs(narrow, numaggs_kwds=kwds)

    assert kwds == {"whiten_means": False}
    assert out_wide.shape[0] == 4
    assert out_narrow.shape[0] == 4
