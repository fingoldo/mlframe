"""Audit 2026-11-20 findings 39-F2 and 39-F3: batched predict must refuse misaligned or truncated output, never return it silently."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core.predict import _run_batched


def _batches_fn(outputs_by_batch):
    """An entry function returning the next prepared output for each batch it is called on."""
    state = {"i": 0}

    def fn(_slice, *args, **kwargs):
        """Return the prepared output for this batch index."""
        out = outputs_by_batch[state["i"]]
        state["i"] += 1
        return out

    return fn


def test_missing_ensemble_key_in_one_batch_raises():
    """Batch 0 has ensemble_predictions, batch 1 does not: concatenating would misalign rows, so it raises (39-F2)."""
    df = pd.DataFrame({"a": np.arange(4.0)})
    outs = [{"ensemble_predictions": np.zeros(2)}, {}]
    with pytest.raises(RuntimeError, match="missing from 1 of 2 batches"):
        _run_batched(_batches_fn(outs), df, 2)


def test_shape_mismatch_across_batches_raises_not_truncates():
    """Batches with incompatible shapes raise with the shapes; the result is never truncated to batch 0 (39-F3)."""
    df = pd.DataFrame({"a": np.arange(4.0)})
    outs = [{"ensemble_predictions": np.zeros((2, 3))}, {"ensemble_predictions": np.zeros((2,))}]
    with pytest.raises(RuntimeError, match="np.concatenate"):
        _run_batched(_batches_fn(outs), df, 2)


def test_aligned_batches_concatenate_in_row_order():
    """Equal-shape batches concatenate to the full row count, in order."""
    df = pd.DataFrame({"a": np.arange(4.0)})
    outs = [{"ensemble_predictions": np.array([1.0, 2.0])}, {"ensemble_predictions": np.array([3.0, 4.0])}]
    merged = _run_batched(_batches_fn(outs), df, 2)
    np.testing.assert_array_equal(merged["ensemble_predictions"], [1.0, 2.0, 3.0, 4.0])
