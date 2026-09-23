"""A model that fails on one batch must not come back as a short, silently misaligned array.

`_concat_probs_dicts` skipped batches a model produced nothing for, so `results["predictions"]["M"]` returned 4/5 of the
rows under the model's own name with no length signal: every caller zipping it against the input frame assigned
predictions to the wrong rows from the failed batch onwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core.predict import _concat_probs_dicts, _run_batched


def test_a_missing_batch_is_nan_filled_to_its_own_row_count():
    """The failed batch's rows must become NaN in place; a shorter array would silently shift every later prediction."""
    parts = [{"ok": np.zeros(10), "flaky": np.ones(10)}, {"ok": np.zeros(7)}, {"ok": np.zeros(10), "flaky": np.ones(10)}]
    partial: set[str] = set()
    out = _concat_probs_dicts(parts, [10, 7, 10], partial)
    assert out["ok"].shape == (27,)
    assert out["flaky"].shape == (27,), "the gap must occupy the failed batch's rows, not vanish"
    assert np.isnan(out["flaky"][10:17]).all()
    assert not np.isnan(np.delete(out["flaky"], np.arange(10, 17))).any()
    assert partial == {"flaky"}


def test_a_two_dimensional_probability_array_keeps_its_columns():
    """The fill has to match the class dimension too, not collapse a probability matrix to one column."""
    parts = [{"m": np.ones((4, 3))}, {}, {"m": np.ones((4, 3))}]
    out = _concat_probs_dicts(parts, [4, 5, 4], set())
    assert out["m"].shape == (13, 3)
    assert np.isnan(out["m"][4:9]).all()


def test_integer_predictions_are_widened_so_the_gap_can_be_represented():
    """An int array cannot hold NaN; silently wrapping to a sentinel integer would be worse than widening."""
    parts = [{"m": np.zeros(4, dtype=np.int64)}, {}]
    out = _concat_probs_dicts(parts, [4, 2], set())
    assert np.issubdtype(out["m"].dtype, np.floating)
    assert np.isnan(out["m"][4:]).all()


def test_nothing_changes_when_every_batch_delivered():
    """The negative control: with no gap the result is a plain concatenation, byte for byte."""
    parts = [{"m": np.arange(3.0)}, {"m": np.arange(3.0)}]
    np.testing.assert_array_equal(_concat_probs_dicts(parts, [3, 3], set())["m"], np.concatenate([np.arange(3.0)] * 2))


def test_an_unbatched_caller_keeps_the_legacy_shape():
    """A caller that passes no row counts cannot have its rows realigned, so it keeps the old behaviour."""
    parts = [{"m": np.arange(3.0)}, {}]
    assert _concat_probs_dicts(parts)["m"].shape == (3,)


def test_run_batched_lists_the_partial_model():
    """A model that produced nothing for some batch is named to the caller, not quietly NaN-filled and forgotten."""
    calls = {"n": 0}

    def entry_fn(frame, **kwargs):
        """Predict every batch but the second, where one model disappears."""
        calls["n"] += 1
        preds = {"good": np.zeros(len(frame))}
        if calls["n"] != 2:  # the second batch loses one model
            preds["flaky"] = np.ones(len(frame))
        return {"predictions": preds}

    df = pd.DataFrame({"x": np.arange(25.0)})
    out = _run_batched(entry_fn, df, 10)
    assert out["partial_models"] == ["flaky"]
    assert out["predictions"]["good"].shape == (25,)
    assert out["predictions"]["flaky"].shape == (25,)
    assert np.isnan(out["predictions"]["flaky"][10:20]).all()
