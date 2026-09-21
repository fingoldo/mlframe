"""Within composite post-processing, each (model, frame) pair must be predicted once.

The wrap pass predicted every composite on val and test; the cross-target ensemble report then predicted the ensemble on
the same frames, the MoE gate predicted that ensemble and the raw model on val again, and the refit pre-screen predicted
each component on val once more - each call re-running the shim's pre-pipeline over the whole frame. The phase now
memoises predictions on (model, frame) for its duration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._prediction_memo import memo_predict, prediction_memo, with_prediction_memo


class _Counting:
    """A model that records every predict call."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.calls = 0

    def predict(self, frame):
        """Return a deterministic function of the frame, counting the call."""
        self.calls += 1
        return np.asarray(frame["x"], dtype=np.float64) * self.scale


def _frame(n: int = 5) -> pd.DataFrame:
    """A tiny frame."""
    return pd.DataFrame({"x": np.arange(n, dtype=np.float64)})


def test_a_pair_is_predicted_once_inside_the_memo():
    """Two requests for the same model and frame run one predict."""
    model, frame = _Counting(), _frame()
    with prediction_memo():
        first = memo_predict(model, frame)
        second = memo_predict(model, frame)
    assert model.calls == 1
    np.testing.assert_array_equal(first, second)


def test_different_frames_and_models_are_kept_apart():
    """The key is the pair: another frame or another model is a fresh predict."""
    a, b = _Counting(1.0), _Counting(2.0)
    f1, f2 = _frame(), _frame()
    with prediction_memo():
        memo_predict(a, f1)
        memo_predict(a, f2)
        out_b = memo_predict(b, f1)
    assert a.calls == 2 and b.calls == 1
    np.testing.assert_array_equal(out_b, 2.0 * np.arange(5))


def test_outside_the_memo_every_call_predicts():
    """No active phase, no caching: behaviour is exactly a plain predict."""
    model, frame = _Counting(), _frame()
    memo_predict(model, frame)
    memo_predict(model, frame)
    assert model.calls == 2


def test_the_memo_is_dropped_when_the_phase_ends():
    """A later phase must not see predictions from an earlier one."""
    model, frame = _Counting(), _frame()
    with prediction_memo():
        memo_predict(model, frame)
    with prediction_memo():
        memo_predict(model, frame)
    assert model.calls == 2


def test_callers_cannot_edit_each_others_predictions():
    """Each caller gets its own copy, so an in-place edit cannot leak into the next caller's result."""
    model, frame = _Counting(), _frame()
    with prediction_memo():
        first = memo_predict(model, frame)
        first[:] = -1.0
        second = memo_predict(model, frame)
    np.testing.assert_array_equal(second, np.arange(5, dtype=np.float64))


def test_the_decorator_activates_the_memo_for_the_call():
    """``with_prediction_memo`` scopes the memo to one call of the decorated phase."""
    model, frame = _Counting(), _frame()

    @with_prediction_memo
    def phase():
        """Predict the same pair twice."""
        memo_predict(model, frame)
        memo_predict(model, frame)

    phase()
    assert model.calls == 1
    memo_predict(model, frame)
    assert model.calls == 2, "the memo must not outlive the decorated call"
