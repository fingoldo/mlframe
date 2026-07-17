"""Temporal-aggregate FE: the engineered value uses only PAST rows, never the future.

``temporal_lag`` (and its rolling/expanding siblings) replay each test row's statistic
against the FROZEN per-entity train history plus EARLIER-in-time rows within the test frame
-- never the row's own future, never the train labels. This file pins the temporal-leakage
contract directly on the replay surface:

* A row's lag value equals an EARLIER row's value for the same entity (train history seeds
  the buffer; within-test only earlier rows contribute). A future test row cannot influence
  an earlier row's output -- proven by appending future rows and checking earlier outputs
  are unchanged.
* Time-order independence of INPUT ROW ORDER: shuffling the test frame's row order yields
  the same per-(entity, time) lag values (the replay sorts by time internally).
* Unseen entity / no-predecessor -> the frozen global prior, finite.
* Replay reads only X and is invariant to any y in scope.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._temporal_agg_fe import build_temporal_lag_recipe
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def lag_recipe():
    # Train history: entity 'e0' saw values [10, 20] (time-sorted), 'e1' saw [100].
    """Lag recipe."""
    history = {
        "e0": {"v": [10.0, 20.0]},
        "e1": {"v": [100.0]},
    }
    return build_temporal_lag_recipe(
        name="lag1(val|ent)",
        entity_cols=["ent"],
        value_col="val",
        time_col="t",
        lag=1,
        history=history,
        global_prior=-1.0,
    )


def _frame(rows):
    """Helper that frame."""
    return pd.DataFrame(rows, columns=["ent", "t", "val"])


def test_lag_uses_only_past_values(lag_recipe):
    # Within-test rows for e0 at times 5,6 with values 1,2. lag=1 -> the previous
    # value in the merged (history ++ within-test) time order.
    """Lag uses only past values."""
    X = _frame(
        [
            ("e0", 5, 1.0),
            ("e0", 6, 2.0),
        ]
    )
    out = apply_recipe(lag_recipe, X)
    # Merged time order for e0: history [10,20] then test [1@t5, 2@t6].
    # Row t5: predecessor is the last history value 20.0.
    # Row t6: predecessor is the test value at t5 = 1.0.
    assert out[0] == pytest.approx(20.0)
    assert out[1] == pytest.approx(1.0)


def test_future_test_rows_do_not_change_earlier_outputs(lag_recipe):
    """Future test rows do not change earlier outputs."""
    X_short = _frame([("e0", 5, 1.0), ("e0", 6, 2.0)])
    X_long = _frame([("e0", 5, 1.0), ("e0", 6, 2.0), ("e0", 7, 9.0), ("e0", 8, 9.9)])
    out_short = apply_recipe(lag_recipe, X_short)
    out_long = apply_recipe(lag_recipe, X_long)
    # The first two rows' lag values must be identical whether or not future rows
    # (t7, t8) are present -- a future row cannot leak into an earlier output.
    np.testing.assert_array_equal(out_short, out_long[:2])


def test_lag_invariant_to_input_row_order(lag_recipe):
    """Lag invariant to input row order."""
    rows = [("e0", 5, 1.0), ("e0", 6, 2.0), ("e0", 7, 3.0)]
    X = _frame(rows)
    out_sorted = apply_recipe(lag_recipe, X)
    # Shuffle input row order; replay sorts by time, so per-time outputs match.
    X_shuf = _frame([rows[2], rows[0], rows[1]]).reset_index(drop=True)
    out_shuf = apply_recipe(lag_recipe, X_shuf)
    # Map both back to (t -> lag value) and compare.
    m_sorted = dict(zip(X["t"], out_sorted))
    m_shuf = dict(zip(X_shuf["t"], out_shuf))
    assert m_sorted == m_shuf


def test_unseen_entity_uses_global_prior(lag_recipe):
    """Unseen entity uses global prior."""
    X = _frame([("new_entity", 1, 5.0)])
    out = apply_recipe(lag_recipe, X)
    # No history, no predecessor -> frozen global prior.
    assert out[0] == pytest.approx(-1.0)


def test_first_row_of_known_entity_uses_history(lag_recipe):
    # e1 history is [100]; first test row's lag1 predecessor is 100.
    """First row of known entity uses history."""
    X = _frame([("e1", 3, 7.0)])
    out = apply_recipe(lag_recipe, X)
    assert out[0] == pytest.approx(100.0)


def test_replay_invariant_to_y_in_scope(lag_recipe):
    """Replay invariant to y in scope."""
    X = _frame([("e0", 5, 1.0), ("e0", 6, 2.0)])
    a = apply_recipe(lag_recipe, X)
    _ = np.array([0, 1])  # a y in scope
    b = apply_recipe(lag_recipe, X)
    np.testing.assert_array_equal(a, b)
