"""biz_value test for ``feature_engineering.state_history.last_k_distinct_states_with_durations``.

Source: dd_3rd_nasa-airport-config.md -- "a loop walking backward through configuration-change events,
recording each of the last 10 distinct configurations and how long (in minutes) each was active before the
next change." An entity's PAST dwell-duration in a state can carry a persistent per-entity trait (e.g. some
entities habitually linger longer in a particular configuration each time they revisit it) -- the duration of
one cycle back for the SAME state should predict the current visit's eventual duration, a signal invisible
to a model that only sees the current state.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.state_history import last_k_distinct_states_with_durations


def _make_cyclical_states_with_entity_trait(n_entities: int, seed: int):
    rng = np.random.default_rng(seed)
    seqs, groups = [], []
    for e in range(n_entities):
        states_cycle = [0, 1, 2]
        dwell_c_long = rng.random() < 0.5  # a fixed per-entity trait: does state 2 linger longer for this entity?
        seq = []
        for _ in range(rng.integers(6, 10)):
            for s in states_cycle:
                dur = rng.integers(6, 9) if (s == 2 and dwell_c_long) else rng.integers(2, 4)
                seq += [s] * dur
        seqs.append(np.array(seq))
        groups.append(np.full(len(seq), e))
    return np.concatenate(seqs), np.concatenate(groups)


def test_biz_val_state_history_duration_lag_predicts_next_visit_duration():
    states, group_ids = _make_cyclical_states_with_entity_trait(n_entities=400, seed=0)

    res = last_k_distinct_states_with_durations(states, group_ids, k=5)
    df = pd.DataFrame(res)
    df["state"] = states
    df["group"] = group_ids
    df["is_run_start"] = (df["state"] != df["state"].shift(1)) | (df["group"] != df["group"].shift(1))
    df["run_id"] = df["is_run_start"].cumsum()
    df["current_run_len"] = df.groupby("run_id")["run_id"].transform("size")

    # rows starting a new visit to state 2 -- predict whether THIS visit will be long, using duration_lag_3
    # (one cycle back, the entity's own PREVIOUS visit to state 2's duration -- a purely historical feature).
    mask = df["is_run_start"] & (df["state"] == 2)
    y = (df.loc[mask, "current_run_len"] >= 5).astype(int)
    X = df.loc[mask, ["duration_lag_3"]].fillna(0)

    assert y.nunique() == 2 and min(y.mean(), 1 - y.mean()) > 0.2  # sanity: not a degenerate label.

    model = LogisticRegression().fit(X, y)
    auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    assert auc >= 0.75, f"expected duration_lag_3 to predict the next same-state visit's duration class, got auc={auc:.4f}"


def test_state_history_hand_computed_segments():
    # states: A A A B B C C C C D  (single group)
    states = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
    groups = np.zeros(10, dtype=int)
    res = last_k_distinct_states_with_durations(states, groups, k=3)

    # before any transition, no history yet.
    assert res["state_lag_1"][0] == -1
    assert np.isnan(res["duration_lag_1"][0])

    # at row 3 (first B), the last completed segment is A x3.
    assert res["state_lag_1"][3] == 0
    assert res["duration_lag_1"][3] == 3.0

    # at row 9 (D), history: lag1=C(4), lag2=B(2), lag3=A(3).
    assert res["state_lag_1"][9] == 2 and res["duration_lag_1"][9] == 4.0
    assert res["state_lag_2"][9] == 1 and res["duration_lag_2"][9] == 2.0
    assert res["state_lag_3"][9] == 0 and res["duration_lag_3"][9] == 3.0


def test_state_history_respects_group_boundaries():
    states = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    res = last_k_distinct_states_with_durations(states, groups, k=2)
    # group 1's first row must NOT see group 0's history.
    assert res["state_lag_1"][4] == -1
