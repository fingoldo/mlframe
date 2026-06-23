"""Regression tests for CON18/CON19: synthetic-probability generators and the redundant-factor builder drew from numpy's process-global RNG (non-reproducible AND polluting
the caller's stream). The fix routes every draw through a seeded ``check_random_state`` RNG (or, for the njit generator, a seeded njit-local RNG that is independent of
numpy's global state). Each test pins: same seed -> identical output, and numpy's global RNG state is untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _global_state_unchanged(before) -> bool:
    after = np.random.get_state()
    return before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.fixture(scope="module")
def _probs():
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(0.5, 0.2, size=300), 1e-3, 1 - 1e-3)
    y = (rng.random(300) < p).astype(int)
    return p, y


def test_con18_logit_space_seeded(_probs):
    from mlframe.calibration.probabilities import generate_similar_probs_logit_space
    p, y = _probs
    st = np.random.get_state()
    a = generate_similar_probs_logit_space(p, y, random_state=5)
    b = generate_similar_probs_logit_space(p, y, random_state=5)
    c = generate_similar_probs_logit_space(p, y, random_state=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert _global_state_unchanged(st)


def test_con18_random_walk_seeded(_probs):
    from mlframe.calibration.probabilities import generate_similar_probs_random_walk
    p, y = _probs
    st = np.random.get_state()
    a = generate_similar_probs_random_walk(p, y, n_steps=3, random_state=5)
    b = generate_similar_probs_random_walk(p, y, n_steps=3, random_state=5)
    c = generate_similar_probs_random_walk(p, y, n_steps=3, random_state=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert _global_state_unchanged(st)


def test_con18_similar_probs_seeded(_probs):
    from mlframe.calibration.probabilities import generate_similar_probs
    p, y = _probs
    st = np.random.get_state()
    a = generate_similar_probs(p, y, random_state=5)
    b = generate_similar_probs(p, y, random_state=5)
    assert np.array_equal(a, b)
    assert _global_state_unchanged(st)


def test_con18_by_ranking_seeded(_probs):
    from mlframe.calibration.probabilities import generate_similar_probs_by_ranking
    p, y = _probs
    st = np.random.get_state()
    a = generate_similar_probs_by_ranking(p, y, random_state=5)
    b = generate_similar_probs_by_ranking(p, y, random_state=5)
    c = generate_similar_probs_by_ranking(p, y, random_state=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert _global_state_unchanged(st)


def test_con18_generate_probs_from_outcomes_seeded():
    from mlframe.calibration.probabilities import generate_probs_from_outcomes
    outcomes = (np.arange(500) % 2).astype(np.int64)
    st = np.random.get_state()
    a = generate_probs_from_outcomes(outcomes, random_state=5)
    b = generate_probs_from_outcomes(outcomes, random_state=5)
    c = generate_probs_from_outcomes(outcomes, random_state=6)
    assert np.array_equal(a, b), "same random_state must give identical synthetic probs"
    assert not np.array_equal(a, c)
    # njit RNG is independent of numpy's global stream, so it must remain untouched.
    assert _global_state_unchanged(st)


def test_con19_create_redundant_continuous_factor_seeded():
    from mlframe.feature_selection.filters.discretization._discretization_dataset import create_redundant_continuous_factor

    base = pd.DataFrame({"a": np.arange(200.0), "b": np.arange(200.0) * 2})

    df1 = base.copy()
    df2 = base.copy()
    df3 = base.copy()
    st = np.random.get_state()
    create_redundant_continuous_factor(df1, ["a", "b"], name="r", random_state=5)
    create_redundant_continuous_factor(df2, ["a", "b"], name="r", random_state=5)
    create_redundant_continuous_factor(df3, ["a", "b"], name="r", random_state=6)

    assert np.array_equal(df1["r"].to_numpy(), df2["r"].to_numpy())
    assert not np.array_equal(df1["r"].to_numpy(), df3["r"].to_numpy())
    assert _global_state_unchanged(st)
