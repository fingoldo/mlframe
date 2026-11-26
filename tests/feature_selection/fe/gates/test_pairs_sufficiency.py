"""``pair_regression_stats``: does a pair's winning engineered column summarise the pair (sufficiency), and does the pair carry an interaction (additive against tensor)?

The fixtures are the cases the plug-in MI ratio of the joint-prevalence gate cannot separate: an additive pair whose winner is exact (sin(x1) + x2 style, ratio 0.77 against a 0.90 bar), a genuine interaction (a^2 / b), and a
winner that is a degraded summary of an additive pair."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_sufficiency import MAX_ROWS, interaction_verdict, pair_regression_stats

N = 6000


def _operands(seed: int = 0):
    """Two positive operands and a noise source."""
    rng = np.random.default_rng(seed)
    return rng.random(N) + 0.1, rng.random(N) + 0.1, rng


def test_an_exact_additive_winner_is_sufficient_and_there_is_no_interaction():
    """y = a + sin(5 b): the winner is the exact sum, the pair is additive."""
    a, b, rng = _operands()
    winner = a + np.sin(5 * b)
    s = pair_regression_stats(winner, a, b, winner + rng.normal(0, 0.3, N))
    assert s["t_g_vs_ten"] < 3.0, s
    assert s["t_add_vs_ten"] < 3.0, s
    assert s["skill_g"] > 0.55


def test_a_genuine_interaction_shows_in_the_additive_against_tensor_statistic():
    """y = 0.2 a^2 / b: no sum of two 1-D effects expresses it, and the winner a^2 / b summarises the pair."""
    a, b, rng = _operands()
    winner = a**2 / b
    s = pair_regression_stats(winner, a, b, 0.2 * winner + rng.normal(0, 0.1, N))
    assert s["t_add_vs_ten"] > 10.0, s
    # the tensor smooth beats the exact winner by a statistically visible but practically negligible margin (a coarse rank-space basis against a function that rises sharply near b = 0): the effect size is what counts
    assert (s["skill_ten"] - s["skill_g"]) / s["skill_ten"] < 0.05, s


def test_a_degraded_winner_is_not_sufficient():
    """The additive target with a winner a * b that does not carry it: the tensor smooth beats the winner by a wide margin."""
    a, b, rng = _operands()
    truth = a + np.sin(5 * b)
    s = pair_regression_stats(a * b, a, b, truth + rng.normal(0, 0.3, N))
    assert s["t_g_vs_ten"] > 20.0, s
    assert s["skill_ten"] - s["skill_g"] > 0.3


def test_the_statistics_are_invariant_to_monotone_rescaling_of_the_inputs():
    """Rank transforms: exp / cube of the operands and the winner leave every number unchanged."""
    a, b, rng = _operands()
    winner = a**2 / b
    y = 0.2 * winner + rng.normal(0, 0.1, N)
    base = pair_regression_stats(winner, a, b, y)
    moved = pair_regression_stats(np.exp(winner / 10.0), a**3, np.exp(b), y)
    assert base
    for key in base:
        assert moved[key] == pytest.approx(base[key], rel=1e-6, abs=1e-6), key


def test_degenerate_inputs_return_an_empty_dict():
    """Too few rows or a constant target cannot be modelled."""
    a, b, rng = _operands()
    assert pair_regression_stats(a[:100], a[:100], b[:100], rng.normal(size=100)) == {}
    assert pair_regression_stats(a, a, b, np.zeros(N)) == {}
    assert pair_regression_stats(a, a, b, np.where(np.arange(N) == 3, np.nan, 1.0)) == {}


def test_a_large_frame_is_subsampled_once_and_the_answer_is_stable():
    """Above ``MAX_ROWS`` the statistics are computed on a seeded draw, so a frame twice the cap gives the verdict of the cap."""
    rng = np.random.default_rng(1)
    n = 2 * MAX_ROWS
    a, b = rng.random(n) + 0.1, rng.random(n) + 0.1
    winner = a**2 / b
    s = pair_regression_stats(winner, a, b, 0.2 * winner + rng.normal(0, 0.1, n))
    assert s["t_add_vs_ten"] > 10.0 and (s["skill_ten"] - s["skill_g"]) / s["skill_ten"] < 0.05
    assert pair_regression_stats(winner, a, b, 0.2 * winner + rng.normal(0, 0.1, n), seed=0)["skill_g"] == pytest.approx(s["skill_g"], abs=0.02)


def test_the_verdict_admits_an_interaction_summarised_by_its_winner():
    """a^2 / b with the exact winner: admitted; the reason names the interaction statistic."""
    a, b, rng = _operands()
    winner = a**2 / b
    ok, why = interaction_verdict(pair_regression_stats(winner, a, b, 0.2 * winner + rng.normal(0, 0.1, N)))
    assert ok, why
    assert "interaction" in why


def test_the_verdict_refuses_an_additive_pair_even_with_an_exact_winner():
    """sin(x1) + x2 style: the winner is exact and sufficient, and the operands already carry the pair (the unary credit of x1 is what is missing, not a pair feature)."""
    a, b, rng = _operands()
    winner = a + np.sin(5 * b)
    ok, why = interaction_verdict(pair_regression_stats(winner, a, b, winner + rng.normal(0, 0.3, N)))
    assert not ok
    assert "additive" in why


def test_the_verdict_refuses_a_winner_that_is_not_a_summary_of_the_pair():
    """An interaction pair whose winner is a * b instead of a^2 / b loses most of the pair's R^2."""
    a, b, rng = _operands()
    ok, why = interaction_verdict(pair_regression_stats(a * b, a, b, 0.2 * a**2 / b + rng.normal(0, 0.1, N)))
    assert not ok
    assert "not a summary" in why


def test_the_verdict_refuses_when_nothing_could_be_modelled():
    """An empty statistics dict (too few rows, constant target) is a refusal, not an exception."""
    assert interaction_verdict({}) == (False, "the pair could not be modelled")
    assert interaction_verdict({"skill_ten": -0.01, "skill_g": -0.02, "skill_add": -0.02, "t_add_vs_ten": 3.0, "t_g_vs_ten": 1.0})[0] is False


def test_the_winner_column_is_rebuilt_where_the_chunk_buffer_is_not_kept():
    """``_winner_column`` (shared by the interaction accept path and the Miller-Madow debias): the buffer column when kept, the device resolver on the deferred path, else the recipe replayed on the operand table; ``None`` for an unknown column. Before, the debias called the resolver unconditionally and it raised where the buffer was not kept."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_score_step4 import _winner_column

    a, b, _ = _operands()
    table = np.column_stack([a, b])
    vars_tr = {("a", "identity"): 0, ("b", "identity"): 1}
    binary = {"mul": lambda u, v: u * v}
    kwargs = {"transformed_vars": table, "vars_transformations": vars_tr, "binary_transformations": binary}
    buffer = np.column_stack([a + b, a * b])
    np.testing.assert_array_equal(_winner_column(1, buffer, False, None, None, **kwargs), a * b)
    np.testing.assert_array_equal(_winner_column(7, None, True, lambda ci: np.full(N, float(ci)), None, **kwargs), np.full(N, 7.0))
    rebuilt = _winner_column(3, None, False, lambda ci: 1 / 0, {3: (("a", "identity"), ("b", "identity"), "mul")}, **kwargs)
    np.testing.assert_allclose(rebuilt, (a * b).astype(np.float32), rtol=1e-6)
    assert _winner_column(9, None, False, lambda ci: 1 / 0, {3: (("a", "identity"), ("b", "identity"), "mul")}, **kwargs) is None
