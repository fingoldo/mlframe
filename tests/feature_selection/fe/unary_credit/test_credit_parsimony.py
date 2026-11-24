"""``_fe_credit_parsimony``: backward elimination of the columns a credited form makes redundant; a column goes only when its removal costs under ``R2_DROP`` of out-of-fold R^2 on BOTH the linear and the smooth leg."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_credit_parsimony import R2_DROP, credit_parsimony_enabled, eliminate, rank_hats

N = 4000


def _design(seed: int = 0):
    """Columns: 0 = exact form, 1 = the other signal, 2 = a re-expression of column 0 plus tiny noise, 3 and 4 = noise, 5 = a column that carries real extra signal (y uses it)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-6, 6, N)
    exact = np.sin(x)
    other = rng.standard_normal(N)
    extra = rng.standard_normal(N)
    y = exact + other + 0.5 * extra + 0.2 * rng.standard_normal(N)
    columns = np.column_stack([exact, other, 0.7 * exact + 0.02 * rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N), extra])
    return columns, y


def test_elimination_drops_redundant_and_noise_columns_and_keeps_the_signal():
    """The re-expression (2) and the two noise columns (3, 4) go; the column that carries extra signal (5) stays even though it is removable."""
    columns, y = _design()
    assert sorted(eliminate(columns, y, removable=[2, 3, 4, 5], seed=0)) == [2, 3, 4]


def test_a_column_needed_only_non_linearly_is_kept():
    """y has a squared term of column 3: a ridge on the columns sees nothing in it, the smooth leg does, so the column stays."""
    rng = np.random.default_rng(2)
    a, b, c = (rng.standard_normal(N) for _ in range(3))
    y = a + c**2 + 0.2 * rng.standard_normal(N)
    columns = np.column_stack([a, b, c])
    assert eliminate(columns, y, removable=[1, 2], seed=0) == [1]


def test_protected_columns_are_never_dropped():
    """Only the columns named removable are candidates, whatever their cost."""
    columns, y = _design()
    assert eliminate(columns, y, removable=[], seed=0) == []
    assert 0 not in eliminate(columns, y, removable=[3], seed=0)


def test_each_removal_is_judged_against_the_shrunken_design():
    """Two near-copies of a signal column: the first copy is removable, and then the second is not (it is now the only carrier)."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(N)
    columns = np.column_stack([base + 0.01 * rng.standard_normal(N), base + 0.01 * rng.standard_normal(N), rng.standard_normal(N)])
    y = base + 0.3 * rng.standard_normal(N)
    dropped = eliminate(columns, y, removable=[0, 1, 2], seed=0)
    assert 2 in dropped and len({0, 1} & set(dropped)) == 1


def test_a_tiny_but_significant_sliver_does_not_keep_a_column():
    """The effect size, not the significance, decides: a column that adds under ``R2_DROP`` of the variance is dropped however many rows make it significant."""
    rng = np.random.default_rng(3)
    big = 20000
    a = rng.standard_normal(big)
    sliver = rng.standard_normal(big)
    y = a + 0.03 * sliver + 0.3 * rng.standard_normal(big)  # 0.03^2 / var(y) = 0.0008 < R2_DROP
    assert eliminate(np.column_stack([a, sliver]), y, removable=[1], seed=0) == [1]
    assert 0.03**2 / float(np.var(y)) < R2_DROP


def test_a_reordering_of_a_variable_already_present_does_not_look_useful():
    """A heavy-tailed reciprocal of x2 re-orders the same variable: the coarse hats of its rank can refine the hats of x2 (measured: a loss of 0.0047 on the smooth leg when the straight line was not part of the block),
    while the linear term of x2 is already there. It must go, with the raw x1 that sin(x1) replaces, and x2 itself must stay."""
    rng = np.random.default_rng(0)
    n = 6000
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.standard_normal(n)
    y = np.sin(x1) + x2 + 0.3 * rng.standard_normal(n)
    columns = np.column_stack([x2, x1, 1.0 / (x2 + 1e-3 * np.where(x2 >= 0, 1.0, -1.0)), np.sin(x1)])
    assert sorted(eliminate(columns, y, removable=[0, 1, 2], seed=0)) == [1, 2]


def test_rank_hats_are_invariant_to_a_monotone_rescaling():
    """The smooth leg sees ranks: exp of a column gives the same basis."""
    x = np.random.default_rng(4).standard_normal(500)
    np.testing.assert_array_equal(rank_hats(x), rank_hats(np.exp(x)))


def test_the_rollback_switch(monkeypatch):
    """``MLFRAME_FE_CREDIT_PARSIMONY=0`` turns the pass off."""
    monkeypatch.setenv("MLFRAME_FE_CREDIT_PARSIMONY", "0")
    assert not credit_parsimony_enabled()
    monkeypatch.setenv("MLFRAME_FE_CREDIT_PARSIMONY", "1")
    assert credit_parsimony_enabled()
