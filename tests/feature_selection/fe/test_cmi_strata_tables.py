"""The table-drawn within-stratum null equals the shuffle null in distribution, is chunk-invariant, and keeps the margins of every stratum."""

import numpy as np

from mlframe.feature_selection.filters import _cmi_strata_njit as S
from mlframe.feature_selection.filters import _cmi_strata_tables_njit as T


def _case(n=20000, seed=0):
    """Dense codes: a 10-stratum s, a binary y, an x that depends on both."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 10, n).astype(np.int64)
    y = (rng.random(n) < 0.4).astype(np.int64)
    x = np.clip((rng.normal(size=n) + 0.3 * y + 0.2 * s) * 2 + 5, 0, 9).astype(np.int64)
    return x, y, s


def test_the_table_null_has_the_distribution_of_the_shuffle_null():
    """Mean, spread and quantiles of 3000 draws agree with 3000 within-stratum shuffles to sampling error (the draws come from the same law)."""
    x, y, s = _case()
    shuffled = S.stratified_cmi_null(x, y, s, S.permutation_sources_chunk(s, 0, 3000, 7, S.strata_layout(s)), 30)
    drawn = T.StratifiedNull(x, y, s, 30).chunk(0, 3000, 7)
    se = shuffled.std() / np.sqrt(3000)
    assert abs(drawn.mean() - shuffled.mean()) < 5 * se
    assert abs(drawn.std() - shuffled.std()) < 0.15 * shuffled.std()
    for q in (0.5, 0.9, 0.99):
        assert abs(np.quantile(drawn, q) - np.quantile(shuffled, q)) < 0.2 * (np.quantile(shuffled, 0.99) - np.quantile(shuffled, 0.01))


def test_a_draw_depends_on_its_seed_alone_not_on_the_chunking():
    """Permutation p is a function of seed + p: slices of the null drawn separately equal the single draw."""
    x, y, s = _case(6000)
    null = T.StratifiedNull(x, y, s, 30)
    whole = null.chunk(0, 70, 3)
    parts = np.concatenate([null.chunk(0, 32, 3), null.chunk(32, 32, 3), null.chunk(64, 6, 3)])
    np.testing.assert_array_equal(whole, parts)


def test_a_drawn_table_keeps_the_margins_of_its_stratum():
    """Row and column sums of every drawn table are the stratum's x level and y level counts."""
    row_m = np.array([40, 25, 0, 35], dtype=np.int64)
    col_m = np.array([55, 45], dtype=np.int64)
    lf = T._log_factorials(int(row_m.sum()))
    state = T.seed_state(5)
    tab = np.zeros((4, 2), dtype=np.int64)
    for _ in range(200):
        state = T._draw_table(state, row_m, col_m, lf, tab)
        assert (tab.sum(axis=1) == row_m).all() and (tab.sum(axis=0) == col_m).all() and (tab >= 0).all()
