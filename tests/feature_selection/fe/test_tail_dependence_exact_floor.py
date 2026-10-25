"""The tail-dependence permutation floor is the exact shuffle expectation, not a Monte-Carlo estimate."""
import numpy as np

from mlframe.feature_selection.filters._orthogonal_tail_dependence_fe import tail_dependence_score


def test_floor_is_seed_free_and_matches_many_shuffles():
    """Different seeds give one value, and it matches a 4000-shuffle null mean to Monte-Carlo error."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    y = 0.6 * x + rng.normal(size=4000)
    scores = {tail_dependence_score(x, y, q=0.9, n_perm=50, random_state=s) for s in range(4)}
    assert len(scores) == 1
    raw = tail_dependence_score(x, y, q=0.9, n_perm=0)
    ranks_x = np.argsort(np.argsort(x)) / (len(x) - 1)
    ranks_y = np.argsort(np.argsort(y)) / (len(y) - 1)
    null = np.mean([np.mean((ranks_x > 0.9) & (ranks_y[rng.permutation(len(y))] > 0.9)) for _ in range(4000)]) / 0.1
    assert abs((raw - null) - scores.pop()) < 0.05
