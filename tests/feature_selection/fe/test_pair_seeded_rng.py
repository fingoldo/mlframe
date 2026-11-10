"""The external-validation factor picks of a pair are a function of the seed and the pair only."""

from mlframe.feature_selection.filters._feature_engineering_pairs._pair_rng import PairSeededRng


def test_pair_picks_do_not_depend_on_other_pairs_or_order():
    """The same pair gives the same draws whatever was drawn before; different pairs and seeds differ."""
    a = PairSeededRng(7)
    first = a.for_pair((3, 5)).choice(100, 4, replace=False).tolist()
    a.for_pair((1, 2)).choice(100, 50, replace=False)
    assert a.for_pair((3, 5)).choice(100, 4, replace=False).tolist() == first
    assert PairSeededRng(7).for_pair((3, 5)).choice(100, 4, replace=False).tolist() == first
    assert PairSeededRng(8).for_pair((3, 5)).choice(100, 4, replace=False).tolist() != first
    assert a.for_pair((5, 3)).choice(100, 4, replace=False).tolist() != first
