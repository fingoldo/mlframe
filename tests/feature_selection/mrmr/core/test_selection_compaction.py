"""The parsimony compaction replaces several engineered features by one that carries at least the same bias-corrected joint information."""

import numpy as np

from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import choose_compaction, corrected_joint_mi


def _conjunction(n=20000, seed=0):
    """y ~ Bernoulli(sigmoid(-1.5 + 2.5 A + 3 A B C)) with the pair codes f(A, B), f(A, C) and the composite ABC, all integer-coded."""
    rng = np.random.default_rng(seed)
    a, b, c = (rng.integers(0, 2, n) for _ in range(3))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.5 + 2.5 * a + 3.0 * (a & b & c))))).astype(np.int64)
    cols = {"A": a, "f_ab": 2 * a + b, "f_ac": 2 * a + c, "ABC": a & b & c}
    sources = {"A": frozenset("A"), "f_ab": frozenset("AB"), "f_ac": frozenset("AC"), "ABC": frozenset("ABC")}
    return cols, sources, y


def test_pair_features_are_replaced_by_the_composite_that_carries_the_same_information():
    """{A, f(A,B), f(A,C)} and {A, ABC} determine y equally; the smaller set wins on the corrected score."""
    cols, sources, y = _conjunction()
    choice = choose_compaction(cols, ["A", "f_ab", "f_ac"], ["f_ab", "f_ac"], sources, ["ABC"], y)
    assert choice is not None and choice[0] == "ABC" and sorted(choice[1]) == ["f_ab", "f_ac"]
    full = corrected_joint_mi([cols["A"], cols["f_ab"], cols["f_ac"]], y)[0]
    small = corrected_joint_mi([cols["A"], cols["ABC"]], y)[0]
    assert small >= full


def test_no_compaction_when_the_composite_loses_information():
    """If y depends on f(A,B) and f(A,C) beyond ABC, the composite is not an equivalent replacement."""
    rng = np.random.default_rng(1)
    n = 20000
    a, b, c = (rng.integers(0, 2, n) for _ in range(3))
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(-1.0 + 1.5 * (a & b) + 1.5 * (a & c) - 1.0 * (a & b & c))))).astype(np.int64)
    cols = {"A": a, "f_ab": 2 * a + b, "f_ac": 2 * a + c, "ABC": a & b & c}
    sources = {"A": frozenset("A"), "f_ab": frozenset("AB"), "f_ac": frozenset("AC"), "ABC": frozenset("ABC")}
    assert choose_compaction(cols, ["A", "f_ab", "f_ac"], ["f_ab", "f_ac"], sources, ["ABC"], y) is None


def test_a_replaced_feature_that_a_surviving_recipe_consumes_is_kept():
    """A feature read by another recipe cannot be dropped, so no replacement is proposed."""
    cols, sources, y = _conjunction()
    consumers = {"f_ab": {"some_other_recipe"}}
    assert choose_compaction(cols, ["A", "f_ab", "f_ac"], ["f_ab", "f_ac"], sources, ["ABC"], y, consumers) is None


def test_a_single_engineered_feature_is_never_replaced():
    """A replacement has to save at least one feature."""
    cols, sources, y = _conjunction()
    assert choose_compaction(cols, ["A", "f_ab"], ["f_ab"], sources, ["ABC"], y) is None


def test_synthesized_triplet_is_built_from_the_three_operands_only():
    """The on-demand triplet candidate is a replayable recipe over exactly the three operands, with integer codes."""
    import pandas as pd

    from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import _synthesized_triplet

    rng = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame({c: rng.normal(size=n) for c in "ABCD"})
    y = (X["A"] * X["B"] * X["C"] > 0).astype(np.int64).to_numpy()
    name, rec, codes = _synthesized_triplet(X, y, frozenset("ABC"))
    assert set(rec.src_names) == set("ABC") and name.startswith("A*B*C") and codes.shape == (n,)


def test_cell_triple_candidate_holds_the_conjunction_and_replays():
    """Binary operands are crossed as a joint cell table whose replay equals its fit-time codes and whose corrected information is that of the AND."""
    import pandas as pd

    from mlframe.feature_selection.filters._mrmr_fit_impl._selection_compaction import _synthesized_cell_triple
    from mlframe.feature_selection.filters.engineered_recipes.shared import apply_recipe

    cols, _, y = _conjunction()
    X = pd.DataFrame({"A": cols["A"].astype(float), "B": (cols["f_ab"] % 2).astype(float), "C": (cols["f_ac"] % 2).astype(float)})
    _, rec, codes = _synthesized_cell_triple(X, frozenset("ABC"))
    assert np.array_equal(np.asarray(apply_recipe(rec, X)).astype(np.int64), codes)
    small = corrected_joint_mi([cols["A"], codes], y)[0]
    assert small >= corrected_joint_mi([cols["A"], cols["f_ab"], cols["f_ac"]], y)[0] - 1e-3
    assert _synthesized_cell_triple(X.assign(B=np.arange(len(X)) % 20), frozenset("ABC")) is None
