"""K-way chained lookup under unknown_strategy='sentinel' and de-duplication of coordinate-ascent results."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_kway_materialize import _build_kway_chained_lookup
from mlframe.feature_selection.filters._cat_post_refine import _refine_kway_coordinate_ascent
from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe
from mlframe.feature_selection.filters.engineered_recipes._recipe_factorize import _apply_factorize_kway
from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes, merge_vars


def _correlated_pair_data():
    """Columns a, b (b == a, so most (a, b) cells are unseen) and an independent c."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 300)
    c = rng.integers(0, 3, 300)
    return np.column_stack([a, a, c]).astype(np.int32), np.array([4, 4, 3])


def test_sentinel_chain_cardinality_counts_the_sentinel_class():
    """The step cardinality includes the sentinel class, so no replayed code can index past the next table."""
    data, nbins = _correlated_pair_data()
    lookups, nuniqs = _build_kway_chained_lookup(data, (0, 1, 2), nbins, "sentinel", np.int32)
    assert int(lookups[0].max()) < nuniqs[0]
    assert int(lookups[0].max()) + (int(nbins[2]) - 1) * nuniqs[0] < len(lookups[1])


def test_sentinel_kway_replay_of_unseen_prefix_does_not_alias_or_overflow():
    """Replaying an unseen (a, b) prefix with the top c value returns the sentinel class instead of raising IndexError."""
    data, nbins = _correlated_pair_data()
    lookups, nuniqs = _build_kway_chained_lookup(data, (0, 1, 2), nbins, "sentinel", np.int32)
    recipe = EngineeredRecipe(
        name="kway(a__b__c)", kind="factorize", src_names=("a", "b", "c"), factorize_nbins=(4, 4, 3), unknown_strategy="sentinel",
        extra={"kway_order": 3, "chain_lookups": lookups, "chain_nuniqs": nuniqs},
    )
    out = _apply_factorize_kway(recipe, pd.DataFrame({"a": [0, 0], "b": [1, 1], "c": [2, 0]}))
    assert out[0] == out[1] == int(lookups[1].max())


def test_coordinate_ascent_does_not_emit_the_same_member_set_twice():
    """Two seeds that refine to the same member set yield one k-way result."""
    rng = np.random.default_rng(1)
    n = 6000
    cols = rng.integers(0, 2, (n, 5))
    y = ((cols[:, 0] + cols[:, 1] + cols[:, 2]) % 2).astype(np.int64)
    data = cols.astype(np.int32)
    nbins = np.full(5, 2)
    freqs_y = np.bincount(y) / n
    results = []
    for tup in ((0, 1, 3), (0, 1, 4)):
        cls, freqs, nu = merge_vars(factors_data=data, vars_indices=np.array(tup, dtype=np.int64), var_is_nominal=None, factors_nbins=nbins, dtype=np.int32)
        mi = compute_mi_from_classes(classes_x=cls, freqs_x=freqs, classes_y=y, freqs_y=freqs_y, dtype=np.int32)
        results.append((tup, cls, nu, mi))
    refined = _refine_kway_coordinate_ascent(data, results, np.arange(5), nbins, y, freqs_y, 2**20, 2, np.int32, 0)
    tuples = [r[0] for r in refined]
    assert len(tuples) == len(set(tuples)) == 1
