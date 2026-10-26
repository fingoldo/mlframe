"""Orth triplet and quadruplet recipes are built for sources whose own name contains a double underscore (a target-encoded column)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe_with_recipes


def test_triplet_recipes_survive_a_dunder_source_name():
    """A noisy-XOR product over a column named ``cat__te`` and two plain columns yields recipes, none lost to a basis/degree parse failure."""
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"cat__te": rng.standard_normal(n), "a": rng.standard_normal(n), "b": rng.standard_normal(n), "z": rng.standard_normal(n)})
    y = (X["cat__te"] * X["a"] * X["b"] + 0.3 * rng.standard_normal(n) > 0).astype(int).to_numpy()
    _x_aug, _uni, _scores, recipes = hybrid_orth_mi_triplet_fe_with_recipes(X, y, top_triplet_count=4, top_triplet_seed_k=4)
    assert recipes, "no triplet recipe was built"
    assert any("cat__te" in str(r.name) for r in recipes)
