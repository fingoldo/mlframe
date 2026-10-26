"""The default number of triplet columns per fit is high enough for several independent three-way structures."""
import inspect

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe
from mlframe.feature_selection.filters.mrmr import MRMR


def test_class_default_is_eight():
    """The constructor default is 8 (2 found only half of four planted triples)."""
    assert inspect.signature(MRMR.__init__).parameters["fe_hybrid_orth_triplet_top_count"].default == 8


def test_four_independent_triples_all_get_a_column():
    """Four disjoint planted triple products over 24 columns: with the default count all four triples are among the emitted columns, with the old count of 2 only two."""
    rng = np.random.default_rng(0)
    n, p = 6000, 24
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"x{i}" for i in range(p)])
    triples = [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(4)]
    y = (sum(X.iloc[:, a] * X.iloc[:, b] * X.iloc[:, c] for a, b, c in triples) + 0.3 * rng.standard_normal(n) > 0).astype(int).to_numpy()

    def found(count):
        """Number of planted triples with an emitted column for this count."""
        _aug, _uni, scores = hybrid_orth_mi_triplet_fe(X, y, top_triplet_count=count, top_triplet_seed_k=12)
        kept = scores[scores["cmi_gain"] > 0].head(count)["engineered_col"].astype(str)
        return sum(any(all(f"x{i}" in name for i in tr) for name in kept) for tr in triples)

    assert found(8) == 4
    assert found(2) <= 2
