"""The default number of cat x cat crosses covers many independent planted pairs."""
import inspect

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_pair_fe import hybrid_cat_pair_fe
from mlframe.feature_selection.filters.mrmr import MRMR


def test_class_default_is_sixteen():
    """The constructor default is 16 (5 kept only five of eight planted crosses)."""
    assert inspect.signature(MRMR.__init__).parameters["fe_cat_pair_top_k"].default == 16


def test_eight_disjoint_planted_crosses_all_emitted_with_wide_top_k():
    """Eight disjoint planted modular crosses over 16 four-level columns are all emitted with top_k 12 and only five with top_k 5."""
    n, m = 30000, 16
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"c{i}": rng.integers(0, 4, n).astype(str) for i in range(m)})
    logit = sum(((X[f"c{2 * k}"].astype(int) + X[f"c{2 * k + 1}"].astype(int)) % 4 == 0).astype(float) for k in range(8))
    y = ((logit - logit.mean() + 0.5 * rng.normal(size=n)) > 0).astype(int).to_numpy()
    planted = {f"cross_c{2 * j}_c{2 * j + 1}" for j in range(8)}
    assert planted <= set(hybrid_cat_pair_fe(X, y, top_k=12)[1])
    assert len(planted & set(hybrid_cat_pair_fe(X, y, top_k=5)[1])) == 5
