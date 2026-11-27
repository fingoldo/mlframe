"""Cluster 41 regression: the ranker-suite column sweep handles object-dtype columns (audit 2026-11-20, 41-F40).

The branch was ``dt is object``, which is never true for a pandas dtype instance, so object columns were neither turned into categories nor
checked for nested embedding elements.
"""

import numpy as np
import pandas as pd

from mlframe.training.ranking._ranker_suite_train_helpers import _train_mlframe_rank_drops_replacements_single_sweep


def test_string_object_column_becomes_a_category_with_a_missing_level():
    """A string object column with a null is replaced by a category column carrying the __MISSING__ level (41-F40)."""
    df = pd.DataFrame({"c": pd.Series(["a", None, "b"], dtype=object), "x": [1.0, 2.0, 3.0]})
    drops, repl = [], {}
    _train_mlframe_rank_drops_replacements_single_sweep(df, drops, repl)
    assert drops == []
    assert isinstance(repl["c"].dtype, pd.CategoricalDtype)
    assert "__MISSING__" in repl["c"].cat.categories


def test_embedding_object_column_is_dropped_even_with_leading_nulls():
    """An object column of arrays is dropped, and the first non-null element decides it (41-F40)."""
    df = pd.DataFrame({"e": pd.Series([None, np.zeros(3), np.ones(3)], dtype=object)})
    drops, repl = [], {}
    _train_mlframe_rank_drops_replacements_single_sweep(df, drops, repl)
    assert drops == ["e"] and repl == {}
