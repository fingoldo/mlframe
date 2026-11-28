"""Cluster 41 regression: the ranker's categorical vocabulary excludes the test split (audit 2026-11-20, finding 41-F38)."""

import pandas as pd

from mlframe.training.ranking._ranker_suite_train_helpers import _train_mlframe_rank_pandas


def test_test_only_level_is_unseen_not_given_a_code():
    """A level that appears only in test is encoded -1, while train/val levels keep their codes."""
    tr = pd.DataFrame({"c": ["a", "b", "a"]})
    va = pd.DataFrame({"c": ["b", "a"]})
    te = pd.DataFrame({"c": ["a", "zzz"]})
    X_te, X_tr, X_va = _train_mlframe_rank_pandas(tr, ["c"], va, te)
    assert X_te["c"].tolist()[1] == -1
    assert X_te["c"].tolist()[0] == X_tr["c"].tolist()[0]
    assert -1 not in X_tr["c"].tolist() and -1 not in X_va["c"].tolist()
