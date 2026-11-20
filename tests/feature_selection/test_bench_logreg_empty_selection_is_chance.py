"""An empty selection scores chance in the benchmark helper, not the full column set (audit 2026-11-20, finding 10-F3)."""
import numpy as np
import pandas as pd

from mlframe.feature_selection._benchmarks._bench_shared import logreg_holdout_auc


def test_empty_selection_is_chance_not_every_column():
    """A selector that kept nothing was scored with all columns; it must score 0.5."""
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame(rng.standard_normal((200, 4)))
    ytr = (Xtr[0] > 0).astype(int).to_numpy()
    Xte = pd.DataFrame(rng.standard_normal((100, 4)))
    yte = (Xte[0] > 0).astype(int).to_numpy()
    assert logreg_holdout_auc(Xtr, ytr, Xte, yte, []) == 0.5
    assert logreg_holdout_auc(Xtr, ytr, Xte, yte, [0]) > 0.8
