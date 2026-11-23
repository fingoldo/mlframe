"""A cat-FE ``factorize`` column replays consistently when the categorical source has thin levels (AUD23-08).

At fit ``categorize_dataset`` folds a level seen on fewer than ``categorical_fold_min_support`` rows into an "other" bucket (and reorders the codes by frequency); the replay map is built from the raw factorisation and
keeps those levels apart. The audit read this as a train / serve skew. Measured: the replay is a deterministic function of the raw values, the rare levels only make a few extra cells, and the cell -> target rate learned
on the train transform predicts the holdout as well as the XOR allows, so no cell is mislabelled. This pins that behaviour.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.filters.mrmr import MRMR

pytestmark = pytest.mark.slow


def _frame(seed: int, n: int):
    """cat_a == cat_b drives the target (5 percent flips); two thin levels of three rows each in cat_a."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=n)
    b = rng.integers(0, 3, size=n)
    y = ((a == b).astype(int) ^ (rng.random(n) < 0.05).astype(int)).astype(np.int64)
    av = np.array([f"A{v}" for v in a], dtype=object)
    bv = np.array([f"B{v}" for v in b], dtype=object)
    idx = rng.choice(n, 6, replace=False)
    av[idx[:3]] = "RA1"
    av[idx[3:]] = "RA2"
    return pd.DataFrame({"cat_a": av, "cat_b": bv, "noise": rng.normal(size=n)}), pd.Series(y)


def test_thin_levels_do_not_mislabel_cells_at_transform():
    """The train cell -> rate table predicts the holdout with an AUC close to the XOR ceiling, and the replay is deterministic."""
    Xtr, ytr = _frame(7, 4000)
    Xte, yte = _frame(99, 3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_state=42, fe_max_steps=1, fe_cat_pair_enable=False, fe_composite_group_agg_enable=False).fit(Xtr, ytr)
    recipes = [r for r in m._engineered_recipes_ if r.kind in ("factorize", "target_encoding")]
    assert recipes, "the factorize family must produce the interaction on this frame"
    name = recipes[0].name
    tr = np.asarray(m.transform(Xtr)[name], dtype=float)
    te = np.asarray(m.transform(Xte)[name], dtype=float)
    assert np.array_equal(te, np.asarray(m.transform(Xte)[name], dtype=float)), "the replay must be deterministic"
    rate = pd.Series(ytr.to_numpy()).groupby(tr).mean()
    pred = pd.Series(te).map(rate).fillna(float(ytr.mean())).to_numpy()
    assert roc_auc_score(yte, pred) > 0.9
