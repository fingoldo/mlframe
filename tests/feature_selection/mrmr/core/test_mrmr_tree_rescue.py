"""Tests for MRMRTreeRescued -- the gated tree-importance rescue for MRMR's selection-gate collapse.

Wide interaction frame (zero-marginal a*b operands) -> MRMR's marginal greedy under-selects -> the rescue fires and
unions the shallow-GBM importance top-K (recovering the operands). Narrow frames + tree_rescue=False -> exact no-op.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR, MRMRTreeRescued

from tests.conftest import fast_n_estimators


def _wide_interaction(n=2000, p_noise=80, seed=0):
    """y driven by a PURE interaction a*b (zero-marginal operands) + a linear term; >60 cols so MRMR under-selects."""
    rng = np.random.default_rng(seed)
    a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    noise = rng.normal(size=(n, p_noise))
    y = ((a * b + 0.7 * c + 0.3 * rng.normal(size=n)) > 0).astype(int)
    X = pd.DataFrame(np.column_stack([a, b, c, noise]), columns=["a", "b", "c"] + [f"n{i}" for i in range(p_noise)])
    return X, pd.Series(y)


@pytest.mark.timeout(300)
def test_rescue_fires_on_wide_underselection_and_recovers_operands():
    X, y = _wide_interaction()
    m0 = MRMR(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    m1 = MRMRTreeRescued(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    assert len(m1.support_) > len(m0.support_), "rescue should add features when MRMR under-selects on a wide frame"
    out = set(m1.get_feature_names_out())
    assert {"a", "b"} <= out, "the rescue should recover the zero-marginal interaction operands the greedy missed"


@pytest.mark.timeout(300)
def test_rescue_noop_on_narrow_frame():
    X, y = _wide_interaction(p_noise=25)  # 28 cols <= tree_rescue_min_p (60) -> gate cannot fire
    m0 = MRMR(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    m1 = MRMRTreeRescued(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    assert list(m1.support_) == list(m0.support_), "narrow frame must be a byte-identical no-op vs MRMR"


@pytest.mark.timeout(300)
def test_rescue_off_equals_mrmr():
    X, y = _wide_interaction()
    m0 = MRMR(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    m1 = MRMRTreeRescued(tree_rescue=False, verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    assert list(m1.support_) == list(m0.support_), "tree_rescue=False must behave exactly like MRMR"


@pytest.mark.timeout(300)
def test_rescue_transform_pickle_and_support_consistency():
    import pickle

    X, y = _wide_interaction()
    m = MRMRTreeRescued(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=0).fit(X, y)
    Z = m.transform(X.iloc[:10])
    assert len(Z.columns) == len(m.support_)  # support_ extension flows through transform
    assert m.get_support().sum() == len(m.support_)  # mask matches the extended support
    m2 = pickle.loads(pickle.dumps(m))  # rescue extends support_ only -> pickle-clean
    assert list(m2.transform(X.iloc[:10]).columns) == list(Z.columns)


def test_varargs_ctor_get_params_and_clone_preserve_tree_rescue_params():
    """The forwarding ``*args/**kwargs`` ctor must not hide the tree-rescue params from sklearn introspection:
    get_params reports them, set_params/clone round-trip a non-default one. Pre-fix this raised RuntimeError."""
    from sklearn.base import clone

    sel = MRMRTreeRescued(tree_rescue_top_k=7, verbose=0)
    params = sel.get_params(deep=False)
    for name in (
        "tree_rescue",
        "tree_rescue_top_k",
        "tree_rescue_min_p",
        "tree_rescue_min_ratio",
        "tree_rescue_min_features",
        "tree_rescue_n_estimators",
        "tree_rescue_max_depth",
    ):
        assert name in params, f"get_params must expose {name}"
    assert params["tree_rescue_top_k"] == 7
    cloned = clone(sel)
    assert cloned.get_params(deep=False)["tree_rescue_top_k"] == 7, "clone must preserve a non-default tree-rescue param"
    cloned.set_params(tree_rescue_top_k=13)
    assert cloned.tree_rescue_top_k == 13


@pytest.mark.timeout(300)
def test_bizvalue_rescue_lifts_downstream_auc_on_interaction_data():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import lightgbm as lgb

    X, y = _wide_interaction(n=3000, p_noise=120, seed=1)  # 123 cols, strong a*b interaction
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=1, stratify=y)

    def auc(sel):
        sel.fit(Xtr, ytr)
        Ztr, Zte = sel.transform(Xtr), sel.transform(Xte)
        m = lgb.LGBMClassifier(n_estimators=fast_n_estimators(200, fast=80), verbose=-1).fit(Ztr, ytr)
        return roc_auc_score(yte, m.predict_proba(Zte)[:, 1])

    a_off = auc(MRMR(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=1))
    a_on = auc(MRMRTreeRescued(verbose=0, fe_max_steps=0, n_jobs=4, random_seed=1))
    # MRMR collapses on the zero-marginal operands; the rescue recovers them -> a clear downstream lift.
    assert a_on >= a_off + 0.05, f"tree-rescue should lift downstream AUC on interaction data: {a_off:.3f} -> {a_on:.3f}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "--no-cov", "-p", "no:randomly", "-p", "no:cacheprovider"]))
