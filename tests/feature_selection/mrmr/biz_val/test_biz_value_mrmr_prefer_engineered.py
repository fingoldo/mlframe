"""Biz-value: directed-FE near-tie tie-break prefers the ENGINEERED transform
over its raw parent.

Setup: an even-symmetric target ``y = sign(x1**2 - 1)``. The raw column ``x1``
and its engineered Hermite transform ``x1__He2 = He_2(x1) ~ x1**2 - 1`` both
fully determine ``y`` under full-mode MRMR, so their selection gains are TIED to
within a fraction of a percent (the only difference is quantization rounding).
Under the legacy index-order tie-break the greedy selector took whichever had
the lower cols-index, which is the raw parent ``x1`` (engineered cols are
appended after raw ones) -- linearly useless for an even-symmetric target.

The business value: a shallow downstream model (logistic regression) can exploit
``x1**2 - 1`` but NOT raw ``x1`` (its sign carries no information about
``x1**2``). So preferring the engineered feature on the MI-tie lifts downstream
5-fold ROC-AUC from ~0.5 (raw x1 wins) to ~0.99 (x1__He2 wins). This test pins
both halves: the selection decision AND the downstream quantitative win. It is
falsifiable -- delete the tie-break (or set ``prefer_engineered_rel_eps=0.0``)
and the AUC assertion drops to ~0.5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")


def _even_symmetric_fixture(seed: int = 0, n: int = 1500):
    """``y = sign(x1**2 - 1)`` with a little label noise + 4 pure-noise columns.

    He_2(x1) (= x1**2 - 1 up to scale) is a monotone function of the decision
    statistic, so the engineered column linearises the target; raw x1 does not.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "noise_0": rng.standard_normal(n),
            "noise_1": rng.standard_normal(n),
            "noise_2": rng.standard_normal(n),
            "noise_3": rng.standard_normal(n),
        }
    )
    y = ((x1**2 - 1.0) + 0.2 * rng.standard_normal(n) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _mrmr_hybrid_kw():
    return dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
        fe_hybrid_orth_enable=True,
        fe_hybrid_orth_pair_enable=False,
        fe_hybrid_orth_basis="hermite",
        fe_hybrid_orth_degrees=(2, 3),
        fe_hybrid_orth_top_k=5,
    )


@pytest.mark.parametrize("backend", ["njit", "njit_par"])
def test_prefer_engineered_he2_over_raw_x1(backend, monkeypatch):
    """On the MI-tie between x1 and x1__He2, MRMR selects the engineered
    x1__He2 (deterministically, under either polyeval backend)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    monkeypatch.setenv("MLFRAME_POLYEVAL_BACKEND", backend)
    X, y = _even_symmetric_fixture(seed=0)
    m = MRMR(**_mrmr_hybrid_kw()).fit(X, y)
    out = list(m.get_feature_names_out())

    assert "x1__He2" in out, f"backend={backend}: directed-FE tie-break must surface the engineered He_2 transform on the MI-tie; got {out}"
    # The raw parent must NOT be the (only) thing selected -- that is the bug
    # this rule fixes. (x1 may legitimately be absent; if present alongside
    # x1__He2 the downstream AUC test below still pins the value.)
    assert out != ["x1"], f"backend={backend}: selecting raw x1 alone is the regression this rule prevents; got {out}"


def test_downstream_logreg_auc_with_engineered_feature():
    """Quantitative business value: a logistic-regression downstream on the
    MRMR-selected feature(s) reaches 5-fold ROC-AUC >= 0.9 because the
    engineered He_2 feature linearises the even-symmetric target. With raw x1
    selected instead, AUC would collapse to ~0.5."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _even_symmetric_fixture(seed=0)
    m = MRMR(**_mrmr_hybrid_kw()).fit(X, y)
    out = list(m.get_feature_names_out())
    assert "x1__He2" in out, f"precondition: x1__He2 must be selected; got {out}"

    X_sel = m.transform(X)
    auc = cross_val_score(
        LogisticRegression(max_iter=1000),
        X_sel.to_numpy(),
        y.to_numpy(),
        cv=5,
        scoring="roc_auc",
    ).mean()
    assert auc >= 0.9, (
        f"downstream LogReg 5-fold ROC-AUC={auc:.3f} on MRMR-selected "
        f"features {out}; expected >= 0.9 because x1__He2 linearises the "
        f"even-symmetric target. ~0.5 would mean raw x1 was selected instead."
    )


def test_raw_x1_alone_is_linearly_useless_control():
    """Negative control proving the test is falsifiable: a LogReg on RAW x1
    alone is near chance for this even-symmetric target (so the >=0.9 gate in
    the positive test genuinely depends on the engineered feature, not on x1
    happening to be informative)."""
    X, y = _even_symmetric_fixture(seed=0)
    auc_raw = cross_val_score(
        LogisticRegression(max_iter=1000),
        X[["x1"]].to_numpy(),
        y.to_numpy(),
        cv=5,
        scoring="roc_auc",
    ).mean()
    assert auc_raw < 0.65, (
        f"control: raw x1 alone should be ~chance for an even-symmetric "
        f"target, got AUC={auc_raw:.3f}. If this is high the fixture no longer "
        f"isolates the engineered-feature value."
    )
