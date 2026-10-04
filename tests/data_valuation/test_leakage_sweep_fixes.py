"""Leakage and methodology fixes in data_valuation: prior-corrected shift weights, group-disjoint DRO folds."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.data_valuation import adversarial_validation
from mlframe.data_valuation._adversarial_reweighting import dro_reweight_fit


def test_adversarial_validation_weights_do_not_collapse_to_the_floor_when_test_is_small():
    """With 5000 train rows and 100 shifted test rows the density-ratio weights track the shifted feature instead of 96 percent sitting at the 0.1 clip floor."""
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(5000, 3))
    x_test = rng.normal(size=(100, 3)) + np.array([1.5, 0.0, 0.0])
    res = adversarial_validation(x_train, x_test, model=LogisticRegression(), rng=np.random.default_rng(1))
    w = res["suggested_weights"]
    assert np.corrcoef(w, x_train[:, 0])[0, 1] > 0.6
    assert (w <= w.min() * 1.0001).mean() < 0.5
    assert w.mean() == pytest.approx(1.0)


def test_dro_reweight_fit_groups_make_oof_folds_group_disjoint():
    """With ``groups`` every OOF refit omits whole groups; without it the folds are i.i.d. rows and every group stays present."""
    n_groups, per = 10, 20
    gid = np.repeat(np.arange(n_groups), per)
    rng = np.random.default_rng(0)
    X = np.column_stack([gid.astype(float), rng.normal(size=gid.size)])
    y = rng.normal(size=gid.size)

    def run(groups):
        """Run two DRO rounds and return the number of distinct groups seen by each fit call."""
        seen: list[int] = []

        def fit_fn(Xf, yf, wf):
            """Record how many groups the training rows cover, then fit a linear model."""
            seen.append(len(np.unique(Xf[:, 0])))
            return LinearRegression().fit(Xf, yf, sample_weight=wf)

        dro_reweight_fit(fit_fn, lambda yt, yp: (yt - yp) ** 2, X, y, n_rounds=2, n_splits=5, rng=np.random.default_rng(0), groups=groups)
        return seen

    grouped = run(gid)
    assert sum(1 for s in grouped if s < n_groups) == 2 * 5
    iid = run(None)
    assert all(s == n_groups for s in iid)


def test_dro_reweight_fit_rejects_misaligned_groups():
    """A group vector of the wrong length raises instead of silently falling back to i.i.d. folds."""
    X = np.zeros((20, 2))
    y = np.zeros(20)
    with pytest.raises(ValueError, match="groups length"):
        dro_reweight_fit(lambda a, b, w: LinearRegression().fit(a, b), lambda yt, yp: (yt - yp) ** 2, X, y, groups=np.arange(19))
