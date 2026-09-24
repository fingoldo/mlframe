"""BorutaShap's per-trial holdout redraw: opt-in, and off by default for a measured reason."""

import numpy as np
import pandas as pd
import pytest

lightgbm = pytest.importorskip("lightgbm")
pytest.importorskip("shap")

from mlframe.feature_selection.boruta_shap import BorutaShap
import mlframe.feature_selection.boruta_shap as bs


def _run(monkeypatch, **kw):
    seeds = []
    real = bs.train_test_split

    def spy(*a, **k):
        seeds.append(k.get("random_state"))
        return real(*a, **k)

    monkeypatch.setattr(bs, "train_test_split", spy)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"f{i}": rng.normal(size=150) for i in range(4)})
    y = (X["f0"] > 0).astype(int)
    BorutaShap(
        model=lightgbm.LGBMClassifier(n_estimators=10, verbose=-1), importance_measure="shap", classification=True,
        n_trials=4, train_or_test="test", random_state=7, **kw,
    ).fit(X, y)
    return seeds


def test_the_opt_in_redraws_the_holdout_each_trial(monkeypatch):
    seeds = _run(monkeypatch, resample_holdout_per_trial=True)
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds), f"trials reused a split seed: {seeds}"
    assert seeds[0] == 7


def test_the_default_keeps_one_holdout(monkeypatch):
    """The per-trial redraw measurably hurt selection on the auto-vs-gini bed; see BorutaShap.__init__."""
    seeds = _run(monkeypatch)
    assert len(seeds) >= 2 and set(seeds) == {7}
