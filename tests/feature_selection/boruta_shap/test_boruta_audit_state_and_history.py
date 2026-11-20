"""BorutaShap audit regressions: shadow history extremes, refit determinism, stale holdout state, seeded row sampling, auto-dispatch on categoricals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def _frame(n: int = 240, seed: int = 0):
    """Small numeric frame with one informative column and a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = (X["a"] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_max_min_shadow_ignore_nan_when_the_first_feature_was_removed():
    """A row whose first entry is NaN (first feature removed) must still report the real max / min shadow, not NaN."""
    bs = BorutaShap()
    bs.all_columns_ = np.array(["a", "b", "c"])
    bs.history_x_ = np.zeros((3, 3))
    bs.history_shadow_ = np.array([[0.0, 0.0, 0.0], [np.nan, 0.5, 0.2], [np.nan, 0.1, 0.9]])
    bs.store_feature_importance()
    assert bs.history_x_["Max_Shadow"].tolist() == [0.0, 0.5, 0.9]
    assert bs.history_x_["Min_Shadow"].tolist() == [0.0, 0.2, 0.1]


def test_refit_on_the_same_instance_reproduces_the_first_fit():
    """fit() re-derives the shadow stream from random_state, so a second fit on identical data gives identical hit histories."""
    X, y = _frame()
    bs = BorutaShap(n_trials=4, verbose=False, random_state=3)
    bs.fit(X, y)
    first = np.array(bs.history_hits_)
    bs.fit(X, y)
    np.testing.assert_array_equal(first, np.array(bs.history_hits_))


def test_refit_in_train_mode_does_not_permute_on_the_previous_holdout(monkeypatch):
    """A refit with train_or_test='train' must score permutation importance on the current frame, not the earlier fit's 30% holdout."""
    import sklearn.inspection as insp

    X, y = _frame()
    seen = []
    real = insp.permutation_importance

    def _spy(model, X_perm, y_perm, **kw):
        """Record the row count handed to permutation_importance."""
        seen.append(len(X_perm))
        return real(model, X_perm, y_perm, **kw)

    monkeypatch.setattr(insp, "permutation_importance", _spy)
    bs = BorutaShap(n_trials=2, verbose=False, importance_measure="permutation", permutation_n_repeats=2, train_or_test="test")
    bs.fit(X, y)
    seen.clear()
    bs.set_params(train_or_test="train")
    bs.fit(X, y)
    assert seen and all(n == len(X) for n in seen)


def test_sample_true_row_draws_follow_random_state():
    """find_sample draws from the seeded generator, so two instances with the same random_state pick the same rows."""
    preds = np.linspace(0.0, 1.0, 400)

    def _draw(seed: int):
        """Index of the rows find_sample returns for one seeded instance."""
        bs = BorutaShap(random_state=seed)
        bs.preds_ = preds
        bs.X_ = pd.DataFrame({"f": preds})
        bs.X_boruta_ = pd.DataFrame({"f": preds, "shadow_f": preds[::-1]})
        return bs.find_sample().index.to_numpy()

    np.testing.assert_array_equal(_draw(5), _draw(5))


def test_auto_importance_measure_accepts_object_columns():
    """importance_measure='auto' probes the frame before ordinal encoding, so a string column must not crash the probe."""
    X, y = _frame()
    X["cat"] = np.random.default_rng(1).choice(["u", "v", "w"], len(X))
    bs = BorutaShap(n_trials=2, verbose=False, importance_measure="auto")
    bs.fit(X, y)
    assert bs.auto_dispatch_diagnostics_["probe_ok"] is True
