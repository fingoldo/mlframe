"""Regression tests for the 2026-06-22 BorutaShap audit fixes.

Each test pins one fix and fails on the pre-fix code:
  B1 -- get_5_percent_splits ZeroDivisionError on small frames (sample=True, n<=18).
  B2/Opt-1 -- default RandomForest surrogate gets n_jobs=-1.
  B3 -- Bonferroni base is the full original feature count, not the shrinking live column set.
  B5 -- empty acceptance on a single-class target (and on all-zero importances) emits a logger warning.
  B6 -- TentativeRoughFix logs (no stdout print).
  B7 -- create_shadow_features pads the shadow side to >= shadow_min_pad on narrow frames (and opts out at 0).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.boruta_shap import BorutaShap


def test_b1_get_5_percent_splits_no_zerodiv_on_small_frame():
    bs = BorutaShap(random_state=0, verbose=False)
    # length<=18 -> round(0.05*length)==0 -> pre-fix np.arange(0, length, 0) raised ZeroDivisionError.
    out = bs.get_5_percent_splits(15)
    assert out.size > 0
    assert int(out[0]) >= 1


def test_b1_fit_with_sample_true_on_small_frame_does_not_crash():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((15, 4)), columns=list("abcd"))
    y = (X["a"] > 0).astype(int)
    bs = BorutaShap(importance_measure="gini", n_trials=3, random_state=0, verbose=False, sample=True)
    bs.fit(X, y)  # pre-fix: ZeroDivisionError in get_5_percent_splits
    assert hasattr(bs, "selected_features_")


def test_b2_default_surrogate_uses_all_cores():
    bs = BorutaShap(classification=True, random_state=0)
    bs.check_model()
    assert bs.model.n_jobs == -1
    bs_reg = BorutaShap(classification=False, random_state=0)
    bs_reg.check_model()
    assert bs_reg.model.n_jobs == -1


def test_b3_bonferroni_base_is_full_feature_count():
    """test_features must correct by the full original feature count, independent of how many
    columns have already been removed. Pre-fix it used len(self.columns) (the shrunk live set),
    which weakens the correction once rejections start dropping columns."""
    bs = BorutaShap(random_state=0, verbose=False)
    n_full = 20
    bs.all_columns = np.array([f"f{i}" for i in range(n_full)])
    bs.columns = np.array([f"f{i}" for i in range(5)])  # already shrunk to 5 live columns
    bs.hits = np.zeros(n_full)
    bs.hits[0] = 30  # a clearly-accepted feature after 30 trials
    bs.pvalue = 0.05
    bs.rejected_columns = []
    bs.accepted_columns = []

    captured = {}
    orig = bs.bonferoni_corrections

    def _spy(pvals, alpha=0.05, n_tests=None):
        captured["n_tests"] = n_tests
        return orig(pvals, alpha=alpha, n_tests=n_tests)

    bs.bonferoni_corrections = _spy
    bs.test_features(iteration=30)
    assert captured["n_tests"] == n_full  # not 5


def test_b5_single_class_target_empty_accept_warns(caplog):
    """A single-class target makes the surrogate rank nothing above the shadow null -> empty accepted set.
    Pre-fix this was silent; the fix emits a logger warning naming the single-class cause."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((80, 4)), columns=list("abcd"))
    y = pd.Series(np.zeros(80, dtype=int))  # single class
    bs = BorutaShap(importance_measure="gini", classification=True, n_trials=4, random_state=0, verbose=False)
    with caplog.at_level(logging.WARNING):
        bs.fit(X, y)
    assert bs.accepted == []
    assert any("accepted 0 features" in r.message and "single class" in r.message for r in caplog.records)


def test_b5_nonempty_accept_does_not_warn(caplog):
    """The warning must NOT fire when features are accepted (no false alarm)."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((400, 4)), columns=list("abcd"))
    y = (X["a"] + X["b"] > 0).astype(int)
    bs = BorutaShap(importance_measure="gini", classification=True, n_trials=20, random_state=1, verbose=False)
    with caplog.at_level(logging.WARNING):
        bs.fit(X, y)
    assert len(bs.accepted) >= 1
    assert not any("accepted 0 features" in r.message for r in caplog.records)


def test_b7_shadow_pad_widens_null_on_narrow_frame():
    """create_shadow_features must extend the shadow side to >= shadow_min_pad columns on a 2-feature frame
    (pre-fix: exactly one shadow per real column -> 2 shadows, a thin null)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 2)), columns=["a", "b"])
    bs = BorutaShap(random_state=0, verbose=False, shadow_min_pad=5)
    bs.X = X
    bs.create_shadow_features()
    assert bs.X_shadow.shape[1] >= 5  # padded
    assert bs.X_boruta.shape[1] == X.shape[1] + bs.X_shadow.shape[1]
    assert len(set(bs.X_shadow.columns)) == bs.X_shadow.shape[1]  # unique names


def test_b7_shadow_pad_opt_out_is_legacy_one_per_column():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 2)), columns=["a", "b"])
    bs = BorutaShap(random_state=0, verbose=False, shadow_min_pad=0)
    bs.X = X
    bs.create_shadow_features()
    assert bs.X_shadow.shape[1] == 2  # no pad


def test_b7_wide_frame_unaffected_by_pad():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 8)), columns=[f"f{i}" for i in range(8)])
    bs = BorutaShap(random_state=0, verbose=False, shadow_min_pad=5)
    bs.X = X
    bs.create_shadow_features()
    assert bs.X_shadow.shape[1] == 8  # already wider than pad -> untouched


def test_b6_tentative_rough_fix_logs_not_prints(capsys, caplog):
    bs = BorutaShap(random_state=0, verbose=False)
    cols = ["a", "b", "c"]
    bs.history_x = pd.DataFrame(
        {"a": [10.0, 11.0], "b": [0.0, 0.1], "c": [9.0, 9.5], "Max_Shadow": [1.0, 1.0]}
    )
    bs.tentative = ["a", "b", "c"]
    bs.rejected = []
    bs.accepted = []
    with caplog.at_level(logging.INFO):
        bs.TentativeRoughFix()
    out = capsys.readouterr().out
    assert "tentative features are now" not in out  # pre-fix wrote this to stdout via print()
    assert any("tentative features are now" in r.message for r in caplog.records)
