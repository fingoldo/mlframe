"""RFECV helpers: border-file line endings, throttled CatBoost fallback warning, negative-score elbow, logged SFFS failure."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers.rfecv import _cb_border_cache as cbc
from mlframe.feature_selection.wrappers.rfecv._diagnostics import n_stability_elbow_
from mlframe.feature_selection.wrappers.rfecv._finalize import _run_sffs_swap_stage
from mlframe.utils.log_throttle import reset_throttle_counts


def test_subset_border_file_uses_lf_line_endings(tmp_path):
    """The border file CatBoost reads back must hold bare LF rows even on Windows."""
    fold = cbc._FoldBorders()
    fold.lines = {"a": ["0.5", "1.5"], "b": ["2.5\tMin"]}
    path = tmp_path / "borders.tsv"
    cbc._write_subset_borders(fold, ["b", "a"], str(path))
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw == b"0\t2.5\tMin\n1\t0.5\n1\t1.5\n"


def test_cached_borders_fallback_warning_is_throttled(caplog):
    """A failing cached-borders fit logs through log_throttle: present at WARNING, bounded in count."""
    reset_throttle_counts("rfecv_cb_borders_fit_failed")

    class CatBoostClassifier:
        """Name-matching stub that passes is_supported then breaks the Pool build."""

        def get_params(self):
            """No tunables."""
            return {}

    src = np.zeros((4, 2))
    with caplog.at_level(logging.WARNING, logger=cbc.logger.name):
        for _ in range(20):
            ok = cbc.fit_catboost_with_cached_borders(
                CatBoostClassifier(), source=src, X_train=None, y_train=None, fit_features=["a"], fit_params={}, train_rows=np.arange(4),
            )
            assert ok is False
    recs = [r for r in caplog.records if "cached-borders fast path failed" in r.getMessage()]
    assert 1 <= len(recs) <= 5


def test_stability_elbow_prefers_the_better_score_when_scores_are_negative():
    """With neg-loss scores the higher-stability, higher-score N must win, not the product's accidental maximum."""
    fake = SimpleNamespace(
        cv_results_={"nfeatures": [2, 5], "cv_mean_perf": [-0.2, -0.1], "cv_std_perf": [0.0, 0.0]},
        stability_vs_n_curve_=lambda metric="jaccard": {2: 0.2, 5: 0.9},
    )
    assert n_stability_elbow_(fake) == 5


def test_sffs_swap_failure_is_logged_without_verbose(caplog):
    """A raising swap pass is reported at WARNING even when verbose is off."""
    reset_throttle_counts("rfecv_sffs_swap_failed")

    def _boom(**_):
        """Always fail."""
        raise RuntimeError("swap exploded")

    fake = SimpleNamespace(swap_top_k=2, early_stopping_val_nsplits=None, swap_top_k_allow_no_es=False, _sffs_swap_pass=_boom)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.wrappers.rfecv"):
        out = _run_sffs_swap_stage(
            fake, X=None, y=None, X_estimator=None, col_pos=None, estimator=LogisticRegression(), cv=3, scoring=None, best_nfeatures=3, best_score=0.5,
            selected_features_per_nfeatures={}, feature_importances={}, original_features=[], evaluated_scores_mean={}, evaluated_scores_std={}, verbose=0, ndigits=4,
        )
    assert out == (3, 0.5)
    assert any("swap exploded" in r.getMessage() for r in caplog.records)
