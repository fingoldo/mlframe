"""CalibratedClassifierCV inner CV is seeded from the suite seed (A7-03), and the post-hoc-calibration metadata flag
reflects reality per model family (A7-05)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_calibrated_classifier_cv_inner_cv_is_seeded():
    """The linear-model CalibratedClassifierCV must carry a seeded, shuffled StratifiedKFold pinned to config.random_state."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    from mlframe.training.models import create_linear_model
    from mlframe.training.configs import LinearModelConfig

    cfg = LinearModelConfig(model_type="sgd", use_calibrated_classifier=True, random_state=123)
    model = create_linear_model("sgd", cfg, use_regression=False)
    assert isinstance(model, CalibratedClassifierCV)
    assert isinstance(model.cv, StratifiedKFold)
    assert model.cv.random_state == 123
    assert model.cv.shuffle is True


def test_calibrated_classifier_cv_seed_varies_with_config():
    """Two configs with different seeds must produce different inner-CV random_state (proves the seed is threaded)."""
    from mlframe.training.models import create_linear_model
    from mlframe.training.configs import LinearModelConfig

    a = create_linear_model("sgd", LinearModelConfig(model_type="sgd", use_calibrated_classifier=True, random_state=1), use_regression=False)
    b = create_linear_model("sgd", LinearModelConfig(model_type="sgd", use_calibrated_classifier=True, random_state=2), use_regression=False)
    assert a.cv.random_state != b.cv.random_state


def test_posthoc_flag_true_for_calibrated_classifier_cv():
    """A CalibratedClassifierCV-wrapped model must be flagged as post-hoc calibrated."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from mlframe.training._calibration_models import _maybe_apply_posthoc_calibration

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 3)), columns=list("abc"))
    y = (X["a"] > 0).astype(int)
    m = CalibratedClassifierCV(LogisticRegression(max_iter=200), cv=3, method="isotonic").fit(X, y)
    out = _maybe_apply_posthoc_calibration(m, fit_params={}, model_type_name="linear")
    assert getattr(out, "_mlframe_probs_posthoc_calibrated") is True


def test_posthoc_flag_false_for_plain_tree():
    """A plain tree (eval-metric calibration-trained, not post-hoc calibrated) must be flagged False."""
    from sklearn.tree import DecisionTreeClassifier
    from mlframe.training._calibration_models import _maybe_apply_posthoc_calibration

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 3)), columns=list("abc"))
    y = (X["a"] > 0).astype(int)
    m = DecisionTreeClassifier(max_depth=3).fit(X, y)
    out = _maybe_apply_posthoc_calibration(m, fit_params={}, model_type_name="dt")
    assert getattr(out, "_mlframe_probs_posthoc_calibrated") is False
