"""The preflight class balance is the rarest class's share, so balanced multiclass and non-0/1 labels are not flagged as imbalanced."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import dataset_diagnostics


def _diag(y):
    """Diagnostics on a small random frame with target ``y``."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(len(y), 3)), columns=list("abc"))
    return dataset_diagnostics(X, y, classification=True, n_estimators=5, max_rows=len(y))


def test_balanced_three_class_target_reports_a_third():
    """{0,1,2} in equal parts: mean(y) = 1.0 used to read as an extreme positive rate; the rarest share is 1/3."""
    y = np.repeat([0, 1, 2], 60)
    assert abs(_diag(y)["class_balance"] - 1.0 / 3.0) < 1e-12


def test_string_labels_do_not_raise_and_report_the_minority_share():
    """String labels raised in np.mean; a 90/10 split reports 0.1."""
    y = np.array(["a"] * 180 + ["b"] * 20)
    assert abs(_diag(y)["class_balance"] - 0.1) < 1e-12
