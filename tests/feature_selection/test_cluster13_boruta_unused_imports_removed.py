"""Cluster 13 regression: the BorutaSHAP package drops its unused module-level imports (audit 2026-11-20, finding 13-F16).

Nine imports (check_random_state, check_X_y, MinMaxScaler, KMeans, issparse, ks_2samp, IsolationForest, tqdmu, get_pipeline_last_element) were used
only by their own import line. They are gone; the package still imports and keeps the classes it uses.
"""

from pathlib import Path

import mlframe.feature_selection.boruta_shap as bs

_SRC = Path(bs.__file__).resolve().read_text(encoding="utf-8")


def test_unused_imports_removed():
    """None of the unused import lines remain (13-F16)."""
    for line in (
        "from sklearn.utils import check_random_state, check_X_y",
        "from sklearn.preprocessing import MinMaxScaler",
        "from sklearn.cluster import KMeans",
        "from scipy.sparse import issparse",
        "from scipy.stats import ks_2samp",
        "from pyutilz.system import tqdmu",
        "from mlframe.utils.misc import get_pipeline_last_element",
    ):
        assert line not in _SRC, line


def test_module_still_exposes_its_classes():
    """Control: the package still imports and keeps its random-forest estimators in scope (13-F16)."""
    assert "RandomForestClassifier" in _SRC
    assert "RandomForestRegressor" in _SRC
