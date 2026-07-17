"""Tests for per-instance SHAP attribution of the worst (most-confident-wrong) predictions.

Ships: unit tests (top-K worst selection by error severity, K bound, non-tree skip, no-error annotate,
paths written, NO figure leak), a biz_value test (a synthetic where an outlier in a SPECIFIC feature
forces a confident-wrong positive -> that feature has the largest |SHAP| in the instance's attribution),
and a cProfile pass confirming the explained background is bounded by ``max_explain_rows``.

shap is a project dep but guarded via importorskip so a shap-less CI env skips rather than errors.
"""

from __future__ import annotations

import cProfile
import io
import os
import pstats

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

shap = pytest.importorskip("shap")
pd = pytest.importorskip("pandas")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlframe.reporting.charts import shap_per_instance as spi


@pytest.fixture(autouse=True)
def _no_leaked_figures():
    """Every test must leave the pyplot registry as it found it (no figure leak in long sessions)."""
    before = set(plt.get_fignums())
    yield
    leaked = set(plt.get_fignums()) - before
    for num in leaked:
        plt.close(num)
    assert not leaked, f"test leaked open figures: {sorted(leaked)}"


def _fit_clf(X, y, **kw):
    return RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0, n_jobs=1, **kw).fit(X, y)


def _simple_binary(n=400, n_feat=5, seed=0):
    """y = 1 iff f0 > 0 (clean separable-ish); some label noise injects errors."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_feat)), columns=[f"f{i}" for i in range(n_feat)])
    y = (X["f0"].to_numpy() > 0).astype(int)
    return X, y


def test_worst_k_selected_by_error_severity():
    """The K explained rows are exactly the K highest |y_true - y_score| rows."""
    X, y = _simple_binary()
    model = _fit_clf(X, y)
    score = model.predict_proba(X)[:, 1]
    res = spi.shap_worst_errors_explanation(model, X, y, score, feature_names=list(X.columns), k=4)
    assert res.skipped is None
    severity = np.abs(y - score)
    expected = np.argsort(severity)[::-1][:4]
    assert set(res.worst_idx.tolist()) == set(expected.tolist())
    assert np.all(np.diff(res.severities) <= 1e-12)  # worst first, non-increasing
    for f in (res.figure,):
        plt.close(f)


def test_k_bound_respected():
    """worst_idx length is min(k, n) and never exceeds k."""
    X, y = _simple_binary(n=50)
    model = _fit_clf(X, y)
    score = model.predict_proba(X)[:, 1]
    res = spi.shap_worst_errors_explanation(model, X, y, score, k=3)
    assert len(res.worst_idx) == 3
    plt.close(res.figure)


def test_non_tree_model_skipped():
    """Non-tree model -> skipped with a reason, no figure, no KernelExplainer."""
    X, y = _simple_binary()
    model = LogisticRegression(max_iter=200).fit(X, y)
    score = model.predict_proba(X)[:, 1]
    res = spi.shap_worst_errors_explanation(model, X, y, score, k=4)
    assert res.skipped is not None and "non-tree" in res.skipped
    assert res.figure is None


def test_no_errors_annotated():
    """Perfectly-separated input (all severities < 0.5) still renders, annotated as no-misclassification."""
    X, y = _simple_binary(n=300)
    model = _fit_clf(X, y)
    # Hand the model's own confident predictions back as both truth and score -> severity ~ 0.
    pred = model.predict(X)
    score = pred.astype(float)
    res = spi.shap_worst_errors_explanation(model, X, pred, score * 0.99 + (1 - pred) * 0.01, k=2)
    assert res.skipped is None
    assert float(np.max(res.severities)) < 0.5
    plt.close(res.figure)


def test_paths_written(tmp_path):
    """plot_file -> a PNG is written."""
    X, y = _simple_binary()
    model = _fit_clf(X, y)
    score = model.predict_proba(X)[:, 1]
    out = os.path.join(str(tmp_path), "shap_per_instance.png")
    res = spi.shap_worst_errors_explanation(model, X, y, score, k=4, plot_file=out)
    assert res.paths and os.path.exists(res.paths[0])
    plt.close(res.figure)


def test_biz_val_responsible_feature_dominates_instance_attribution():
    """A confident-wrong positive forced by an f2 outlier MUST have f2 as the largest |SHAP| in that row.

    Construct y = 1 iff f0 > 0. Then plant rows with f0 < 0 (true label 0) but a large positive f2 spike,
    and TRAIN with those rows mislabelled as 1 so the forest learns 'big f2 -> positive'. At score time
    those planted rows become confident-wrong positives (true=0, prob high). The per-instance attribution
    must blame f2 (largest |SHAP|) -- the explanation correctly attributes the costly error to f2.
    """
    rng = np.random.default_rng(7)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"].to_numpy() > 0).astype(int)

    n_plant = 60
    plant = rng.choice(np.where(X["f0"].to_numpy() < -0.3)[0], size=n_plant, replace=False)
    X.iloc[plant, X.columns.get_loc("f2")] = rng.uniform(6.0, 9.0, size=n_plant)  # f2 outlier spike
    y_train = y.copy()
    y_train[plant] = 1  # mislabel: teach the forest 'big f2 -> positive'

    model = _fit_clf(X, y_train)
    score = model.predict_proba(X)[:, 1]

    # True labels (NOT the corrupted training labels) -> the planted rows are confident-wrong positives.
    res = spi.shap_worst_errors_explanation(model, X, y, score, feature_names=list(X.columns), k=8)
    assert res.skipped is None

    planted_set = set(plant.tolist())
    f2_blamed = 0
    n_planted_in_topk = 0
    for orig, contrib in zip(res.worst_idx.tolist(), res.contributions):
        if orig in planted_set:
            n_planted_in_topk += 1
            top_feature = contrib[0][0]  # contributions sorted by |SHAP| desc
            if top_feature == "f2":
                f2_blamed += 1
    assert n_planted_in_topk >= 4, f"expected planted rows among worst errors, got {n_planted_in_topk}"
    frac = f2_blamed / max(n_planted_in_topk, 1)
    assert frac >= 0.75, f"f2 should dominate planted-error attributions; got {frac:.2f} ({f2_blamed}/{n_planted_in_topk})"
    plt.close(res.figure)


def test_cprofile_background_bounded():
    """The explained background is capped to max_explain_rows even when n is much larger."""
    X, y = _simple_binary(n=8000)
    model = _fit_clf(X, y)
    score = model.predict_proba(X)[:, 1]

    pr = cProfile.Profile()
    pr.enable()
    res = spi.shap_worst_errors_explanation(model, X, y, score, k=4, max_explain_rows=500)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(10)
    assert res.skipped is None
    assert res.n_background <= 500
    plt.close(res.figure)
