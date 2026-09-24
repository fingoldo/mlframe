"""select_column_transforms: a meaningful multiclass score, and context columns picked from train rows only."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.preprocessing.auto_transform_select import _classification_probe_score, _top_correlated_context_columns, select_column_transforms


def test_a_multiclass_probe_is_scored_over_every_class():
    """Scoring [:, 1] alone on a 4-class y compared the labels against one class's probability."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 1))
    y = np.digitize(X[:, 0], [-0.7, 0.0, 0.7])  # 4 ordered classes, fully determined by X
    model = LogisticRegression(max_iter=500).fit(X, y)
    score = _classification_probe_score(model, y, model.predict_proba(X))
    assert score > 0.9, f"a probe that separates all four classes must score high, got {score:.3f}"


def test_binary_still_uses_roc_auc():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 1))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    assert _classification_probe_score(model, y, model.predict_proba(X)) > 0.95


def test_the_selection_runs_end_to_end_on_a_multiclass_target():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = np.digitize(df["a"].to_numpy(), [-0.5, 0.5])
    out = select_column_transforms(df, y, columns=["a"], task="classification")
    assert "a" in out and out["a"]["best_score"] > 0.6


def test_context_ranking_only_sees_the_rows_it_is_given():
    """A column correlated with the audited one only in the held-out rows must not be chosen from the train rows."""
    n = 200
    rng = np.random.default_rng(3)
    base = rng.normal(size=n)
    leaky = rng.normal(size=n)
    leaky[100:] = base[100:]  # identical to the column, but only in the second (held-out) half
    mild = 0.5 * base + rng.normal(size=n)  # a genuine, moderate relationship everywhere
    df = pd.DataFrame({"x": base, "leaky": leaky, "mild": mild})

    assert _top_correlated_context_columns(df, "x", ["leaky", "mild"], 1) == ["leaky"], "over the whole frame the leak wins"
    assert _top_correlated_context_columns(df, "x", ["leaky", "mild"], 1, rows=np.arange(100)) == ["mild"], (
        "on the train rows alone the leak is noise, so the genuine relationship must be chosen"
    )
