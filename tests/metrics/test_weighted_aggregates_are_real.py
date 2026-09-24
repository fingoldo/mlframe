"""weighted_* needs real class supports, and p-values / dof / base rates are never averaged."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf


def _run(y, X):
    model = LogisticRegression(max_iter=500).fit(X, y)
    metrics: dict = {}
    report_probabilistic_model_perf(
        targets=y, columns=[f"f{i}" for i in range(X.shape[1])], model_name="m", model=model,
        probs=model.predict_proba(X), preds=model.predict(X), plot_file="", show_perf_chart=False,
        print_report=False, metrics=metrics, classes=list(model.classes_),
    )
    return metrics


def _imbalanced(labels):
    rng = np.random.default_rng(0)
    counts = [300, 60, 20]
    y = np.concatenate([[lab] * c for lab, c in zip(labels, counts)])
    X = rng.normal(size=(len(y), 3)) + np.repeat(np.arange(3), counts)[:, None]
    return y, X


def test_string_labels_get_real_supports():
    """A failed int cast used to leave every weight at 1, so weighted_* silently equalled macro_*."""
    y, X = _imbalanced(["a", "b", "c"])
    m = _run(y, X)
    # Aggregates only: a key like weighted_kappa is a metric in its own right, with no macro_ twin.
    weighted = {k: v for k, v in m.items() if isinstance(k, str) and k.startswith("weighted_") and k.replace("weighted_", "macro_", 1) in m and np.isfinite(v)}
    assert weighted, "weighted aggregates must be emitted when supports are known"
    differing = [k for k, v in weighted.items() if not np.isclose(v, m[k.replace("weighted_", "macro_", 1)])]
    assert differing, "with 300/60/20 supports at least one weighted mean must differ from the macro mean"


def test_p_values_and_dof_are_not_aggregated():
    y, X = _imbalanced([0, 1, 2])
    m = _run(y, X)
    bad = [k for k in m if isinstance(k, str) and (k.startswith("macro_") or k.startswith("weighted_"))
           and (k.endswith("_p") or k.endswith("_dof") or k.endswith("base_rate"))]
    assert bad == [], f"a mean of p-values / dof / base rates is not a statistic: {bad}"
