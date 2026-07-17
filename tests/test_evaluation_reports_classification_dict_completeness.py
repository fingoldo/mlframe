"""Regression test: classification-report dict assembly in evaluation/reports.py no longer drops
computed aggregates.

Pre-fix: two call sites in evaluation/reports.py unpacked balanced_accuracy/macro_avgs/(one also
weighted_avgs) from fast_classification_report() and then silently discarded them -- the assembled
output dict only ever contained per-class rows + "accuracy" (+ "weighted avg" at the second site).
Both aggregates were genuinely computed by the kernel, just never surfaced, mirroring
sklearn.metrics.classification_report's "macro avg"/"weighted avg" rows plus balanced_accuracy.

Both call sites are private closures nested inside larger functions (not independently importable),
so this test exercises the fix at the level that IS testable: the fast_classification_report kernel's
output values, and that sklearn agrees with the macro/weighted/balanced_accuracy numbers now being
threaded into the dicts (pinning the values themselves, not the private assembly code).
"""

from __future__ import annotations

import numpy as np


def test_fast_classification_report_macro_and_weighted_avgs_match_sklearn():
    """Fast classification report macro and weighted avgs match sklearn."""
    from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

    from mlframe.metrics._core_precision_mape import fast_classification_report

    rng = np.random.default_rng(0)
    n, k = 500, 4
    y_true = rng.integers(0, k, n)
    y_pred = rng.integers(0, k, n)

    (_hits, _misses, _accuracy, balanced_accuracy, _supports, _precisions, _recalls, _f1s, macro_avgs, weighted_avgs) = fast_classification_report(
        y_true, y_pred, nclasses=k
    )

    # These are exactly the values the fixed evaluate_estimators/_report_dict call sites now write
    # into their output dicts under "balanced_accuracy"/"macro avg"/"weighted avg" -- pre-fix they
    # were computed here and then thrown away.
    sk_bal_acc = balanced_accuracy_score(y_true, y_pred)
    assert abs(float(balanced_accuracy) - sk_bal_acc) < 1e-9

    sk_macro_p, sk_macro_r, sk_macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    assert abs(float(macro_avgs[0]) - sk_macro_p) < 1e-9
    assert abs(float(macro_avgs[1]) - sk_macro_r) < 1e-9
    assert abs(float(macro_avgs[2]) - sk_macro_f1) < 1e-9

    sk_w_p, sk_w_r, sk_w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    assert abs(float(weighted_avgs[0]) - sk_w_p) < 1e-9
    assert abs(float(weighted_avgs[1]) - sk_w_r) < 1e-9
    assert abs(float(weighted_avgs[2]) - sk_w_f1) < 1e-9


def test_evaluate_grouped_still_runs_with_widened_report_dict():
    """Smoke: evaluate_grouped's internal _report_dict now carries extra keys (balanced_accuracy/macro
    avg); confirm the existing min_population gate + weighted-avg consumption still works end to end."""
    import pandas as pd

    from mlframe.evaluation.reports import evaluate_grouped

    class _FixedPredictor:
        """Groups tests covering FixedPredictor."""
        def predict(self, X):
            """Helper that predict."""
            return (X["score"].to_numpy() > 0).astype(int)

    rng = np.random.default_rng(1)
    n = 400
    X_test = pd.DataFrame({"group": rng.choice(["a", "b"], size=n), "score": rng.standard_normal(n)})
    y_test = pd.Series((X_test["score"].to_numpy() + rng.normal(0, 0.5, n) > 0).astype(int))

    out = evaluate_grouped(_FixedPredictor(), X_test, y_test, by_column="group", ntop=5, min_population=1)
    assert not out.empty
    assert {"group", "Откликов", "Точность", "Полнота"}.issubset(out.columns)
