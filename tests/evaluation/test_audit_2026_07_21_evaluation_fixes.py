"""Regression tests for audits/full_audit_2026-07-21/evaluation.md findings F1-F10, P1a, and proposals
PR1/PR2/PR4/PR5/PR6/PR7.

F8/PR3 (splitting evaluate_estimators into named helpers) was assessed and deferred: it is a 327-line,
25-parameter, 5-nesting-level function-complexity concern, not a file-size violation (reports.py itself is
well under the project's 1k-LOC file gate) -- a full extraction carries real regression risk disproportionate
to a P2 architecture observation with no independent functional bug (F1's rename already makes the concrete
pos_label bug structurally fixed). No code change; documented here, not silently dropped.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# F1: evaluate_estimators' pos_label parameter clobbered by a same-named calibration-plot loop variable
# ---------------------------------------------------------------------------


def test_f1_pos_label_not_clobbered_across_estimators():
    """Control-vs-paired comparison: an IDENTICALLY-CONFIGURED estimator evaluated ALONE (where
    `pos_label` can never be clobbered -- the calibration loop only runs once) must produce the EXACT
    same classification_report_dict as when evaluated as the SECOND of two estimators. Pre-fix, the
    first estimator's `for pos_label in range(nclasses):` calibration loop left `pos_label` at
    `nclasses - 1` (=1) by the time the second estimator's own preds were computed, silently
    overriding the caller's requested `pos_label=0` for every estimator after the first."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.evaluation.reports import evaluate_estimators

    rng = np.random.default_rng(0)
    n = 300
    X_train = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y_train = (X_train["f0"].to_numpy() + rng.normal(scale=0.2, size=n) > 0).astype(np.int64)
    X_test = pd.DataFrame(rng.normal(size=(150, 4)), columns=[f"f{i}" for i in range(4)])
    y_test = (X_test["f0"].to_numpy() + rng.normal(scale=0.2, size=150) > 0).astype(np.int64)

    def make_est():
        """Builds one estimator instance for this test's per-estimator loop."""
        return LogisticRegression(max_iter=200, random_state=0)

    common_kwargs = dict(
        pos_label=0,
        use_sklearn_calibration=True,
        show_calibration_plot=True,
        show_classification_report=True,
        show_confusion_matrix=False,
        plot=False,
        display_labels=["neg", "pos"],
    )

    results_log_alone = {"results": {}}
    evaluate_estimators(X_train, X_test, y_train, y_test, estimators=[make_est()], results_log=results_log_alone, **common_kwargs)
    dict_alone = results_log_alone["results"]["classification_report_dict"]

    results_log_paired = {"results": {}}
    evaluate_estimators(X_train, X_test, y_train, y_test, estimators=[make_est(), make_est()], results_log=results_log_paired, **common_kwargs)
    dict_paired = results_log_paired["results"]["classification_report_dict"]

    assert dict_alone == dict_paired


def test_f1_calibration_loop_variable_renamed_no_shadowing():
    """F1: calibration loop variable renamed no shadowing."""
    import inspect

    from mlframe.evaluation import reports

    src = inspect.getsource(reports.evaluate_estimators)
    assert "for pos_label in range(nclasses)" not in src
    assert "for _calib_class_label in range(nclasses)" in src


# ---------------------------------------------------------------------------
# F2 / F9 / PR1: optimize_group_blend_weight didn't actually hold previously-tuned groups fixed
# ---------------------------------------------------------------------------


def test_f2_optimize_group_blend_weight_two_disjoint_groups_applies_cumulatively(monkeypatch):
    """Spy on label_correlation_rerank to confirm the SECOND group's CV search reranks BOTH groups
    (previously-tuned group A included), not just group B in isolation."""
    # importlib.import_module (not `import X.Y as alias` / `from X import Y as alias`) -- mlframe.evaluation's
    # own __init__.py imports the FUNCTION `label_correlation_rerank` and binds it as the PACKAGE attribute
    # `mlframe.evaluation.label_correlation_rerank`, shadowing the submodule at that same attribute path (both
    # `import a.b.c as x` and `from a.b import c` resolve via attribute traversal, which sees the function, not
    # the module -- only sys.modules / importlib.import_module reach the real submodule object).
    import importlib

    lcr_mod = importlib.import_module("mlframe.evaluation.label_correlation_rerank")

    seen_groups_by_call = []
    real_rerank = lcr_mod.label_correlation_rerank

    def spy_rerank(*args, **kwargs):
        """Records rerank calls for this test's assertions."""
        seen_groups_by_call.append(list(kwargs.get("correlated_groups", [])))
        return real_rerank(*args, **kwargs)

    monkeypatch.setattr(lcr_mod, "label_correlation_rerank", spy_rerank)

    rng = np.random.default_rng(0)
    n, n_labels = 400, 6
    y_true = rng.integers(0, 2, size=(n, n_labels))
    pred_scores = np.clip(y_true.astype(np.float64) + rng.normal(scale=0.3, size=(n, n_labels)), 0, 1)
    group_a = (0, 1)
    group_b = (2, 3)

    weights = lcr_mod.optimize_group_blend_weight(
        y_true, pred_scores, correlated_groups=[group_a, group_b], n_splits=2, random_state=0,
    )
    assert set(weights.keys()) == {group_a, group_b}

    # Calls made while tuning group_b (i.e. AFTER weights[group_a] exists) must include group_a too.
    group_b_calls = [g for g in seen_groups_by_call if group_b in g]
    assert group_b_calls, "expected at least one label_correlation_rerank call while tuning group_b"
    assert any(group_a in g for g in group_b_calls), (
        "group_a must be included in group_b's CV evaluation once group_a has an optimized weight "
        "(coordinate ascent holding it fixed) -- pre-fix, only [group_b] was ever passed"
    )


def test_f2_default_single_group_behavior_unaffected():
    """A single group must still work exactly as before (nothing to hold fixed)."""
    from mlframe.evaluation.label_correlation_rerank import optimize_group_blend_weight

    rng = np.random.default_rng(1)
    n, n_labels = 300, 4
    y_true = rng.integers(0, 2, size=(n, n_labels))
    pred_scores = np.clip(y_true.astype(np.float64) + rng.normal(scale=0.3, size=(n, n_labels)), 0, 1)
    weights = optimize_group_blend_weight(y_true, pred_scores, correlated_groups=[(0, 1, 2, 3)], n_splits=2, random_state=0)
    assert set(weights.keys()) == {(0, 1, 2, 3)}
    assert 0.0 <= weights[(0, 1, 2, 3)] <= 1.0


# ---------------------------------------------------------------------------
# P1a / PR2: constant_group_target_scan's overall_var was not NaN-safe
# ---------------------------------------------------------------------------


def test_p1a_nan_in_y_raises_clear_error_instead_of_silent_nan_ratios():
    """P1a nan in y raises clear error instead of silent nan ratios."""
    from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"group_col": rng.integers(0, 5, size=n)})
    y = rng.normal(size=n)
    y[10] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        constant_group_target_scan(df, y, candidate_cols=["group_col"])


def test_p1a_no_nan_still_works():
    """P1a no nan still works."""
    from mlframe.evaluation.constant_group_leak_scan import constant_group_target_scan

    rng = np.random.default_rng(0)
    n = 300
    # group_col strongly predicts y's mean -> low within-group variance ratio -> flagged.
    group_col = rng.integers(0, 3, size=n)
    y = group_col.astype(np.float64) * 10.0 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"group_col": group_col})
    result = constant_group_target_scan(df, y, candidate_cols=["group_col"], min_group_size=5)
    assert not result.empty
    assert bool(result.iloc[0]["flagged"])


# ---------------------------------------------------------------------------
# F3: expanding_window_leakage's auto_remediate re-check could break under duplicate time_col ties
# ---------------------------------------------------------------------------


def test_f3_stable_sort_makes_verification_resort_a_true_identity():
    """F3: stable sort makes verification resort a true identity."""
    from sklearn.linear_model import LinearRegression

    from mlframe.evaluation.expanding_window_leakage import detect_expanding_window_feature_leakage

    rng = np.random.default_rng(0)
    n = 600
    # Heavy ties in time_col (only 15 distinct days over 600 rows) -- the identity-critical case for the
    # auto_remediate verification re-check's internal re-sort.
    time_col_vals = rng.integers(0, 15, size=n)
    n_cats = 10
    cat_rate = rng.uniform(0.5, 5.0, n_cats)
    cat = rng.integers(0, n_cats, size=n)
    df = pd.DataFrame({"t": time_col_vals, "cat": cat})
    y = cat_rate[cat] * 3.0 + rng.normal(scale=1.0, size=n)

    def fit_transform_fn(fit_df, transform_df):
        """Fake fit_transform callable used to control the resort input."""
        counts = fit_df["cat"].value_counts()
        return transform_df["cat"].map(counts).fillna(0).to_numpy(dtype=np.float64)

    result = detect_expanding_window_feature_leakage(
        df, "t", y, fit_transform_fn, lambda: LinearRegression(), n_splits=5, scoring="r2", auto_remediate=True,
    )
    assert "remediated_feature" in result
    assert len(result["remediated_feature"]) == n
    # remediation_verified must be a genuine bool (the re-check ran without an index-misalignment crash
    # or silently-corrupted verification), and re-running the SAME detector with the leakage-safe
    # remediated_feature substituted must no longer detect the original leak.
    assert isinstance(result["remediation_verified"], bool)
    assert result["remediation_verified"] is True


# ---------------------------------------------------------------------------
# F4 / PR4: bootstrap_metrics missing from __all__ and package re-exports
# ---------------------------------------------------------------------------


def test_f4_bootstrap_metrics_in_all_and_reexported():
    """F4: bootstrap metrics in all and reexported."""
    from mlframe.evaluation import bootstrap as bootstrap_mod
    from mlframe.evaluation import bootstrap_metrics

    assert "bootstrap_metrics" in bootstrap_mod.__all__
    assert bootstrap_metrics is bootstrap_mod.bootstrap_metrics


# ---------------------------------------------------------------------------
# F5 / PR5: subpopulation_ratio_drift_check's undocumented NaN-drop, now opt-out
# ---------------------------------------------------------------------------


def test_f5_dropna_false_surfaces_nan_prevalence_shift():
    """F5: dropna false surfaces nan prevalence shift."""
    from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check

    train_df = pd.DataFrame({"g": [1, 1, 1, 1, np.nan]})  # 20% NaN
    test_df = pd.DataFrame({"g": [1, np.nan, np.nan, np.nan, np.nan]})  # 80% NaN

    out = subpopulation_ratio_drift_check(train_df, test_df, "g", dropna=False)
    nan_rows = out[out["subgroup_value"].isna()]
    assert len(nan_rows) == 1
    assert nan_rows.iloc[0]["train_prevalence"] == pytest.approx(0.2)
    assert nan_rows.iloc[0]["test_prevalence"] == pytest.approx(0.8)


def test_f5_default_dropna_true_unaffected():
    """F5: default dropna true unaffected."""
    from mlframe.evaluation.subpopulation_drift import subpopulation_ratio_drift_check

    train_df = pd.DataFrame({"g": [1, 1, 2, np.nan]})
    test_df = pd.DataFrame({"g": [1, 2, 2, np.nan]})
    out = subpopulation_ratio_drift_check(train_df, test_df, "g")
    assert not out["subgroup_value"].isna().any()


# ---------------------------------------------------------------------------
# F6: evaluate_grouped's hardcoded Russian output column labels, now overridable
# ---------------------------------------------------------------------------


def test_f6_output_labels_overridable():
    """F6: output labels overridable."""
    from mlframe.evaluation.reports import evaluate_grouped

    class _FixedPredictor:
        """Stub predictor returning a fixed prediction array."""
        def predict(self, X):
            """No-op / recording stub matching the estimator's predict() signature."""
            return (X["score"].to_numpy() > 0).astype(int)

    rng = np.random.default_rng(1)
    n = 300
    X_test = pd.DataFrame({"group": rng.choice(["a", "b"], size=n), "score": rng.standard_normal(n)})
    y_test = pd.Series((X_test["score"].to_numpy() + rng.normal(0, 0.5, n) > 0).astype(int))

    out = evaluate_grouped(
        _FixedPredictor(), X_test, y_test, by_column="group", ntop=5, min_population=1,
        count_col_name="count", precision_col_name="precision", recall_col_name="recall",
    )
    assert {"group", "count", "precision", "recall"}.issubset(out.columns)
    assert not {"Откликов", "Точность", "Полнота"}.intersection(out.columns)


def test_f6_default_labels_unchanged():
    """F6: default labels unchanged."""
    from mlframe.evaluation.reports import evaluate_grouped

    class _FixedPredictor:
        """Stub predictor returning a fixed prediction array."""
        def predict(self, X):
            """No-op / recording stub matching the estimator's predict() signature."""
            return (X["score"].to_numpy() > 0).astype(int)

    rng = np.random.default_rng(1)
    n = 300
    X_test = pd.DataFrame({"group": rng.choice(["a", "b"], size=n), "score": rng.standard_normal(n)})
    y_test = pd.Series((X_test["score"].to_numpy() + rng.normal(0, 0.5, n) > 0).astype(int))
    out = evaluate_grouped(_FixedPredictor(), X_test, y_test, by_column="group", ntop=5, min_population=1)
    assert {"group", "Откликов", "Точность", "Полнота"}.issubset(out.columns)


# ---------------------------------------------------------------------------
# F7: reports.py's baseline_model.predict(...).astype(int) truncation
# ---------------------------------------------------------------------------


def test_f7_continuous_baseline_not_truncated():
    """F7: continuous baseline not truncated."""
    import inspect

    from mlframe.evaluation import reports

    src = inspect.getsource(reports)
    assert "baseline_model.predict(X_test_val).astype(int)" not in src
    assert "eval_set.set_baseline(np.asarray(baseline_model.predict(X_test_val), dtype=np.float64))" in src


# ---------------------------------------------------------------------------
# F10 / PR7: leak_scan.py's stale refactor-history comment
# ---------------------------------------------------------------------------


def test_f10_stale_double_argsort_comment_removed():
    """F10: stale double argsort comment removed."""
    import inspect

    from mlframe.evaluation import leak_scan

    src = inspect.getsource(leak_scan._spearman_against)
    assert "still had the old double-argsort" not in src


# ---------------------------------------------------------------------------
# PR6: O(n^2) memory in check_pairwise_score_correlation / _energy_distance now warns above a size guard
# ---------------------------------------------------------------------------


def test_pr6_pairwise_score_correlation_warns_above_size_guard(monkeypatch):
    """PR6: pairwise score correlation warns above size guard."""
    from mlframe.evaluation import blend_source_selection as bss_mod

    monkeypatch.setattr(bss_mod, "_PAIRWISE_MATRIX_WARN_N", 50)
    rng = np.random.default_rng(0)
    n = 100
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.1, size=n)
    with pytest.warns(UserWarning, match="pairwise-comparison matrices"):
        bss_mod.check_pairwise_score_correlation(a, b)


def test_pr6_pairwise_score_correlation_silent_under_guard():
    """PR6: pairwise score correlation silent under guard."""
    from mlframe.evaluation.blend_source_selection import check_pairwise_score_correlation

    rng = np.random.default_rng(0)
    n = 50
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.1, size=n)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_pairwise_score_correlation(a, b)
    assert not any("pairwise-comparison matrices" in str(w.message) for w in caught)


def test_pr6_energy_distance_warns_above_size_guard(monkeypatch):
    """PR6: energy distance warns above size guard."""
    import importlib

    # importlib (see test_f2's comment above) -- mlframe.evaluation's __init__.py imports the FUNCTION
    # `distribution_matching_subset_search` into the package namespace, shadowing the submodule.
    dmss_mod = importlib.import_module("mlframe.evaluation.distribution_matching_subset_search")

    monkeypatch.setattr(dmss_mod, "_ENERGY_DISTANCE_WARN_N", 20)
    rng = np.random.default_rng(0)
    sample_mat = rng.normal(size=(50, 3))
    target_mat = rng.normal(size=(50, 3))
    with pytest.warns(UserWarning, match="pairwise-distance matrices"):
        dmss_mod._energy_distance(sample_mat, target_mat, target_target_dist=0.0)
