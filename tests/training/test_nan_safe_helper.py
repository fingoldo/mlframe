"""Wave-21 sensor cluster: mlframe.utils.nan_safe helper + 11 production
call sites migrated to it.

Audit wave 21 found a missing central nan-safe helper across the
codebase: ~200 raw `np.argmax` / `np.median` / `np.percentile` /
`np.quantile` calls vs ~32 `nan*` variants. Each non-nan call was a
potential silent-NaN-propagation bug.

This sensor pins both:
1. The helper's contract (``argmax_classes_safe`` / ``quantile_safe`` /
   ``median_safe`` in ``mlframe.utils.nan_safe``).
2. The 11 production sites that now use the helper or the nan-aware
   variants (training/core/predict.py x2, training/_reporting.py,
   evaluation/reports.py, reporting/charts/multiclass.py x2,
   reporting/charts/ltr.py, metrics/core.py, models/ensembling.py x4,
   feature_selection/general.py, feature_engineering/numerical.py,
   training/_classif_helpers.py, calibration/quality.py x2).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

# ---- mlframe.utils.nan_safe contract -----------------------------------


def test_argmax_classes_safe_all_finite():
    """Argmax classes safe all finite."""
    from mlframe.utils.nan_safe import argmax_classes_safe

    p = np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]])
    out = argmax_classes_safe(p, context="test")
    np.testing.assert_array_equal(out, [1, 0])


def test_argmax_classes_safe_with_all_nan_row_uses_fallback(caplog):
    """Argmax classes safe with all nan row uses fallback."""
    from mlframe.utils.nan_safe import argmax_classes_safe

    p = np.array(
        [
            [0.1, 0.7, 0.2],
            [np.nan, np.nan, np.nan],
            [0.5, 0.4, 0.3],
        ]
    )
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = argmax_classes_safe(p, fallback_class=99, context="test")
    np.testing.assert_array_equal(out, [1, 99, 0])
    assert any("1/3 rows contain NO finite probabilities" in r.message for r in caplog.records)


def test_argmax_classes_safe_mixed_finite_nan_row():
    """A row with SOME finite entries uses nanargmax (picks max-finite)."""
    from mlframe.utils.nan_safe import argmax_classes_safe

    p = np.array(
        [
            [0.1, np.nan, 0.5],
            [np.nan, 0.7, 0.2],
        ]
    )
    out = argmax_classes_safe(p, context="test")
    np.testing.assert_array_equal(out, [2, 1])


def test_argmax_classes_safe_1d_array():
    """Argmax classes safe 1d array."""
    from mlframe.utils.nan_safe import argmax_classes_safe

    p = np.array([0.1, 0.7, 0.2])
    out = argmax_classes_safe(p, context="test")
    # Shape is asserted alongside the value: all three 1-D branches must agree on it, and they did not --
    # only the all-finite one returned (1,), so the same `int(out)` that passes here raised
    # "only 0-dimensional arrays can be converted to Python scalars" on the all-NaN sibling below.
    assert out.shape == (1,), f"the documented (N,) contract is broken: got shape {out.shape}"
    assert out.dtype == np.int64
    assert out[0] == 1


def test_argmax_classes_safe_1d_all_nan(caplog):
    """Argmax classes safe 1d all nan."""
    from mlframe.utils.nan_safe import argmax_classes_safe

    p = np.array([np.nan, np.nan, np.nan])
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = argmax_classes_safe(p, fallback_class=7, context="test")
    assert out.shape == (1,), f"the all-NaN branch disagrees with its all-finite sibling: got shape {out.shape}"
    assert out.dtype == np.int64
    assert out[0] == 7


def test_argmax_classes_safe_1d_partial_nan():
    """The third 1-D branch: some entries finite, so nanargmax picks the max FINITE index.

    Untested until now, and it was the branch that quietly returned a 0-d array -- the shape assertion is the
    point, since the value alone reads the same whether the result is 0-d or (1,).
    """
    from mlframe.utils.nan_safe import argmax_classes_safe

    p = np.array([0.1, np.nan, 0.9, np.nan])
    out = argmax_classes_safe(p, context="test")
    assert out.shape == (1,), f"the partial-NaN branch disagrees with its siblings: got shape {out.shape}"
    assert out.dtype == np.int64
    assert out[0] == 2, "nanargmax must pick the index of the largest FINITE entry, not a NaN slot"


def test_quantile_safe_finite_input():
    """Quantile safe finite input."""
    from mlframe.utils.nan_safe import quantile_safe

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert quantile_safe(arr, 0.5) == 3.0


def test_quantile_safe_with_nan_input():
    """Quantile safe with nan input."""
    from mlframe.utils.nan_safe import quantile_safe

    arr = np.array([1.0, 2.0, np.nan, 4.0])
    # nanquantile interpolation: median of [1,2,4] is 2.0.
    assert quantile_safe(arr, 0.5) == 2.0


def test_quantile_safe_all_nan_returns_fallback(caplog):
    """Quantile safe all nan returns fallback."""
    from mlframe.utils.nan_safe import quantile_safe

    arr = np.array([np.nan, np.nan])
    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = quantile_safe(arr, 0.5, fallback=-1.0)
    assert out == -1.0


def test_quantile_safe_q_sequence():
    """Quantile safe q sequence."""
    from mlframe.utils.nan_safe import quantile_safe

    arr = np.array([1.0, 2.0, 3.0, 4.0])
    out = quantile_safe(arr, [0.25, 0.5, 0.75])
    np.testing.assert_array_almost_equal(out, [1.75, 2.5, 3.25])


def test_median_safe_finite():
    """Median safe finite."""
    from mlframe.utils.nan_safe import median_safe

    assert median_safe(np.array([1.0, 2.0, 3.0])) == 2.0


def test_median_safe_with_nan():
    """Median safe with nan."""
    from mlframe.utils.nan_safe import median_safe

    assert median_safe(np.array([1.0, np.nan, 3.0])) == 2.0


def test_median_safe_all_nan_fallback(caplog):
    """Median safe all nan fallback."""
    from mlframe.utils.nan_safe import median_safe

    with caplog.at_level(logging.WARNING, logger="mlframe.utils.nan_safe"):
        out = median_safe(np.array([np.nan, np.nan]), fallback=0.0)
    assert out == 0.0


# ---- Production-site behavioural guards ---------------------------------

_NAN_PROBS = np.array([[0.1, np.nan, 0.9], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])


def test_wave21_evaluation_reports_multiclass_preds_skip_nan_column():
    """evaluate_estimator's multiclass hard labels ignore a NaN probability instead of treating it as the maximum."""
    from mlframe.evaluation.reports import _evaluate_estimator_nclasses

    np.testing.assert_array_equal(_evaluate_estimator_nclasses(3, None, _NAN_PROBS, 1), [2, 0, 2])


def test_wave21_classif_helpers_multiclass_decision_skips_nan_column():
    """The multiclass decision rule maps a NaN-bearing row to its largest finite probability, through classes_."""
    from mlframe.training._classif_helpers import _predict_from_probs
    from mlframe.training.configs import TargetTypes

    out = _predict_from_probs(_NAN_PROBS, TargetTypes.MULTICLASS_CLASSIFICATION, classes_=np.array([10, 20, 30]))
    np.testing.assert_array_equal(out, [30, 10, 30])


def test_wave21_suite_predict_multiclass_preds_skip_nan_column():
    """The suite predict path labels a NaN-bearing multiclass row with its largest finite probability."""
    from mlframe.training.core._predict_main_suite import _predict_mlframe_mo_never_test_falls_back

    np.testing.assert_array_equal(_predict_mlframe_mo_never_test_falls_back(_NAN_PROBS, 0.5, "m"), [2, 0, 2])


def test_wave21_multiclass_confusion_panel_skips_nan_column():
    """The multiclass report's confusion matrix scores a NaN-bearing row by its largest finite probability."""
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    fig = compose_multiclass_figure(np.array([2, 0, 2]), _NAN_PROBS, [0, 1, 2], panels_template="CONFUSION")
    matrix = np.asarray(fig.panels[0][0].matrix)
    np.testing.assert_allclose(matrix, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_wave21_ltr_top1_panel_ignores_nan_score():
    """A NaN score is never picked as a query's top document, so the finite top-scored document decides correctness."""
    from mlframe.reporting.charts.ltr import _top1_by_qsize_panel

    y_score = np.array([0.2, np.nan, 0.9])
    groups = np.zeros(3, dtype=int)
    wrong_top = _top1_by_qsize_panel(np.array([0, 1, 0]), y_score, groups)
    np.testing.assert_allclose(wrong_top.values, [0.0])
    right_top = _top1_by_qsize_panel(np.array([0, 0, 1]), y_score, groups)
    np.testing.assert_allclose(right_top.values, [1.0])


def test_wave21_fairness_metrics_survive_a_nan_bin():
    """A subgroup whose metric is NaN does not turn the across-subgroup mean NaN."""
    import pandas as pd

    from mlframe.metrics._fairness_metrics import compute_fairness_metrics

    bins = pd.Series(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    y_true = np.arange(12, dtype=np.float64)

    def metric(yt, yp):
        """Return NaN for the group holding the largest targets, otherwise the mean absolute error."""
        return float("nan") if yt.max() == 11 else float(np.mean(np.abs(yt - yp)))

    y_pred = y_true + np.repeat([1.0, 3.0, 0.0], 4)
    table = compute_fairness_metrics(
        metrics={"mae": metric},
        metrics_higher_is_better={"mae": False},
        subgroups={"g": {"bins": bins}},
        subset_index=bins.index,
        y_true=y_true,
        y_pred=y_pred,
    )
    assert len(table) == 1
    assert table["metric_mean"].iloc[0] == pytest.approx(2.0)


def test_wave21_member_quality_gate_median_ignores_nan_member():
    """A member with non-finite predictions is excluded and the cross-member median stays finite, so the relative gate still works."""
    from mlframe.models.ensembling.quality_gate import compute_member_quality_gate

    base = np.linspace(0.1, 0.9, 50)
    members = [base, base + 0.01, base + 0.02, np.full(50, np.nan)]
    kept, excluded, stats = compute_member_quality_gate(members)
    assert kept == [0, 1]
    assert sorted(i for i, _ in excluded) == [2, 3]
    assert np.isfinite(stats["median_mae"]) and stats["median_mae"] == pytest.approx(0.01)
    assert stats["rel_mae_threshold"] == pytest.approx(2.5 * 0.01)


def test_wave21_numaggs_quantiles_ignore_nan():
    """Quantile features of a NaN-bearing column equal the quantiles of its finite values."""
    from mlframe.feature_engineering._numerical_counts import compute_nunique_modes_quantiles_numpy, default_quantiles

    values = np.array([1.0, 2.0, 2.0, 3.0, np.nan, 5.0, 5.0, 5.0, 7.0])
    res = compute_nunique_modes_quantiles_numpy(values)
    finite = values[np.isfinite(values)]
    expected = np.nanquantile(finite, default_quantiles, method="median_unbiased")
    np.testing.assert_allclose(np.array(res[5 : 5 + len(expected)], dtype=np.float64), expected)


def test_wave21_calibration_bins_use_nan_aware_means():
    """A NaN prediction inside a bin leaves that bin's mean prediction and mean outcome finite."""
    from mlframe.calibration.quality import bin_predictions

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 1.0])
    y_pred = np.array([0.1, 0.2, np.nan, 0.4, 0.6, 0.7, 0.8, 0.9])
    pockets_pred, pockets_true, data = bin_predictions(y_true, y_pred, np.argsort(y_pred), 2)
    np.testing.assert_allclose(pockets_pred, [0.325, 0.8])
    np.testing.assert_allclose(pockets_true, [0.75, 0.5])
    assert np.isfinite(data).all()


class _NanProbaMulticlass:
    """Fitted-model stand-in whose predict_proba emits a NaN probability in the first row."""

    classes_ = np.array([0, 1, 2])

    def __init__(self):
        """Expose the feature names the predict-time column check reads."""
        self.feature_names_in_ = np.array(["x0", "x1"], dtype=object)

    def predict_proba(self, X):
        """Three-class probabilities, row 0 carries a NaN."""
        return _NAN_PROBS.copy()

    def predict(self, X):
        """Hard labels from the finite probabilities."""
        return np.array([2, 0, 2])


def test_wave21_predict_from_models_per_model_and_ensemble_preds_skip_nan_column():
    """predict_from_models labels NaN-bearing rows by their largest finite probability for each member and for the ensemble."""
    import pandas as pd

    from mlframe.training.configs import TargetTypes
    from mlframe.training.core.predict import predict_from_models

    members = [SimpleNamespace(model=_NanProbaMulticlass(), pre_pipeline=None, model_name=f"m{i}") for i in range(2)]
    models = {TargetTypes.MULTICLASS_CLASSIFICATION: {"y": members}}
    df = pd.DataFrame({"x0": [0.0, 1.0, 2.0], "x1": [1.0, 2.0, 3.0]})
    metadata = {"columns": ["x0", "x1"], "raw_input_columns": ["x0", "x1"]}
    results = predict_from_models(df=df, models=models, metadata=metadata, features_and_targets_extractor=None, return_probabilities=True, verbose=0)
    per_model = {name: preds for name, preds in results["predictions"].items() if name != "ensemble"}
    assert len(per_model) == 2
    for name, preds in per_model.items():
        np.testing.assert_array_equal(np.asarray(preds), [2, 0, 2], err_msg=name)
    np.testing.assert_array_equal(np.asarray(results["ensemble_predictions"]), [2, 0, 2])


def test_wave21_efs_permuted_mi_baseline_ignores_nan_samples():
    """NaN permuted-MI samples do not turn the permutation baseline NaN: a clearly informative feature is still credited."""
    from mlframe.feature_selection.general import _aggregate_permuted_mis_per_target

    usefulness = _aggregate_permuted_mis_per_target(
        target_indices=[2],
        target_columns=["t"],
        all_permuted_mis={"t": [np.array([0.01, np.nan, 0.02, 0.015])]},
        current_permuted_mis={"t": [np.array([[0.01, 0.0], [np.nan, 0.0], [0.02, 0.01]])]},
        verbose=0,
        occupied_bins_per_col=np.array([4, 4, 2]),
        original_mi_results=[np.array([0.5, 0.0])],
        n_samples=1000,
        bins=np.zeros((10, 2)),
        permuted_max_mi_quantile=0.99,
        min_mi_prevalence=1.0,
        max_permuted_prevalence_percent=0.05,
        fdr_alpha=0.05,
        features_usefulness=np.zeros(2, dtype=np.int32),
    )
    np.testing.assert_array_equal(usefulness, [1, 0])


def test_wave21_probabilistic_report_preds_are_nan_safe():
    """The probabilistic report derives multiclass preds with a NaN-safe argmax.

    Raw ``np.argmax`` treats NaN as the maximum, so a row ``[0.1, nan, 0.9]`` was labelled class 1 (the NaN
    column); the NaN-safe argmax picks the largest finite probability (class 2), and an all-NaN row gets the
    fallback class rather than an arbitrary one.
    """
    from mlframe.training.reporting._reporting_probabilistic_helpers import _report_probabilist_preds_none

    probs = np.array([[0.1, np.nan, 0.9], [0.7, 0.2, 0.1], [np.nan, np.nan, np.nan]])
    targets = np.array([2, 0, 1])
    model = SimpleNamespace(classes_=np.array([10, 20, 30]))
    preds, out_probs = _report_probabilist_preds_none(None, targets, probs, None, model)
    assert out_probs is probs
    np.testing.assert_array_equal(preds, np.array([30, 10, 10]))
