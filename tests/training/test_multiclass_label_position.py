"""Regression: multiclass reporting must treat class labels by POSITION.

Multiclass targets are not label-encoded to 0..K-1 anywhere in mlframe, so the
raw labels reaching the reporting layer may be e.g. [1, 2, 3] or [10, 20, 30].

#2  reporting/charts/multiclass.py: every panel builder indexes a K-sized
    structure positionally (``matrix[int(t)]``, ``y_true == k``,
    ``labels=range(K)``). With raw non-0..K-1 labels this IndexErrors (the whole
    multiclass figure is then silently dropped by the dispatcher) or builds a
    wrong confusion matrix / zero top-k. compose_multiclass_figure now remaps
    y_true to positions once before dispatching to the panels.

#3  _reporting_probabilistic.py: the weighted-metric supports counted
    ``_yt_all == cid`` where ``cid`` is the per-class ENUMERATE position, not the
    class label -> for non-0-indexed integer labels the supports shift and
    ``weighted_*`` aggregates are silently wrong. Now counts against the actual
    class label ``classes[cid]``.
"""

from __future__ import annotations

import numpy as np


def test_compose_multiclass_figure_handles_non_contiguous_labels():
    """Non-0..K-1 class labels (e.g. [10, 20, 30]) must index by position, not by raw label value, or figure composition crashes."""
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    classes = [10, 20, 30]  # non-0..K-1 labels: pre-fix matrix[int(10)] -> IndexError
    y_true = np.array([10, 10, 20, 20, 30, 30])
    # Perfect predictions: one-hot on each row's true-class POSITION.
    y_proba = np.eye(3)[[0, 0, 1, 1, 2, 2]].astype(float)

    fig = compose_multiclass_figure(
        y_true,
        y_proba,
        classes,
        panels_template="CONFUSION PR_F1 ROC PR_CURVES CALIB_GRID PROB_DIST TOP_K_ACC",
    )
    assert fig is not None

    mats = [p for row in fig.panels for p in row if p is not None and hasattr(p, "matrix")]
    assert mats, "confusion heatmap panel missing"
    conf = np.asarray(mats[0].matrix)
    # Perfect predictions -> row-normalised confusion is the identity.
    assert np.allclose(np.diag(conf), 1.0), f"confusion diagonal not all 1.0: {np.diag(conf)}"


def test_compose_multiclass_figure_handles_string_and_unseen_labels():
    """Vectorized label->position remap (argsort + searchsorted) must handle string
    classes and unseen labels (-> -1, excluded) exactly like the dict.get listcomp
    it replaced. Guards the large-n optimization on its non-integer / unseen edges."""
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    classes = ["low", "med", "high"]  # non-numeric, non-sorted display labels
    # 6th sample's TRUE label is UNSEEN -> remaps to -1 -> excluded from every panel.
    y_true = np.array(["low", "low", "med", "med", "high", "unseen"])
    # Perfect predictions at each row's true-class POSITION (last row irrelevant: excluded).
    y_proba = np.eye(3)[[0, 0, 1, 1, 2, 2]].astype(float)

    fig = compose_multiclass_figure(
        y_true,
        y_proba,
        classes,
        panels_template="CONFUSION PR_F1 ROC PR_CURVES CALIB_GRID PROB_DIST TOP_K_ACC",
    )
    assert fig is not None
    mats = [p for row in fig.panels for p in row if p is not None and hasattr(p, "matrix")]
    assert mats, "confusion heatmap panel missing"
    conf = np.asarray(mats[0].matrix)
    # The 5 seen rows are perfectly predicted at their class position; the unseen
    # sample is dropped, so every populated class row is the identity.
    assert np.allclose(np.diag(conf), 1.0), f"confusion diagonal not all 1.0: {np.diag(conf)}"


def test_weighted_metric_supports_use_class_label_not_enumerate_index():
    """A weighted multiclass metric must index support/weights by class label, not by enumeration position, under non-0-indexed labels."""
    from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf

    classes = [1, 2, 3]  # non-0-indexed integer multiclass labels
    # Skewed supports: 60 of class 1, 30 of class 2, 10 of class 3.
    targets = np.array([1] * 60 + [2] * 30 + [3] * 10)
    # Predict class 1 (position 0) for everyone -> class-1 recall=1.0, others 0.0.
    probs = np.zeros((100, 3))
    probs[:, 0] = 1.0
    preds = np.full(100, classes[0])  # predicted label = 1
    metrics: dict = {}

    report_probabilistic_model_perf(
        targets=targets,
        columns=["f"],
        model_name="m",
        model=None,
        classes=classes,
        preds=preds,
        probs=probs,
        metrics=metrics,
        print_report=False,
        show_perf_chart=False,
        verbose=False,
    )

    # Correct support-weighted recall = (1.0*60 + 0*30 + 0*10)/100 = 0.6.
    # Pre-fix the class-1 block (enumerate position 0) got support
    # count(targets==0)=0 -> dropped -> weighted_recall collapsed to 0.0.
    assert "weighted_recall" in metrics, f"weighted_recall absent; keys={sorted(metrics)}"
    assert abs(float(metrics["weighted_recall"]) - 0.6) < 0.05, f"weighted_recall={metrics['weighted_recall']} (correct=0.6, pre-fix buggy=0.0)"


def test_multiclass_ice_metric_auto_maps_non_0_indexed_integer_labels():
    """compute_probabilistic_multiclass_error with non-0-indexed integer labels and labels=None.

    The ICE scorer (training/_helpers_training_configs.integral_calibration_error) wraps this with labels=None, so a
    multiclass target labelled [1,2,3] reaches it raw. Pre-fix every per-class indicator was ``y_true == column_index``
    (column 0 vs label 0 -> empty), collapsing ICE to a no-skill ~+1.8 instead of the correct ~-0.17 for well-separated
    probs -- silently corrupting model selection AND the report TOTAL INTEGRAL ERROR. The auto-map now mirrors an
    explicit labels=[1,2,3] and the 0-indexed equivalent bit-for-bit.
    """
    from mlframe.metrics.core import compute_probabilistic_multiclass_error

    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(1, 4, size=n)  # labels {1,2,3}, NOT 0..K-1
    probs = np.zeros((n, 3))
    for i, c in enumerate(y):
        probs[i, c - 1] = 0.8
        others = [k for k in range(3) if k != c - 1]
        probs[i, others[0]] = 0.1
        probs[i, others[1]] = 0.1
    probs = np.clip(probs + rng.normal(0, 0.02, probs.shape), 1e-6, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)

    err_auto = compute_probabilistic_multiclass_error(y_true=y, y_score=probs)  # labels=None -> auto-map
    err_explicit = compute_probabilistic_multiclass_error(y_true=y, y_score=probs, labels=np.array([1, 2, 3]))
    err_0indexed = compute_probabilistic_multiclass_error(y_true=y - 1, y_score=probs)

    assert np.isclose(err_auto, err_explicit), f"auto-map ICE {err_auto} != explicit-labels ICE {err_explicit} (pre-fix auto ~+1.8)"
    assert np.isclose(err_auto, err_0indexed), f"auto-map ICE {err_auto} != 0-indexed ICE {err_0indexed}"
    assert err_auto < 0.0, f"well-separated multiclass ICE should be negative; got {err_auto} (pre-fix ~+1.8)"

    # return_per_class must also surface the corrected per-class vector keyed by column position.
    total, per_class = compute_probabilistic_multiclass_error(y_true=y, y_score=probs, return_per_class=True)
    assert set(per_class) == {0, 1, 2} and np.isclose(total, err_explicit)


def test_dummy_baselines_non_0_indexed_integer_multiclass_labels():
    """compute_dummy_baselines on integer multiclass labels {1,2,3} (NOT 0..K-1).

    Integer multiclass targets are not label-encoded upstream (only string/object are), so {1,2,3}
    reaches the dummy-baseline builders raw. Pre-fix two label-position assumptions silently broke it:
      * ``np.bincount(train_y, minlength=K)`` returns max(label)+1 == K+1 wide -> a phantom class-0
        column -> the prior/most_frequent/uniform prob matrices come out (N, K+1) not (N, K).
      * the metrics table calls ``log_loss(y, p, labels=np.arange(K))``; raw label 3 is absent from
        {0,1,2} so sklearn raises -> every classification metric becomes NaN -> the whole table fails.
    The dispatch now searchsorts the labels to positions 0..K-1, so the prob matrices are (N, K) and
    at least one baseline reports a finite log_loss.
    """
    from mlframe.training.baselines import compute_dummy_baselines
    from mlframe.training.configs import DummyBaselinesConfig

    rng = np.random.default_rng(0)
    n_tr, n_va, n_te = 300, 120, 120
    train_y = rng.integers(1, 4, size=n_tr)  # labels {1,2,3}
    val_y = rng.integers(1, 4, size=n_va)
    test_y = rng.integers(1, 4, size=n_te)
    import pandas as pd

    train_X = pd.DataFrame({"f": rng.normal(size=n_tr)})
    val_X = pd.DataFrame({"f": rng.normal(size=n_va)})
    test_X = pd.DataFrame({"f": rng.normal(size=n_te)})

    rep = compute_dummy_baselines(
        target_type="multiclass_classification",
        target_name="t",
        train_X=train_X,
        val_X=val_X,
        test_X=test_X,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
        config=DummyBaselinesConfig(),
    )
    assert rep.extras.get("n_classes") == 3
    # At least one baseline must have a finite val_log_loss: pre-fix every row was NaN
    # (log_loss labels=arange(3) rejects label 3) -> the table is all-failed.
    tbl = rep.table
    finite_ll = tbl["val_log_loss"].apply(lambda v: np.isfinite(v) if v is not None else False)
    assert bool(finite_ll.any()), f"no baseline produced a finite val_log_loss for non-0-indexed labels; table=\n{tbl}"
    # Width check via uniform: 1/K per class on K=3 columns.
    assert not tbl["failed"].all(), f"all baselines failed for {{1,2,3}} labels; table=\n{tbl}"


def test_classification_report_no_phantom_class_for_non_0_indexed_labels(caplog):
    """The printed classification_report table must carry exactly K rows (in label order) with the correct macro avg.

    Pre-fix the report inferred ``nclasses = max(label) + 1`` and called the njit table with raw labels, so labels
    [1,2,3] produced a 4-row table with a phantom 0-support class-0 row. That phantom row is averaged into ``macro avg``
    (the per-class mean), dragging it BELOW sklearn's: fast macro-precision 0.67 vs sklearn 0.89 on the fixture below.
    The remap to positions 0..K-1 against ``classes`` removes the phantom row and matches sklearn.
    """
    import logging

    from sklearn.metrics import classification_report as _skl_report
    from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf

    classes = [1, 2, 3]
    targets = np.array([1, 1, 2, 2, 3, 3] * 40)
    preds = np.array([1, 1, 2, 3, 3, 3] * 40)
    pos = {1: 0, 2: 1, 3: 2}
    probs = np.zeros((len(targets), 3))
    for i, p in enumerate(preds):
        probs[i, pos[p]] = 1.0

    logger_name = "mlframe.training.reporting._reporting_probabilistic"
    with caplog.at_level(logging.INFO, logger=logger_name):
        report_probabilistic_model_perf(
            targets=targets,
            columns=["f"],
            model_name="m",
            model=None,
            classes=classes,
            preds=preds,
            probs=probs,
            metrics={},
            print_report=True,
            show_perf_chart=False,
            report_ndigits=2,
        )
    report_block = next(r.getMessage() for r in caplog.records if "f1-score" in r.getMessage())
    table_lines = report_block.splitlines()

    # No phantom class-0 row (labels start at 1): only the header may contain the substring, so check row prefixes.
    assert not any(line.strip().startswith("0 ") for line in table_lines), f"phantom class-0 row present:\n{report_block}"
    # macro avg matches sklearn's (0.89 precision here), NOT the phantom-diluted 0.67.
    skl = _skl_report(targets, preds, zero_division=0, digits=2)
    skl_macro = next(l for l in skl.splitlines() if "macro avg" in l).split()
    our_macro = next(l for l in table_lines if "macro avg" in l).split()
    # last 4 numeric tokens are precision recall f1 support
    assert our_macro[-4:-1] == skl_macro[-4:-1], f"macro avg mismatch: ours={our_macro[-4:-1]} sklearn={skl_macro[-4:-1]}"
