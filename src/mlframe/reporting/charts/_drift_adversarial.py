"""Adversarial-validation drift detection: can a classifier tell the two frames apart?

Carved out of ``drift.py``, which had grown to 11 lines short of the house 1000-LOC backstop. The three drift
families in that module answer the same question by different means and share almost nothing: PSI compares
marginal distributions bin by bin, the residual/CUSUM panels look for drift over TIME within one frame, and this
one fits an actual model to separate frame A from frame B. Only the frame-column plumbing is common, and it stays
in the parent.

An adversarial AUC near 0.5 means the frames are exchangeable; well above it means a model trained on A is being
evaluated on a different population, and the per-feature importances name which columns give it away.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.spec import AnnotationPanelSpec, BarPanelSpec, FigureSpec, LinePanelSpec

from ._drift_shared import (
    ADV_MAX_ROWS_PER_SIDE, ADV_N_ESTIMATORS, ADV_TOP_FEATURES, MIN_ADV_ROWS_PER_SIDE,
    _adversarial_auc_bar, _frame_columns, _frame_rows,
)

def _subsample_rows(n: int, cap: int, seed: int) -> np.ndarray:
    """Return a sorted random index subsample of size ``min(n, cap)`` (sorted so downstream row-order-sensitive ops stay stable); the full index range if ``n`` is already within ``cap``."""
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))


def adversarial_auc(
    feature_frame_a: Any,
    feature_frame_b: Any,
    *,
    feature_names: Optional[Sequence[str]] = None,
    max_rows_per_side: int = ADV_MAX_ROWS_PER_SIDE,
    n_splits: int = 3,
    seed: int = 0,
    lgbm_params: Optional[dict] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Train a LightGBM classifier to separate side-A (label 0) from side-B (label 1) on a shuffled union.

    Returns ``(auc, fpr, tpr, importances, names)`` where ``auc`` is the cross-validated out-of-fold ROC AUC
    (the honest "can a model tell the two sets apart" estimate -- in-sample AUC overstates separability), ``fpr/tpr``
    are the OOF ROC-curve points, and ``importances`` are the model's gain importances aligned to ``names``. Each side
    is subsampled to ``max_rows_per_side`` first so a 1M-row union stays cheap. AUC ~0.5 => same distribution;
    AUC >> 0.5 => the sets are distinguishable (CV will not transfer / covariate shift present).
    """
    import lightgbm as lgb
    import pandas as pd
    from mlframe.metrics.core import fast_roc_auc, fast_roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cols_a, names_a = _frame_columns(feature_frame_a, feature_names)
    cols_b, names_b = _frame_columns(feature_frame_b, feature_names)
    if names_a != names_b:
        raise ValueError("adversarial_auc: the two sides must share the same feature columns")
    names = tuple(names_a)

    na = cols_a[0].shape[0] if cols_a else 0
    nb = cols_b[0].shape[0] if cols_b else 0
    ia = _subsample_rows(na, max_rows_per_side, seed)
    ib = _subsample_rows(nb, max_rows_per_side, seed + 1)

    def _encode_pair(ca, cb):
        """Return (a, b) as 1-D float64 arrays, or ``None`` to skip a non-scalar column. Numeric columns pass through;
        string / categorical / object columns are label-encoded against the A+B union so the same category maps to the
        same code on both sides (categorical drift -- a level present only in one side -- is a real adversarial signal,
        not a reason to crash). NaN / None map to the -1 sentinel, which LightGBM treats as missing. Non-scalar columns
        (embedding ``List(...)`` columns materialised as object arrays of Python lists / ndarrays) are skipped: they
        have no single scalar value to feed the separating classifier."""
        a = np.asarray(ca)
        b = np.asarray(cb)
        if a.ndim > 1 or b.ndim > 1:
            return None  # 2-D (fixed-width embedding) -> not a scalar drift feature
        if a.dtype == object and len(a) and isinstance(a.flat[0], (list, tuple, np.ndarray, dict)):
            return None  # object column of per-row sequences (ragged embedding / nested)
        try:
            return a.astype(np.float64), b.astype(np.float64)
        except (ValueError, TypeError):
            sa = pd.Series(a).astype("string")
            sb = pd.Series(b).astype("string")
            codes, _ = pd.factorize(pd.concat([sa, sb], ignore_index=True), use_na_sentinel=True)
            return codes[: len(a)].astype(np.float64), codes[len(a) :].astype(np.float64)

    _enc, _kept_names = [], []
    for j, nm in enumerate(names):
        e = _encode_pair(cols_a[j], cols_b[j])
        if e is not None:
            _enc.append(e)
            _kept_names.append(nm)
    names = tuple(_kept_names)
    Xa = np.column_stack([e[0][ia] for e in _enc]) if _enc else np.empty((len(ia), 0))
    Xb = np.column_stack([e[1][ib] for e in _enc]) if _enc else np.empty((len(ib), 0))
    X = np.vstack([Xa, Xb])
    y = np.concatenate([np.zeros(len(ia), dtype=np.int64), np.ones(len(ib), dtype=np.int64)])

    # n_jobs=1, not -1: this diagnostic classifier is cheap (small subsampled X/y, serial cross_val_predict
    # with no outer parallelism to feed) and never benefited from claiming every core -- under any
    # concurrent-worker environment (CI xdist shards, several dev-box sessions at once) an unbounded LightGBM
    # thread pool here causes severe CPU oversubscription, which can block the native LGBM_BoosterUpdateOneIter
    # call indefinitely. pytest-timeout's thread-based method can't preempt a blocked native call -- it marks
    # the test "timed out" for reporting but the OS thread keeps running, hanging the worker until an external
    # job-level cap kills it (see ci.yml's own "native-call hang" hypothesis; this pins the exact call site).
    params: dict = dict(n_estimators=ADV_N_ESTIMATORS, num_leaves=31, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, n_jobs=1, random_state=seed, verbosity=-1, importance_type="gain")
    if lgbm_params:
        params.update(lgbm_params)
    clf = lgb.LGBMClassifier(**params)

    # Need at least 2 of each class per fold; clamp n_splits to the minority count so a tiny synthetic still runs.
    k = max(2, min(int(n_splits), int(min(len(ia), len(ib)))))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(fast_roc_auc(y, oof))
    fpr, tpr, _ = fast_roc_curve(y, oof)

    # Importances come from a single full-data fit (the per-fold models are discarded by cross_val_predict); a full fit
    # gives the most stable ranking of which features carry the separating signal.
    clf.fit(X, y)
    importances = np.asarray(clf.feature_importances_, dtype=np.float64)
    return auc, fpr, tpr, importances, names


def adversarial_validation(
    train_frame: Any,
    test_frame: Any,
    *,
    val_frame: Any = None,
    feature_names: Optional[Sequence[str]] = None,
    max_rows_per_side: int = ADV_MAX_ROWS_PER_SIDE,
    top_features: int = ADV_TOP_FEATURES,
    n_splits: int = 3,
    seed: int = 0,
    lgbm_params: Optional[dict] = None,
    figsize: Tuple[float, float] = (12.0, 5.0),
) -> FigureSpec:
    """Adversarial-validation panel: "will my CV transfer?".

    Trains a LightGBM classifier to separate train (label 0) from test (label 1) -- and, when ``val_frame`` is given,
    train-vs-val too -- on a shuffled union, reports the out-of-fold ROC + AUC, and ranks the top-``top_features``
    drifting features by classifier importance. AUC ~0.5 means train and test are indistinguishable (CV estimates
    transfer); AUC well above ~0.6-0.7 means the sets differ and the top-importance features are the drift drivers.

    Returns a 2-panel FigureSpec: left a ROC LinePanelSpec (train-vs-test, plus train-vs-val when supplied, + the
    chance diagonal, AUCs in the title), right a BarPanelSpec of the top drifting features (train-vs-test importances).
    """
    # Stratified CV needs >= MIN_ADV_ROWS_PER_SIDE rows per side and >= 1 feature column; an empty / tiny side makes
    # cross_val_predict raise on a 0-sample fold. Surface an honest placeholder instead of crashing the report.
    cols_a, _ = _frame_columns(train_frame, feature_names)
    cols_b, _ = _frame_columns(test_frame, feature_names)
    na = cols_a[0].shape[0] if cols_a else 0
    nb = cols_b[0].shape[0] if cols_b else 0
    if not cols_a or not cols_b or min(na, nb) < MIN_ADV_ROWS_PER_SIDE:
        ann = AnnotationPanelSpec(
            text=f"Adversarial validation skipped: needs >= {MIN_ADV_ROWS_PER_SIDE} rows/side and >= 1 feature "
            f"(got train={na}, test={nb}, n_features={len(cols_a)})",
            title="Adversarial validation",
        )
        return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)

    auc_tt, fpr_tt, tpr_tt, imp_tt, names = adversarial_auc(
        train_frame, test_frame, feature_names=feature_names,
        max_rows_per_side=max_rows_per_side, n_splits=n_splits, seed=seed, lgbm_params=lgbm_params,
    )

    series_x = [fpr_tt, np.array([0.0, 1.0])]
    series_y = [tpr_tt, np.array([0.0, 1.0])]
    labels = [f"train-vs-test (AUC={auc_tt:.3f})", "chance"]
    styles = ["-", "--"]
    colors = ["crimson", "gray"]
    title_bits = [f"train-vs-test AUC={auc_tt:.3f}"]
    auc_tv: Optional[float] = None  # stays None when no val frame was supplied; the verdict below reads it

    if val_frame is not None:
        auc_tv, fpr_tv, tpr_tv, _, _ = adversarial_auc(
            train_frame, val_frame, feature_names=feature_names,
            max_rows_per_side=max_rows_per_side, n_splits=n_splits, seed=seed + 100, lgbm_params=lgbm_params,
        )
        series_x.insert(1, fpr_tv)
        series_y.insert(1, tpr_tv)
        labels.insert(1, f"train-vs-val (AUC={auc_tv:.3f})")
        styles.insert(1, "-")
        colors.insert(1, "steelblue")
        title_bits.append(f"train-vs-val AUC={auc_tv:.3f}")

    # Each ROC curve has its own fpr grid (different per train-vs-test / train-vs-val pair); LinePanelSpec carries a
    # tuple of per-series x arrays so every curve keeps its native vertices instead of being resampled onto a shared grid.
    series_x = [np.asarray(fx, dtype=np.float64) for fx in series_x]
    # A fixed AUC >= 0.6 bar ignores how many rows produced the AUC. Under the null (identical distributions) the
    # AUC's standard error is sqrt((n_a + n_b + 1) / (12 * n_a * n_b)), so on a few hundred rows per side an AUC of
    # 0.60 is ordinary noise while on 200k rows per side 0.52 is a real, reproducible shift.
    _n_a = min(_frame_rows(train_frame), max_rows_per_side)
    _n_b = min(_frame_rows(test_frame), max_rows_per_side)
    adv_bar = 0.5 + _adversarial_auc_bar(_n_a, _n_b)
    _pairs = [("train-vs-test", float(auc_tt))]
    if auc_tv is not None and np.isfinite(auc_tv):
        _pairs.append(("train-vs-val", float(auc_tv)))
    _worst_name, _worst_auc = max(_pairs, key=lambda kv: kv[1])
    verdict = (
        f"worst pair {_worst_name}: AUC {_worst_auc:.3f} > {adv_bar:.3f} (the no-shift noise bar at {_n_a:,} vs "
        f"{_n_b:,} rows/side) => shift, CV may NOT transfer"
        if _worst_auc > adv_bar
        else f"worst pair {_worst_name}: AUC {_worst_auc:.3f} within the no-shift noise bar of {adv_bar:.3f} at "
        f"{_n_a:,} vs {_n_b:,} rows/side => indistinguishable, CV transfers"
    )
    roc = LinePanelSpec(
        x=tuple(series_x),
        y=tuple(series_y),
        series_labels=tuple(labels),
        line_styles=tuple(styles),
        colors=tuple(colors),
        title="Adversarial validation: " + "; ".join(title_bits) + f"\n({verdict})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    order = np.argsort(imp_tt)[::-1][: max(1, min(top_features, imp_tt.size))]
    bar = BarPanelSpec(
        categories=tuple(names[i] for i in order),
        values=imp_tt[order],
        title=f"Top {len(order)} drifting features (train-vs-test gain importance)",
        xlabel="feature",
        ylabel="LightGBM gain importance",
        colors=("crimson",),
        xtick_rotation=60.0,
    )
    caption = (
        "A classifier is trained to tell TRAIN rows from TEST rows. If the two sets share a distribution it cannot, "
        "and the out-of-fold ROC hugs the diagonal at AUC 0.5. An AUC well above 0.5 means they genuinely differ, "
        "and the bars rank the features carrying that difference by gain importance -- those are the drift drivers "
        f"to investigate first. Measured AUC = {auc_tt:.3f} on out-of-fold predictions over {n_splits} folds. Read "
        "the AUC as a distance from 0.5, not as a model-quality score: higher is WORSE here."
    )
    return FigureSpec(suptitle="", panels=((roc, bar),), figsize=figsize, caption=caption)


__all__ = ["adversarial_auc", "adversarial_validation"]
