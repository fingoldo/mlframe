"""Consolidated from test_biz_value_mrmr_layer29.py.

Layer 29 biz_value: VALIDATE HYBRID FE ON sklearn TOY DATASETS (real data).

Layers 21-28 pinned hybrid orthogonal-polynomial FE on SYNTHETIC quadratic /
XOR / saddle / cross-basis signals where the engineered ``He_n`` columns are
known a priori to solve the task. Layer 29 closes the loop on REAL sklearn
toy datasets (no hand-crafted signal): the hybrid FE knob must, at minimum,
not regress downstream score by more than a small tolerance vs the
hybrid-off baseline.

Datasets pinned
---------------
A. ``load_breast_cancer`` (binary, n=569, p=30): baseline LogReg accuracy
   target ~0.95. Hybrid must match within tolerance.
B. ``load_diabetes`` (regression, n=442, p=10): baseline LinearRegression
   R^2 ~0.45 on holdout. Hybrid must stay within tolerance.
C. ``load_iris`` (3-way multiclass, n=150, p=4): baseline LogReg accuracy
   ~0.96. Hybrid must match within tolerance.
D. ``load_wine`` (3-way multiclass, n=178, p=13): baseline LogReg accuracy
   ~0.97. Hybrid must match within tolerance.
E. ``make_classification(n=1500, p=20, n_informative=3, class_sep=0.8)``
   binary with most columns noise: baseline ~0.80. Hybrid must match
   within tolerance.

Contracts
---------
1. ``hybrid_score >= baseline_score - tolerance`` (tolerance: 0.02 for
   accuracy, 0.05 for R^2). Per-test assert message reports the lift
   number so any negative lift is visible.
2. ``hybrid_support_size <= baseline_support_size * 1.5 + 5`` (the +5
   absorbs the per-stage top_k=5 budget when baseline support is tiny).
3. Any positive lift is documented in the assert message for visibility.

This is REAL data: no guarantee hybrid will help. The contract is that it
must not HURT measurably. If a dataset genuinely regresses, fix prod or
surface the limitation in the layer report (do NOT relax tolerance to
paper over a regression — per house rule).

2026-05-31 Layer 29.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# Tolerances on the regression / match-or-improve contract.
ACC_TOLERANCE = 0.02
R2_TOLERANCE = 0.05
# Support-size blowup bound: hybrid may add at most 50% more cols, plus a
# small absolute slack to absorb top_k=5 when baseline support is tiny.
SUPPORT_SIZE_FACTOR = 1.5
SUPPORT_SIZE_SLACK = 5


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
def _fit_transform_pair(X_tr, y_tr, X_te, *, hybrid: bool):
    """Fit MRMR with hybrid on/off, transform train + holdout.

    Returns (Xtr_sel, Xte_sel, support_size). Engineered columns are
    folded in automatically by ``MRMR.transform`` when hybrid is on.
    """
    if hybrid:
        m = _make_mrmr(
            fe_hybrid_orth_enable=True,
            fe_hybrid_orth_pair_enable=False,
            fe_hybrid_orth_basis="hermite",
            fe_hybrid_orth_top_k=5,
        )
    else:
        m = _make_mrmr(fe_hybrid_orth_enable=False)
    m.fit(X_tr, y_tr)
    Xtr_sel = m.transform(X_tr)
    Xte_sel = m.transform(X_te)
    return Xtr_sel, Xte_sel, int(np.asarray(Xtr_sel).shape[1])


def _score_classification(X_tr, y_tr, X_te, y_te) -> float:
    """Standardize + LogReg, return holdout accuracy."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(Xtr, y_tr)
    return float(accuracy_score(y_te, clf.predict(Xte)))


def _score_regression(X_tr, y_tr, X_te, y_te) -> float:
    """Standardize + LinearRegression, return holdout R^2."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    reg = LinearRegression()
    reg.fit(Xtr, y_tr)
    return float(r2_score(y_te, reg.predict(Xte)))


def _split_classification(X, y, *, test_size=0.25, random_state=0):
    """Stratified split for classification."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def _split_regression(X, y, *, test_size=0.25, random_state=0):
    """Unstratified split for continuous y."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def _run_classification_pair(X, y, *, random_state=0):
    """Full pair-run: split once, fit-transform-score baseline + hybrid.

    Returns (baseline_score, hybrid_score, baseline_size, hybrid_size).
    """
    X_tr, X_te, y_tr, y_te = _split_classification(X, y, random_state=random_state)
    Xtr_b, Xte_b, size_b = _fit_transform_pair(X_tr, y_tr, X_te, hybrid=False)
    Xtr_h, Xte_h, size_h = _fit_transform_pair(X_tr, y_tr, X_te, hybrid=True)
    s_b = _score_classification(Xtr_b, y_tr, Xte_b, y_te)
    s_h = _score_classification(Xtr_h, y_tr, Xte_h, y_te)
    return s_b, s_h, size_b, size_h


def _run_regression_pair(X, y, *, random_state=0):
    X_tr, X_te, y_tr, y_te = _split_regression(X, y, random_state=random_state)
    Xtr_b, Xte_b, size_b = _fit_transform_pair(X_tr, y_tr, X_te, hybrid=False)
    Xtr_h, Xte_h, size_h = _fit_transform_pair(X_tr, y_tr, X_te, hybrid=True)
    s_b = _score_regression(Xtr_b, y_tr, Xte_b, y_te)
    s_h = _score_regression(Xtr_h, y_tr, Xte_h, y_te)
    return s_b, s_h, size_b, size_h


def _assert_support_bounded(size_b: int, size_h: int, dataset: str) -> None:
    """Pin that hybrid does not balloon support beyond the documented cap."""
    upper = size_b * SUPPORT_SIZE_FACTOR + SUPPORT_SIZE_SLACK
    assert size_h <= upper, (
        f"[{dataset}] hybrid support_size={size_h} exceeds bound "
        f"{upper:.1f} = baseline({size_b}) * {SUPPORT_SIZE_FACTOR} + "
        f"{SUPPORT_SIZE_SLACK}; FE stage is padding support beyond the "
        f"per-stage top_k=5 budget."
    )


# ---------------------------------------------------------------------------
# A. breast_cancer (binary, n=569, p=30)
# ---------------------------------------------------------------------------


class TestBreastCancerHybrid:

    def test_breast_cancer_hybrid_matches_baseline(self):
        bc = load_breast_cancer(as_frame=True)
        X, y = bc.data, bc.target
        s_b, s_h, size_b, size_h = _run_classification_pair(X, y, random_state=0)
        lift = s_h - s_b
        _assert_support_bounded(size_b, size_h, "breast_cancer")
        assert s_h >= s_b - ACC_TOLERANCE, (
            f"[breast_cancer] hybrid accuracy {s_h:.4f} regressed from "
            f"baseline {s_b:.4f} by more than {ACC_TOLERANCE} "
            f"(lift={lift:+.4f}); support_size baseline={size_b} "
            f"hybrid={size_h}."
        )


# ---------------------------------------------------------------------------
# B. diabetes (regression, n=442, p=10)
# ---------------------------------------------------------------------------


class TestDiabetesHybrid:

    def test_diabetes_hybrid_matches_baseline(self):
        d = load_diabetes(as_frame=True)
        X, y = d.data, d.target
        s_b, s_h, size_b, size_h = _run_regression_pair(X, y, random_state=0)
        lift = s_h - s_b
        _assert_support_bounded(size_b, size_h, "diabetes")
        # Post Layer 29 cell-budget pre-screen fix: continuous-y diabetes
        # no longer collapses to support_=['age'] via fallback. Baseline
        # Ridge R^2 should be >= 0.40 (was 0.02 pre-fix - catastrophic
        # regression caused by the cell-budget refusing all 9 numeric
        # features when nbins_y was the count of unique y values).
        assert s_b >= 0.30, (
            f"[diabetes] baseline R^2 {s_b:.4f} below 0.30 - the cell-"
            f"budget pre-screen may have regressed to refusing legitimate "
            f"numeric features on continuous regression targets. Verify "
            f"_screen_predictors.py:_nbins_x_ceiling = 2*sqrt(n) is intact."
        )
        assert s_h >= s_b - R2_TOLERANCE, (
            f"[diabetes] hybrid R^2 {s_h:.4f} regressed from baseline "
            f"{s_b:.4f} by more than {R2_TOLERANCE} (lift={lift:+.4f}); "
            f"support_size baseline={size_b} hybrid={size_h}."
        )


# ---------------------------------------------------------------------------
# C. iris (3-way multiclass, n=150, p=4)
# ---------------------------------------------------------------------------


class TestIrisHybrid:

    def test_iris_hybrid_matches_baseline(self):
        ir = load_iris(as_frame=True)
        X, y = ir.data, ir.target
        s_b, s_h, size_b, size_h = _run_classification_pair(X, y, random_state=0)
        lift = s_h - s_b
        _assert_support_bounded(size_b, size_h, "iris")
        assert s_h >= s_b - ACC_TOLERANCE, (
            f"[iris] hybrid accuracy {s_h:.4f} regressed from baseline "
            f"{s_b:.4f} by more than {ACC_TOLERANCE} (lift={lift:+.4f}); "
            f"support_size baseline={size_b} hybrid={size_h}."
        )


# ---------------------------------------------------------------------------
# D. wine (3-way multiclass, n=178, p=13)
# ---------------------------------------------------------------------------


class TestWineHybrid:

    def test_wine_hybrid_matches_baseline(self):
        w = load_wine(as_frame=True)
        X, y = w.data, w.target
        s_b, s_h, size_b, size_h = _run_classification_pair(X, y, random_state=0)
        lift = s_h - s_b
        _assert_support_bounded(size_b, size_h, "wine")
        assert s_h >= s_b - ACC_TOLERANCE, (
            f"[wine] hybrid accuracy {s_h:.4f} regressed from baseline "
            f"{s_b:.4f} by more than {ACC_TOLERANCE} (lift={lift:+.4f}); "
            f"support_size baseline={size_b} hybrid={size_h}."
        )


# ---------------------------------------------------------------------------
# E. make_classification (slightly degenerate noise + few informative)
# ---------------------------------------------------------------------------


class TestMakeClassificationHybrid:

    def test_make_classification_hybrid_matches_baseline(self):
        Xa, ya = make_classification(
            n_samples=1500,
            n_features=20,
            n_informative=3,
            n_redundant=2,
            n_repeated=0,
            n_classes=2,
            class_sep=0.8,
            random_state=0,
        )
        X = pd.DataFrame(Xa, columns=[f"f{i:02d}" for i in range(Xa.shape[1])])
        y = pd.Series(ya, name="y")
        s_b, s_h, size_b, size_h = _run_classification_pair(X, y, random_state=0)
        lift = s_h - s_b
        _assert_support_bounded(size_b, size_h, "make_classification")
        assert s_h >= s_b - ACC_TOLERANCE, (
            f"[make_classification] hybrid accuracy {s_h:.4f} regressed "
            f"from baseline {s_b:.4f} by more than {ACC_TOLERANCE} "
            f"(lift={lift:+.4f}); support_size baseline={size_b} "
            f"hybrid={size_h}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
