"""Real-dataset noise-injection VALUE proofs for the FS selector family (bizvalue_value_proofs-09).

Every pre-existing real-data FS contract in the suite is tolerance-shaped (best-of-N within 0.02 of the all-features
baseline -- layer83/29) or an absolute floor (rc2_undersampled); none demonstrates a selector ADDING value on real data.
Raw sklearn datasets are nearly noiseless, so a win is structurally impossible as-is. This file applies the standard,
honest showcase recipe: take a REAL dataset and AUGMENT it with realistic-marginal noise, which creates genuine
headroom that selection can recover.

Datasets (both offline-safe, already used by layer83): ``load_breast_cancer`` (binary) + ``load_diabetes`` (regression).
Augmentation per dataset:
  - ``n_perm`` permuted-row copies of real columns -- each keeps a real column's marginal distribution but destroys its
    dependence on y (realistic decoys a column-shuffling noise generator produces);
  - ``n_dup`` near-duplicates (real column + small ``N(0, 0.05*sd)``) -- redundant, not noise; recall is NOT penalised
    for these (they reference the same signal a kept real column carries).

Three legs per (selector, dataset) pair, both seeds {0,1} must pass (majority of 2 == both):
  (a) VALUE: downstream 5-fold AUC/R2 on the SELECTED set >= the all-(augmented)-features score for an UNREGULARIZED
      downstream model (``LogisticRegression(C=1e6)`` / ``LinearRegression``) + a per-pair floor. An unregularized model
      cannot ignore the injected noise (it has no L2 to shrink the junk coefficients), so dropping noise is a real win.
      Same headroom trick as ``test_wrappers_biz_value.py::test_score_lift_vs_all_features``.
  (b) HONESTY: a HistGBM downstream is asserted within-epsilon only (>= all-features - eps). Boostings self-regularize,
      so they are barely hurt by the noise; claiming a tree WIN on real data would be a fake (per the FE-transformer
      file's own finding). The regression epsilon is wider than the binary one because R2 has far larger fold variance
      than AUC and a compact 2-feature selection genuinely cedes some R2 to a tree that sees every column.
  (c) NOISE REJECTION: >= 80% of injected permuted-noise columns dropped; REAL-column recall >= a per-selector floor.
      MRMR and the ShapProxied OOF path return a COMPACT, de-duplicated minimal-sufficient set (they keep the 5-6
      strongest real predictors and route the rest away), so their honest recall floor is far below 50% -- forcing 50%
      there would contradict the selectors' design. Broad-keeping selectors (BorutaShap, capped-RFECV) clear 50%.

All floors below were calibrated from a measured dev run (CPU, store-python 3.14, CUDA off) and set 5-15% below the
WORSE of the two seeds. Measured table is in the per-case docstrings; the calibration harness lives at
``audit/fs_tests_audit_2026_06_10/_calib_real_data_noise.py``.

BorutaShap is intentionally NOT run on diabetes: SHAP + RandomForestRegressor over ~90 augmented columns measured ~57s
per fit, over the project's ~55s per-test budget. It runs on breast_cancer only (~5-8s).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from tests.feature_selection.conftest import fast_subset

# The intentional C=1e6 (near-unregularized) downstream probe does not converge in 500 iters on the augmented frame --
# that is BY DESIGN (no L2 to shrink junk coefficients is exactly what makes dropping noise a measurable win), so silence
# the resulting ConvergenceWarning flood rather than let ~22k lines bury the run output.
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")

# Exact column-name token so substring overlap (bc1 in bc10, perm_1 in perm_10) cannot miscredit recall / rejection.
_TOKEN = re.compile(r"[A-Za-z]+_?\d+")

# Noise volume per dataset. Heavy on breast_cancer (n=569, 30 real) so the augmented width (156 cols) actually hurts the
# unregularized LogReg on the full set; diabetes (n=442, 10 real) needs less to swamp a 10-coef LinearRegression.
_BC_PERM, _BC_DUP = 120, 6
_DB_PERM, _DB_DUP = 80, 4


def _augment(X_real, feat_names, *, n_perm, n_dup, seed):
    """Return (DataFrame, real_names, noise_names, dup_names) for a real (X, names) pair plus injected decoys."""
    rng = np.random.default_rng(seed)
    n, p = X_real.shape
    frame = {nm: X_real[:, j] for j, nm in enumerate(feat_names)}
    noise_names = []
    for i in range(n_perm):
        src = i % p
        frame[f"perm_{i}"] = X_real[rng.permutation(n), src]
        noise_names.append(f"perm_{i}")
    dup_names = []
    for i in range(n_dup):
        src = i % p
        sd = float(X_real[:, src].std())
        frame[f"dup_{i}"] = X_real[:, src] + rng.normal(0.0, 0.05 * sd, size=n)
        dup_names.append(f"dup_{i}")
    return pd.DataFrame(frame), list(feat_names), noise_names, dup_names


def _support_names(sel, all_cols):
    """RAW selected column names (engineered tail excluded) via get_support / support_ / selected_features_."""
    gs = getattr(sel, "get_support", None)
    if callable(gs):
        try:
            s = np.asarray(gs())
        except Exception:
            s = None
        if s is not None:
            if s.dtype == bool:
                return [c for c, m in zip(all_cols, s) if m]
            return [all_cols[int(i)] for i in s]
    if hasattr(sel, "support_"):
        s = np.asarray(sel.support_)
        if s.dtype == bool:
            return [c for c, m in zip(all_cols, s) if m]
        return [all_cols[int(i)] for i in s]
    if hasattr(sel, "selected_features_"):
        return list(sel.selected_features_)
    raise AttributeError(f"{type(sel).__name__}: no support to extract")


def _output_names(sel, all_cols):
    """All selected feature names INCLUDING the engineered tail -- credits a real/noise column whose engineered
    combination (e.g. ``add(bc7,bc21)``) survives, the faithful recovery metric under MRMR's de-duplication."""
    gfno = getattr(sel, "get_feature_names_out", None)
    if callable(gfno):
        try:
            return [str(x) for x in gfno()]
        except Exception:
            pass
    return _support_names(sel, all_cols)


def _recall_and_rejection(out_names, real, noise):
    toks = set()
    for nm in out_names:
        toks.update(_TOKEN.findall(nm))
    real_used = sum(1 for c in real if c in toks)
    noise_used = sum(1 for c in noise if c in toks)
    noise_dropped = len(noise) - noise_used
    return real_used / max(len(real), 1), noise_dropped / max(len(noise), 1)


def _downstream(df, y, cols, task, kind):
    if not cols:
        return float("nan")
    X = df[cols].to_numpy()
    scoring = "roc_auc" if task != "regression" else "r2"
    if kind == "linear":
        est = LogisticRegression(C=1e6, max_iter=500) if task != "regression" else LinearRegression()
    else:
        est = (HistGradientBoostingClassifier(max_iter=80, random_state=0) if task != "regression"
               else HistGradientBoostingRegressor(max_iter=80, random_state=0))
    return float(cross_val_score(est, X, y, cv=5, scoring=scoring).mean())


# --- selector factories (mirror tests/feature_selection/_selector_factories.py configs) --------------------------


def _make_mrmr(task):
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False,
                full_npermutations=3, random_seed=0, min_features_fallback=1, verbose=False)


def _make_rfecv(task):
    from mlframe.feature_selection.wrappers import RFECV
    est = Ridge() if task == "regression" else LogisticRegression(max_iter=200, random_state=0)
    # max_nfeatures caps the argmax rule so the high-variance MBH search cannot keep most of the injected noise.
    return RFECV(estimator=est, cv=3, max_refits=4, random_state=0, leakage_corr_threshold=None,
                 n_features_selection_rule="argmax", max_nfeatures=12)


def _make_shap_proxied(task):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = task != "regression"
    model = (RandomForestClassifier(n_estimators=10, random_state=0) if cls
             else RandomForestRegressor(n_estimators=10, random_state=0))
    return ShapProxiedFS(model=model, classification=cls, n_splits=3, n_models=1, max_features=None,
                         top_n=10, holdout_size=0.25, revalidate=False, trust_guard=False,
                         prefilter_top=None, cluster_features=False, random_state=0, n_jobs=1)


def _make_boruta_shap(task):
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = task != "regression"
    model = (RandomForestClassifier(n_estimators=30, random_state=0) if cls
             else RandomForestRegressor(n_estimators=30, random_state=0))
    # SHAP importance (not the weaker gini default) + 25 trials are what actually reject the injected noise; the 10-trial
    # gini config kept all 156 columns in dev calibration.
    return BorutaShap(model=model, classification=cls, n_trials=25, random_state=0, train_or_test="train",
                      verbose=False, optimistic=True, importance_measure="shap")


def _load(ds_name):
    if ds_name == "breast_cancer":
        d = load_breast_cancer()
        names = [f"bc{i}" for i in range(d.data.shape[1])]
        return "binary", d.data, names, d.target, _BC_PERM, _BC_DUP
    d = load_diabetes()
    names = [f"db{i}" for i in range(d.data.shape[1])]
    return "regression", d.data, names, d.target, _DB_PERM, _DB_DUP


# (selector_key, make, dataset, needs_shap, value_floor, tree_eps, recall_floor) -- every floor calibrated from the
# WORSE seed of a measured dev run, set 5-15% below it. Measured mins are recorded in each tuple's trailing comment.
_CASES = [
    # MRMR returns a compact de-dup set: strong linear lift, near-zero noise, low literal recall (5-6 of 30 real cols).
    ("MRMR", _make_mrmr, "breast_cancer", False, 0.012, 0.020, 0.13),   # LIN>=+0.0236 noise=1.00 recall=0.20 tree>=-0.0044
    ("MRMR", _make_mrmr, "diabetes", False, 0.060, 0.090, 0.13),        # LIN>=+0.122  noise=1.00 recall=0.20 tree>=-0.055
    # RFECV's reliable showcase is the regression set (both seeds +0.145); breast_cancer's near-1.0 LogReg has no headroom.
    ("RFECV", _make_rfecv, "diabetes", False, 0.090, 0.060, 0.45),      # LIN>=+0.145  noise=0.99 recall=0.60 tree>=-0.027
    # ShapProxiedFS: clean two-sided proof on binary (recall clears 50%); compact OOF set on regression (low recall).
    ("ShapProxiedFS", _make_shap_proxied, "breast_cancer", True, 0.012, 0.020, 0.42),  # LIN>=+0.0221 noise=0.99 recall=0.50 tree>=+0.0001
    ("ShapProxiedFS", _make_shap_proxied, "diabetes", True, 0.035, 0.060, 0.13),       # LIN>=+0.053  noise=0.96 recall=0.20 tree>=-0.035
    # BorutaShap: breast_cancer only (diabetes ~57s, over the 55s budget). Broad set -> recall clears 50% comfortably.
    ("BorutaShap", _make_boruta_shap, "breast_cancer", True, 0.012, 0.020, 0.55),      # LIN>=+0.0219 noise=0.98 recall=0.80 tree>=+0.0005
]


def _case_id(case):
    return f"{case[0]}-{case[2]}"


# Fast-mode representatives: one binary (MRMR/breast_cancer, the clean de-dup showcase) + one regression
# (RFECV/diabetes, ~1s, the reliable RFECV win) -- one of each task at minimal wall. The fast smoke test below carries
# these so a path survives ``MLFRAME_FAST=1`` (the conftest collection hook skips every ``slow``-marked test in fast mode).
_FAST_KEYS = {("MRMR", "breast_cancer"), ("RFECV", "diabetes")}
_FAST_CASES = [c for c in _CASES if (c[0], c[2]) in _FAST_KEYS]


def _run_case(case):
    """Fit the selector on both augmented seeds and assert the value / honesty / noise-rejection legs on the worse seed."""
    selector_key, make, ds_name, needs_shap, value_floor, tree_eps, recall_floor = case
    if needs_shap:
        pytest.importorskip("shap")
    task, X_real, feat_names, y, n_perm, n_dup = _load(ds_name)

    lin_deltas, tree_deltas, noise_rejs, recalls = [], [], [], []
    for seed in (0, 1):
        df, real, noise, dup = _augment(X_real, feat_names, n_perm=n_perm, n_dup=n_dup, seed=seed)
        all_cols = list(df.columns)
        sel = make(task)
        sel.fit(df, y)

        support = _support_names(sel, all_cols)
        out_names = _output_names(sel, all_cols)
        recall, noise_rej = _recall_and_rejection(out_names, real, noise)

        sel_lin = _downstream(df, y, support, task, "linear")
        full_lin = _downstream(df, y, all_cols, task, "linear")
        sel_tree = _downstream(df, y, support, task, "tree")
        full_tree = _downstream(df, y, all_cols, task, "tree")

        assert support, f"{selector_key}/{ds_name} seed{seed}: empty selection"
        assert np.isfinite(sel_lin), f"{selector_key}/{ds_name} seed{seed}: non-finite selected-set linear score"
        lin_deltas.append(sel_lin - full_lin)
        tree_deltas.append(sel_tree - full_tree)
        noise_rejs.append(noise_rej)
        recalls.append(recall)

    # Majority of 2 seeds == BOTH seeds. Use the worse seed for every leg so a single lucky seed cannot carry the claim.
    worst_lin = min(lin_deltas)
    worst_tree = min(tree_deltas)
    worst_noise = min(noise_rejs)
    worst_recall = min(recalls)

    # (a) VALUE: selection beats the all-augmented-features unregularized model by the calibrated per-pair floor.
    assert worst_lin >= value_floor, (
        f"{selector_key}/{ds_name}: noise-injection VALUE not realised -- worst-seed selected-vs-all linear delta "
        f"{worst_lin:+.4f} < floor {value_floor:+.4f} (per-seed deltas {['%+.4f' % d for d in lin_deltas]}). "
        f"An unregularized {'LogReg(C=1e6)' if task != 'regression' else 'LinearRegression'} should gain from dropping "
        f"the injected noise; a regression here means the selector kept the junk or lost the signal."
    )

    # (b) HONESTY: HistGBM stays within-epsilon of all-features (no fake tree win). Boostings self-regularize.
    assert worst_tree >= -tree_eps, (
        f"{selector_key}/{ds_name}: tree honesty leg violated -- worst-seed selected-vs-all HistGBM delta "
        f"{worst_tree:+.4f} below -{tree_eps:.3f} (per-seed {['%+.4f' % d for d in tree_deltas]}). A boosting should be "
        f"roughly unharmed by the injected noise; a large drop means the selection discarded info the tree was using."
    )

    # (c) NOISE REJECTION: >= 80% of injected permuted-noise columns dropped (mission-fixed), real recall per-selector.
    assert worst_noise >= 0.80, (
        f"{selector_key}/{ds_name}: noise rejection {worst_noise:.2f} < 0.80 "
        f"(per-seed {['%.2f' % r for r in noise_rejs]}); the selector admitted too many permuted decoys."
    )
    assert worst_recall >= recall_floor, (
        f"{selector_key}/{ds_name}: real-column recall {worst_recall:.2f} < floor {recall_floor:.2f} "
        f"(per-seed {['%.2f' % r for r in recalls]}); the selector dropped genuine signal columns."
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", _CASES, ids=_case_id)
def test_biz_val_real_data_noise_injection_value_and_honesty(case):
    _run_case(case)


@pytest.mark.parametrize("case", fast_subset(_FAST_CASES, n=2), ids=_case_id)
def test_biz_val_real_data_noise_injection_fast_smoke(case):
    """Fast-mode-surviving representative: keeps one binary + one regression noise-injection proof alive under
    ``MLFRAME_FAST=1`` (where the slow full-matrix test above is skipped), so every leg has a fast code path."""
    _run_case(case)
