"""Pre-flight "will-it-shine?" diagnostics for ShapProxiedFS (Batch C).

Before spending the full SHAP + search + re-validation budget, cheap dataset statistics predict
whether the SHAP-coalition proxy is trustworthy on THIS data and where it sits vs alternatives:

  - full-model fit quality (depth-4 booster CV score vs a trivial baseline): the proxy can only be as
    good as the model it explains -- a model that can't learn the target yields garbage attributions.
  - additive-vs-deep ratio (depth-1 'stumps' score / depth-4 score): low ratio => interactions
    dominate (XOR-like), where the plain main-effect proxy struggles -> recommend ``interaction_aware``.
  - redundancy (max |correlation| among a feature sample): high => recommend ``cluster_features``.
  - width (n_features) and class balance: route to clustering / the AUC objective.

Returns a recommendation in {"run", "caution", "fallback"} with human-readable reasons. Cheap by
design (subsamples rows, 3-fold CV of two small boosters); the full proxy-fidelity Spearman is the
trust guard measured during ``fit``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _cv_score(estimator, X, y, classification):
    from sklearn.model_selection import cross_val_score

    scoring = "roc_auc" if classification else "r2"
    try:
        return float(np.mean(cross_val_score(estimator, X, y, cv=3, scoring=scoring)))
    except Exception:
        return float("nan")


def dataset_diagnostics(X, y, *, classification, max_rows=5000, max_corr_features=400, random_state=0):
    import pandas as pd
    from xgboost import XGBClassifier, XGBRegressor

    rng = np.random.default_rng(random_state)
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y = np.asarray(y)
    n, f = X.shape
    if n > max_rows:  # subsample rows for the cheap probes
        sel = rng.choice(n, size=max_rows, replace=False)
        Xs, ys = X.iloc[sel], y[sel]
    else:
        Xs, ys = X, y

    balance = float(np.mean(y)) if classification else float("nan")

    # Redundancy: max |corr| over a random feature sample (cap for wide data).
    cols = np.arange(f)
    if f > max_corr_features:
        cols = rng.choice(f, size=max_corr_features, replace=False)
    Xc = np.nan_to_num(Xs.iloc[:, cols].to_numpy(dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc, rowvar=False)
    np.fill_diagonal(C, 0.0)
    max_abs_corr = float(np.nanmax(np.abs(C))) if C.size else 0.0

    common = dict(n_estimators=150, learning_rate=0.1, n_jobs=-1, random_state=random_state, tree_method="hist")
    if classification:
        deep = XGBClassifier(max_depth=4, eval_metric="logloss", **common)
        stump = XGBClassifier(max_depth=1, eval_metric="logloss", **common)
        trivial = max(balance, 1 - balance)  # accuracy-free AUC baseline is 0.5
        base = 0.5
    else:
        deep = XGBRegressor(max_depth=4, **common)
        stump = XGBRegressor(max_depth=1, **common)
        base = 0.0  # r2 of the mean predictor
    deep_score = _cv_score(deep, Xs, ys, classification)
    stump_score = _cv_score(stump, Xs, ys, classification)

    # additive-vs-deep ratio in (improvement over trivial) space; ~1 => additive, <<1 => interactions.
    num = stump_score - base
    den = deep_score - base
    additive_ratio = float(np.clip(num / den, 0.0, 1.5)) if (np.isfinite(den) and den > 1e-6) else float("nan")

    return dict(n_features=int(f), n_samples=int(n), n_over_p=float(n / max(f, 1)),
                class_balance=balance, max_abs_corr=max_abs_corr,
                full_model_fit=deep_score, stump_fit=stump_score, additive_ratio=additive_ratio,
                base_score=base)


def preflight(
    X, y, *, classification, cluster_auto_threshold=40, redundancy_threshold=0.7,
    additive_ratio_floor=0.6, min_fit_gain=0.03, imbalance_floor=0.05, random_state=0,
):
    """Cheap recommendation on whether / how to run ShapProxiedFS. Returns a dict with
    ``recommendation`` in {run, caution, fallback}, the diagnostics, and the reasons."""
    d = dataset_diagnostics(X, y, classification=classification, random_state=random_state)
    reasons, suggestions = [], []
    rec = "run"

    if not np.isfinite(d["full_model_fit"]) or (d["full_model_fit"] - d["base_score"]) < min_fit_gain:
        rec = "fallback"
        reasons.append(f"full-model fit barely beats trivial ({d['full_model_fit']:.3f}); the proxy can "
                       f"only be as good as the model it explains -> prefer a different selector.")
    if np.isfinite(d["additive_ratio"]) and d["additive_ratio"] < additive_ratio_floor:
        reasons.append(f"interaction-heavy (additive/deep ratio {d['additive_ratio']:.2f} < "
                       f"{additive_ratio_floor}); the main-effect proxy will struggle.")
        suggestions.append("enable interaction_aware=True")
        if rec != "fallback":
            rec = "caution"
    if d["max_abs_corr"] >= redundancy_threshold:
        reasons.append(f"high feature redundancy (max|corr|={d['max_abs_corr']:.2f}).")
        suggestions.append("enable cluster_features=True")
    if d["n_features"] > cluster_auto_threshold:
        reasons.append(f"{d['n_features']} features exceed the exhaustive budget.")
        suggestions.append("cluster_features + pre-screen (auto)")
    if classification and np.isfinite(d["class_balance"]) and min(d["class_balance"], 1 - d["class_balance"]) < imbalance_floor:
        reasons.append(f"imbalanced target (pos rate {d['class_balance']:.3f}).")
        suggestions.append("use metric='auc'")
        if rec == "run":
            rec = "caution"
    if rec == "run" and not reasons:
        reasons.append(f"additive ratio {d['additive_ratio']:.2f}, fit {d['full_model_fit']:.3f}, "
                       f"max|corr| {d['max_abs_corr']:.2f} -- favourable regime.")

    return dict(recommendation=rec, diagnostics=d, reasons=reasons, suggestions=sorted(set(suggestions)))
