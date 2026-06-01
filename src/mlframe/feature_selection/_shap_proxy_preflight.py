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
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


def _cv_score(estimator, X, y, classification):
    scoring = "roc_auc" if classification else "r2"
    try:
        return float(np.mean(cross_val_score(estimator, X, y, cv=3, scoring=scoring)))
    except Exception:
        return float("nan")


def dataset_diagnostics(X, y, *, classification, max_rows=2000, max_rows_corr=5000,
                        max_corr_features=400, n_estimators=100, random_state=0,
                        inner_n_jobs_cap=False):
    """Cheap statistics that gate ShapProxiedFS's full pipeline. See module docstring.

    ``max_rows`` (iter25, lowered 5000 -> 2000): row subsample for the cheap booster probes. The
    additive-vs-deep ratio is a RANK-only signal (deep vs stump on the same data) and stabilises at
    far fewer rows than a typical SHAP / refine fit needs. Live-test measurement (2026-05-28,
    width=1000 / n_rows=5000) showed preflight=77s vs fit=86s -- the gate cost the same as the thing
    it gated. Halving rows + capping trees (see ``n_estimators``) restores the gate's cheap-check
    purpose without flipping the recommendation on the iter17 5-regime calibration set.

    ``max_rows_corr`` (iter25, kept at 5000): row subsample for the ``max_abs_corr`` redundancy probe
    -- DECOUPLED from ``max_rows`` because the correlation estimate's variance falls as ``1/sqrt(n)``
    and a 2000-row sample shrinks the sampling-variance ceiling on ``max|corr|`` enough that the 0.7
    redundancy gate trips inconsistently between full and subsampled views on the same data. The
    correlation pass is cheap (one O(n * m^2) numpy call against ``max_corr_features <= 400`` columns)
    so capping rows here gives no measurable savings -- we only sub-sample to bound RAM on very large
    inputs. ``n_estimators`` is what actually pays for the speed.

    ``n_estimators`` (iter25, default 100; was 150): RANKING-only cap for the depth-4 + depth-1
    boosters whose ``cross_val_score`` ratio drives ``additive_ratio``. Same cap-the-ranker pattern as
    iter9/iter10/iter19 -- we read RANKS / RATIOS, not deployed predictions, so a smaller tree count
    delivers the same signal at ~1/3 the cost. ``full_model_fit`` may drop a few thousandths, but the
    additive/deep RATIO and the gate-trip points around it are stable across all 5 iter17 regimes
    (additive_highSNR / redundancy_heavy / interaction_heavy / xor_interaction / noise_heavy).

    iter26: the two booster CV calls (deep + stump) over the SAME (Xs, ys) are parallelised via a
    2-thread ``joblib.Parallel(prefer='threads')`` pool; each fit's inner xgboost ``n_jobs`` is
    capped to ``n_cores // 2`` so the outer-x-inner product matches the box's core count (mirrors
    ``_parallel_honest_losses`` from iter4). The two estimators don't share state and xgboost
    releases the GIL during tree-build, so this is a free wall-clock cut. Measured width=1000 /
    n_rows=5000 booster portion: 20.4s -> 17.8s (1.15x; deep d=4 dominates so the smaller
    stump d=1 gets hidden behind it -- still ~13% off the gate's hottest slice). Scores
    byte-identical vs the serial path (same seeds + same inner ``n_jobs`` is enough).

    iter27: the ``deep`` booster's ``max_depth`` is capped 4 -> 3. iter26 left the parallel wall
    bounded by the deep d=4 branch (~14s at width=1000 / n_rows=5000 -- the stump d=1 finishes
    well before deep so the parallel pool is essentially deep's wall plus the dispatch overhead).
    Same cap-the-ranker pattern as iter9/iter10/iter19/iter25: the deep probe is RANKING-ONLY --
    we read the RATIO ``(stump_score - base) / (deep_score - base)`` to gate "interaction-heavy
    vs additive", NEVER deploy the model. A shallower deep model still captures all 2-way and the
    bulk of 3-way interactions (the additive-ratio differentiation), and the ratio's gate-trip
    point at the 0.6 floor is preserved across the iter17 5-regime calibration set
    (additive_highSNR / redundancy_heavy / interaction_heavy / xor_interaction / noise_heavy --
    measured 2026-05-28: d=4 ratios 1.078/1.060/0.827/0.000/1.147, d=3 ratios
    1.053/1.049/0.759/0.000/1.161; recommendation + suggestion set IDENTICAL on all 5). The
    closest-to-floor regime (interaction_heavy) drops 0.827 -> 0.759 -- still well above the 0.6
    floor. Wall-clock at width=1000: median 17.82s -> 12.9s (1.38x; 28% off iter26's parallel
    baseline)."""
    import pandas as pd
    from xgboost import XGBClassifier, XGBRegressor

    rng = np.random.default_rng(random_state)
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y = np.asarray(y)
    n, f = X.shape

    balance = float(np.mean(y)) if classification else float("nan")

    # Redundancy: max |corr| over a random feature sample (cap for wide data). Rows are sampled
    # INDEPENDENTLY of ``max_rows`` (the booster cap) and FIRST so the column-sample rng draw is
    # bit-for-bit identical to the pre-iter25 path -- ``max_rows_corr`` defaults to the old 5000-row
    # behaviour, so this preserves the corr-gate's tripping points byte-equivalent to the legacy
    # implementation while the new ``max_rows`` knob only affects the booster CV calls below.
    if n > max_rows_corr:
        sel_corr = rng.choice(n, size=max_rows_corr, replace=False)
        Xc_src = X.iloc[sel_corr]
    else:
        Xc_src = X
    cols = np.arange(f)
    if f > max_corr_features:
        cols = rng.choice(f, size=max_corr_features, replace=False)
    Xc = np.nan_to_num(Xc_src.iloc[:, cols].to_numpy(dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc, rowvar=False)
    np.fill_diagonal(C, 0.0)
    max_abs_corr = float(np.nanmax(np.abs(C))) if C.size else 0.0

    # Booster row subsample (ranking-only -- see ``max_rows`` docstring). Sampled AFTER the corr-pass
    # rng draws so the corr gate's bit-for-bit determinism vs the pre-iter25 path is preserved (the
    # legacy ordering was rows->corr->boosters; iter25 reorders to corr-rows->corr-cols->booster-rows
    # but ``max_rows_corr`` defaults to the legacy 5000 so the corr rng prefix matches).
    if n > max_rows:
        sel = rng.choice(n, size=max_rows, replace=False)
        Xs, ys = X.iloc[sel], y[sel]
    else:
        Xs, ys = X, y

    # Two independent boosters (deep d=4 and stump d=1) over the SAME (Xs, ys); each fit already
    # parallelises across cores via xgboost's own n_jobs, so we cap inner threads to n_cores//2 and
    # run the two _cv_score calls concurrently in a 2-thread joblib pool. xgboost releases the GIL
    # during training so prefer="threads" shares (Xs, ys) without pickling overhead. Iter4's
    # ``_parallel_honest_losses`` uses the same outer-x-inner pattern. ``random_state`` is set
    # explicitly on both estimators so CV folds and tree seeds remain deterministic regardless of
    # which worker finishes first -- byte-identical scores vs the prior serial path.
    import os

    n_cores = os.cpu_count() or 1
    # iter54: default lets xgboost manage all cores via its own thread pool (inner=-1). The 2 fits
    # (deep + stump) run concurrently in a 2-worker joblib pool; A/B at width 4000+10000 measured the
    # iter4 ``n_cores // outer`` cap as 8-9% e2e slower (per-stage table in ``_shap_proxy_explain``).
    # ``inner_n_jobs_cap=True`` restores legacy ``n_cores // 2`` for HW where the cap helps.
    inner = max(1, n_cores // 2) if inner_n_jobs_cap else -1
    common = dict(n_estimators=int(n_estimators), learning_rate=0.1, n_jobs=inner,
                  random_state=random_state, tree_method="hist")
    # iter27 cap-the-ranker: deep max_depth 4 -> 3. The deep probe is RANKING-ONLY (we read the
    # additive_ratio gate, not deployed predictions); a depth-3 booster still captures all 2-way
    # and most 3-way interactions so the additive-vs-deep ratio's gate-trip points at the 0.6
    # floor stay stable across all 5 iter17 calibration regimes. ~28% off the gate's parallel wall
    # at width=1000 / n_rows=5000 (deep dominates the parallel pool, see ``dataset_diagnostics``
    # docstring for the per-regime ratio table).
    if classification:
        deep = XGBClassifier(max_depth=3, eval_metric="logloss", **common)
        stump = XGBClassifier(max_depth=1, eval_metric="logloss", **common)
        base = 0.5  # AUC of a constant predictor
    else:
        deep = XGBRegressor(max_depth=3, **common)
        stump = XGBRegressor(max_depth=1, **common)
        base = 0.0  # r2 of the mean predictor

    if n_cores >= 2:
        from joblib import Parallel, delayed

        deep_score, stump_score = Parallel(n_jobs=2, prefer="threads")(
            delayed(_cv_score)(est, Xs, ys, classification) for est in (deep, stump)
        )
    else:
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
    max_rows=2000, max_rows_corr=5000, n_estimators=100, inner_n_jobs_cap=False,
):
    """Cheap recommendation on whether / how to run ShapProxiedFS. Returns a dict with
    ``recommendation`` in {run, caution, fallback}, the diagnostics, and the reasons.

    ``max_rows`` / ``max_rows_corr`` / ``n_estimators``: see :func:`dataset_diagnostics` --
    ranking-only booster caps that keep the gate cheap at wide regimes (the iter25 fix for
    preflight=fit wall-clock parity).

    ``inner_n_jobs_cap`` (iter54, default False): see :func:`dataset_diagnostics`."""
    d = dataset_diagnostics(X, y, classification=classification, random_state=random_state,
                            max_rows=max_rows, max_rows_corr=max_rows_corr, n_estimators=n_estimators,
                            inner_n_jobs_cap=inner_n_jobs_cap)
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
