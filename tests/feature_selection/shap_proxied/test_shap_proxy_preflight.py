"""biz_val for the pre-flight decision rule (Batch C): it must correctly distinguish the regime where
ShapProxiedFS shines (additive, high-SNR, well-fit) from where it struggles (pure interaction / XOR),
from cheap statistics alone -- so a user can decide whether/how to run it before paying the full cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


def _additive(n=2500, seed=0):
    """Helper that additive."""
    rng = np.random.default_rng(seed)
    inf = rng.normal(size=(n, 5))
    noise = rng.normal(size=(n, 5))
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"f{i}" for i in range(10)])
    logit = 1.0 * inf[:, 0] + 0.9 * inf[:, 1] - 0.8 * inf[:, 2] + 0.6 * inf[:, 3] + 0.4 * inf[:, 4]
    y = (logit > 0).astype(int)  # high-SNR additive -> the favourable regime
    return X, y


def test_preflight_corr_gate_bit_identical_under_iter25_cap():
    """Iter25 booster cap (max_rows + n_estimators) must NOT alter ``max_abs_corr`` -- the
    redundancy gate is driven by an independent ``max_rows_corr`` knob (default 5000, matches the
    legacy implementation's row sample) so the corr column sample's rng draw is bit-for-bit identical
    to the pre-iter25 path. Fast unit test (small synthetic; xgboost ranking call dominates but on
    n=300 / f=20 runs <5s).
    """
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import preflight

    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(size=(n, 3))
    # 9 redundant copies + 8 noise -> max|corr| well above 0.7 floor.
    refl = np.hstack([z[:, [k]] + 0.05 * rng.normal(size=(n, 3)) for k in range(3)])
    noise = rng.normal(size=(n, 8))
    X = pd.DataFrame(np.column_stack([refl, noise]), columns=[f"f{i}" for i in range(refl.shape[1] + noise.shape[1])])
    y = (z[:, 0] + 0.5 * z[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int)

    legacy = preflight(X, y, classification=True, random_state=0, max_rows=5000, n_estimators=150)
    capped = preflight(X, y, classification=True, random_state=0, max_rows=2000, n_estimators=100)
    # Redundancy gate output bit-for-bit identical (corr-pass determinism preserved).
    assert capped["diagnostics"]["max_abs_corr"] == legacy["diagnostics"]["max_abs_corr"]
    # Recommendation invariant on this small synthetic.
    assert capped["recommendation"] == legacy["recommendation"]


def _xor(n=2500, seed=0):
    """Helper that xor."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(int)  # pure interaction
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(8)]), y


@pytest.mark.slow
def test_biz_val_preflight_favours_additive_flags_interaction():
    """Biz val preflight favours additive flags interaction."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    add = ShapProxiedFS.preflight(*_additive(), classification=True, random_state=0)
    xor = ShapProxiedFS.preflight(*_xor(), classification=True, random_state=0)

    add_ratio = add["diagnostics"]["additive_ratio"]
    xor_ratio = xor["diagnostics"]["additive_ratio"]
    print(f"[NT-09] additive_ratio: additive={add_ratio:.4f} xor={xor_ratio:.4f} gap={add_ratio - xor_ratio:.4f}")

    # The ORDERING, computed in the same process with the same xgboost build, rather than two absolute cuts
    # only 0.1 apart. The ratio comes out of a fitted booster, so a different xgboost version or thread count
    # shifts both arms together -- which a paired comparison cancels and an absolute cut does not. The
    # recommendation each fixture drives is the decision that actually matters and is asserted separately.
    assert add_ratio > xor_ratio + 0.15, (
        f"the additive fixture no longer scores a materially higher additive ratio than the XOR one "
        f"(additive {add_ratio:.4f} vs xor {xor_ratio:.4f}); the diagnostic has stopped separating the two regimes"
    )
    assert add["recommendation"] == "run", add
    assert xor["diagnostics"]["additive_ratio"] < add["diagnostics"]["additive_ratio"], xor["diagnostics"]
    assert xor["recommendation"] in ("caution", "fallback")
    assert any("interaction" in r for r in xor["reasons"]), xor["reasons"]
    assert "enable interaction_aware=True" in xor["suggestions"]


@pytest.mark.slow
def test_biz_val_preflight_flags_redundancy_and_width():
    """Biz val preflight flags redundancy and width."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=(n, 3))
    refl = np.hstack([z[:, [k]] + 0.1 * rng.normal(size=(n, 20)) for k in range(3)])  # 60 redundant
    X = pd.DataFrame(refl, columns=[f"f{i}" for i in range(refl.shape[1])])
    y = (z[:, 0] + 0.8 * z[:, 1] - 0.7 * z[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    rep = ShapProxiedFS.preflight(X, y, classification=True, random_state=0)
    assert rep["diagnostics"]["max_abs_corr"] >= 0.7
    assert "enable cluster_features=True" in rep["suggestions"]
    assert rep["diagnostics"]["n_features"] > 40


# iter25 cap: the preflight booster CV calls were dominating the gate's wall-clock (live test
# 2026-05-28 width=1000 / n_rows=5000 measured preflight=77s vs full fit=86s). Capping
# ``n_estimators=100`` + booster row subsample to 2000 cuts ~30% off the gate on the same regime
# without flipping the recommendation -- this test pins both invariants.

# Match the iter17 calibration regimes (additive/redundancy/interaction/xor/noise_heavy) -- these are
# the gate's calibration anchor and must produce identical recommendations under the cap.
_ITER17_REGIMES = (
    dict(
        name="additive_highSNR",
        kwargs=dict(n_samples=2000, n_informative=8, n_redundant=0, n_noise=400, interaction_order=0, interaction_strength=0.0, snr=5.0, task="binary", seed=0),
    ),
    dict(
        name="redundancy_heavy", kwargs=dict(n_samples=1200, n_informative=8, n_redundant=24, redundancy_rho=0.9, n_noise=400, snr=2.5, task="binary", seed=1)
    ),
    dict(
        name="interaction_heavy",
        kwargs=dict(n_samples=1200, n_informative=8, n_redundant=0, n_noise=400, interaction_order=2, interaction_strength=0.7, snr=3.0, task="binary", seed=2),
    ),
    dict(
        name="xor_interaction",
        kwargs=dict(
            n_samples=1500, n_informative=6, n_redundant=0, n_noise=400, interaction_order="xor", interaction_strength=0.9, snr=3.0, task="binary", seed=4
        ),
    ),
    dict(name="noise_heavy", kwargs=dict(n_samples=1500, n_informative=8, n_redundant=0, n_noise=1200, snr=2.0, task="binary", seed=3)),
)


@pytest.mark.slow
@pytest.mark.parametrize("regime", _ITER17_REGIMES, ids=lambda r: r["name"])
def test_preflight_capped_recommendation_matches_legacy_iter17(regime):
    """Iter25 cap (max_rows=2000, n_estimators=100) must produce identical recommendation to the
    legacy params (max_rows=5000, n_estimators=150) across the iter17 calibration regimes. The
    ``additive_ratio`` and ``full_model_fit`` are allowed to differ slightly (smaller cap = slightly
    lower CV score on average) but the gate decision and the suggestion set must be stable.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(**regime["kwargs"])
    legacy = ShapProxiedFS.preflight(X, y, classification=True, random_state=0, max_rows=5000, n_estimators=150)
    capped = ShapProxiedFS.preflight(X, y, classification=True, random_state=0, max_rows=2000, n_estimators=100)

    assert capped["recommendation"] == legacy["recommendation"], (
        f"regime {regime['name']}: cap flipped recommendation "
        f"{legacy['recommendation']!r} -> {capped['recommendation']!r}; "
        f"ratios legacy={legacy['diagnostics']['additive_ratio']:.3f} "
        f"capped={capped['diagnostics']['additive_ratio']:.3f}"
    )
    assert sorted(capped["suggestions"]) == sorted(
        legacy["suggestions"]
    ), f"regime {regime['name']}: suggestion set changed under cap legacy={legacy['suggestions']} capped={capped['suggestions']}"
    # max_abs_corr uses an independent ``max_rows_corr`` knob (default 5000) so the redundancy gate
    # remains bit-for-bit identical to the legacy pre-iter25 implementation.
    assert capped["diagnostics"]["max_abs_corr"] == legacy["diagnostics"]["max_abs_corr"]


def _run_diagnostics_with_deep_depth(X, y, deep_max_depth):
    """Run the same logic as ``dataset_diagnostics`` but pin the deep booster's ``max_depth`` to the
    given value. Used to compare iter27's d=3 default against the legacy d=4 across regimes -- the
    rest of the orchestration (parallel pool, inner n_jobs, n_estimators, row caps) stays
    bit-for-bit identical so the comparison isolates the depth knob.
    """
    import os, numpy as np, pandas as pd
    from xgboost import XGBClassifier
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import _cv_score

    max_rows, max_rows_corr, max_corr_features, n_estimators = 2000, 5000, 400, 100
    rng = np.random.default_rng(0)
    Xf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y = np.asarray(y)
    n, f = Xf.shape
    balance = float(np.mean(y))
    if n > max_rows_corr:
        sel_corr = rng.choice(n, size=max_rows_corr, replace=False)
        Xc_src = Xf.iloc[sel_corr]
    else:
        Xc_src = Xf
    cols = np.arange(f)
    if f > max_corr_features:
        cols = rng.choice(f, size=max_corr_features, replace=False)
    Xc = np.nan_to_num(Xc_src.iloc[:, cols].to_numpy(dtype=np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc, rowvar=False)
    np.fill_diagonal(C, 0.0)
    max_abs_corr = float(np.nanmax(np.abs(C))) if C.size else 0.0
    if n > max_rows:
        sel = rng.choice(n, size=max_rows, replace=False)
        Xs, ys = Xf.iloc[sel], y[sel]
    else:
        Xs, ys = Xf, y
    n_cores = os.cpu_count() or 1
    inner = max(1, n_cores // 2)
    common = dict(n_estimators=n_estimators, learning_rate=0.1, n_jobs=inner, random_state=0, tree_method="hist")
    deep = XGBClassifier(max_depth=deep_max_depth, eval_metric="logloss", **common)
    stump = XGBClassifier(max_depth=1, eval_metric="logloss", **common)
    if n_cores >= 2:
        from joblib import Parallel, delayed

        deep_score, stump_score = Parallel(n_jobs=2, prefer="threads")(delayed(_cv_score)(est, Xs, ys, True) for est in (deep, stump))
    else:
        deep_score = _cv_score(deep, Xs, ys, True)
        stump_score = _cv_score(stump, Xs, ys, True)
    base = 0.5
    num = stump_score - base
    den = deep_score - base
    additive_ratio = float(np.clip(num / den, 0.0, 1.5)) if (np.isfinite(den) and den > 1e-6) else float("nan")
    diag = dict(
        n_features=int(f),
        n_samples=int(n),
        n_over_p=float(n / max(f, 1)),
        class_balance=balance,
        max_abs_corr=max_abs_corr,
        full_model_fit=deep_score,
        stump_fit=stump_score,
        additive_ratio=additive_ratio,
        base_score=base,
    )

    # Re-run preflight's reason/recommendation logic on the synthetic diag (mirrors preflight() body
    # in src so the test pins the gate-decision path, not just the booster scores).
    import numpy as _np

    reasons, suggestions = [], []
    rec = "run"
    if not _np.isfinite(diag["full_model_fit"]) or (diag["full_model_fit"] - diag["base_score"]) < 0.03:
        rec = "fallback"
        reasons.append("full-model fit barely beats trivial.")
    if _np.isfinite(diag["additive_ratio"]) and diag["additive_ratio"] < 0.6:
        suggestions.append("enable interaction_aware=True")
        if rec != "fallback":
            rec = "caution"
    if diag["max_abs_corr"] >= 0.7:
        suggestions.append("enable cluster_features=True")
    if diag["n_features"] > 40:
        suggestions.append("cluster_features + pre-screen (auto)")
    if min(diag["class_balance"], 1 - diag["class_balance"]) < 0.05:
        suggestions.append("use metric='auc'")
        if rec == "run":
            rec = "caution"
    return dict(recommendation=rec, diagnostics=diag, suggestions=sorted(set(suggestions)))


# Regimes where the deep-vs-stump fit gap is comfortably above the ``min_fit_gain=0.03`` floor, so
# dropping deep max_depth 4 -> 3 cannot flip the recommendation. ``xor_interaction`` is excluded: on a
# pure-XOR target the deep model sits right ON the 0.03 fit-gain boundary (d=4 AUC ~0.55 -> deep-base
# ~0.050; d=3 AUC ~0.52 -> deep-base ~0.019), so the depth cut legitimately moves it across the
# fallback threshold. That boundary crossing is NOT a regression -- both depths still correctly flag
# XOR as a hard regime (a guard recommendation, never "run") and the additive ratio is degenerate-low
# (stump below base => interactions fully dominate). The cross-depth recommendation EQUALITY only
# holds away from the threshold; the XOR case is pinned by the guard-invariant assertions below.
_DEPTH_EQ_REGIMES = tuple(r for r in _ITER17_REGIMES if r["name"] != "xor_interaction")


@pytest.mark.slow
@pytest.mark.parametrize("regime", _DEPTH_EQ_REGIMES, ids=lambda r: r["name"])
def test_preflight_deep_depth3_recommendation_matches_depth4_iter27(regime):
    """Iter27 cap-the-ranker: the deep booster's ``max_depth`` was lowered 4 -> 3. On the calibration
    regimes whose deep-vs-trivial fit gap clears the ``min_fit_gain=0.03`` floor the recommendation +
    suggestion set must match the legacy depth=4 behaviour. We rebuild ``dataset_diagnostics`` with
    both depths via ``_run_diagnostics_with_deep_depth`` (everything else -- inner n_jobs, joblib pool,
    n_estimators, row caps -- stays identical). The boundary ``xor_interaction`` regime is covered
    separately by ``test_preflight_deep_depth3_xor_guard_invariant`` because its deep fit straddles the
    fit-gain threshold, so cross-depth equality there is not a meaningful contract.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(**regime["kwargs"])
    legacy = _run_diagnostics_with_deep_depth(X, y, 4)
    new = _run_diagnostics_with_deep_depth(X, y, 3)

    assert new["recommendation"] == legacy["recommendation"], (
        f"regime {regime['name']}: deep d=4 -> d=3 flipped recommendation "
        f"{legacy['recommendation']!r} -> {new['recommendation']!r}; "
        f"ratios legacy(d=4)={legacy['diagnostics']['additive_ratio']:.3f} "
        f"new(d=3)={new['diagnostics']['additive_ratio']:.3f}"
    )
    assert sorted(new["suggestions"]) == sorted(
        legacy["suggestions"]
    ), f"regime {regime['name']}: suggestion set changed under d=3 legacy={legacy['suggestions']} new={new['suggestions']}"
    # The corr-pass is depth-independent (it doesn't fit a booster) so max_abs_corr must be
    # bit-for-bit identical between the two depths.
    assert new["diagnostics"]["max_abs_corr"] == legacy["diagnostics"]["max_abs_corr"]


@pytest.mark.slow
def test_preflight_deep_depth3_xor_guard_invariant():
    """On pure-XOR the depth-3 deep probe sits on the ``min_fit_gain`` boundary, so the cap-the-ranker
    cut can flip the recommendation between "caution" (d=4) and "fallback" (d=3). That is a legitimate
    boundary crossing, not a regression -- the prod default is d=3 and is deterministic. What MUST hold
    for the proxy's gate to be trustworthy on XOR: both depths emit a GUARD recommendation (never
    "run"), the additive ratio stays degenerate-low (a depth-1 stump cannot model XOR so num<=0 ->
    ratio clipped to 0), and the corr-pass is depth-independent. A real regression -- XOR recommended
    "run", or the additive ratio breaking above the 0.6 interaction floor -- still fails here.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    regime = next(r for r in _ITER17_REGIMES if r["name"] == "xor_interaction")
    X, y, _ = make_regime_dataset(**regime["kwargs"])
    legacy = _run_diagnostics_with_deep_depth(X, y, 4)
    new = _run_diagnostics_with_deep_depth(X, y, 3)

    for label, rep in (("d=4", legacy), ("d=3", new)):
        assert rep["recommendation"] in ("caution", "fallback"), f"XOR {label} must flag a guard recommendation, got {rep['recommendation']!r}: {rep}"
        assert (
            rep["diagnostics"]["additive_ratio"] < 0.6
        ), f"XOR {label} additive ratio should be interaction-low (<0.6), got {rep['diagnostics']['additive_ratio']}"
    # The claim this test is named for is that capping the ranker's depth 4 -> 3 does not move the verdict,
    # and that is a comparison between the two arms rather than two absolute cuts. Both are fitted-booster
    # diagnostics, so a different xgboost build shifts them together -- which this cancels. (Measured on the
    # XOR fixture the ratio is ~0.03, so the 0.6 cuts above have ~20x of margin and stay as a regime check.)
    d4_ratio = legacy["diagnostics"]["additive_ratio"]
    d3_ratio = new["diagnostics"]["additive_ratio"]
    assert abs(d4_ratio - d3_ratio) <= 0.25, (
        f"the depth cap moved the additive ratio from {d4_ratio:.4f} (d=4) to {d3_ratio:.4f} (d=3); the cut was "
        "supposed to leave the interaction verdict where it was"
    )
    # Not equality: measured, the cap moves the verdict from 'caution' to 'fallback', which is a STRICTER
    # guard on an interaction-heavy fixture and not a regression. What must never happen is the cap relaxing
    # the verdict -- letting a XOR frame through as "run" -- so the ordering is what gets asserted.
    _severity = {"run": 0, "caution": 1, "fallback": 2}
    assert _severity[new["recommendation"]] >= _severity[legacy["recommendation"]], (
        f"the depth cap RELAXED the recommendation from {legacy['recommendation']!r} (d=4) to "
        f"{new['recommendation']!r} (d=3); capping the ranker must not let an interaction-heavy frame through"
    )
    # The corr-pass is depth-independent (it doesn't fit a booster) so max_abs_corr must be
    # bit-for-bit identical between the two depths.
    assert new["diagnostics"]["max_abs_corr"] == legacy["diagnostics"]["max_abs_corr"]


@pytest.mark.slow
def _captured_booster_kwarg(monkeypatch, name: str) -> list:
    """Record `name` from every XGB estimator the preflight probe constructs.

    Returns the (initially empty) list the spies append to, so the caller can run the preflight and then
    read what was actually asked for. Spying the constructor answers "is the cap in effect" exactly, which
    a wall-clock budget can only guess at from the outside.
    """
    import xgboost

    # `dataset_diagnostics` imports the estimators inside the function body, so the name to replace is the
    # one on the xgboost module itself, not a module-level alias in the preflight module.
    seen: list = []
    for cls_name in ("XGBClassifier", "XGBRegressor"):
        real = getattr(xgboost, cls_name)

        def _spy(*args, _real=real, **kwargs):
            """Record the kwarg, then build the real estimator."""
            if name in kwargs:
                seen.append(kwargs[name])
            return _real(*args, **kwargs)

        monkeypatch.setattr(xgboost, cls_name, _spy)
    return seen


def _captured_booster_depths(monkeypatch) -> list:
    """The `max_depth` of every XGB estimator the preflight probe constructs."""
    return _captured_booster_kwarg(monkeypatch, "max_depth")


def test_the_deep_ranker_is_capped_at_depth_three(monkeypatch):
    """The iter27 cap-the-ranker cut, asserted on the booster rather than on the clock.

    The previous form asserted `elapsed < 25.0` and its own failure message named the real property
    ("check deep booster max_depth is 3"). xgboost's training time moves with thread count, OpenMP build
    and CPU, so that budget was false-red on a 2-vCPU runner with the cap correctly applied and false-green
    on a large box with it removed. The constructor kwargs are exact and free.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(
        n_samples=5000,
        n_informative=12,
        n_redundant=8,
        n_noise=980,
        snr=8.0,
        task="binary",
        seed=0,
    )
    depths = _captured_booster_depths(monkeypatch)
    rep = ShapProxiedFS.preflight(X, y, classification=True, random_state=0)

    assert rep["recommendation"] in ("run", "caution"), rep
    assert sorted(depths) == [1, 3], f"the preflight probe built boosters at depths {sorted(depths)}; the ranking pair must be depth-1 (stump) and depth-3 (deep)"


def test_preflight_parallel_booster_byte_identical_across_calls():
    """Iter26 parallelises the deep+stump booster CV via joblib(n_jobs=2, prefer='threads') with
    xgboost's inner ``n_jobs`` capped to ``n_cores // 2`` (mirrors iter4's outer-x-inner pattern).

    Two pins on the new orchestration:
      (a) two back-to-back calls return byte-identical scores -- the joblib pool doesn't introduce
          non-determinism even though deep/stump can finish in either order;
      (b) the booster results match a direct serial reference (same seeds, same hyper-params, same
          inner ``n_jobs``) -- the joblib wrapper itself is score-neutral.
    """
    import os
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import dataset_diagnostics

    rng = np.random.default_rng(0)
    n = 600
    nfeat = 25
    X = pd.DataFrame(rng.normal(size=(n, nfeat)), columns=[f"f{i}" for i in range(nfeat)])
    y = (X.iloc[:, 0] + 0.5 * X.iloc[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(int).to_numpy()

    a = dataset_diagnostics(X, y, classification=True, random_state=0)
    b = dataset_diagnostics(X, y, classification=True, random_state=0)
    # (a) determinism of the parallel path itself.
    assert a["full_model_fit"] == b["full_model_fit"]
    assert a["stump_fit"] == b["stump_fit"]
    assert a["additive_ratio"] == b["additive_ratio"]
    assert a["max_abs_corr"] == b["max_abs_corr"]

    # (b) match a direct serial xgboost+cross_val_score reference with the exact same inner n_jobs
    # the parallel path passes to xgb. The orchestration (joblib pool vs sequential) is score-neutral.
    n_cores = os.cpu_count() or 1
    inner = max(1, n_cores // 2)
    common = dict(n_estimators=100, learning_rate=0.1, n_jobs=inner, random_state=0, tree_method="hist")
    # Recreate the booster subsample exactly as ``dataset_diagnostics`` does: first the corr-pass
    # rng draws (none here because n=600 < max_rows_corr=5000 default), then the booster row sample
    # (none because n=600 < max_rows=2000 default). So Xs/ys == X/y. Deep ``max_depth=3`` matches
    # the iter27 cap-the-ranker choice (was 4 pre-iter27); see ``dataset_diagnostics`` docstring.
    deep = XGBClassifier(max_depth=3, eval_metric="logloss", **common)
    stump = XGBClassifier(max_depth=1, eval_metric="logloss", **common)
    deep_ref = float(np.mean(cross_val_score(deep, X, y, cv=3, scoring="roc_auc")))
    stump_ref = float(np.mean(cross_val_score(stump, X, y, cv=3, scoring="roc_auc")))
    assert a["full_model_fit"] == deep_ref, (a["full_model_fit"], deep_ref)
    assert a["stump_fit"] == stump_ref, (a["stump_fit"], stump_ref)


@pytest.mark.slow
def test_the_booster_tree_cap_is_in_effect(monkeypatch):
    """The iter25 cheap-check cap, asserted on the boosters rather than on the clock.

    Pre-iter25 the gate took 77s on the user's machine against the 86s full fit it was supposed to gate
    cheaply; iter25 capped the booster CV calls (n_estimators 150 -> 100, max_rows 5000 -> 2000). The
    previous form asserted `elapsed < 30.0` and its own failure message named the real property ("check
    booster cap (n_estimators) is in effect"). xgboost training time moves with thread count, OpenMP build
    and CPU, so that budget was false-red on a small runner with the cap applied and false-green on a
    large one without it.
    """
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y, _ = make_regime_dataset(
        n_samples=5000,
        n_informative=12,
        n_redundant=8,
        n_noise=980,
        snr=8.0,
        task="binary",
        seed=0,
    )
    trees = _captured_booster_kwarg(monkeypatch, "n_estimators")
    rep = ShapProxiedFS.preflight(X, y, classification=True, random_state=0)

    # Recommendation should still be "run" on this favourable wide regime (high SNR, low informative
    # density relative to noise but additive signal dominates).
    assert rep["recommendation"] in ("run", "caution"), rep
    assert trees, "no booster was constructed, so the cap this test names was never exercised"
    assert set(trees) == {100}, f"the preflight probe built boosters with n_estimators {sorted(set(trees))}; the iter25 ranking cap is 100"
