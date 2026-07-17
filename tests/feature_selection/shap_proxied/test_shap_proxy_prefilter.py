"""Unit tests for the ShapProxiedFS native-importance pre-filter (``_shap_proxy_prefilter``).

Covers: method routing (``auto`` smart default by width/rows + CUDA presence, explicit pass-through,
unknown -> ValueError), each concrete ranking method keeps the planted informatives, the
original-column mapping (``working_cols`` is sorted original indices and the selector's ``support_``
stays in original-column space after a prefilter), and graceful fall-through when no importances exist.

Fast by design: tiny widths so the four methods run in well under the test budget; the heavy
speed/quality characterization lives in the benchmark + the slow biz_value test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")


# --------------------------------------------------------------------------- routing (no fit needed)
def test_resolve_explicit_methods_pass_through():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import PREFILTER_METHODS, resolve_prefilter_method

    for m in PREFILTER_METHODS:
        assert resolve_prefilter_method(m, n_features=10000, n_rows=4000) == m


def test_resolve_unknown_method_raises():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import resolve_prefilter_method

    with pytest.raises(ValueError):
        resolve_prefilter_method("not_a_method", n_features=100, n_rows=100)


def test_resolve_auto_keeps_model_for_moderate_width():
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import _auto_fast_width, resolve_prefilter_method

    narrow = _auto_fast_width() - 1
    assert resolve_prefilter_method("auto", n_features=narrow, n_rows=4000) == "model"


def test_resolve_auto_switches_two_stage_for_wide_when_no_gpu(monkeypatch):
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    # No CUDA -> auto must pick two_stage at/above auto_fast_width. iter21 unified two_stage_min_width
    # with auto_fast_width based on the prefilter micro-bench (5.10x faster than "model" at width 6000
    # with identical 8/8 informative recall); the sub-4k sweep then lowered the unified threshold to
    # 1000 (4.47-5.25x speedup with parity recall at every sub-4k width tested), so the legacy
    # fast_model auto window stays collapsed and starts further down.
    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    wide = PF._auto_fast_width()
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=4000) == "two_stage"
    # Even with many rows, no device means no gpu_model -> still two_stage on the wide side.
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=10**6) == "two_stage"


def test_resolve_auto_routes_gpu_when_device_and_enough_rows(monkeypatch):
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: True)
    # Wide-data (>= auto_fast_width): with a CUDA device + enough rows, auto picks gpu_model over
    # two_stage (full-fidelity ranking on the GPU outranks the cheap funnel on big-n hardware).
    wide = PF._auto_fast_width()
    big_n = PF._gpu_model_min_rows()
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n) == "gpu_model"
    # Too few rows -> GPU transfer overhead not worth it -> two_stage takes over (iter21).
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n - 1) == "two_stage"


def test_resolve_auto_falls_back_to_fast_model_when_two_stage_gated_above_auto_fast(monkeypatch):
    """Safety-net branch: when a user pushes ``two_stage_min_width`` ABOVE ``auto_fast_width`` via
    kernel_tuning_cache, the window in between routes to ``"fast_model"`` (still interaction-aware,
    just slower than two_stage). Defaults unify the two thresholds (iter21) so this branch is dormant
    on the shipped config but the wiring is exercised here for safety."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    afw = PF._auto_fast_width()
    monkeypatch.setattr(PF, "_two_stage_min_width", lambda: afw + 4000)
    mid_wide = afw + 1000  # in the dormant window between the two thresholds
    assert PF.resolve_prefilter_method("auto", n_features=mid_wide, n_rows=4000) == "fast_model"
    # Above the (raised) two_stage threshold -> two_stage again.
    assert PF.resolve_prefilter_method("auto", n_features=afw + 5000, n_rows=4000) == "two_stage"


# --------------------------------------------------------------------------- ranking correctness
def _wide_xy(seed=0, width=300, n_informative=5):
    """A few strong informatives + lots of independent noise (no clustering needed here)."""
    rng = np.random.default_rng(seed)
    n = 1200
    inf = rng.normal(size=(n, n_informative))
    noise = rng.normal(size=(n, width - n_informative))
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"inf{i}" for i in range(n_informative)] + [f"noise{i}" for i in range(width - n_informative)])
    coefs = np.array([0.9, 0.8, -0.7, 0.6, 0.4])[:n_informative]
    logit = (inf * coefs).sum(axis=1)
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("method", ["model", "univariate", "fast_model"])
def test_prefilter_keeps_informatives_and_returns_sorted_original_indices(method):
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(width=300, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    keep_k = 40
    working_cols, info = prefilter_columns(model, X, y.astype(np.float64), method=method, prefilter_top=keep_k, classification=True, n_features=X.shape[1])

    # working_cols: sorted, unique, within range, length == kept.
    assert working_cols.ndim == 1 and len(working_cols) == keep_k == info["kept"]
    assert list(working_cols) == sorted(set(int(c) for c in working_cols))
    assert working_cols.min() >= 0 and working_cols.max() < X.shape[1]
    assert info["method"] == method and info["of"] == X.shape[1]

    # All 5 informatives (original indices 0..4) survive the cut for every interaction-aware /
    # marginal-strong method on this main-effect-driven target.
    kept = set(int(c) for c in working_cols)
    assert {0, 1, 2, 3, 4} <= kept, f"{method}: lost informatives, kept head={sorted(kept)[:8]}"


def test_prefilter_no_importance_model_falls_through_to_identity():
    """A model exposing neither feature_importances_ nor coef_ -> keep all columns (identity)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    class _NoImportance:
        def get_params(self, deep=False):
            return {}

        def fit(self, X, y):
            return self

    X, y = _wide_xy(width=50)
    working_cols, info = prefilter_columns(
        _NoImportance(), X, y.astype(np.float64), method="model", prefilter_top=10, classification=True, n_features=X.shape[1]
    )
    np.testing.assert_array_equal(working_cols, np.arange(X.shape[1]))
    assert info["kept"] == X.shape[1] and info.get("skipped") == "no_importance"


# --------------------------------------------------------------------------- end-to-end selector wire
@pytest.mark.parametrize("method", ["model", "univariate", "fast_model"])
def test_selector_support_stays_in_original_space_under_prefilter(method):
    """The prefilter restricts the working frame; support_ / selected_features_ must still be reported
    in ORIGINAL column space (and name-based transform must match)."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    X, y = _wide_xy(seed=1, width=200, n_informative=5)
    sel = ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="auto",
        prefilter_top=60,
        prefilter_method=method,
        cluster_features=True,
        cluster_corr_threshold=0.7,
        top_n=8,
        n_splits=3,
        n_revalidation_models=1,
        trust_guard=False,
        run_importance_ablation=False,
        random_state=0,
        verbose=False,
    )
    sel.fit(X, pd.Series(y))

    assert sel.support_.shape == (X.shape[1],)
    assert sel.support_.dtype == bool
    assert sel.n_features_in_ == X.shape[1]
    assert len(sel.selected_features_) == int(sel.support_.sum())
    # names map back to ORIGINAL columns, and support_ agrees with selected names.
    assert set(sel.selected_features_) <= set(X.columns)
    support_named = {c for c, m in zip(X.columns, sel.support_) if m}
    assert support_named == set(sel.selected_features_)
    # prefilter actually fired and recorded the resolved method.
    pf = sel.shap_proxy_report_["prefilter"]
    assert pf["method"] == method and pf["kept"] == 60 and pf["of"] == X.shape[1]
    # name-based transform returns exactly the selected columns.
    out = sel.transform(X)
    assert list(out.columns) == list(sel.selected_features_)


def test_default_prefilter_method_is_auto():
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    assert ShapProxiedFS().prefilter_method == "auto"


# ----------------------------------------------------------- prefilter_n_estimators cap (iter10)
def test_default_prefilter_n_estimators_cap_is_100():
    """The facade ships a tree-count cap on the pre-filter's ranking booster (same cap-the-ranker
    pattern iter9 applied to refine + trust-guard); the default makes the speed win opt-out, not
    opt-in. ``None`` disables the cap."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    assert ShapProxiedFS().prefilter_n_estimators == 100


def test_prefilter_cap_preserves_top_k_jaccard_vs_uncapped():
    """The cap reduces the ranking booster's tree count to ~100 to cut the prefilter wall-clock by ~3x.
    Importance attribution stabilises well below the default 300 trees -- so on a signal-bearing
    top-K slice (K close to the planted informative cardinality, where the importances are not
    in the noise-tail tie regime) the capped ranking must agree with the uncapped to Jaccard >= 0.95.

    Why measure Jaccard near the signal cardinality, not at the production ``prefilter_top``:
    once K extends deep into the noise tail the importances are essentially tied (differences are
    below tree-build noise across ANY seed perturbation), so even the SAME booster run twice with
    different RNGs would produce sub-0.95 Jaccard there. The decision-relevant quantity is whether
    the SIGNAL ranks survive the cap -- this is asserted directly via the informative-recovery set.

    Independently checked for ``"model"`` and ``"fast_model"`` so neither method silently degrades."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    # Clean separable synthetic: enough rows + low noise so the importances escape the tail-tie regime.
    rng = np.random.default_rng(0)
    n, width, n_inf = 3000, 200, 8
    inf = rng.normal(size=(n, n_inf))
    noise = rng.normal(size=(n, width - n_inf))
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"inf{i}" for i in range(n_inf)] + [f"noise{i}" for i in range(width - n_inf)])
    coefs = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8])
    logit = (inf * coefs).sum(axis=1)
    y = (logit + 0.05 * rng.normal(size=n) > 0).astype(np.float64)

    # Signal-aware K: the importances above this threshold carry the real ranking signal; beyond it,
    # the noise tail is rank-degenerate (same booster + different RNG would also disagree).
    signal_k = n_inf
    # Larger K used to capture informative-recovery (the prefilter's actual product).
    prod_k = 60

    for method in ("model", "fast_model"):
        # Build TWO templates with identical seeds so capped vs uncapped only differ in n_estimators.
        model_full = make_default_estimator(classification=True, random_state=0, n_estimators=300)
        model_cap = make_default_estimator(classification=True, random_state=0, n_estimators=300)
        uncapped, _ = prefilter_columns(
            model_full, X, y, method=method, prefilter_top=signal_k, classification=True, n_features=X.shape[1], n_estimators_cap=None
        )
        capped, info = prefilter_columns(
            model_cap, X, y, method=method, prefilter_top=signal_k, classification=True, n_features=X.shape[1], n_estimators_cap=100
        )
        a, b = set(map(int, uncapped)), set(map(int, capped))
        jaccard = len(a & b) / len(a | b)
        assert jaccard >= 0.95, f"{method}: top-{signal_k} Jaccard {jaccard:.3f} < 0.95 (capped vs uncapped)"

        # Production-K view: at the user-facing prefilter_top, the cap MUST keep all informatives
        # (the prefilter's decision-relevant output -- it doesn't need rank-stability among noise).
        capped_prod, info_prod = prefilter_columns(
            make_default_estimator(classification=True, random_state=0, n_estimators=300),
            X,
            y,
            method=method,
            prefilter_top=prod_k,
            classification=True,
            n_features=X.shape[1],
            n_estimators_cap=100,
        )
        kept_prod = set(map(int, capped_prod))
        assert set(range(n_inf)) <= kept_prod, f"{method}: capped prefilter lost informatives at prod K={prod_k}, got head={sorted(kept_prod)[:12]}"
        # Info dict surfaces the applied cap so downstream report consumers can see what ran.
        assert info["n_estimators_cap"] == 100
        assert info_prod["n_estimators_cap"] == 100


def test_prefilter_cap_none_is_legacy_uncapped_behaviour():
    """``n_estimators_cap=None`` must preserve the legacy path: the cloned ranking booster is fit at
    the template's own ``n_estimators`` (300 by default). Regression guard so the new knob is a pure
    extension."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(width=150, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=300)
    _, info = prefilter_columns(
        model, X, y.astype(np.float64), method="model", prefilter_top=30, classification=True, n_features=X.shape[1], n_estimators_cap=None
    )
    assert info["n_estimators_cap"] is None


# ----------------------------------------------------------- two_stage prefilter (iter12)
def test_resolve_auto_routes_two_stage_at_wide_threshold_no_gpu(monkeypatch):
    """auto routing edges around ``two_stage_min_width`` when no CUDA is available (iter21 unified
    two_stage_min_width with auto_fast_width so the two edges collapse onto one threshold):
    - n_features < auto_fast_width                    -> "model"      (faithful single-shot fit)
    - n_features >= auto_fast_width (== tsmw default) -> "two_stage"  (cheap funnel + capped booster).
    The dormant ``"fast_model"`` fallback for the case where a tuning cache override raises tsmw above
    afw is covered by ``test_resolve_auto_falls_back_to_fast_model_when_two_stage_gated_above_auto_fast``.
    GPU branch is verified separately to keep this test deterministic."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    afw = PF._auto_fast_width()
    tsmw = PF._two_stage_min_width()
    assert tsmw <= afw, (
        "default tsmw must not exceed afw -- iter21 unified them; if you raise tsmw via kernel_tuning_cache, the fast_model fallback test covers that branch."
    )

    # Below auto_fast_width -> "model"
    assert PF.resolve_prefilter_method("auto", n_features=afw - 1, n_rows=4000) == "model"
    # At/above auto_fast_width -> "two_stage" (default tsmw == afw, so the same width crosses both)
    assert PF.resolve_prefilter_method("auto", n_features=afw, n_rows=4000) == "two_stage"
    assert PF.resolve_prefilter_method("auto", n_features=afw + 1000, n_rows=4000) == "two_stage"


def test_resolve_auto_gpu_wins_over_two_stage_when_device_available(monkeypatch):
    """When the row count clears the GPU crossover AND a CUDA device is enumerable, gpu_model
    outranks two_stage (faithful full ranking on the GPU beats the funnel on big-n hardware)."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: True)
    wide = PF._two_stage_min_width() + 5000
    big_n = PF._gpu_model_min_rows()
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n) == "gpu_model"
    # Below GPU min rows -> two_stage (not fast_model) because the width clears the two-stage threshold.
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n - 1) == "two_stage"


def test_two_stage_returns_original_indices_and_honors_prefilter_top():
    """``working_cols`` must be in ORIGINAL positional space (mapped back through stage A's keep set);
    the kept count is ``min(prefilter_top, stage1_keep)`` and ``stage1_kept`` reflects stage A's funnel."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    # Width 600 - tiny, fast enough for the unit suite (not exercising the >=8000 auto threshold but
    # we call two_stage EXPLICITLY here so routing is bypassed; the contract test is independent of width).
    X, y = _wide_xy(seed=2, width=600, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    working_cols, info = prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=50, classification=True, n_features=X.shape[1], n_estimators_cap=80, stage1_keep=200
    )

    # sorted, in range, mapped to ORIGINAL positional indices.
    assert working_cols.ndim == 1 and len(working_cols) == info["kept"] == 50
    assert list(working_cols) == sorted(set(int(c) for c in working_cols))
    assert working_cols.min() >= 0 and working_cols.max() < X.shape[1]
    # Stage A bookkeeping surfaced for the report.
    assert info["method"] == "two_stage"
    assert info["of"] == X.shape[1]
    assert info["stage1_kept"] == 200 and info["stage1_of"] == X.shape[1]
    assert info["stage_a_seconds"] >= 0.0 and info["stage_b_seconds"] >= 0.0
    assert info["n_estimators_cap"] == 80
    # All 5 informatives (original indices 0..4) survive on a main-effect target with planted signal.
    kept = set(int(c) for c in working_cols)
    assert set(range(5)) <= kept, f"lost informatives, kept head={sorted(kept)[:8]}"


def test_two_stage_stage1_keep_defaults_to_min_2000_or_20pct(monkeypatch):
    """When ``stage1_keep`` is None, the prefilter computes ``min(2000, 0.2*n_features)``. Probe the
    info dict for both regimes (0.2 cap binds vs the 2000 ceiling binds)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns, _default_stage1_keep

    # Direct unit on the helper -- avoids paying for a stage-B fit on huge widths.
    assert _default_stage1_keep(100) == 20  # 0.2 binds (20 < 2000)
    assert _default_stage1_keep(8000) == 1600  # 0.2 binds (1600 < 2000)
    assert _default_stage1_keep(20000) == 2000  # 2000 ceiling binds (4000 > 2000)
    assert _default_stage1_keep(0) == 1  # degenerate guard

    # End-to-end at small width: default kicks in, recorded in info.
    X, y = _wide_xy(seed=3, width=300, n_informative=4)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=80)
    _, info = prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=25, classification=True, n_features=X.shape[1], n_estimators_cap=80, stage1_keep=None
    )
    assert info["stage1_kept"] == _default_stage1_keep(X.shape[1])


def test_two_stage_recovery_matches_single_stage_on_main_effect_target():
    """Recovery contract: on a main-effect-driven target (the regime where two_stage's stage A is
    sound) the two-stage funnel recovers at least as many planted informatives as the single-stage
    ``"model"`` path. Both methods are run on the SAME small fixture; this is the unit-level
    recovery guard. The 6k biz_value test exercises the same property under the production regime."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(seed=4, width=400, n_informative=5)
    # Force single-thread fits: xgboost's hist tree builds histograms with a thread-count-dependent
    # reduction order, so under CPU contention the multi-thread importance ranking can drift enough
    # to flip recovery past the 1-feature slack -- a scheduler artefact, not a recovery-contract
    # change. n_jobs=1 is deterministic given random_state, so the comparison tests the funnel, not
    # the thread pool. (Same robustness motive as the best-of-N timing on the speed gates.)
    model_a = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    model_b = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    model_a.set_params(n_jobs=1)
    model_b.set_params(n_jobs=1)
    keep = 40
    informative = set(range(5))

    single, _ = prefilter_columns(
        model_a, X, y.astype(np.float64), method="model", prefilter_top=keep, classification=True, n_features=X.shape[1], n_estimators_cap=80
    )
    two_stage, _ = prefilter_columns(
        model_b,
        X,
        y.astype(np.float64),
        method="two_stage",
        prefilter_top=keep,
        classification=True,
        n_features=X.shape[1],
        n_estimators_cap=80,
        stage1_keep=120,
    )
    rec_single = len(informative & set(map(int, single)))
    rec_two = len(informative & set(map(int, two_stage)))
    # 1-feature slack: same contract as the biz_value test.
    assert rec_two >= rec_single - 1, f"two_stage recovery {rec_two}/5 < single-stage {rec_single}/5 - 1 slack"


def test_prefilter_cap_does_not_increase_fast_model_budget():
    """``fast_model`` already reduces the budget (template / 4 with a floor of 50). If the user passes
    a LARGER cap than fast_model's own reduced budget, the cap must NOT push it back up -- the
    fast_model path keeps its own (smaller) budget. Enforced via ``min(current, cap)`` in the ranker."""
    from sklearn.base import clone

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _unwrap_estimator, make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import _rank_fast_model

    X, y = _wide_xy(width=120, n_informative=5)
    # Template default is 300 -> fast_model would set ~75 (300//4). Cap at 200 must NOT raise it.
    model = make_default_estimator(classification=True, random_state=0, n_estimators=300)
    # Instrument by inspecting the cloned estimator's n_estimators after fast_model's set_params chain.
    # Drive _rank_fast_model with a HUGE cap and re-derive the budget it would have used.
    captured = {}

    def _spy_fit(self, *a, **kw):  # type: ignore[no-redef]
        captured["n_estimators"] = self.get_params().get("n_estimators")
        return orig_fit(self, *a, **kw)

    pf = clone(model)
    orig_fit = type(_unwrap_estimator(pf)).fit
    # Easier path: call the ranker and just read what got set via a small probe model.
    import xgboost as xgb

    class _ProbeXGB(xgb.XGBClassifier):
        n_estimators_seen = None

        def fit(self, X, y, **kw):
            type(self).n_estimators_seen = int(self.get_params()["n_estimators"])
            return super().fit(X, y, **kw)

    probe = _ProbeXGB(n_estimators=300, random_state=0, verbosity=0, tree_method="hist")
    _rank_fast_model(probe, X, y.astype(np.float64), n_features=X.shape[1], n_estimators_cap=200)
    # fast_model would compute max(50, 300//4) = 75; cap=200 cannot raise it above 75.
    assert _ProbeXGB.n_estimators_seen == 75, f"fast_model n_estimators wrongly raised by cap: got {_ProbeXGB.n_estimators_seen}, expected 75"

    # And when the cap is SMALLER than fast_model's own reduced budget, the cap DOES bind.
    _ProbeXGB.n_estimators_seen = None
    probe2 = _ProbeXGB(n_estimators=300, random_state=0, verbosity=0, tree_method="hist")
    _rank_fast_model(probe2, X, y.astype(np.float64), n_features=X.shape[1], n_estimators_cap=40)
    assert _ProbeXGB.n_estimators_seen == 40


# ------------------------- shared stage-A cohort accessors (iter13)
def test_two_stage_report_exposes_stage1_survivors_and_f_scores():
    """``two_stage`` must publish two NEW fields on the info dict so downstream stages can read the
    cheap marginal-strength ranking WITHOUT a second f_classif / f_regression pass:

      * ``stage1_survivors`` : sorted positional indices of the stage-A cohort (a SUPERSET of the
        final ``working_cols``; length == ``stage1_kept``).
      * ``stage1_f_scores``  : dense length-n_features ANOVA F-score vector (-inf for constant cols).

    Plus the stable public accessors ``get_cached_f_scores`` / ``get_stage1_survivors`` must read the
    same data back from the report (the future SHAP-on-stage-A / trust-guard / clustering consumers
    must not have to dig into the report's internal key names)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import (
        get_cached_f_scores,
        get_stage1_survivors,
        prefilter_columns,
    )

    X, y = _wide_xy(seed=5, width=600, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    working_cols, info = prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=50, classification=True, n_features=X.shape[1], n_estimators_cap=80, stage1_keep=180
    )

    # New fields present.
    assert "stage1_survivors" in info and "stage1_f_scores" in info
    survivors = info["stage1_survivors"]
    fscores = info["stage1_f_scores"]
    # stage1_survivors: sorted, unique, in-range, length matches the funnel count.
    assert survivors.ndim == 1 and len(survivors) == info["stage1_kept"] == 180
    assert list(survivors) == sorted(set(int(c) for c in survivors))
    assert int(survivors.min()) >= 0 and int(survivors.max()) < X.shape[1]
    # working_cols is a SUBSET of stage1_survivors (stage B narrows further); recovery invariant.
    assert set(int(c) for c in working_cols) <= set(int(c) for c in survivors), "working_cols escaped the stage-A cohort"
    # F-scores: dense length-n_features, finite for non-constant columns, sentinel handled.
    assert fscores.ndim == 1 and fscores.shape[0] == X.shape[1]
    assert np.isfinite(fscores).sum() >= X.shape[1] - 5, "too many non-finite F-scores on a clean synthetic"
    # The stage-A cohort is precisely the top-stage1_keep by F-score (a contract anchor for any
    # downstream consumer that wants to RE-rank the stage-A cohort by a different scorer).
    top_by_f = np.sort(np.argsort(-fscores, kind="stable")[: info["stage1_kept"]])
    np.testing.assert_array_equal(survivors, top_by_f)

    # Public accessors round-trip the same arrays (no key-name coupling for callers).
    np.testing.assert_array_equal(get_stage1_survivors(info), survivors)
    np.testing.assert_array_equal(get_cached_f_scores(info), fscores)
    # None-tolerant: missing dict / wrong key returns None instead of raising.
    assert get_cached_f_scores(None) is None
    assert get_stage1_survivors(None) is None
    assert get_cached_f_scores({}) is None
    assert get_stage1_survivors({}) is None


def test_univariate_report_exposes_f_scores_for_downstream_reuse():
    """``univariate`` already computes the same F-score vector as ``two_stage``'s stage A. The two
    paths must agree on the cached-scores contract so a downstream consumer that reads
    ``get_cached_f_scores`` does NOT have to special-case the method. ``stage1_survivors`` is
    two-stage-only by design (univariate has no funnel)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import (
        get_cached_f_scores,
        get_stage1_survivors,
        prefilter_columns,
    )

    X, y = _wide_xy(seed=6, width=400, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=80)
    working_cols, info = prefilter_columns(model, X, y.astype(np.float64), method="univariate", prefilter_top=40, classification=True, n_features=X.shape[1])

    fscores = get_cached_f_scores(info)
    assert fscores is not None and fscores.shape[0] == X.shape[1]
    # The kept set is precisely the top-K by F-score (univariate's own ranking).
    top_by_f = np.sort(np.argsort(-fscores, kind="stable")[: info["kept"]])
    np.testing.assert_array_equal(working_cols, top_by_f)
    # univariate has no stage-A funnel -> no stage1_survivors.
    assert get_stage1_survivors(info) is None


def test_biz_value_cached_f_scores_avoid_recomputation():
    """biz_value (iter13): the cached F-scores from a ``two_stage`` prefilter let downstream stages
    skip a duplicate ``f_classif`` pass. On a moderately wide synthetic (width=4000, n=2000) this
    measures the cheap-now win: reading the cached vector is ~100x+ faster than recomputing the same
    F-scores from scratch. The bound is loose (10x) to absorb dev-HW jitter; the actual gap on the dev
    machine is much larger because the cache lookup is a dict-get vs sklearn's full O(n*f) scorer."""
    import time as _time
    from sklearn.feature_selection import f_classif

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import (
        get_cached_f_scores,
        prefilter_columns,
    )

    rng = np.random.default_rng(11)
    n, width, n_inf = 2000, 4000, 8
    inf = rng.normal(size=(n, n_inf)).astype(np.float32)
    noise = rng.normal(size=(n, width - n_inf)).astype(np.float32)
    X = pd.DataFrame(np.column_stack([inf, noise]), columns=[f"f{i}" for i in range(width)])
    logit = (inf * np.linspace(1.2, 0.4, n_inf)).sum(axis=1)
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(np.float64)

    model = make_default_estimator(classification=True, random_state=0, n_estimators=80)
    # Run two_stage (the path that caches F-scores) -- inside it sklearn runs f_classif once.
    _, info = prefilter_columns(model, X, y, method="two_stage", prefilter_top=200, classification=True, n_features=width, n_estimators_cap=80, stage1_keep=800)

    # 1) Cached path: just read the published vector (free).
    t0 = _time.perf_counter()
    cached = get_cached_f_scores(info)
    t_cached = _time.perf_counter() - t0
    assert cached is not None and cached.shape[0] == width

    # 2) Naive path a downstream stage would have used pre-iter13: recompute f_classif from scratch.
    Xv = X.values
    t0 = _time.perf_counter()
    fresh, _ = f_classif(Xv, y)
    t_fresh = _time.perf_counter() - t0
    fresh = np.asarray(fresh, dtype=np.float64)
    fresh[~np.isfinite(fresh)] = -np.inf

    # Cached vector matches a fresh recomputation (same scorer, same -inf sentinel).
    np.testing.assert_allclose(cached, fresh, rtol=1e-6, atol=1e-6, err_msg="cached F-scores diverged from a fresh f_classif call")
    # Cheap-now win: the cache hit is at least 10x faster than the recomputation. Print the actual
    # numbers so future iterations can see the win evolve under load (-s captures stdout).
    print(
        f"[iter13 f_cache] cached={t_cached * 1e3:.3f}ms recompute={t_fresh * 1e3:.3f}ms speedup={t_fresh / max(t_cached, 1e-9):.1f}x width={width} n={n}",
        flush=True,
    )
    assert t_fresh > t_cached * 10, (
        f"cached F-score lookup is not measurably faster than recompute: cached={t_cached * 1e3:.2f}ms vs recompute={t_fresh * 1e3:.2f}ms"
    )


@pytest.mark.parametrize("method", ["model", "fast_model"])
def test_booster_prefilter_methods_do_not_publish_f_scores(method):
    """``model`` / ``fast_model`` rank by booster ``feature_importances_``, NOT F-scores -- so the
    cached-scores accessor must return None for them (no silent reuse of stale / absent data).
    Future "I want marginal strength" consumers fall back to recomputing f_classif themselves when
    the prefilter took the booster path; the explicit None signals that."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import (
        get_cached_f_scores,
        get_stage1_survivors,
        prefilter_columns,
    )

    X, y = _wide_xy(seed=7, width=300, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=80)
    _, info = prefilter_columns(
        model, X, y.astype(np.float64), method=method, prefilter_top=30, classification=True, n_features=X.shape[1], n_estimators_cap=60
    )
    assert get_cached_f_scores(info) is None
    assert get_stage1_survivors(info) is None


# ----------------------------------------------------------- stage-B GPU dispatch (iter47)
def test_gpu_model_available_requires_xgboost_use_cuda_build(monkeypatch):
    """An enumerable cupy device is necessary but NOT sufficient: when xgboost was built without
    USE_CUDA (the default pip wheel on many platforms), passing ``device="cuda"`` silently downgrades
    to CPU. ``gpu_model_available`` must return False in that case so routers don't drop work onto a
    non-existent GPU path."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    PF.reset_gpu_model_available_cache()

    # Stub the cupy device probe to succeed and toggle xgboost build_info.
    class _CudaRT:
        @staticmethod
        def getDeviceCount():
            return 1

    class _FakeCupy:
        cuda = type("cuda", (), {"runtime": _CudaRT})

    import sys

    monkeypatch.setitem(sys.modules, "cupy", _FakeCupy)

    import xgboost as xgb

    orig_build_info = xgb.build_info
    monkeypatch.setattr(xgb, "build_info", lambda: {**orig_build_info(), "USE_CUDA": False})
    PF.reset_gpu_model_available_cache()
    assert PF.gpu_model_available() is False

    monkeypatch.setattr(xgb, "build_info", lambda: {**orig_build_info(), "USE_CUDA": True})
    PF.reset_gpu_model_available_cache()
    assert PF.gpu_model_available() is True


def test_gpu_model_available_caches_result(monkeypatch):
    """The probe is per-process invariant (build flag + device topology don't change at runtime), so
    repeated calls must hit the cache and skip the cupy / xgboost imports after the first call."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    PF.reset_gpu_model_available_cache()
    # First call materializes the cache; subsequent calls must not re-probe.
    first = PF.gpu_model_available()
    calls = {"n": 0}

    def _exploding_build_info():
        calls["n"] += 1
        raise RuntimeError("build_info should not be re-called once cached")

    import xgboost as xgb

    monkeypatch.setattr(xgb, "build_info", _exploding_build_info)
    # Cache hit: no probe, returns the same answer.
    assert PF.gpu_model_available() == first
    assert calls["n"] == 0


def test_stage_b_should_route_gpu_gates_on_rows_and_features(monkeypatch):
    """Gate must require ALL of (a) gpu_model_available True, (b) n_rows >= floor, (c) n_features_b >= floor.
    Each individual miss leaves the booster on CPU -- the small-problem GPU upload would be slower than
    the CPU fit, and a missing xgboost-CUDA build makes ``device="cuda"`` a silent no-op."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: True)
    n_rows_ok = PF._stage_b_gpu_min_rows()
    n_feat_ok = PF._stage_b_gpu_min_features()

    # All three gates pass.
    assert PF._stage_b_should_route_gpu(n_rows=n_rows_ok, n_features_b=n_feat_ok) is True
    # Each gate alone blocks.
    assert PF._stage_b_should_route_gpu(n_rows=n_rows_ok - 1, n_features_b=n_feat_ok) is False
    assert PF._stage_b_should_route_gpu(n_rows=n_rows_ok, n_features_b=n_feat_ok - 1) is False
    # No GPU available -> always False regardless of size.
    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    assert PF._stage_b_should_route_gpu(n_rows=n_rows_ok * 10, n_features_b=n_feat_ok * 10) is False


def test_two_stage_calls_gpu_path_when_gate_fires(monkeypatch):
    """When the stage-B gate fires, ``_rank_two_stage`` MUST route through ``_rank_gpu_model`` (not
    ``_rank_model``); when it does not fire, the legacy ``_rank_model`` path is taken. This is the
    wiring contract -- recovery / ranking parity is covered by the existing two_stage recovery test."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import make_default_estimator

    X, y = _wide_xy(seed=11, width=600, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=80)

    calls = {"cpu": 0, "gpu": 0}
    real_cpu = PF._rank_model

    def _wrap_cpu(*a, **kw):
        calls["cpu"] += 1
        return real_cpu(*a, **kw)

    def _wrap_gpu(*a, **kw):
        # Substitute the CPU path so the test doesn't need a real GPU build (the wiring contract is
        # which function gets called, not how it runs). Counts a gpu invocation.
        calls["gpu"] += 1
        return real_cpu(*a, **kw)

    monkeypatch.setattr(PF, "_rank_model", _wrap_cpu)
    monkeypatch.setattr(PF, "_rank_gpu_model", _wrap_gpu)

    # Gate FORCED ON -> stage-B routes to gpu wrapper.
    monkeypatch.setattr(PF, "_stage_b_should_route_gpu", lambda **_: True)
    _, info_gpu = PF.prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=30, classification=True, n_features=X.shape[1], n_estimators_cap=40, stage1_keep=120
    )
    assert info_gpu["stage_b_routed_gpu"] is True
    assert calls["gpu"] == 1 and calls["cpu"] == 0

    # Gate FORCED OFF -> stage-B stays on CPU model path.
    calls["cpu"] = 0
    calls["gpu"] = 0
    monkeypatch.setattr(PF, "_stage_b_should_route_gpu", lambda **_: False)
    _, info_cpu = PF.prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=30, classification=True, n_features=X.shape[1], n_estimators_cap=40, stage1_keep=120
    )
    assert info_cpu["stage_b_routed_gpu"] is False
    assert calls["cpu"] == 1 and calls["gpu"] == 0


def test_stage_b_gpu_thresholds_overridable_via_kernel_tuning_cache(monkeypatch):
    """Both stage-B GPU thresholds (rows + features) must be overridable via the shared
    ``kernel_tuning_cache`` entry under ``shap_proxy_prefilter``, NOT hardcoded. This is the per-HW
    tuning contract; the cache lookup is the single point where users override conservative defaults."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter as PF

    # Override the cheap tuning helper with a synthetic dict; this is the only public hook on the
    # prefilter side and the same pattern existing tests use (e.g. auto_fast_width override).
    monkeypatch.setattr(PF, "_prefilter_tuning", lambda: {"stage_b_gpu_min_rows": 1234, "stage_b_gpu_min_features": 77})
    assert PF._stage_b_gpu_min_rows() == 1234
    assert PF._stage_b_gpu_min_features() == 77

    monkeypatch.setattr(PF, "_prefilter_tuning", lambda: {})
    # Defaults restored when no entry.
    assert PF._stage_b_gpu_min_rows() == PF._STAGE_B_GPU_MIN_ROWS
    assert PF._stage_b_gpu_min_features() == PF._STAGE_B_GPU_MIN_FEATURES
