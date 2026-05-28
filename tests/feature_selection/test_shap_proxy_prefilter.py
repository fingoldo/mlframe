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
    from mlframe.feature_selection._shap_proxy_prefilter import PREFILTER_METHODS, resolve_prefilter_method

    for m in PREFILTER_METHODS:
        assert resolve_prefilter_method(m, n_features=10000, n_rows=4000) == m


def test_resolve_unknown_method_raises():
    from mlframe.feature_selection._shap_proxy_prefilter import resolve_prefilter_method

    with pytest.raises(ValueError):
        resolve_prefilter_method("not_a_method", n_features=100, n_rows=100)


def test_resolve_auto_keeps_model_for_moderate_width():
    from mlframe.feature_selection._shap_proxy_prefilter import _auto_fast_width, resolve_prefilter_method

    narrow = _auto_fast_width() - 1
    assert resolve_prefilter_method("auto", n_features=narrow, n_rows=4000) == "model"


def test_resolve_auto_switches_fast_for_very_wide_when_no_gpu(monkeypatch):
    import mlframe.feature_selection._shap_proxy_prefilter as PF

    # No CUDA -> auto must pick fast_model in the mid-wide window (between auto_fast_width and
    # two_stage_min_width). Above two_stage_min_width the new iter12 routing takes over -- that edge
    # is covered by test_resolve_auto_routes_two_stage_at_wide_threshold_no_gpu.
    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    mid_wide = (PF._auto_fast_width() + PF._two_stage_min_width()) // 2
    assert PF._auto_fast_width() <= mid_wide < PF._two_stage_min_width()
    assert PF.resolve_prefilter_method("auto", n_features=mid_wide, n_rows=4000) == "fast_model"
    # Even with many rows, no device means no gpu_model in that window.
    assert PF.resolve_prefilter_method("auto", n_features=mid_wide, n_rows=10 ** 6) == "fast_model"


def test_resolve_auto_routes_gpu_when_device_and_enough_rows(monkeypatch):
    import mlframe.feature_selection._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: True)
    # Pick a width in the mid-wide window (>= auto_fast_width, < two_stage_min_width) so the GPU
    # vs fast_model crossover is decided by row count alone; the two_stage edge (width >= 8000) is
    # covered by test_resolve_auto_gpu_wins_over_two_stage_when_device_available.
    mid_wide = (PF._auto_fast_width() + PF._two_stage_min_width()) // 2
    assert PF._auto_fast_width() <= mid_wide < PF._two_stage_min_width()
    big_n = PF._gpu_model_min_rows()
    assert PF.resolve_prefilter_method("auto", n_features=mid_wide, n_rows=big_n) == "gpu_model"
    # Too few rows -> GPU transfer overhead not worth it -> fast_model even with a device.
    assert PF.resolve_prefilter_method("auto", n_features=mid_wide, n_rows=big_n - 1) == "fast_model"


# --------------------------------------------------------------------------- ranking correctness
def _wide_xy(seed=0, width=300, n_informative=5):
    """A few strong informatives + lots of independent noise (no clustering needed here)."""
    rng = np.random.default_rng(seed)
    n = 1200
    inf = rng.normal(size=(n, n_informative))
    noise = rng.normal(size=(n, width - n_informative))
    X = pd.DataFrame(np.column_stack([inf, noise]),
                     columns=[f"inf{i}" for i in range(n_informative)]
                     + [f"noise{i}" for i in range(width - n_informative)])
    coefs = np.array([0.9, 0.8, -0.7, 0.6, 0.4])[:n_informative]
    logit = (inf * coefs).sum(axis=1)
    y = (logit + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("method", ["model", "univariate", "fast_model"])
def test_prefilter_keeps_informatives_and_returns_sorted_original_indices(method):
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(width=300, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    keep_k = 40
    working_cols, info = prefilter_columns(
        model, X, y.astype(np.float64), method=method, prefilter_top=keep_k,
        classification=True, n_features=X.shape[1])

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
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    class _NoImportance:
        def get_params(self, deep=False):
            return {}

        def fit(self, X, y):
            return self

    X, y = _wide_xy(width=50)
    working_cols, info = prefilter_columns(
        _NoImportance(), X, y.astype(np.float64), method="model", prefilter_top=10,
        classification=True, n_features=X.shape[1])
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
        classification=True, metric="brier", optimizer="auto", prefilter_top=60,
        prefilter_method=method, cluster_features=True, cluster_corr_threshold=0.7,
        top_n=8, n_splits=3, n_revalidation_models=1, trust_guard=False,
        run_importance_ablation=False, random_state=0, verbose=False)
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
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    # Clean separable synthetic: enough rows + low noise so the importances escape the tail-tie regime.
    rng = np.random.default_rng(0)
    n, width, n_inf = 3000, 200, 8
    inf = rng.normal(size=(n, n_inf))
    noise = rng.normal(size=(n, width - n_inf))
    X = pd.DataFrame(np.column_stack([inf, noise]),
                     columns=[f"inf{i}" for i in range(n_inf)]
                     + [f"noise{i}" for i in range(width - n_inf)])
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
            model_full, X, y, method=method, prefilter_top=signal_k, classification=True,
            n_features=X.shape[1], n_estimators_cap=None)
        capped, info = prefilter_columns(
            model_cap, X, y, method=method, prefilter_top=signal_k, classification=True,
            n_features=X.shape[1], n_estimators_cap=100)
        a, b = set(map(int, uncapped)), set(map(int, capped))
        jaccard = len(a & b) / len(a | b)
        assert jaccard >= 0.95, f"{method}: top-{signal_k} Jaccard {jaccard:.3f} < 0.95 (capped vs uncapped)"

        # Production-K view: at the user-facing prefilter_top, the cap MUST keep all informatives
        # (the prefilter's decision-relevant output -- it doesn't need rank-stability among noise).
        capped_prod, info_prod = prefilter_columns(
            make_default_estimator(classification=True, random_state=0, n_estimators=300),
            X, y, method=method, prefilter_top=prod_k, classification=True,
            n_features=X.shape[1], n_estimators_cap=100)
        kept_prod = set(map(int, capped_prod))
        assert set(range(n_inf)) <= kept_prod, (
            f"{method}: capped prefilter lost informatives at prod K={prod_k}, "
            f"got head={sorted(kept_prod)[:12]}")
        # Info dict surfaces the applied cap so downstream report consumers can see what ran.
        assert info["n_estimators_cap"] == 100
        assert info_prod["n_estimators_cap"] == 100


def test_prefilter_cap_none_is_legacy_uncapped_behaviour():
    """``n_estimators_cap=None`` must preserve the legacy path: the cloned ranking booster is fit at
    the template's own ``n_estimators`` (300 by default). Regression guard so the new knob is a pure
    extension."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(width=150, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=300)
    _, info = prefilter_columns(
        model, X, y.astype(np.float64), method="model", prefilter_top=30,
        classification=True, n_features=X.shape[1], n_estimators_cap=None)
    assert info["n_estimators_cap"] is None


# ----------------------------------------------------------- two_stage prefilter (iter12)
def test_resolve_auto_routes_two_stage_at_wide_threshold_no_gpu(monkeypatch):
    """auto routing edges around ``two_stage_min_width`` when no CUDA is available:
    - n_features < auto_fast_width -> "model" (legacy)
    - auto_fast_width <= n_features < two_stage_min_width -> "fast_model" (legacy)
    - n_features >= two_stage_min_width -> "two_stage" (new path).
    GPU branch is verified separately to keep this test deterministic."""
    import mlframe.feature_selection._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: False)
    afw = PF._auto_fast_width()
    tsmw = PF._two_stage_min_width()
    assert tsmw > afw, "two_stage threshold must be wider than the fast_model crossover by design"

    # Below auto_fast_width -> "model"
    assert PF.resolve_prefilter_method("auto", n_features=afw - 1, n_rows=4000) == "model"
    # Between auto_fast_width and two_stage_min_width -> "fast_model"
    assert PF.resolve_prefilter_method("auto", n_features=tsmw - 1, n_rows=4000) == "fast_model"
    # At/above two_stage_min_width -> "two_stage"
    assert PF.resolve_prefilter_method("auto", n_features=tsmw, n_rows=4000) == "two_stage"
    assert PF.resolve_prefilter_method("auto", n_features=tsmw + 1000, n_rows=4000) == "two_stage"


def test_resolve_auto_gpu_wins_over_two_stage_when_device_available(monkeypatch):
    """When the row count clears the GPU crossover AND a CUDA device is enumerable, gpu_model
    outranks two_stage (faithful full ranking on the GPU beats the funnel on big-n hardware)."""
    import mlframe.feature_selection._shap_proxy_prefilter as PF

    monkeypatch.setattr(PF, "gpu_model_available", lambda: True)
    wide = PF._two_stage_min_width() + 5000
    big_n = PF._gpu_model_min_rows()
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n) == "gpu_model"
    # Below GPU min rows -> two_stage (not fast_model) because the width clears the two-stage threshold.
    assert PF.resolve_prefilter_method("auto", n_features=wide, n_rows=big_n - 1) == "two_stage"


def test_two_stage_returns_original_indices_and_honors_prefilter_top():
    """``working_cols`` must be in ORIGINAL positional space (mapped back through stage A's keep set);
    the kept count is ``min(prefilter_top, stage1_keep)`` and ``stage1_kept`` reflects stage A's funnel."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    # Width 600 - tiny, fast enough for the unit suite (not exercising the >=8000 auto threshold but
    # we call two_stage EXPLICITLY here so routing is bypassed; the contract test is independent of width).
    X, y = _wide_xy(seed=2, width=600, n_informative=5)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    working_cols, info = prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=50,
        classification=True, n_features=X.shape[1], n_estimators_cap=80, stage1_keep=200)

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
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns, _default_stage1_keep

    # Direct unit on the helper -- avoids paying for a stage-B fit on huge widths.
    assert _default_stage1_keep(100) == 20       # 0.2 binds (20 < 2000)
    assert _default_stage1_keep(8000) == 1600    # 0.2 binds (1600 < 2000)
    assert _default_stage1_keep(20000) == 2000   # 2000 ceiling binds (4000 > 2000)
    assert _default_stage1_keep(0) == 1          # degenerate guard

    # End-to-end at small width: default kicks in, recorded in info.
    X, y = _wide_xy(seed=3, width=300, n_informative=4)
    model = make_default_estimator(classification=True, random_state=0, n_estimators=80)
    _, info = prefilter_columns(
        model, X, y.astype(np.float64), method="two_stage", prefilter_top=25,
        classification=True, n_features=X.shape[1], n_estimators_cap=80, stage1_keep=None)
    assert info["stage1_kept"] == _default_stage1_keep(X.shape[1])


def test_two_stage_recovery_matches_single_stage_on_main_effect_target():
    """Recovery contract: on a main-effect-driven target (the regime where two_stage's stage A is
    sound) the two-stage funnel recovers at least as many planted informatives as the single-stage
    ``"model"`` path. Both methods are run on the SAME small fixture; this is the unit-level
    recovery guard. The 6k biz_value test exercises the same property under the production regime."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import prefilter_columns

    X, y = _wide_xy(seed=4, width=400, n_informative=5)
    model_a = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    model_b = make_default_estimator(classification=True, random_state=0, n_estimators=120)
    keep = 40
    informative = set(range(5))

    single, _ = prefilter_columns(
        model_a, X, y.astype(np.float64), method="model", prefilter_top=keep,
        classification=True, n_features=X.shape[1], n_estimators_cap=80)
    two_stage, _ = prefilter_columns(
        model_b, X, y.astype(np.float64), method="two_stage", prefilter_top=keep,
        classification=True, n_features=X.shape[1], n_estimators_cap=80, stage1_keep=120)
    rec_single = len(informative & set(map(int, single)))
    rec_two = len(informative & set(map(int, two_stage)))
    # 1-feature slack: same contract as the biz_value test.
    assert rec_two >= rec_single - 1, (
        f"two_stage recovery {rec_two}/5 < single-stage {rec_single}/5 - 1 slack")


def test_prefilter_cap_does_not_increase_fast_model_budget():
    """``fast_model`` already reduces the budget (template / 4 with a floor of 50). If the user passes
    a LARGER cap than fast_model's own reduced budget, the cap must NOT push it back up -- the
    fast_model path keeps its own (smaller) budget. Enforced via ``min(current, cap)`` in the ranker."""
    from sklearn.base import clone

    from mlframe.feature_selection._shap_proxy_explain import _unwrap_estimator, make_default_estimator
    from mlframe.feature_selection._shap_proxy_prefilter import _rank_fast_model

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
    assert _ProbeXGB.n_estimators_seen == 75, (
        f"fast_model n_estimators wrongly raised by cap: got {_ProbeXGB.n_estimators_seen}, expected 75")

    # And when the cap is SMALLER than fast_model's own reduced budget, the cap DOES bind.
    _ProbeXGB.n_estimators_seen = None
    probe2 = _ProbeXGB(n_estimators=300, random_state=0, verbosity=0, tree_method="hist")
    _rank_fast_model(probe2, X, y.astype(np.float64), n_features=X.shape[1], n_estimators_cap=40)
    assert _ProbeXGB.n_estimators_seen == 40
