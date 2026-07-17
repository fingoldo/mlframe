"""Regression sensors pinning the composite-all-FE-on UNDER-SELECTION bugs (root-caused 2026-06-05).

Investigation summary (7-step ablation on the Layer-101 mega fixture):
  * With FE OFF, the core greedy screen is HEALTHY -- every seed selects ``[group_id, x1, x2, x3]`` and the
    binned MI ranks the signals correctly (x1 = 0.033-0.057 >> noise = 0.001). ``test_core_screen_*`` pins this
    (a POSITIVE sensor: if a future change breaks the core screen, it fails).
  * With the full ``_all_on_kwargs`` FE family set ON, selection degrades badly via plug-in-MI bias on the wide
    engineered candidate pool (the same bias class as Layer-15 + the FE accuracy gate):
      1. the empty-support ``min_features_fallback=1`` top-1-raw fallback does NOT fire when engineered features
         exist (``_mrmr_fit_impl.py`` ``if selected_vars or n_engineered_out``), so on some seeds support is EMPTY
         (0 raw features) despite recoverable signal (oracle AUC ~0.77);
      2. the maxT FDR floor over-rejects on the wide pool;
      3. raw x1 (the STRONGEST signal) is out-ranked by overfit-in-sample-MI engineered / high-card columns and
         dropped, and a 50-level noise categorical (``cat_b``) is sometimes selected over real signal.

These sensors assert the CORRECT contract. Fix A1 (fallback fires on 0-raw), Fix A2 (floor never
empties), and Fix B (wide-pool MI debiasing) have since landed -- all sensors in this file now pass
(2026-07-10: confirmed 22/22 green after fixing an unrelated ``_support_names`` helper bug that had been
masking the real assertions behind a spurious ``ValueError`` on every run, including this file's own
POSITIVE sensor). They were never xfail while the bugs were open -- they surfaced real production bugs
and stayed red on purpose until fixed; kept as plain (non-xfail) regression sensors now that they pass,
so a future regression in any of Fix A1/A2/B fails loudly again.
"""

import importlib.util
import os

import numpy as np
import pytest

_L101_PATH = os.path.join(os.path.dirname(__file__), "test_biz_value_mrmr_regression_union", "test_full_suite_regression.py")
_spec = importlib.util.spec_from_file_location("_l101_underselect_helpers", _L101_PATH)
_l101 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_l101)
_build_mega = _l101._build_mega
_make_mega_mrmr = _l101._make_mega_mrmr

# Seeds observed to collapse (empty support / x1 dropped / noise selected) under the all-on config.
COLLAPSE_SEEDS = [0, 7, 13, 42]


def _support_names(m):
    """Check support names."""
    _ni = getattr(m, "feature_names_in_", None)
    ni = list(_ni) if _ni is not None else []
    return [ni[i] if i < len(ni) else f"idx{i}" for i in np.asarray(m.support_).ravel()]


def _sources(names):
    """Raw source token of each selected column (``x1__He2`` -> ``x1``; raw ``x1`` -> ``x1``)."""
    return {str(n).split("__", 1)[0] for n in names}


@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_core_screen_selects_signal_with_fe_off(seed):
    """POSITIVE sensor: the core greedy screen, with FE disabled, recovers the planted signal."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _build_mega(seed)
    Xn = X[[c for c in X.columns if c not in ("cat_a", "cat_b", "ts")]]
    m = MRMR(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        quantization_nbins=10,
        random_seed=seed,
        fe_univariate_basis_enable=False,
        fe_univariate_fourier_enable=False,
        fe_hybrid_orth_enable=False,
    ).fit(Xn, y)
    names = _support_names(m)
    assert "x1" in names, f"seed={seed}: FE-off core screen dropped the strongest raw signal x1: {names}"
    assert sum(s in names for s in ("x1", "x2", "x3")) >= 2, f"seed={seed}: FE-off core screen missed signal: {names}"


@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_composite_fe_support_is_never_empty(seed):
    """Fix A1/A2: a feature selector must never return EMPTY support when signal is recoverable."""
    X, y = _build_mega(seed)
    m = _make_mega_mrmr(random_seed=seed).fit(X, y)
    n_raw = len(np.asarray(m.support_).ravel())
    assert n_raw >= 1, (
        f"seed={seed}: composite all-FE-on returned EMPTY raw support despite recoverable signal "
        f"(oracle x1/x2/x3 AUC ~0.77); the top-1-raw fallback must fire when 0 raw features are selected."
    )


@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_composite_fe_retains_strongest_signal(seed):
    """Fix B (IMPLEMENTED 2026-06-16): the strongest raw signal x1 must survive composite FE -- as raw x1 or an x1-derived column.

    FIXED by the raw-feature floor-drop protection (``_fit_impl_core`` near the hinge/orth protection blocks):
    the Westfall-Young maxT relevance floor, computed over the wide all-FE-on candidate pool, floored x1 out;
    rather than lower the floor (which would admit high-card raw noise like the 50-level cat_b), the protection
    re-adds a screen-dropped raw IFF it lifts a held-out linear fit over the already-selected design. seed=0 now
    selects [x1,x2,x3,group_id,cat_a,cat_a__te_std]; cat_b stays out; CASE1 clean recovery intact.

    Historical root-cause (kept for context):

    CONFIRMED seed=0 root-cause (2026-06-15, n=3000, nbins=10; instrumented, not hypothesised):
      * x1's binned MI = 0.0572 and is PRESENT in ``cached_MIs`` (singleton key) at fit time -- ~30x above noise
        (x4=0.0012). x1 is genuinely scorable, NOT weak and NOT missing from the screen.
      * there is NO feature-budget cap in play (max_features / n_features_to_select / max_runtime_mins all None,
        fallback_used_=False). So x1 was not crowded out by a cap -- it was REJECTED by the relevance floor.
      * the order-1 maxT acceptance floor (``_permutation_null.maxt_*`` order-1, returns the q-quantile of the
        per-shuffle MAX corrected marginal MI over ALL ``candidate_indices``) is computed over the FULL screening
        pool, which the all-FE-on config widens to ~hundreds of (mostly overfit) engineered columns. A wider pool
        raises the per-shuffle max, so the floor a genuine raw feature must clear rises ABOVE x1's true 0.0572 and
        x1 is dropped. This is Westfall-Young FWER control behaving as designed -- the defect is that overfit
        engineered candidates are admitted into the SAME family that judges raw-feature significance.
    Concrete fix recipe (for the dedicated, full-suite-validated effort): judge raw-feature acceptance against a
    maxT null whose candidate pool is the RAW (or pre-screened) feature set, not the engineered-inflated full pool
    -- i.e. give engineered candidates their own family rather than letting them widen the raw-feature null. A
    naive CMI-given-source redundancy gate on the kfold-TE column is the WRONG fix: TE's value is linear-usability
    of the same partition (CMI(te; y | cat) ~ 0 by construction), so gating on CMI would regress legitimate TE use.
    This change touches the core FWER null for EVERY selection -> high regression risk -> must be benchmarked across
    the whole layer suite before shipping. seeds 7/13/42 pass; seed=0 is the hard tail. Tracked, not masked (no xfail)."""
    X, y = _build_mega(seed)
    m = _make_mega_mrmr(random_seed=seed).fit(X, y)
    names = _support_names(m)
    assert "x1" in _sources(names), (
        f"seed={seed}: composite all-FE-on dropped the strongest signal x1 (neither raw nor x1-derived in "
        f"support); it was out-ranked by overfit-in-sample-MI engineered columns. support={names}"
    )


@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_composite_fe_no_highcard_noise_over_signal(seed):
    """Fix B: the 50-level pure-noise categorical cat_b must not be selected while real signal x1 is absent."""
    X, y = _build_mega(seed)
    m = _make_mega_mrmr(random_seed=seed).fit(X, y)
    names = _support_names(m)
    srcs = _sources(names)
    if "cat_b" in srcs:
        assert "x1" in srcs, (
            f"seed={seed}: high-card noise cat_b was selected while real signal x1 was dropped -- its 50-level "
            f"in-sample binned MI is finite-sample-inflated above genuine signal. support={names}"
        )


@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_composite_fe_fallback_rescues_real_signal_not_pure_noise(seed):
    """Index-space fix: the empty-support fallback ranks raw features by ``cached_MIs``, which is keyed in COLS-SPACE (categorize_dataset reorders columns when
    categoricals exist), NOT in ``feature_names_in_`` space. Pre-fix the fallback read ``cached_MIs[(input_idx,)]`` directly, mis-aligning every column once the
    screen reordered, so it rescued a pure-noise positive leg (``num_pos_a``, raw MI ~0) whose slot happened to carry a strong signal's MI. Pin that the rescued
    support carries genuine signal (one of the planted ``x*`` legs / group / cat effects) and never the known pure-noise legs while a real signal is absent."""
    X, y = _build_mega(seed)
    m = _make_mega_mrmr(random_seed=seed).fit(X, y)
    srcs = _sources(_support_names(m))
    real_signal = {"x1", "x2", "x3", "group_id", "cat_a"}
    pure_noise = {"num_pos_a", "num_pos_b", "num_heavy"}
    assert srcs & real_signal, f"seed={seed}: support carries no planted signal at all: {sorted(srcs)}"
    leaked_noise = srcs & pure_noise
    if leaked_noise:
        assert srcs & real_signal, (
            f"seed={seed}: pure-noise leg(s) {sorted(leaked_noise)} rescued while no real signal present "
            f"-- the cols-space index translation in the fallback regressed. support_sources={sorted(srcs)}"
        )


@pytest.mark.parametrize("seed", [0, 13])
def test_empty_screen_rescue_conditions_on_engineered_survivors(seed):
    """Regression (2026-06-08): the empty-RAW-screen rescue must NOT re-inject a raw
    operand whose y-information is fully carried by a SURVIVING engineered child.

    On a composite target ``y = a**2/b + 3*log(c)*sin(d) + noise`` at n>20000 the greedy
    screen leaves 0 raw survivors but builds the engineered children ``div(sqr(a),abs(b))``
    and ``mul(log(c),sin(d))``; the empty-screen rescue then fires. Each raw operand
    a,b,c,d individually clears the relevance floor AND its own permutation null (it IS a
    genuine operand) and -- being a mutually-independent uniform -- is not redundant with
    any OTHER raw operand, so the raw-only redundancy dedup admitted ALL FOUR, re-injecting
    exactly the operands the engineered children subsume. The rescue now seeds its
    redundancy-dedup conditioning set with the surviving engineered features, so an operand
    fully captured by its engineered child fails the redundancy test and drops. We assert
    the rescue does NOT re-admit the full {a,b,c,d} operand block alongside the two correct
    engineered pairs (the pre-fix selection carried all four)."""
    import numpy as np
    import pandas as pd
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(seed)
    n = 25_000
    a = rng.uniform(1, 5, n)
    b = rng.uniform(1, 5, n)
    c = rng.uniform(1, 5, n)
    d = rng.uniform(0, 2 * np.pi, n)
    e = rng.normal(0, 1, n)
    y = a**2 / b + 3.0 * np.log(c) * np.sin(d) + 0.3 * e
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    fs = MRMR(verbose=0, random_seed=seed).fit(df, pd.Series(y, name="y"))
    selected = list(fs.get_feature_names_out())
    raw_operands = {"a", "b", "c", "d"}
    re_admitted = raw_operands & set(selected)
    assert len(re_admitted) <= 1, (
        f"seed={seed}: empty-screen rescue re-injected raw operands {sorted(re_admitted)} "
        f"fully subsumed by surviving engineered children; the rescue redundancy-dedup must "
        f"condition on the engineered survivors. selected={selected}"
    )
    # The genuine signal must still be captured by at least one engineered child (we never
    # drop signal -- only the redundant raw operands).
    eng = [s for s in selected if s not in df.columns]
    assert eng, f"seed={seed}: no engineered child survived; signal lost. selected={selected}"
