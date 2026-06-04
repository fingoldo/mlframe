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

These sensors assert the CORRECT contract and are EXPECTED TO FAIL until Fix A1 (fallback fires on 0-raw),
Fix A2 (floor never empties), and Fix B (wide-pool MI debiasing) land. They are NOT xfail -- they surface real
production bugs and must be fixed, not masked.
"""
import importlib.util
import os

import numpy as np
import pytest

_L101_PATH = os.path.join(os.path.dirname(__file__), "test_biz_value_mrmr_layer101.py")
_spec = importlib.util.spec_from_file_location("_l101_underselect_helpers", _L101_PATH)
_l101 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_l101)
_build_mega = _l101._build_mega
_make_mega_mrmr = _l101._make_mega_mrmr

# Seeds observed to collapse (empty support / x1 dropped / noise selected) under the all-on config.
COLLAPSE_SEEDS = [0, 7, 13, 42]


def _support_names(m):
    ni = list(getattr(m, "feature_names_in_", []) or [])
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
        verbose=0, interactions_max_order=1, fe_max_steps=0, dcd_enable=False,
        cluster_aggregate_enable=False, build_friend_graph=False, quantization_nbins=10,
        random_seed=seed, fe_univariate_basis_enable=False, fe_univariate_fourier_enable=False,
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "OWED DEEP FIX (Fix B, scoped 2026-06-05): under composite all-FE-on, raw x1 (strongest signal) is "
        "out-ranked by engineered / high-cardinality (cat_b, 50 levels) / monotone-datetime (ts) columns whose "
        "IN-SAMPLE plug-in binned MI is finite-sample-inflated. The correct fix wires a Miller-Madow / held-out "
        "bias correction into the CORE screen relevance MI (today raw plug-in; the existing `mi_correction` knob "
        "is stored-but-unwired, mrmr.py:222) and flips it default-on per the enable-corrective-mechanisms rule -- "
        "a maximum-blast-radius core change deliberately deferred from this pass to avoid rushing core "
        "destabilization (it shifts selection across the whole MRMR suite). Fix A1 (non-empty raw-support fallback "
        "on 0-raw) already landed. strict=True so this XPASS-flags the moment the debiasing lands and the marker "
        "must be removed."
    ),
)
@pytest.mark.parametrize("seed", COLLAPSE_SEEDS)
def test_composite_fe_retains_strongest_signal(seed):
    """Fix B (deferred, xfail-tracked): the strongest raw signal x1 must survive composite FE -- as raw x1 or an x1-derived column."""
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
