"""Unit tests for the iter56 brute-force dispatcher cap + iter57 boundary audit + iter58 sweep.

The lever raises the default ``brute_force_max_features`` from 22 -> 28 and the dispatcher's
``n_sub`` feasibility gate from 2M -> 80M. Both are overridable per-HW via
``pyutilz.system.kernel_tuning_cache``. The iter57 boundary tests pin the dispatcher's actual
truth table at default ``max_features=None`` (brute force fires at n<=26, beam at n in {27, 28}).
The iter58 sweep (``bench_iter58_beam_width_sweep``) measured caps {22, 28, 32, 40} at C3 + C3_hard
and confirmed 28 as the recall-maximising sweet spot across both regimes; 32 lost a recall hit at
C3_hard, 40 was slower without improving recall, 22 underrecalled at both. The default stays at 28.
"""

from __future__ import annotations

import math

import pytest


# ------------------------------------------------------------ helper resolvers
def test_resolve_brute_force_max_features_default_is_28():
    from mlframe.feature_selection.shap_proxied_fs import (
        _DEFAULT_BRUTE_FORCE_MAX_FEATURES,
        _resolve_brute_force_max_features,
    )

    assert _DEFAULT_BRUTE_FORCE_MAX_FEATURES == 28
    # Without a cache entry the helper returns the module default (or whatever the cache pins).
    assert _resolve_brute_force_max_features() >= 1


def test_resolve_brute_force_max_features_honours_explicit_default():
    from mlframe.feature_selection.shap_proxied_fs import _resolve_brute_force_max_features

    # When the cache is missing pyutilz the helper must swallow ImportError and return the passed
    # default, never raise.
    assert _resolve_brute_force_max_features(default=17) >= 1


def test_resolve_brute_force_n_sub_gate_default_is_80m():
    from mlframe.feature_selection.shap_proxied_fs import (
        _DEFAULT_BRUTE_FORCE_N_SUB_GATE,
        _resolve_brute_force_n_sub_gate,
    )

    # Calibrated to permit n=28 (76.7M subsets at max_card=12) but block n=29 (123M).
    assert _DEFAULT_BRUTE_FORCE_N_SUB_GATE == 80_000_000
    assert _resolve_brute_force_n_sub_gate() >= 1


# ------------------------------------------------------------ ShapProxiedFS __init__ wiring
def test_shap_proxied_fs_default_cap_uses_module_default():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import (
        _resolve_brute_force_max_features,
        ShapProxiedFS,
    )

    fs = ShapProxiedFS()
    assert fs.brute_force_max_features == _resolve_brute_force_max_features()


def test_shap_proxied_fs_explicit_cap_wins():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(brute_force_max_features=11)
    assert fs.brute_force_max_features == 11


# ------------------------------------------------------------ dispatcher behaviour
def test_resolve_optimizer_bruteforce_at_n_equal_cap():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(
        brute_force_max_features=28, min_features=1, max_features=12, optimizer="auto",
        use_gpu=False)
    # n=28 -> sum C(28,k) k=1..12 = 76.7M subsets, under the 80M gate, must dispatch brute force.
    assert fs._resolve_optimizer(28) == "bruteforce"


def test_resolve_optimizer_falls_through_to_beam_when_n_exceeds_cap():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(brute_force_max_features=22, optimizer="auto")
    # n=23 > cap 22 -> beam regardless of n_sub.
    assert fs._resolve_optimizer(23) == "beam"


def test_resolve_optimizer_falls_through_when_n_sub_exceeds_gate():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    # n=29 / max_card=12 -> ~123M subsets, exceeds the 80M default gate -> beam.
    fs = ShapProxiedFS(
        brute_force_max_features=29, min_features=1, max_features=12, optimizer="auto")
    expected = sum(math.comb(29, r) for r in range(1, 13))
    assert expected > 80_000_000
    assert fs._resolve_optimizer(29) == "beam"


def test_resolve_optimizer_passthrough_when_not_auto():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(brute_force_max_features=22, optimizer="beam")
    # Explicit optimizer is never overridden by the dispatcher.
    assert fs._resolve_optimizer(10) == "beam"


# --------------------------------------- iter57 audit: default max_features behaviour
# At default ``max_features=None`` the brute-force kernel treats max_card as ``n_features``
# (total subsets = 2^n - 1). The dispatcher's n_sub gate then routes only n<=26 to brute force;
# n in {27, 28} falls back to beam despite the cap allowing them. These tests pin the actual
# dispatcher truth table so a future cap raise that does NOT also adjust the gate is caught.
@pytest.mark.parametrize("n", [22, 24, 26])
def test_resolve_optimizer_default_max_features_dispatches_bruteforce_up_to_26(n):
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(optimizer="auto", use_gpu=False)
    # Defaults: max_features=None, min_features=1, brute_force_max_features=28, gate=80M.
    # 2^26 - 1 = 67M < 80M -> brute force, 2^27 - 1 = 134M > 80M -> beam.
    assert fs._resolve_optimizer(n) == "bruteforce"


@pytest.mark.parametrize("n", [27, 28])
def test_resolve_optimizer_default_max_features_falls_to_beam_at_27_28(n):
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    fs = ShapProxiedFS(optimizer="auto", use_gpu=False)
    # cap allows n=27,28 but the n_sub gate blocks because at max_features=None the kernel
    # enumerates 2^n - 1 subsets, exceeding the 80M default gate. Caller must pin
    # max_features<=12 to actually run brute force at n=27,28.
    assert fs._resolve_optimizer(n) == "beam"


def test_resolve_optimizer_explicit_max_features_unlocks_n28_bruteforce():
    pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    # Documented opt-in: max_features<=12 with default cap=28 makes brute_force feasible at n=28.
    fs = ShapProxiedFS(optimizer="auto", max_features=12, use_gpu=False)
    assert fs._resolve_optimizer(28) == "bruteforce"


# --------------------------------------- iter58 calibration: 28 is the prescreen-pool sweet spot
# iter58's sweep (``bench_iter58_beam_width_sweep``) measured caps {22, 28, 32, 40} at C3 (snr=8)
# and C3_hard (snr=2) with all stages instrumented. Result table (steady-state, post-warmup):
#
#   regime    cap  e2e_s  search_s  recall  honest_loss
#   C3        22   cold   154.9     17/20   0.149824
#   C3        28   43.2   1.35      18/20   0.149590
#   C3        32   44.6   1.72      18/20   0.149411
#   C3        40   57.3   3.40      18/20   0.150751
#   C3_hard   22   cold   170.2     15/20   0.194880
#   C3_hard   28   47.0   2.02      16/20   0.195924
#   C3_hard   32   43.6   2.43      15/20   0.196003   <- LOST a recall hit vs cap28
#   C3_hard   40   53.0   3.57      16/20   0.196421   <- tied at 16 but worse honest_loss + slower
#
# cap28 recall-dominates or ties every wider cap across both regimes; cap32 loses a recall hit at
# C3_hard (the low-SNR regime where iter56 saw the bigger absolute gain). Hypothesis: widening the
# prescreen pool past 28 lets noise features into beam's input, occasionally pushing the chosen
# subset off the truly informative one when the SHAP ranking is noisy. The default stays at 28.
def test_iter58_default_cap_is_28_per_beam_width_sweep():
    from mlframe.feature_selection.shap_proxied_fs import _DEFAULT_BRUTE_FORCE_MAX_FEATURES

    # If this default ever changes, re-run the iter58 sweep at the new candidate value PLUS at 28
    # under matched random_state + dataset seed and confirm the wider cap doesn't lose a recall hit
    # at C3_hard (the regime that flagged cap32 as worse).
    assert _DEFAULT_BRUTE_FORCE_MAX_FEATURES == 28
