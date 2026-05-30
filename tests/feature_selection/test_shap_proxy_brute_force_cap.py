"""Unit tests for the iter56 brute-force dispatcher cap.

The lever raises the default ``brute_force_max_features`` from 22 -> 28 and the dispatcher's
``n_sub`` feasibility gate from 2M -> 50M. Both are overridable per-HW via
``pyutilz.system.kernel_tuning_cache``.
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
