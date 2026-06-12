"""Unit tests for the iter41 width-dependent ``revalidation_ucb_stdev_multiplier`` resolver.

Pure-Python contract checks: no fit, no joblib, no booster work. Validates that:
  - ``None`` (sentinel) routes to 0.6 at ``n_features >= 10000`` and 1.0 below.
  - Explicit user value overrides the auto for any width.
  - The auto threshold is exactly 10000 (inclusive: >=10000 -> 0.6).
  - The instance still works with legacy float arguments (back-compat).
"""

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def test_default_is_none_sentinel():
    sel = ShapProxiedFS()
    assert sel.revalidation_ucb_stdev_multiplier is None


def test_auto_below_threshold_returns_one_point_zero():
    sel = ShapProxiedFS()
    assert sel._resolve_revalidation_ucb_stdev_multiplier(1) == 1.0
    assert sel._resolve_revalidation_ucb_stdev_multiplier(100) == 1.0
    assert sel._resolve_revalidation_ucb_stdev_multiplier(9999) == 1.0


def test_auto_at_and_above_threshold_returns_zero_point_six():
    sel = ShapProxiedFS()
    assert sel._resolve_revalidation_ucb_stdev_multiplier(10000) == 0.6
    assert sel._resolve_revalidation_ucb_stdev_multiplier(20000) == 0.6
    assert sel._resolve_revalidation_ucb_stdev_multiplier(100000) == 0.6


def test_user_pinned_overrides_auto_below_threshold():
    sel = ShapProxiedFS(revalidation_ucb_stdev_multiplier=0.4)
    assert sel.revalidation_ucb_stdev_multiplier == 0.4
    assert sel._resolve_revalidation_ucb_stdev_multiplier(500) == 0.4


def test_user_pinned_overrides_auto_above_threshold():
    sel = ShapProxiedFS(revalidation_ucb_stdev_multiplier=1.5)
    assert sel.revalidation_ucb_stdev_multiplier == 1.5
    assert sel._resolve_revalidation_ucb_stdev_multiplier(50000) == 1.5


def test_user_pinned_zero_is_respected_not_none_coerced():
    # 0.0 is a legitimate (no-std-margin) calibration; must not be treated as falsy sentinel.
    sel = ShapProxiedFS(revalidation_ucb_stdev_multiplier=0.0)
    assert sel.revalidation_ucb_stdev_multiplier == 0.0
    assert sel._resolve_revalidation_ucb_stdev_multiplier(20000) == 0.0


def test_resolver_returns_float_type():
    sel = ShapProxiedFS()
    out = sel._resolve_revalidation_ucb_stdev_multiplier(10000)
    assert isinstance(out, float)


def test_refine_multiplier_is_independent():
    # iter35's within_cluster_refine multiplier is independently configured; tightening reval should
    # not perturb the refine default. Guards the same-helper-but-independent-knobs invariant.
    sel = ShapProxiedFS()
    assert sel.refine_ucb_stdev_multiplier == 1.0
    sel2 = ShapProxiedFS(revalidation_ucb_stdev_multiplier=0.3)
    assert sel2.refine_ucb_stdev_multiplier == 1.0
