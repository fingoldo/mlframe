"""Group-aware relevance MI in ``MRMR.fit(groups=...)``: regression + biz_value.

The kernel itself is unit-tested in ``test_group_mi_kernel.py``; here we pin the END-TO-END wiring through the joblib
worker path (thread-local publish/republish) and the constructor / strict-block / no-harm contracts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR


def _panel(seed: int, G: int = 150, per: int = 40):
    """Panel where ``x_leak`` is a pure between-group LEVEL (high GLOBAL MI, ~0 within-group) and ``x_within`` is the
    genuine within-group signal. A group-naive MRMR picks up ``x_leak``; a group-aware one must demote it."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(G), per)
    n = groups.size
    gmean = rng.normal(size=G)[groups]
    x_within = rng.normal(size=n)
    y = gmean + 0.9 * x_within + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x_leak": gmean.copy(), "x_within": x_within.copy(), "x_noise": rng.normal(size=n)})
    return X, y, groups


def _selected(model) -> list:
    names = getattr(model, "selected_features_names_", None)
    if names:
        return list(names)
    return list(model.get_feature_names_out())


def _fit(X, y, groups, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=2, fe_max_steps=0, verbose=0, **kw)
        m.fit(X, y, groups=groups)
    return m


def test_group_naive_selects_the_leak_feature_baseline():
    """Pre-fix baseline: group-naive MRMR SELECTS the between-group-level leak feature (the bug this feature fixes)."""
    X, y, groups = _panel(1)
    sel = _selected(_fit(X, y, groups, group_aware_mi=False))
    assert "x_leak" in sel, f"group-naive should pick the leak level; got {sel}"


def test_group_aware_demotes_the_leak_feature():
    """biz_value: with ``group_aware_mi=True`` the pure between-group-level leak feature is DROPPED while the genuine
    within-group signal is retained."""
    X, y, groups = _panel(1)
    m = _fit(X, y, groups, group_aware_mi=True)
    sel = _selected(m)
    assert "x_within" in sel, f"within-group signal must be retained; got {sel}"
    assert "x_leak" not in sel, f"between-group-level leak must be demoted; got {sel}"
    assert m.groups_ignored_ is False


def test_group_aware_off_is_noop_and_warns_group_ignored():
    """Default off: groups are ignored (group-naive), ``groups_ignored_=True`` + the warn-only fallback fires."""
    X, y, groups = _panel(2)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = MRMR(max_runtime_mins=2, fe_max_steps=0, verbose=0, group_aware_mi=False)
        m.fit(X, y, groups=groups)
    assert m.groups_ignored_ is True
    assert any("does NOT consume them" in str(w.message) for w in rec)


def test_strict_groups_with_group_aware_proceeds():
    """(strict_groups=True, group_aware_mi=True) must PROCEED (groups consumed) rather than raise."""
    X, y, groups = _panel(3)
    m = _fit(X, y, groups, group_aware_mi=True, strict_groups=True)
    assert m.groups_ignored_ is False


def test_strict_groups_without_group_aware_still_raises():
    """(strict_groups=True, group_aware_mi=False) keeps raising -- the honest 'I dropped your groups' contract."""
    X, y, groups = _panel(3)
    with pytest.raises(NotImplementedError):
        MRMR(max_runtime_mins=2, fe_max_steps=0, verbose=0, strict_groups=True).fit(X, y, groups=groups)


def test_no_harm_single_group_matches_group_naive():
    """No-harm control: one group (no group structure) -> group-aware selection == group-naive selection."""
    X, y, _ = _panel(4)
    one = np.zeros(len(X), dtype=np.int64)
    naive = _selected(_fit(X, y, one, group_aware_mi=False))
    aware = _selected(_fit(X, y, one, group_aware_mi=True))
    assert set(aware) == set(naive), f"single-group aware {aware} != naive {naive}"


def test_group_aware_keeps_sign_flipping_within_group_signal():
    """biz_value (the case a global-then-DEMEAN approach DROPS but the per-group MI kernel KEEPS): a feature whose
    within-group slope FLIPS sign across groups still has high per-group MI in EVERY group, so group-aware MRMR retains
    it. (A demean guard subtracts the group mean and would see ~0 residual association; per-group MI is sign-blind and
    keeps it. Global MI is also sign-blind, so this pins group-aware as NO-HARM on the sign-flip case, not a regression
    -- the strictly-better-than-demean claim itself is unit-pinned in ``test_group_mi_kernel.py``.)"""
    rng = np.random.default_rng(11)
    G, per = 160, 40
    groups = np.repeat(np.arange(G), per)
    n = groups.size
    slope = np.where((np.arange(G) % 2) == 0, 1.0, -1.0)[groups]
    x_flip = rng.normal(size=n)
    y = slope * x_flip + 0.05 * rng.normal(size=n)
    X = pd.DataFrame({"x_flip": x_flip.copy(), "x_noise": rng.normal(size=n)})
    aware = _selected(_fit(X, y, groups, group_aware_mi=True))
    assert "x_flip" in aware, f"per-group MI must retain the sign-flipping signal; got {aware}"


def test_fs_config_lever_folds_group_aware_mi_into_mrmr_kwargs():
    """The ``FeatureSelectionConfig.mrmr_group_aware_mi`` first-class lever folds into ``mrmr_kwargs`` (D-surface)."""
    from mlframe.training._feature_selection_config import FeatureSelectionConfig

    cfg = FeatureSelectionConfig(
        use_mrmr_fs=True, mrmr_group_aware_mi=True, mrmr_group_mi_min_rows=25, mrmr_group_mi_aggregate="equal"
    )
    assert cfg.mrmr_kwargs["group_aware_mi"] is True
    assert cfg.mrmr_kwargs["group_mi_min_rows"] == 25
    assert cfg.mrmr_kwargs["group_mi_aggregate"] == "equal"
    # default off -> merges nothing (byte-identical to today)
    assert not (FeatureSelectionConfig().mrmr_kwargs or {}).get("group_aware_mi")


def test_fit_cache_key_folds_groups_signature():
    """Regression: under group_aware_mi the process-wide _FIT_CACHE keys on the groups content too, so two fits on the
    SAME X/y with DIFFERENT groups cannot replay one another's I(X;Y|G)-dependent selection. This pins the mechanism:
    the folded groups content-signature distinguishes distinct group assignments (a naive key would collide)."""
    from mlframe.feature_selection.filters._mrmr_fingerprints import _content_array_signature

    rng = np.random.default_rng(21)
    n = 4000
    groups_a = np.repeat(np.arange(100), 40)
    groups_b = groups_a[rng.permutation(n)]
    assert _content_array_signature(groups_a) != _content_array_signature(groups_b), "groups signature must differ"


def test_group_aware_non_uniform_sample_weight_disables_with_warning(caplog):
    """Row resampling under non-uniform sample_weight reshuffles X but not groups -> group-aware is disabled (with a
    log warning) rather than mis-aligning rows to groups."""
    import logging

    X, y, groups = _panel(5)
    sw = np.ones(len(X))
    sw[: len(X) // 2] = 3.0
    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(max_runtime_mins=2, fe_max_steps=0, verbose=0, group_aware_mi=True)
        m.fit(X, y, groups=groups, sample_weight=sw)
    assert "group_aware_mi disabled" in caplog.text
