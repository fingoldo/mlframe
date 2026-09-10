"""``tests.conftest._install_macos_lgb_crash_skip`` turns real LightGBM Dataset construction into a
clean pytest skip on macOS, and is a no-op everywhere else.

Per-test ``skipif`` markers on the handful of tests first found crashing (audits/ci_review_2026-09-08/
_TRACKER.md, X5) turned out to badly undercount the real exposure -- a repo-wide grep found
``mlframe_models=["lgb"]`` in 39+ test files. This conftest-level patch is the actual fix: it
intercepts the one choke point every LightGBM fit path passes through
(``lightgbm.basic.Dataset.construct``), so it has to be proven to actually fire for a real fit, not
just for a synthetic direct call to ``.construct()``.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("lightgbm")

from tests.conftest import _install_macos_lgb_crash_skip


@pytest.fixture(autouse=True)
def _restore_lightgbm_construct():
    """``_install_macos_lgb_crash_skip`` mutates a class attribute directly (not via monkeypatch), so
    each test restores it manually -- otherwise a darwin-simulating test would leave the real pytest
    session's LightGBM permanently patched for every test collected afterward."""
    import lightgbm.basic

    original = lightgbm.basic.Dataset.construct
    yield
    lightgbm.basic.Dataset.construct = original


def test_real_lgb_fit_is_skipped_when_platform_is_darwin(monkeypatch):
    """A genuine LGBMClassifier.fit() call raises Skipped once the patch is installed under darwin --
    not just a synthetic direct call to Dataset.construct()."""
    import numpy as np
    import lightgbm as lgb

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("MLFRAME_TESTS_ALLOW_MACOS_LGB_FIT", raising=False)
    _install_macos_lgb_crash_skip()

    rng = np.random.default_rng(0)
    X = rng.random((50, 4)).astype(np.float64)
    y = (X[:, 0] > 0.5).astype(int)
    with pytest.raises(pytest.skip.Exception):
        lgb.LGBMClassifier(n_estimators=3, verbose=-1).fit(X, y)


def test_noop_on_non_darwin(monkeypatch):
    """On linux/win32 the patch is never installed; a real fit runs and completes normally."""
    import numpy as np
    import lightgbm as lgb
    import lightgbm.basic

    original = lightgbm.basic.Dataset.construct
    for plat in ("linux", "win32"):
        monkeypatch.setattr(sys, "platform", plat)
        _install_macos_lgb_crash_skip()
        assert lightgbm.basic.Dataset.construct is original

    rng = np.random.default_rng(0)
    X = rng.random((50, 4)).astype(np.float64)
    y = (X[:, 0] > 0.5).astype(int)
    m = lgb.LGBMClassifier(n_estimators=3, verbose=-1).fit(X, y)
    assert m.predict(X).shape == (50,)


def test_opt_out_env_var_skips_installing_the_patch(monkeypatch):
    """MLFRAME_TESTS_ALLOW_MACOS_LGB_FIT=1 leaves Dataset.construct untouched even on darwin."""
    import lightgbm.basic

    original = lightgbm.basic.Dataset.construct
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("MLFRAME_TESTS_ALLOW_MACOS_LGB_FIT", "1")
    _install_macos_lgb_crash_skip()
    assert lightgbm.basic.Dataset.construct is original
