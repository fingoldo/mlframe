"""Shim-sanity test: mlframe still imports normality tests after the
2026-05-27 move to ``pyutilz.stats.normality``. Full calibration suite
lives in pyutilz/tests/stats/test_normality.py.
"""
from __future__ import annotations

import numpy as np


def test_shim_reexports_normality_verdict() -> None:
    from mlframe.training.targets._residual_normality_tests import normality_verdict

    rng = np.random.default_rng(0)
    out = normality_verdict(rng.standard_normal(1000))
    assert "reject_normal" in out
    assert out["reject_normal"] is False


def test_shim_reexports_underscored_aliases() -> None:
    from mlframe.training.targets._residual_normality_tests import (
        _dagostino_k2,
        _anderson_darling_normal,
    )

    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    K2, p, _, _ = _dagostino_k2(x)
    assert isinstance(K2, float)
    A2, p_ad = _anderson_darling_normal(x)
    assert isinstance(A2, float)
