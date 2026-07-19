"""Regression (MRMR critique N-F6): Chao-Shen (CS) MI estimator / null-path parity.

N-F6 was originally filed as the CS analogue of N-F1 (a permutation null must use the SAME estimator as
the observed statistic), back when CS was a standalone estimator with no production null caller and
mi_correction='chao_shen' silently degraded to plug-in ('none') with a warning. CS is now fully wired
into BOTH the observed-relevance and permutation-null paths (mirroring Miller-Madow's wiring exactly --
see ``MRMR.fit``'s ``set_mi_chao_shen`` call site), closing the exact N-F1/N-F6 mismatch this file exists
to catch. This test pins the CURRENT contract:

1. mi_correction='chao_shen' does not enable Miller-Madow (the two corrections are mutually exclusive).
2. Requesting it no longer emits the stale "falls back to plug-in" no-op warning -- it is genuinely engaged.
"""

import numpy as np


def test_chao_shen_does_not_enable_miller_madow_toggle():
    """Chao shen does not enable miller madow toggle."""
    from mlframe.feature_selection.filters.info_theory._state_and_dispatch import use_mi_miller_madow, set_mi_miller_madow

    try:
        for corr, expect_mm in (("none", False), ("miller_madow", True), ("chao_shen", False)):
            set_mi_miller_madow(corr == "miller_madow")  # exactly MRMR.fit's wiring
            assert use_mi_miller_madow() is expect_mm, f"mi_correction={corr!r} unexpectedly set MM={use_mi_miller_madow()}"
    finally:
        set_mi_miller_madow(False)


def test_chao_shen_mi_is_standalone_matches_plugin_shape():
    # CS estimator is a direct call, not routed through the null kernels; on a dependent pair it returns a finite MI.
    """Chao shen mi is standalone matches plugin shape."""
    from mlframe.feature_selection.filters._chao_shen import chao_shen_mi

    rng = np.random.default_rng(0)
    n = 1500
    x = rng.integers(0, 6, n).astype(np.int32)
    y = ((x + rng.integers(0, 2, n)) % 2).astype(np.int32)
    cs = chao_shen_mi(x, y)
    assert np.isfinite(cs) and cs >= 0.0


def test_mrmr_chao_shen_no_longer_warns_as_unwired_noop(caplog):
    """``mi_correction='chao_shen'`` is now fully wired (observed + null); requesting it must NOT emit
    the old "falls back to plug-in" no-op warning."""
    import logging
    from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR

    rng = np.random.default_rng(1)
    n = 400
    import pandas as pd

    X = pd.DataFrame({f"f{i}": rng.integers(0, 4, n) for i in range(4)})
    y = (X["f0"] % 2).to_numpy()
    with caplog.at_level(logging.WARNING):
        MRMR(mi_correction="chao_shen", max_runtime_mins=0.2).fit(X, y)
    assert not any(
        "chao_shen" in r.message and "plug-in" in r.message for r in caplog.records
    ), "MRMR still warns that mi_correction='chao_shen' falls back to plug-in, but CS is now fully wired -- stale no-op warning should have been removed."
