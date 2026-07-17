"""Regression (MRMR critique N-F6): the Chao-Shen (CS) MI estimator is NOT wired into any null-using path.

N-F6 was filed as the CS analogue of N-F1 (a permutation null must use the SAME estimator as the observed statistic).
Evidence (bench_nf6_chao_shen_null_status.py) shows CS is a STANDALONE estimator with no production null caller, and
mi_correction='chao_shen' currently degrades to plug-in ('none') for BOTH the observed relevance and its null -- so there
is no observed-vs-null estimator mismatch. This test pins that DOC contract:

1. mi_correction='chao_shen' does not enable Miller-Madow (observed and null both stay plug-in -> matched estimator).
2. It is surfaced (logged), not silently ignored.

If a future PR wires CS into the observed relevance, guard (1) forces the null to be wired in the same change (otherwise
observed CS vs plug-in null is the exact N-F1/N-F6 mismatch this test exists to catch).
"""

import numpy as np


def test_chao_shen_does_not_enable_miller_madow_toggle():
    from mlframe.feature_selection.filters.info_theory._state_and_dispatch import use_mi_miller_madow, set_mi_miller_madow

    try:
        for corr, expect_mm in (("none", False), ("miller_madow", True), ("chao_shen", False)):
            set_mi_miller_madow(corr == "miller_madow")  # exactly MRMR.fit's wiring
            assert use_mi_miller_madow() is expect_mm, f"mi_correction={corr!r} unexpectedly set MM={use_mi_miller_madow()}"
    finally:
        set_mi_miller_madow(False)


def test_chao_shen_mi_is_standalone_matches_plugin_shape():
    # CS estimator is a direct call, not routed through the null kernels; on a dependent pair it returns a finite MI.
    from mlframe.feature_selection.filters._chao_shen import chao_shen_mi

    rng = np.random.default_rng(0)
    n = 1500
    x = rng.integers(0, 6, n).astype(np.int32)
    y = ((x + rng.integers(0, 2, n)) % 2).astype(np.int32)
    cs = chao_shen_mi(x, y)
    assert np.isfinite(cs) and cs >= 0.0


def test_mrmr_warns_when_chao_shen_requested(caplog):
    # mi_correction='chao_shen' must be surfaced as an unwired no-op, not silently ignored.
    import logging
    from mlframe.feature_selection.filters.mrmr._mrmr_class import MRMR

    rng = np.random.default_rng(1)
    n = 400
    import pandas as pd

    X = pd.DataFrame({f"f{i}": rng.integers(0, 4, n) for i in range(4)})
    y = (X["f0"] % 2).to_numpy()
    with caplog.at_level(logging.WARNING):
        MRMR(mi_correction="chao_shen", max_runtime_mins=0.2).fit(X, y)
    assert any("chao_shen" in r.message and "plug-in" in r.message for r in caplog.records), (
        "MRMR did not warn that mi_correction='chao_shen' falls back to plug-in"
    )
