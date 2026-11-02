"""Triplet CMI scoring says so when n is too small for any stratum to reach the row floor (audit 2026-10-13, D14/R15)."""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_triplet_fe import score_triplet_cross_basis_by_cmi


def _frame(n):
    """Three continuous legs, their product as the engineered column, and a target driven by the product."""
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    eng = pd.DataFrame({"a*b*c__T1_T1_T1": raw["a"] * raw["b"] * raw["c"]})
    y = (eng.iloc[:, 0] > 0).astype(int).to_numpy()
    return raw, eng, y


def test_small_n_warns_and_scores_zero(caplog):
    """At n=150 with 10 bins every stratum is far below 30 rows: a warning names the cause, the gain stays at or below zero."""
    raw, eng, y = _frame(150)
    with caplog.at_level(logging.WARNING):
        out = score_triplet_cross_basis_by_cmi(raw, eng, y, nbins=10, n_perm=20)
    assert any("fewer than 30 rows" in r.getMessage() for r in caplog.records)
    assert out["conditional_mi"].iloc[0] == 0.0


def test_large_n_does_not_warn(caplog):
    """At n=6000 the strata are large and nothing is logged."""
    raw, eng, y = _frame(6000)
    with caplog.at_level(logging.WARNING):
        score_triplet_cross_basis_by_cmi(raw, eng, y, nbins=10, n_perm=20)
    assert not any("fewer than 30 rows" in r.getMessage() for r in caplog.records)
