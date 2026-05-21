"""Regression test for the vectorised leakage detector in
``analyze_feature_distribution`` (iter114, 2026-05-21).

The pre-iter114 leakage detector was an O(F) loop calling ``np.corrcoef``
per column (~145 ms on the c0139 fuzz shape, 200k rows / 15 numeric cols).
A naive vectorised replacement that mean-imputed NaN cells in features
diluted the correlation enough to drop legitimate leakage below the 0.99
threshold on columns with sparse NaN -- a real correctness regression.

The shipped version uses ``pandas.DataFrame.corrwith`` which handles per-
column NaN with pairwise-complete-obs semantics (bit-exact with the old
per-column loop) AND vectorises the C-level reduction across columns (~65
ms on the same shape). These tests pin the correctness properties so a
future "let's just impute the NaN" refactor fails fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training._target_distribution_analyzer import analyze_feature_distribution


def test_strong_leakage_is_detected():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.standard_normal(n)
    X = pd.DataFrame({
        "leaky": y + 0.001 * rng.standard_normal(n),
        "unrelated": rng.standard_normal(n),
        "mild": 0.5 * y + rng.standard_normal(n),
    })
    rep = analyze_feature_distribution(X, y)
    assert "leaky" in rep.leakage_candidates
    assert "unrelated" not in rep.leakage_candidates
    # corr ~0.45 is below default 0.99 threshold -> not flagged
    assert "mild" not in rep.leakage_candidates


def test_leakage_survives_10pct_nan_in_feature():
    """Naive mean-imputation would dilute corr below 0.99 here; corrwith uses
    pairwise complete obs so the 900 finite rows carry the correlation."""
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.standard_normal(n)
    leaky = y + 0.001 * rng.standard_normal(n)
    leaky[::10] = np.nan  # 10% NaN
    X = pd.DataFrame({"leaky": leaky, "unrelated": rng.standard_normal(n)})
    rep = analyze_feature_distribution(X, y)
    assert "leaky" in rep.leakage_candidates


def test_leakage_survives_20pct_nan_in_y():
    rng = np.random.default_rng(0)
    n = 1000
    y = rng.standard_normal(n)
    X = pd.DataFrame({
        "leaky": y + 0.001 * rng.standard_normal(n),
        "unrelated": rng.standard_normal(n),
    })
    y_with_nan = y.copy()
    y_with_nan[::5] = np.nan
    rep = analyze_feature_distribution(X, y_with_nan)
    assert "leaky" in rep.leakage_candidates


def test_constant_feature_does_not_crash_or_flag_leakage():
    """std=0 -> corrwith returns NaN -> skipped, not flagged."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n)
    X = pd.DataFrame({
        "leaky": y + 0.001 * rng.standard_normal(n),
        "const": np.full(n, 5.0),
    })
    rep = analyze_feature_distribution(X, y)
    assert "leaky" in rep.leakage_candidates
    assert "const" not in rep.leakage_candidates
    # The constant column DOES get flagged as low-variance (separate detector)
    assert "const" in rep.drop_candidates


def test_no_y_skips_leakage_detection_entirely():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "col_0": rng.standard_normal(n),
        "col_1": rng.standard_normal(n),
    })
    rep = analyze_feature_distribution(X, y=None)
    assert rep.leakage_candidates == []
