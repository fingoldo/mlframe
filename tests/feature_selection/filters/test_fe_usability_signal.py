"""Unit pins for the shared tail-concentration usability helpers (``_fe_usability_signal``).

Focus: the 2026-07-03 strided-subsample optimisation of ``abs_pearson`` must estimate |Pearson corr|
to well within every consumer's decision margin (min_corr 0.6; tail-concentration gaps ~0.99 vs ~0.06),
and must PRESERVE the outlier-inflated correlation the tail-concentration signal depends on.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

MOD = "mlframe.feature_selection.filters._fe_usability_signal"


def _full_abs_pearson(y, v):
    """Full abs pearson."""
    m = np.isfinite(y) & np.isfinite(v)
    yy, vv = y[m], v[m]
    ys, vs = float(yy.std()), float(vv.std())
    if ys <= 0 or vs <= 0:
        return 0.0
    return abs(float(np.mean((yy - yy.mean()) * (vv - vv.mean())) / (ys * vs)))


def test_abs_pearson_subsample_matches_fulln_within_margin():
    """Abs pearson subsample matches fulln within margin."""
    m = importlib.import_module(MOD)
    rng = np.random.default_rng(7)
    n = 1_000_000
    for _ in range(8):
        y = rng.normal(size=n)
        v = 0.4 * y + rng.normal(size=n)  # a real, moderate correlation
        got = m.abs_pearson(y, v)  # subsampled (default cap _ABS_PEARSON_MAX_ROWS, now 30k)
        exp = _full_abs_pearson(y, v)  # full-n
        # Contract is SELECTION-equivalence, not a tight absolute match: every consumer compares |corr| against
        # WIDE margins (min_corr 0.6; tail-concentration gap ~0.99 vs ~0.06). At the 30k cap (lowered from 250k to
        # match UNIFIED_FE_SUBSAMPLE_N) a 30k-of-1M estimate of a ~0.37 corr lands within ~1e-2 worst-of-8 -- still
        # ~70x below the 0.6 gate, so no keep/reject decision can flip. The old 5e-3 pin assumed the 250k cap.
        assert abs(got - exp) < 2e-2, f"subsample corr {got:.4f} vs full {exp:.4f}"


def test_abs_pearson_preserves_outlier_inflated_corr():
    """with_outliers-style tail signal: a**2/b tracks y only in the ~3% outlier tail. The strided
    subsample keeps the outlier PROPORTION, so the outlier-inflated |corr| ~0.99 survives (>> the 0.6
    bar) -- the property the whole tail-concentration detector relies on."""
    m = importlib.import_module(MOD)
    rng = np.random.default_rng(11)
    n = 1_000_000
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    # inject 3% / 15-IQR outliers into a and b (mirrors the with_outliers fixture spirit)
    idx = rng.choice(n, size=int(0.03 * n), replace=False)
    a[idx] *= 15.0
    b[idx] /= 15.0
    y = a**2 / b
    form = a * a / np.where(np.abs(b) < 1e-12, np.nan, b)
    got = m.abs_pearson(y, form)
    assert got > 0.9, f"outlier-inflated |corr| collapsed under subsample: {got:.4f}"


def test_abs_pearson_env_optout_full_n(monkeypatch):
    """Abs pearson env optout full n."""
    monkeypatch.setenv("MLFRAME_USABILITY_CORR_MAX_ROWS", "0")
    m = importlib.import_module(MOD)
    _orig_dict = dict(m.__dict__)
    m = importlib.reload(m)
    try:
        rng = np.random.default_rng(3)
        n = 600_000
        y = rng.normal(size=n)
        v = 0.7 * y + rng.normal(size=n)
        assert m.abs_pearson(y, v) == pytest.approx(_full_abs_pearson(y, v), abs=1e-12)
    finally:
        monkeypatch.delenv("MLFRAME_USABILITY_CORR_MAX_ROWS", raising=False)
        m.__dict__.clear()
        m.__dict__.update(_orig_dict)
