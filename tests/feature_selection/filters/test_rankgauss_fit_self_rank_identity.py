"""Regression sensor: the FIT path of ``generate_rankgauss_features`` computes the fit values' ranks among themselves
via a single ``argsort`` mid-rank scatter (``_self_avg_tie_rank``) instead of the two full-array ``searchsorted`` sweeps
of ``_avg_tie_rank(np.sort(x), x)``. At fit time the query set IS the reference set, so the cheaper self-rank is
bit-identical (~2.6x on the full 6-col 1M generate). The replay path keeps ``searchsorted`` (test values are not the fit
values). This pins (a) ``_self_avg_tie_rank`` bit-identity vs the two-sweep reference on continuous AND tied data, and
(b) that the fit path uses NO ``searchsorted`` over the data column.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _extra_fe_families as fam
from mlframe.feature_selection.filters._extra_fe_families import (
    _self_avg_tie_rank,
    _avg_tie_rank,
    _rank_to_gauss,
    generate_rankgauss_features,
)


def _reference_fit_self_rank(x_finite: np.ndarray) -> np.ndarray:
    fs = np.sort(x_finite)
    return _avg_tie_rank(fs, x_finite)


def test_self_avg_tie_rank_identical_continuous():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50_000)
    assert np.array_equal(_self_avg_tie_rank(x), _reference_fit_self_rank(x))


def test_self_avg_tie_rank_identical_tied():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 40, 50_000).astype(np.float64)  # heavy exact ties
    assert np.array_equal(_self_avg_tie_rank(x), _reference_fit_self_rank(x))


def test_self_avg_tie_rank_singleton_and_allequal():
    assert np.array_equal(_self_avg_tie_rank(np.array([7.0])), np.array([0.0]))
    x = np.full(1000, 2.5)
    assert np.array_equal(_self_avg_tie_rank(x), _reference_fit_self_rank(x))


def test_generate_rankgauss_fit_bit_identical_and_no_searchsorted(monkeypatch):
    import pandas as pd

    rng = np.random.default_rng(11)
    df = pd.DataFrame(
        {
            "cont": rng.standard_normal(20_000),
            "disc": rng.integers(0, 25, 20_000).astype(np.float64),
            "nanc": np.where(rng.random(20_000) < 0.1, np.nan, rng.standard_normal(20_000)),
        }
    )

    # Reference outputs via the exact two-sweep fit path.
    ref = {}
    for c in df.columns:
        x = np.asarray(df[c].to_numpy(), dtype=np.float64)
        fin = np.isfinite(x)
        fs = np.sort(x[fin]) if fin.any() else np.array([0.0])
        nf = int(fs.size)
        out = np.zeros_like(x)
        if nf > 0 and fin.any():
            out[fin] = _rank_to_gauss(_avg_tie_rank(fs, x[fin]), nf)
        ref["rankgauss__" + c] = out

    calls = {"n": 0}
    real = np.searchsorted

    def spy(a, v, side="left", sorter=None):
        calls["n"] += 1
        return real(a, v, side=side, sorter=sorter)

    monkeypatch.setattr(fam.np, "searchsorted", spy)
    enc, _ = generate_rankgauss_features(df, list(df.columns))
    monkeypatch.undo()

    assert calls["n"] == 0, f"fit path must not call searchsorted on the data, used {calls['n']}"
    for name, exp in ref.items():
        assert np.array_equal(enc[name].to_numpy(), exp), f"{name} fit output diverged from two-sweep reference"
