"""The raw-column pre-screen keeps columns with a marginal binned-MI signal (a U shape included), drops most noise, and keeps non-finite columns unjudged."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._candidate_pool import marginal_mi_pvalues, raw_column_pool

_spec = importlib.util.spec_from_file_location("pool_planted", Path(__file__).resolve().parents[1] / "_synth" / "pool_planted.py")
_planted = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_planted)


def test_weak_u_shape_column_is_in_the_pool_and_most_noise_is_not():
    """``sig_x`` of the U-shape generator passes at the loose bound; with 99 noise columns about p_loose of them do, plus the top-k."""
    X, y, _planted_names = _planted.make_weak_marginal_shape(n=20000, p=100, seed=0, strength=1.0)
    pool = raw_column_pool(X, y, top_k=5)
    assert "sig_x" in pool
    assert len(pool) < 45


def test_hidden_periodic_and_xor_like_columns_are_not_promised_by_the_prescreen():
    """A fast oscillation is invisible to the binned MI, which is why those families keep all columns: the pre-screen does not rank ``sig_x`` above the noise."""
    X, y, _ = _planted.make_hidden_periodic(n=20000, p=40, seed=0, freq=20.0)
    codes_pool = raw_column_pool(X, y, top_k=1, p_loose=0.0)
    assert len(codes_pool) <= 2  # at most the one top-by-MI column (not necessarily sig_x)


def test_non_finite_columns_are_kept_unjudged():
    """A column with a NaN on the subsample cannot be coded here and stays in the pool."""
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    X.loc[3, "b"] = np.nan
    y = rng.integers(0, 2, n)
    assert "b" in raw_column_pool(X, y, p_loose=0.0, top_k=0)


def test_mi_and_pvalue_of_an_informative_and_an_independent_column():
    """An informative column has a small p-value and a larger MI than an independent one."""
    rng = np.random.default_rng(1)
    m = 8000
    y = rng.integers(0, 2, m)
    informative = np.clip((y * 3 + rng.integers(0, 7, m)), 0, 9)
    independent = rng.integers(0, 10, m)
    mi, p = marginal_mi_pvalues(np.vstack([informative, independent]).astype(np.int16), np.array([10, 10]), y)
    assert mi[0] > mi[1] and p[0] < 1e-6 and p[1] > 0.001
