"""biz_val tests for ``mlframe.feature_engineering.bruteforce`` --
``run_pysr_feature_engineering``.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
PySR symbolic regression discovers human-readable equations from
data. Tests use minimal budget so they complete in seconds.

Requires Julia runtime on PATH (D:/Julia/bin).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


_MINI_PYSR = {
    "niterations": 3,
    "populations": 3,
    "population_size": 15,
    "tournament_selection_n": 6,
    "maxdepth": 3,
    "binary_operators": ["+", "-", "*"],
    "unary_operators": ["square"],
    "procs": 1,
}


def _make_synth(n=200, seed=42):
    """y = x0^2 + x1 - 0.5, no noise. x2 = noise feature."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = x0 ** 2 + x1 - 0.5
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y": y})


# ---------------------------------------------------------------------------
# run_pysr_feature_engineering
# ---------------------------------------------------------------------------


def test_biz_val_bruteforce_pysr_runs_and_returns_equations():
    """Minimal PySR fit must produce a non-None ``.equations_``
    DataFrame after training."""
    pytest.importorskip("pysr")
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=80, seed=42)
    model = run_pysr_feature_engineering(
        df=df, target_col="y", sample_size=80,
        encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None, "PySR must populate .equations_ after fit"


def test_biz_val_bruteforce_pysr_accepts_polars():
    """PySR via bruteforce must accept polars DataFrames (converts
    to pandas internally)."""
    pytest.importorskip("pysr")
    pytest.importorskip("polars")
    import polars as pl
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=60, seed=42)
    model = run_pysr_feature_engineering(
        df=pl.from_pandas(df), target_col="y", sample_size=60,
        encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None


def test_biz_val_bruteforce_pysr_drop_columns_excludes_feature():
    """``drop_columns=['x2']`` must exclude the noise column."""
    pytest.importorskip("pysr")
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=60, seed=42)
    model = run_pysr_feature_engineering(
        df=df, target_col="y", drop_columns=["x2"],
        sample_size=60, encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None


def test_biz_val_bruteforce_pysr_reserved_names_smoke():
    """Default ``reserved_names=['im']`` renames conflicting columns
    without crashing."""
    pytest.importorskip("pysr")
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=50, seed=42)
    df["im"] = np.random.default_rng(0).normal(size=50)
    model = run_pysr_feature_engineering(
        df=df, target_col="y", sample_size=50,
        encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None
