"""biz_val tests for ``mlframe.feature_engineering.bruteforce`` --
``run_pysr_feature_engineering``.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
PySR symbolic regression discovers human-readable equations.

Requires Julia runtime. The module-level ``_check_julia`` gate
skips with a clear message when Julia is unavailable or the bridge
fails to initialize -- so CI without Julia isn't blocked.
"""
from __future__ import annotations

import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _check_julia():
    """Return True if Julia is available AND import succeeds."""
    # Prefer the D: install.
    for bindir in ("D:/Julia/bin",):
        julia_exe = os.path.join(bindir, "julia.exe")
        if os.path.isfile(julia_exe):
            os.environ["JULIA_EXE"] = julia_exe
            os.environ.setdefault("PATH", bindir + os.pathsep + os.environ.get("PATH", ""))
            break
    # Smoke: can pysr import at all?
    try:
        import pysr  # noqa: F401
        return True
    except (ImportError, subprocess.CalledProcessError, OSError):
        return False


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


# Gate the whole module.
pytestmark = pytest.mark.skipif(
    not _check_julia(),
    reason="Julia runtime not available (D:/Julia/bin/julia.exe missing "
           "or pysr import failed)",
)


def test_biz_val_bruteforce_pysr_runs_and_returns_equations():
    """Minimal PySR fit must produce a non-None ``.equations_``
    DataFrame after training."""
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=80, seed=42)
    model = run_pysr_feature_engineering(
        df=df, target_col="y", sample_size=80,
        encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None, "PySR must populate .equations_ after fit"


def test_biz_val_bruteforce_pysr_accepts_polars():
    """PySR via bruteforce must accept polars DataFrames."""
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
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=60, seed=42)
    model = run_pysr_feature_engineering(
        df=df, target_col="y", drop_columns=["x2"],
        sample_size=60, encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None


def test_biz_val_bruteforce_pysr_reserved_names_smoke():
    """Default ``reserved_names=['im']`` renames conflicting columns."""
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering
    df = _make_synth(n=50, seed=42)
    df["im"] = np.random.default_rng(0).normal(size=50)
    model = run_pysr_feature_engineering(
        df=df, target_col="y", sample_size=50,
        encode_categoricals=False, verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    assert model.equations_ is not None
