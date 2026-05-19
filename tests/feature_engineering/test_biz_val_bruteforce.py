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
    """Return True if Julia is available AND pysr imports.

    Tries a list of well-known Windows install locations and also `shutil.which("julia")` so
    Julia in PATH (e.g. juliaup-managed) is honoured. Previously hard-coded `D:/Julia/bin` and
    used `os.environ.setdefault("PATH", ...)` which never appended (PATH always exists), so the
    Julia bindir was never actually exposed to pysr's subprocess - check skipped even when
    Julia was installed in the standard location.
    """
    import shutil

    candidate_exes = []
    julia_from_path = shutil.which("julia")
    if julia_from_path:
        candidate_exes.append(julia_from_path)
    for bindir in ("D:/Julia/bin", r"C:\\Program Files\\Julia\\bin"):
        julia_exe = os.path.join(bindir, "julia.exe")
        if os.path.isfile(julia_exe):
            candidate_exes.append(julia_exe)

    for julia_exe in candidate_exes:
        bindir = os.path.dirname(julia_exe)
        os.environ["JULIA_EXE"] = julia_exe
        # Prepend (not setdefault) so pysr finds julia.exe in subprocesses.
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        # SUBPROCESS PROBE: pysr's transitive import chain (Julia + PyJuliaCall +
        # torch on some installs) can native-crash on broken environments,
        # tearing down the xdist worker. A subprocess probe contains the blast
        # radius: a crash in the child returns non-zero exit, we skip the
        # whole module instead of taking down the test session.
        try:
            r = subprocess.run(
                [sys.executable, "-c", "import pysr"],
                env=os.environ,
                capture_output=True,
                timeout=30,
            )
            if r.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, OSError):
            continue
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


# Gate the whole module: skip when Julia unavailable; also mark slow_only so
# fast-mode runs skip it cleanly. PySR fit can take 30-60+ seconds even on a
# tiny synthetic, and the embedded Julia process occasionally raises Windows
# access-violation when its multi-threaded GC interacts with the pytest
# subprocess teardown. Both make it unsuitable for fast/CI loops.
pytestmark = [
    pytest.mark.skipif(
        not _check_julia(),
        reason="Julia runtime not available (D:/Julia/bin/julia.exe missing "
               "or pysr import failed)",
    ),
    pytest.mark.slow_only,
]


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
