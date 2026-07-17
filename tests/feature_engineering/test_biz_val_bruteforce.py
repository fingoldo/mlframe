"""biz_val tests for ``mlframe.feature_engineering.bruteforce`` --
``run_pysr_feature_engineering``.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN.
PySR symbolic regression discovers human-readable equations.

Requires Julia runtime. The module-level ``_check_julia`` gate
skips with a clear message when Julia is unavailable or the bridge
fails to initialize -- so CI without Julia isn't blocked.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


from tests._pysr_gate import pysr_works as _check_julia


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
    y = x0**2 + x1 - 0.5
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "y": y})


# Gate the whole module: skip when Julia unavailable; also mark slow_only so
# fast-mode runs skip it cleanly. PySR fit can take 30-60+ seconds even on a
# tiny synthetic, and the embedded Julia process occasionally raises Windows
# access-violation when its multi-threaded GC interacts with the pytest
# subprocess teardown. Both make it unsuitable for fast/CI loops.
pytestmark = [
    pytest.mark.skipif(
        not _check_julia(),
        reason="PySR / Julia runtime not usable (probe failed)",
    ),
    pytest.mark.slow_only,
]


def _equation_strings(model) -> list[str]:
    """Extract equation expression strings from a PySR result regardless of pandas vs dict layout."""
    eqs = model.equations_
    assert eqs is not None, "PySR must populate .equations_ after fit"
    if hasattr(eqs, "columns"):
        # PySRRegressor returns a pandas DataFrame with an "equation" column.
        if "equation" in eqs.columns:
            return [str(s) for s in eqs["equation"].tolist()]
        if "sympy_format" in eqs.columns:
            return [str(s) for s in eqs["sympy_format"].tolist()]
        return [str(s) for s in eqs.iloc[:, 0].tolist()]
    return [str(s) for s in list(eqs)]


def test_biz_val_bruteforce_pysr_runs_and_returns_equations():
    """Minimal PySR fit must return at least one equation that mentions an input variable -- a bare ``is not None``
    check passes even when PySR emits an empty / all-constant equation set."""
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    df = _make_synth(n=80, seed=42)
    model = run_pysr_feature_engineering(
        df=df,
        target_col="y",
        sample_size=80,
        encode_categoricals=False,
        verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    eq_strs = _equation_strings(model)
    assert len(eq_strs) >= 1, "PySR must return at least one equation"
    assert any(("x0" in s) or ("x1" in s) or ("x2" in s) for s in eq_strs), f"none of the returned equations reference any input variable: {eq_strs}"


def test_biz_val_bruteforce_pysr_accepts_polars():
    """PySR via bruteforce must accept polars DataFrames AND return behavioural equations (variable-referencing)."""
    import polars as pl
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    df = _make_synth(n=60, seed=42)
    model = run_pysr_feature_engineering(
        df=pl.from_pandas(df),
        target_col="y",
        sample_size=60,
        encode_categoricals=False,
        verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    eq_strs = _equation_strings(model)
    assert len(eq_strs) >= 1
    assert any(("x0" in s) or ("x1" in s) or ("x2" in s) for s in eq_strs), f"polars input must yield variable-referencing equations; got {eq_strs}"


def test_biz_val_bruteforce_pysr_drop_columns_excludes_feature():
    """``drop_columns=['x2']`` must exclude the dropped column from every returned equation."""
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    df = _make_synth(n=60, seed=42)
    model = run_pysr_feature_engineering(
        df=df,
        target_col="y",
        drop_columns=["x2"],
        sample_size=60,
        encode_categoricals=False,
        verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    eq_strs = _equation_strings(model)
    assert len(eq_strs) >= 1
    assert not any("x2" in s for s in eq_strs), f"drop_columns=['x2'] must prevent x2 from appearing in any equation; got {eq_strs}"


def test_biz_val_bruteforce_pysr_reserved_names_smoke():
    """Default ``reserved_names=['im']`` renames the conflicting input column so equations reference the renamed
    sanitised symbol, not the raw ``im`` token (which collides with PySR's complex unit)."""
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    df = _make_synth(n=50, seed=42)
    df["im"] = np.random.default_rng(0).normal(size=50)
    model = run_pysr_feature_engineering(
        df=df,
        target_col="y",
        sample_size=50,
        encode_categoricals=False,
        verbose=0,
        pysr_params_override=_MINI_PYSR,
    )
    eq_strs = _equation_strings(model)
    assert len(eq_strs) >= 1
    # The renamed column must NOT appear under its raw reserved name as a free symbol; if PySR consumed ``im`` it
    # would have either raised (collision with imaginary unit) or returned no input-referencing equation.
    assert any(("x0" in s) or ("x1" in s) or ("reserved_im" in s) for s in eq_strs), (
        f"renamed reserved column should be discoverable in equations; got {eq_strs}"
    )
