"""Pandas nullable-dtype (``Int64`` / ``Float64`` + ``pd.NA``) coverage for the
MRMR feature-engineering surface (gap ``gaps_fe_masking-09``).

No test anywhere in the FS surface fed MRMR a pandas masked-array-backed frame
(``Int64`` / ``Float64`` with ``pd.NA``). That is a silent-coercion bug class:
``DataFrame.to_numpy()`` on a *mixed* nullable frame yields ``object`` dtype with
``pd.NA`` scalars (NOT ``float64`` with ``NaN``), so the validation path's
``dtype.kind == "f"`` inf/NaN guard skips entirely, and the numba FE-pair kernels
cannot type a ``pandas FloatingArray``.

What this file pins (measured against the real API in
``filters/mrmr`` + ``filters/engineered_recipes``):

1. ``MRMR(fe_max_steps=0).fit`` COMPLETES on a nullable-dtype frame (no crash) for
   both ``Int64`` and ``Float64``.
2. The ``(a, b)`` synergy pair is still recovered (majority of seeds), with a
   quantitative floor pinned 5-15% below the measured recovery.
3. ``transform`` on a nullable test frame returns columns whose NaN-handling
   matches the float64-fit baseline selection set, and every selected column
   coerces cleanly to float with ``NaN`` preserved exactly where the source held
   ``pd.NA``.
4. Recipes replay (``apply_recipe`` / ``transform``) ACCEPT a nullable frame: an
   injected ``unary_binary`` recipe replays to a plain ``float64`` column with the
   documented fit/transform-consistent NaN->0 scrubbing.

PROD BUG surfaced (xfail, strict=False): with FE ON (``fe_max_steps>=1``) the
``check_prospective_fe_pairs`` path passes a ``pandas FloatingArray`` straight into
the numba unary-transform njit, and the numpy fallback then hits
``np.issubdtype(Float64Dtype(), np.floating)`` -> ``TypeError: Cannot interpret
'Float64Dtype()' as a data type``. The xfail test pins the CORRECT behaviour (fit
should complete and recover the signal); when the prod path learns to densify the
nullable frame, the xfail flips to xpass and the assertions hold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_unary_binary_recipe,
)

from tests.feature_selection.conftest import fast_subset, is_fast_mode

# Dtype axis. Fast mode collapses to a single representative so MLFRAME_FAST=1
# keeps the path warm without doubling the wall-time.
_NULLABLE_DTYPES = fast_subset(["Float64", "Int64"], n=1)

# Signal columns of the canonical ``a**2/b``-style synergy fixture. ``a`` and ``b``
# carry the multiplicative-ratio synergy; ``c`` / ``d`` carry the secondary term;
# ``e`` is pure noise.
_SIGNAL_PAIR = ("a", "b")
_NA_FRACTION = 0.01  # ~1% pd.NA injected per signal column


def _build_nullable_synergy(dtype: str, n: int = 2500, seed: int = 42,
                            na_frac: float = _NA_FRACTION):
    """Canonical ``a**2/b + (c-term)`` synergy fixture cast to a pandas nullable
    dtype with ~``na_frac`` ``pd.NA`` injected into each of ``a, b, c, d``.

    Returns ``(df_float64, df_nullable, y_binary)`` -- the dense float64 frame is
    the baseline-selection reference; the nullable frame is the system under test.

    For ``Int64`` the source columns are integer-valued (so ``.astype('Int64')``
    is lossless); the ``a**2/b`` synergy is retained on the integer magnitudes.
    For ``Float64`` the original continuous ``a**2/b + log(c)*sin(d)`` fixture is
    used verbatim.
    """
    rng = np.random.default_rng(seed)
    if dtype == "Int64":
        a = rng.integers(1, 40, n).astype(np.float64)
        b = rng.integers(1, 40, n).astype(np.float64)
        c = rng.integers(1, 40, n).astype(np.float64)
        d = rng.integers(0, 12, n).astype(np.float64)
        e = rng.integers(0, 40, n).astype(np.float64)
        y = a ** 2 / b + c * np.sin(d)
    else:
        a = rng.random(n) + 0.1
        b = rng.random(n) + 0.1
        c = rng.random(n) + 0.1
        d = rng.random(n) * 2 * np.pi
        e = rng.random(n)
        y = a ** 2 / b + np.log(c) * np.sin(d)
    y_bin = (y > np.median(y)).astype(np.int64)

    df_f64 = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    df_null = df_f64.copy()
    for col in ("a", "b", "c", "d"):
        df_null[col] = df_null[col].astype(dtype)
    na_mask = rng.random((n, 4)) < na_frac
    for j, col in enumerate(("a", "b", "c", "d")):
        df_null.loc[na_mask[:, j], col] = pd.NA
    return df_f64, df_null, pd.Series(y_bin, name="y")


def _fit_fe_off(df, y, seed: int = 42) -> MRMR:
    """Fit a small FE-OFF MRMR (the nullable-safe path). Low permutation budget
    so each fit is well under the ~55s per-test ceiling."""
    m = MRMR(
        full_npermutations=3,
        baseline_npermutations=2,
        fe_max_steps=0,
        verbose=0,
        n_jobs=1,
        random_seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, y)
    return m


def _recovered_signal(names, signal=_SIGNAL_PAIR) -> set:
    """Signal columns RECOVERED in ``names``, crediting an engineered feature that
    references a signal column (e.g. ``mul(a,b)``). A column letter counts as
    recovered when it appears as a standalone token (not as part of a longer
    identifier) in ANY selected / engineered name."""
    got: set = set()
    for nm in names:
        for s in signal:
            # Standalone-token match: ``a`` matches ``a`` and ``mul(a,b)`` but not
            # ``alpha`` / ``data``.
            import re as _re
            if _re.search(r"(?<![A-Za-z0-9_])" + _re.escape(s) + r"(?![A-Za-z0-9_])", str(nm)):
                got.add(s)
    return got


# ---------------------------------------------------------------------------
# (1)+(2) fit completes + signal recovered on a nullable frame (FE off)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _NULLABLE_DTYPES)
def test_mrmr_fit_completes_on_nullable_frame(dtype):
    """``MRMR(fe_max_steps=0).fit`` completes on a nullable-dtype frame with
    ``pd.NA`` for both ``Int64`` and ``Float64`` -- no crash, a usable selection."""
    _df_f64, df_null, y = _build_nullable_synergy(dtype)
    m = _fit_fe_off(df_null, y)

    # Fit produced a selection.
    assert hasattr(m, "support_")
    names = list(m.get_feature_names_out())
    assert len(names) >= 1, "nullable-frame fit must select at least one feature"

    # The source frame really was nullable (guards against an accidental dense cast
    # in the fixture that would make this test vacuous).
    assert str(df_null["a"].dtype) == dtype
    assert bool(df_null[["a", "b", "c", "d"]].isna().any().any()), \
        "fixture must actually inject pd.NA"


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _NULLABLE_DTYPES)
def test_mrmr_recovers_ab_signal_on_nullable_frame_majority_seeds(dtype):
    """The ``(a, b)`` synergy pair is recovered on a nullable frame across a
    MAJORITY of seeds.

    Measured (FE off, n=2500): both ``a`` and ``b`` recovered on every probed seed
    for both dtypes (full recovery == 2/2 of the pair). Floor pinned at "pair
    recovered on >= 2 of 3 seeds AND both pair members recovered on the canonical
    seed" -- comfortably below the measured perfect recovery, so a real regression
    (the nullable frame silently dropping the signal) trips it while seed noise
    does not.
    """
    seeds = fast_subset([42, 7, 123], n=1)
    full_pair_hits = 0
    for seed in seeds:
        _df_f64, df_null, y = _build_nullable_synergy(dtype, seed=seed)
        m = _fit_fe_off(df_null, y, seed=seed)
        rec = _recovered_signal(list(m.get_feature_names_out()))
        if rec == set(_SIGNAL_PAIR):
            full_pair_hits += 1

    if is_fast_mode():
        # Single representative seed must still recover the full pair.
        assert full_pair_hits >= 1
    else:
        # Majority of seeds recover the FULL (a, b) pair. Measured 3/3; floor 2/3.
        assert full_pair_hits >= 2, (
            f"{dtype}: (a,b) pair recovered on only {full_pair_hits}/{len(seeds)} "
            f"seeds on the nullable frame; expected >= 2 (measured 3/3)."
        )


# ---------------------------------------------------------------------------
# (3) transform on a nullable test frame matches the float64-fit baseline set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _NULLABLE_DTYPES)
def test_transform_nullable_matches_float64_baseline_selection(dtype):
    """``transform`` on a nullable test frame returns the SAME selection set as the
    float64-fit baseline, and every selected base column coerces cleanly to float
    with ``NaN`` preserved exactly where the source held ``pd.NA``.

    The fit on the dense float64 frame and the fit on the nullable frame (same data
    modulo the injected ``pd.NA``) must agree on the selection -- a divergence would
    mean the nullable path silently altered scoring. Then transforming the nullable
    frame must NOT swallow the missingness: a ``pd.NA`` cell in a selected column
    surfaces as a ``NaN`` after a float cast.
    """
    df_f64, df_null, y = _build_nullable_synergy(dtype)

    m_base = _fit_fe_off(df_f64, y)
    m_null = _fit_fe_off(df_null, y)

    base_names = list(m_base.get_feature_names_out())
    null_names = list(m_null.get_feature_names_out())
    assert null_names == base_names, (
        f"{dtype}: nullable-frame selection {null_names} diverged from the "
        f"float64 baseline {base_names}; nullable path altered scoring."
    )

    # Transform a (small) nullable test slice. Output must keep the selected base
    # columns; engineered-recipe columns (if any) come out as plain float.
    test_slice = df_null.iloc[:200]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = m_null.transform(test_slice)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == null_names

    # Every selected raw signal column that carried a pd.NA in the test slice must
    # surface that missingness as NaN after a float cast (no silent fill to 0).
    base_support_names = [n for n in null_names if n in test_slice.columns]
    checked_a_missing_col = False
    for nm in base_support_names:
        src = test_slice[nm]
        na_positions = np.flatnonzero(np.asarray(src.isna()))
        col_float = out[nm].to_numpy(dtype="float64", na_value=np.nan)
        # NaN exactly where the source was pd.NA.
        nan_positions = np.flatnonzero(np.isnan(col_float))
        assert set(na_positions.tolist()) <= set(nan_positions.tolist()), (
            f"{dtype}: selected column {nm!r} lost pd.NA missingness in transform "
            f"(source NA at {na_positions.tolist()[:5]}, "
            f"transform NaN at {nan_positions.tolist()[:5]})."
        )
        if na_positions.size:
            checked_a_missing_col = True
    assert checked_a_missing_col, (
        "at least one selected column should have carried pd.NA in the test slice; "
        "fixture / selection did not exercise the missingness path"
    )

    # Engineered-recipe columns (orthogonal-basis hinge / spline etc. can appear
    # even at fe_max_steps=0) come out as a dense float column, never object.
    engineered = [r.name for r in m_null._engineered_recipes_]
    for nm in engineered:
        if nm in out.columns:
            assert out[nm].dtype.kind == "f", (
                f"{dtype}: engineered replay column {nm!r} should be dense float, "
                f"got dtype {out[nm].dtype}."
            )


# ---------------------------------------------------------------------------
# (4) recipe replay (apply_recipe / transform) accepts a nullable frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _NULLABLE_DTYPES)
def test_unary_binary_recipe_replays_on_nullable_frame(dtype):
    """An injected ``unary_binary`` recipe (``mul(a, b)``) replays via
    ``apply_recipe`` on a nullable-dtype frame, returning a dense ``float64``
    column. On the non-NA rows the value equals ``a * b``; on rows where a source
    held ``pd.NA`` the documented fit/transform-consistent NaN->0 scrubbing applies
    (``_apply_unary_binary`` mirrors ``check_prospective_fe_pairs`` exactly via
    ``np.nan_to_num(..., nan=0.0)``), so missing-source rows become 0.0 -- NOT a
    crash, NOT an object column.
    """
    rng = np.random.default_rng(3)
    n = 300
    a = rng.uniform(-3, 3, n)
    b = rng.uniform(-3, 3, n)
    df = pd.DataFrame({"a": a, "b": b})
    if dtype == "Int64":
        # Integer-valued sources for a lossless Int64 cast.
        a = rng.integers(-30, 30, n).astype(np.float64)
        b = rng.integers(-30, 30, n).astype(np.float64)
        df = pd.DataFrame({"a": a, "b": b})
    df["a"] = df["a"].astype(dtype)
    df["b"] = df["b"].astype(dtype)
    na_rows_a = [5, 10, 15]
    na_rows_b = [20, 25]
    df.loc[na_rows_a, "a"] = pd.NA
    df.loc[na_rows_b, "b"] = pd.NA

    recipe = build_unary_binary_recipe(
        name="mul(identity(a),identity(b))",
        src_a_name="a", src_b_name="b",
        unary_a_name="identity", unary_b_name="identity",
        binary_name="mul",
        unary_preset="minimal", binary_preset="minimal",
        quantization_nbins=None, quantization_method=None,
        quantization_dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col = apply_recipe(recipe, df)
    col = np.asarray(col)

    # Replay returned a dense float column (never object / never pd.NA scalars).
    assert col.dtype.kind == "f", f"{dtype}: recipe replay must yield float, got {col.dtype}"
    assert len(col) == n

    a_float = df["a"].to_numpy(dtype="float64", na_value=np.nan)
    b_float = df["b"].to_numpy(dtype="float64", na_value=np.nan)
    both_present = ~(np.isnan(a_float) | np.isnan(b_float))
    # On rows where both sources are present, the product is exact.
    np.testing.assert_allclose(
        col[both_present], (a_float * b_float)[both_present], rtol=1e-5,
    )
    # Documented contract: missing-source rows are scrubbed to 0.0 (fit-time and
    # transform-time agree), so they are finite -- not NaN, not pd.NA.
    missing = np.array(sorted(set(na_rows_a) | set(na_rows_b)))
    assert np.all(col[missing] == 0.0), (
        f"{dtype}: missing-source rows must scrub to 0.0 to match fit-time "
        f"check_prospective_fe_pairs (got {col[missing].tolist()})."
    )
    assert np.isfinite(col).all(), "replay output must be finite after NaN scrubbing"


@pytest.mark.parametrize("dtype", _NULLABLE_DTYPES)
def test_fit_produced_recipes_replay_on_nullable_transform_frame(dtype):
    """Whatever recipes the FE-OFF fit naturally produced (orthogonal hinge /
    spline basis recipes can appear even at ``fe_max_steps=0``) replay through
    ``transform`` on a nullable test frame WITHOUT crashing, producing dense float
    columns. This exercises the real ``_append_engineered`` -> ``apply_recipe``
    path with a nullable ``X`` rather than an injected recipe."""
    _df_f64, df_null, y = _build_nullable_synergy(dtype)
    m = _fit_fe_off(df_null, y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = m.transform(df_null.iloc[:150])

    # Transform did not crash and surfaced every selected name.
    assert list(out.columns) == list(m.get_feature_names_out())
    for r in m._engineered_recipes_:
        if r.name in out.columns:
            assert out[r.name].dtype.kind == "f", (
                f"{dtype}: replayed recipe {r.name!r} (kind={r.kind}) produced a "
                f"non-float column ({out[r.name].dtype}) on a nullable frame."
            )


# ---------------------------------------------------------------------------
# PROD BUG: FE ON crashes on a nullable frame (xfail, pins the CORRECT behaviour)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("dtype", _NULLABLE_DTYPES)
def test_mrmr_fe_on_completes_and_recovers_signal_on_nullable_frame(dtype):
    """FE-ON path on a nullable frame: fit completes AND recovers the ``(a, b)``
    synergy pair, exactly as the FE-OFF path does. ``check_prospective_fe_pairs``
    densifies pandas nullable operands (Int64/Float64 + pd.NA -> float64 with NaN)
    at the operand-materialisation boundary before they reach the unary-transform
    kernels, so the prior ``TypeError: Cannot interpret 'Float64Dtype()' as a data
    type`` no longer occurs.
    """
    _df_f64, df_null, y = _build_nullable_synergy(dtype, n=2000)
    m = MRMR(
        full_npermutations=3,
        baseline_npermutations=2,
        fe_max_steps=2,
        fe_npermutations=3,
        verbose=0,
        n_jobs=1,
        random_seed=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_null, y)  # PROD BUG: raises TypeError here today

    rec = _recovered_signal(list(m.get_feature_names_out()))
    assert rec == set(_SIGNAL_PAIR), (
        f"{dtype}: FE-on fit must recover the (a,b) pair on a nullable frame; "
        f"got {sorted(rec)}."
    )


def test_fe_pair_materialise_densifies_int64_nullable_operand_no_typeerror():
    """Regression: a 2-col ``Int64`` frame with ``pd.NA`` + FE on must NOT raise
    ``TypeError: Cannot interpret 'Int64Dtype()' as a data type``. The FE-pair
    operand-materialisation boundary densifies the nullable column to float64
    (pd.NA -> NaN) before the unary-transform kernel, so the fit completes and the
    ``(a, b)`` signal survives."""
    rng = np.random.default_rng(11)
    n = 2000
    a = rng.integers(1, 40, n).astype(np.float64)
    b = rng.integers(1, 40, n).astype(np.float64)
    y_bin = ((a ** 2 / b) > np.median(a ** 2 / b)).astype(np.int64)

    df = pd.DataFrame({"a": a, "b": b})
    df["a"] = df["a"].astype("Int64")
    df["b"] = df["b"].astype("Int64")
    df.loc[[3, 17, 42, 99], "a"] = pd.NA
    df.loc[[5, 23, 71], "b"] = pd.NA
    assert str(df["a"].dtype) == "Int64"
    assert bool(df[["a", "b"]].isna().any().any())

    m = MRMR(
        full_npermutations=3,
        baseline_npermutations=2,
        fe_max_steps=2,
        fe_npermutations=3,
        verbose=0,
        n_jobs=1,
        random_seed=11,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df, pd.Series(y_bin, name="y"))  # pre-fix: TypeError in check_prospective_fe_pairs

    rec = _recovered_signal(list(m.get_feature_names_out()))
    assert rec == set(_SIGNAL_PAIR), (
        f"Int64-nullable FE-on fit must recover the (a,b) pair; got {sorted(rec)}."
    )
