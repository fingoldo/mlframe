"""Regression tests for MRMR / screen / discretization fixes.

Each test targets one fix from the batch. Tests are designed to FAIL on pre-fix code
and PASS once the corresponding fix lands.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _toy_dataset(n_rows: int = 200, n_cols: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float64)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int64)
    cols = [f"f{i}" for i in range(n_cols)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


# ----------------------------------------------------------------------
# Fix 1: target_prefix must not depend on global np.random state.
# Two MRMR(random_seed=42) instances should agree on the injected target
# column names; before the fix, the column name depended on the process
# global RNG state, which is non-deterministic across test orderings.
# ----------------------------------------------------------------------


def _capture_target_prefix(mrmr: MRMR, X: pd.DataFrame, y) -> str:
    """Trigger the prefix-generation path and return the prefix string used."""
    # The injected names start with the prefix. Probe the implementation directly:
    # _resolve_target_prefix is the seam introduced by the fix. The pre-fix
    # implementation has no such helper, so we fall back to running fit() once
    # and inspecting the columns mutation via a monkey-patched categorize_dataset.
    # Post-fix MRMR exposes _resolve_target_prefix as a stable seam; pre-fix this skip-mask hid
    # the absence of the seam. The fix has landed (mrmr.py:632), so the attribute is required.
    assert hasattr(mrmr, "_resolve_target_prefix"), "MRMR must expose _resolve_target_prefix; the fix at mrmr.py:632 regressed."
    return mrmr._resolve_target_prefix()


def test_fix1_target_prefix_reproducible_across_instances():
    """Two MRMR(random_seed=42) instances must produce identical target prefix."""
    X, y = _toy_dataset()

    m1 = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2)
    m2 = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2)

    p1 = _capture_target_prefix(m1, X, y)
    p2 = _capture_target_prefix(m2, X, y)
    assert p1 == p2, f"target prefix non-deterministic: {p1!r} vs {p2!r}"

    # Independent: a third instance with a different seed should differ.
    m3 = MRMR(random_seed=43, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2)
    p3 = _capture_target_prefix(m3, X, y)
    assert p3 != p1, "different seeds must produce different prefixes"


def test_fix1_target_prefix_no_global_rng_consumption():
    """Constructing/using MRMR(random_seed=42) must not advance global np.random state."""
    np.random.seed(12345)
    before = np.random.get_state()

    m = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2)
    _ = _capture_target_prefix(m, *_toy_dataset())

    after = np.random.get_state()
    # state tuple: ('MT19937', uint32 array, int, int, float) -- compare element-wise.
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2] == after[2]
    assert before[3] == after[3]
    assert before[4] == after[4]


# ----------------------------------------------------------------------
# Fix 2: pandas target injection must not leak when fit raises.
# ----------------------------------------------------------------------


def test_fix2_pandas_target_columns_cleaned_after_fit_exception(monkeypatch):
    """If fit raises after injection, caller's DataFrame must not retain targ_* cols."""
    X, y = _toy_dataset()
    original_columns = list(X.columns)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced failure inside fit() after target injection")

    # Force a failure inside the post-injection / pre-drop section by patching
    # categorize_dataset (called immediately after injection).
    monkeypatch.setattr(
        "mlframe.feature_selection.filters.mrmr.categorize_dataset",
        _boom,
    )

    m = MRMR(verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, skip_retraining_on_same_shape=False)
    with pytest.raises(RuntimeError, match="forced failure"):
        m.fit(X, y)

    assert list(X.columns) == original_columns, f"caller's DataFrame leaked target columns: extra={set(X.columns) - set(original_columns)}"


# ----------------------------------------------------------------------
# Fix 3: screen_predictors must not mutate global numpy RNG state.
# Direct call with a small dataset is heavyweight; instead exercise the
# entry-point branch responsible for the RNG side effect via the public
# MRMR.fit path with a seeded run. We compare global RNG state before and
# after a fit().
# ----------------------------------------------------------------------


def test_fix3_screen_does_not_mutate_global_numpy_rng():
    """A fit(random_seed=...) must NOT change the process-global numpy RNG state."""
    X, y = _toy_dataset()
    np.random.seed(99)
    before = np.random.get_state()

    m = MRMR(random_seed=42, verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, skip_retraining_on_same_shape=False, fe_max_steps=0)
    m.fit(X.copy(), y)

    after = np.random.get_state()
    assert before[0] == after[0]
    # Critical assertion: byte-identical uint32 state array.
    assert np.array_equal(before[1], after[1]), (
        "screen_predictors / MRMR.fit reseeded the global numpy RNG; use np.random.default_rng(seed) instead of np.random.seed(seed)."
    )


# ----------------------------------------------------------------------
# Fix 4: cv / cv_shuffle must be wired through to the RFECV call, not dead.
# We assert that constructing MRMR with cv=N records N, and that the RFECV
# call site reads it. Behavior-level: when run_additional_rfecv_minutes is
# off, cv is purely a stored constructor arg; we verify wiring via the
# private RFECV-builder helper.
# ----------------------------------------------------------------------


def test_fix4_cv_param_wired_into_rfecv_kwargs():
    """MRMR.cv and MRMR.cv_shuffle must be threaded into RFECV constructor kwargs."""
    m = MRMR(cv=7, cv_shuffle=True, verbose=0)
    # Post-fix MRMR exposes _rfecv_cv_kwargs as a stable seam; the fix at mrmr.py:674 has landed.
    assert hasattr(m, "_rfecv_cv_kwargs"), "MRMR must expose _rfecv_cv_kwargs; the fix at mrmr.py:674 regressed."
    kwargs = m._rfecv_cv_kwargs()
    assert kwargs.get("cv") == 7
    assert kwargs.get("cv_shuffle") is True


# ----------------------------------------------------------------------
# Fix 5: int64 -> int16 downcast notice must NOT go to stdout under verbose=0.
# ----------------------------------------------------------------------


def test_fix5_int64_downcast_silent_under_verbose0(capsys):
    """No stdout pollution when verbose=0 and int64 targets are downcast."""
    X, y = _toy_dataset()
    assert y.dtype == np.int64

    m = MRMR(verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, skip_retraining_on_same_shape=False, fe_max_steps=0)
    try:
        m.fit(X.copy(), y)
    except Exception:
        # We only care about stdout, not the (possibly noisy) fit completing.
        pass
    captured = capsys.readouterr()
    assert "Converted targets from int64 to int16" not in captured.out, f"stdout pollution under verbose=0: {captured.out!r}"


def test_fix5_int64_downcast_logged_under_verbose1(caplog):
    """Verbose >= 1 must route the downcast notice through logger.info, not print."""
    X, y = _toy_dataset()
    m = MRMR(verbose=1, n_jobs=1, full_npermutations=2, baseline_npermutations=2, skip_retraining_on_same_shape=False, fe_max_steps=0)
    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.filters.mrmr"):
        try:
            m.fit(X.copy(), y)
        except Exception:
            pass
    # Allow either substring match.
    msgs = " ".join(r.getMessage() for r in caplog.records)
    assert "int64" in msgs and "int16" in msgs, f"downcast not logged at INFO: {msgs!r}"


# ----------------------------------------------------------------------
# Fix 6: int64 -> int16 downcast must NOT silently truncate out-of-range values.
# ----------------------------------------------------------------------


def test_fix6_int64_downcast_skipped_when_out_of_int16_range():
    """Targets with values outside int16 must keep their original dtype."""
    X, _ = _toy_dataset(n_rows=200, n_cols=6, seed=0)
    rng = np.random.default_rng(0)
    # Values clearly outside [-32768, 32767].
    y = rng.integers(low=40_000, high=60_000, size=len(X)).astype(np.int64)
    y_orig_unique = sorted(set(y.tolist()))

    m = MRMR(verbose=0, n_jobs=1, full_npermutations=2, baseline_npermutations=2, skip_retraining_on_same_shape=False, fe_max_steps=0)
    # The downcast happens early in fit, BEFORE any categorisation. Reach in via the
    # newly-introduced helper.
    assert hasattr(m, "_coerce_target_dtype"), "MRMR must expose _coerce_target_dtype; the fix at mrmr.py:650 regressed."
    coerced = m._coerce_target_dtype(y)
    assert coerced.dtype != np.int16, f"out-of-range int64 silently downcast to int16; original max={max(y)}"
    # Round-trip preservation: unique values match (no truncation).
    assert sorted(set(coerced.tolist())) == y_orig_unique


# ----------------------------------------------------------------------
# Fix 7: random_state must alias to random_seed (or warn) so neither value is silently dropped.
# ----------------------------------------------------------------------


def test_fix7_random_state_aliases_random_seed():
    """MRMR(random_state=7) must surface 7 as the EFFECTIVE seed without dropping it.

    The alias is now resolved LAZILY (``_effective_random_seed``) instead of by mutating the ctor locals,
    so the stored constructor attributes stay byte-identical to what the user passed (sklearn ``get_params``
    round-trip contract): ``random_seed`` stays None, ``random_state`` echoes 7. The real contract -- no seed
    silently dropped -- is checked on the EFFECTIVE value, which is what fit consumes."""
    m = MRMR(random_state=7)
    # Stored ctor attrs stay pristine (no cross-contamination): random_state-only leaves random_seed None.
    assert m.random_seed is None
    assert m.random_state == 7
    # The effective seed surfaces 7 and agrees with the random_seed-direct API.
    m2 = MRMR(random_seed=7)
    assert m._effective_random_seed() == 7
    assert m._effective_random_seed() == m2._effective_random_seed()


# ----------------------------------------------------------------------
# Fix 8: _lazy_chunks helper must remain in place (parallel session refactor).
# This is a forward-protection assertion.
# ----------------------------------------------------------------------


def test_fix8_lazy_chunks_helper_present():
    """The lazy combinations chunker must remain at module scope."""
    from mlframe.feature_selection.filters import mrmr as mrmr_mod

    assert hasattr(mrmr_mod, "_lazy_chunks"), "_lazy_chunks helper missing"
    # Functional sanity: chunking [0..10] by 3 yields 4 chunks.
    chunks = list(mrmr_mod._lazy_chunks(range(10), 3))
    assert len(chunks) == 4
    assert chunks[0] == [0, 1, 2]
    assert chunks[-1] == [9]
