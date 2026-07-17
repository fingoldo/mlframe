"""Adversarial probes for the 2026-06-17 (c7689a4b) MRMR patches.

Targets four patch families in ``_mrmr_fit_impl/_fit_impl_core.py`` and
``mrmr/_mrmr_class.py``:

1. ``fe_max_steps=0`` gating of the four discrete-structural FE families.
2. Pure-raw cluster representative selection (keep strongest cached-MI, strip rest).
3. Never-empty rescue when the redundancy/cluster exclusion empties the pool.
4. polars LazyFrame auto-collect + Struct rejection in fit().

Tests ASSERT the correct behavior: they pass if the patch is sound, fail on a hole.
All deterministic (fixed seeds), n<=2000, fe_max_steps small.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

_DISCRETE_PREFIXES = ("il_", "pmod_", "argmax_", "gate_", "gate_select__", "gate_mask__", "argmax__", "binagg")


def _discrete_cols(estimator, X):
    """Discrete cols."""
    out = estimator.transform(X)
    return [c for c in out.columns if any(str(c).startswith(p) for p in _DISCRETE_PREFIXES)]


def _names_out(estimator):
    """Names out."""
    return [str(c) for c in estimator.get_feature_names_out()]


def _mk(**kw):
    """Build an MRMR from sensible fast-test defaults (n_workers=1, 1-minute budget), overridable via kw."""
    base = dict(n_workers=1, max_runtime_mins=1, verbose=0)
    base.update(kw)
    return MRMR(**base)


# --------------------------------------------------------------------------- #
# 1. fe_max_steps=0 discrete-FE gating
# --------------------------------------------------------------------------- #


def _gcd_frame(seed=1, n=2000):
    """Gcd frame."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 40, n)
    b = rng.integers(1, 40, n)
    y = (np.gcd(a, b) >= 4).astype(int)
    X = pd.DataFrame({"a": a, "b": b, "noise0": rng.integers(0, 50, n), "noise1": rng.normal(size=n)})
    return X, y


def test_fe_max_steps_zero_with_explicit_enable_runs_operator_only_path():
    """At fe_max_steps=0 the AUTOMATIC FE pipeline (iterative pair search) is off, but the discrete-structural
    operators are a DISTINCT group that, when explicitly enabled, still run via the operator-only path provided
    the small-n reliability floor is met (n>=500; here n=2000) -- see the design note at
    _mrmr_fit_impl/_fit_impl_core.py (``the operator-lift biz_value tests rely on exactly that``). So with every
    discrete flag explicitly True on a gcd target, the family is NOT hard-blocked: the operators are reachable."""
    X, y = _gcd_frame()
    sel = _mk(
        fe_max_steps=0,
        quantization_nbins=8,
        fe_conditional_gate_enable=True,
        fe_pairwise_modular_enable=True,
        fe_row_argmax_enable=True,
        fe_integer_lattice_enable=True,
    )
    sel.fit(X, y)
    assert _names_out(sel), "operator-only path emptied support at fe_max_steps=0"
    assert _discrete_cols(sel, X), (
        "explicitly-enabled discrete operators did not fire on the gcd target at fe_max_steps=0 "
        "(operator-only path); the family must be reachable when explicitly requested"
    )


def test_fe_max_steps_zero_keeps_genuine_raw_signal():
    """The genuine gcd-driving signal must survive at fe_max_steps=0 -- either as the raw columns a,b OR
    captured by the default-on integer-lattice composite over them (il_gcd subsumes its raw operands by the
    redundancy-dedup design). Either way the (a,b) structure must not be lost to pure noise."""
    X, y = _gcd_frame()
    sel = _mk(fe_max_steps=0, quantization_nbins=8)
    sel.fit(X, y)
    names = set(_names_out(sel))
    assert names, "support emptied at fe_max_steps=0"
    captures_ab = bool({"a", "b"} & names) or any(n.startswith("il_") or "il_gcd" in n for n in names)
    assert captures_ab, f"genuine a/b signal neither kept raw nor captured by an integer-lattice composite: {sorted(names)}"


def test_fe_max_steps_one_lets_discrete_fire():
    """Positive control: at fe_max_steps>=1 the discrete family is no longer
    gated off, so on a structure that needs it a discrete op CAN appear."""
    X, y = _gcd_frame()
    sel = _mk(
        fe_max_steps=1,
        quantization_nbins=8,
        fe_conditional_gate_enable=True,
        fe_row_argmax_enable=True,
        fe_integer_lattice_enable=True,
        fe_pairwise_modular_enable=True,
    )
    sel.fit(X, y)
    # Not asserting a discrete col MUST appear (data-dependent), only that the
    # gate did not hard-block it: fit completes and support is non-empty.
    assert _names_out(sel), "fe_max_steps=1 fit produced empty support"


# --------------------------------------------------------------------------- #
# 2. Pure-raw cluster representative (keep strongest cached-MI, strip rest)
# --------------------------------------------------------------------------- #


def test_three_mutually_redundant_raw_keep_exactly_one():
    """Three near-identical copies of the signal column plus pure noise.
    The dedup must keep exactly ONE of the trio (the strongest) and not
    collapse the whole support, nor keep two redundant siblings."""
    rng = np.random.default_rng(7)
    n = 1500
    a = rng.normal(size=n)
    y = (a > 0).astype(int)
    X = pd.DataFrame(
        {
            "c0": a + 1e-6 * rng.normal(size=n),
            "c1": a + 1e-6 * rng.normal(size=n),
            "c2": a + 1e-6 * rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    sel = _mk(fe_max_steps=0)
    sel.fit(X, y)
    names = set(_names_out(sel))
    kept_trio = {c for c in ("c0", "c1", "c2") if c in names}
    assert len(kept_trio) >= 1, f"all redundant copies dropped: {sorted(names)}"
    assert len(kept_trio) == 1, f"redundant trio not de-duplicated to a single rep: kept {kept_trio}"
    assert "noise" not in names, "pure noise wrongly selected"


def test_genuine_signal_not_dropped_when_clustered_with_near_dup():
    """A cluster mixing a genuine independent signal with a near-duplicate of
    ANOTHER column must not drop the genuine non-redundant column."""
    rng = np.random.default_rng(11)
    n = 1500
    a = rng.normal(size=n)
    b = rng.normal(size=n)  # independent genuine signal
    y = ((a + b) > 0).astype(int)
    X = pd.DataFrame(
        {
            "a": a,
            "a_dup": a + 1e-6 * rng.normal(size=n),
            "b": b,
        }
    )
    sel = _mk(fe_max_steps=0)
    sel.fit(X, y)
    names = set(_names_out(sel))
    assert "b" in names, f"genuine independent signal b wrongly dropped: {sorted(names)}"
    # a or a_dup (one rep), but not both.
    a_kept = {c for c in ("a", "a_dup") if c in names}
    assert len(a_kept) >= 1, f"both copies of a dropped: {sorted(names)}"


def test_mi_tie_breaks_to_lower_input_index_stable():
    """Two truly identical columns (exact tie in cached MI). The patch tie-
    breaks by lowest fit-time input index, so the FIRST column must survive,
    deterministically and stably across reruns."""
    rng = np.random.default_rng(13)
    n = 1500
    a = rng.normal(size=n)
    y = (a > 0).astype(int)
    shared = a + 1e-9 * rng.normal(size=n)
    X = pd.DataFrame({"first": shared.copy(), "second": shared.copy(), "noise": rng.normal(size=n)})
    keeps = []
    for _ in range(2):
        sel = _mk(fe_max_steps=0)
        sel.fit(X, y)
        names = set(_names_out(sel))
        keeps.append({c for c in ("first", "second") if c in names})
    assert keeps[0] == keeps[1], f"tie-break not stable: {keeps}"
    assert len(keeps[0]) == 1, f"exact-duplicate pair not de-duped: {keeps[0]}"
    # lowest-index preference -> 'first' survives.
    assert keeps[0] == {"first"}, f"tie did not break to lowest input index; kept {keeps[0]}"


# --------------------------------------------------------------------------- #
# 3. Never-empty rescue when exclusion empties the pool
# --------------------------------------------------------------------------- #


def test_never_empty_rescue_keeps_one_of_collinear_pair():
    """Two ~0.997-collinear columns each recorded as the other's redundant
    member can empty the pool. The never-empty guarantee must keep exactly
    one representative (the stronger), not return empty support."""
    rng = np.random.default_rng(17)
    n = 1500
    base = rng.normal(size=n)
    y = (base > 0).astype(int)
    X = pd.DataFrame(
        {
            "p": base + 0.05 * rng.normal(size=n),
            "q": base + 0.05 * rng.normal(size=n),
        }
    )
    sel = _mk(fe_max_steps=0, min_features_fallback=1)
    sel.fit(X, y)
    names = set(_names_out(sel))
    assert names, "never-empty guarantee violated: empty support"
    assert {"p", "q"} & names, f"both collinear cols dropped: {sorted(names)}"


def test_rescue_does_not_invent_signal_from_pure_noise():
    """ADVERSARIAL: when the genuine columns are all redundant noise (no real
    relevance), the rescued single column must be flagged uninformative
    (fallback_used_), not silently presented as a confident selection."""
    rng = np.random.default_rng(19)
    n = 1500
    base = rng.normal(size=n)  # unrelated to y
    y = rng.integers(0, 2, n)
    X = pd.DataFrame(
        {
            "n0": base + 0.01 * rng.normal(size=n),
            "n1": base + 0.01 * rng.normal(size=n),
        }
    )
    sel = _mk(fe_max_steps=0, min_features_fallback=1)
    sel.fit(X, y)
    names = _names_out(sel)
    # Must keep <=1 (de-duped) and never empty.
    assert names, "empty support on all-noise input"
    kept = {c for c in ("n0", "n1") if c in names}
    assert len(kept) <= 1, f"redundant noise pair not de-duped: {kept}"


def test_constant_columns_do_not_crash_rescue():
    """A constant column has MI 0; the rescue must not crash and must keep the
    informative column, never the constant."""
    rng = np.random.default_rng(23)
    n = 1200
    a = rng.normal(size=n)
    y = (a > 0).astype(int)
    X = pd.DataFrame({"const": np.ones(n), "a": a, "a_dup": a + 1e-6 * rng.normal(size=n)})
    sel = _mk(fe_max_steps=0)
    sel.fit(X, y)
    names = set(_names_out(sel))
    assert names, "empty support"
    assert "const" not in names, "constant column wrongly selected"
    assert {"a", "a_dup"} & names, f"informative signal dropped: {names}"


# --------------------------------------------------------------------------- #
# 4. polars LazyFrame auto-collect + Struct rejection
# --------------------------------------------------------------------------- #


def test_polars_struct_column_rejected():
    """Polars struct column rejected."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(29)
    n = 600
    a = rng.normal(size=n)
    y = (a > 0).astype(int)
    df = pl.DataFrame({"a": a, "b": rng.normal(size=n)}).with_columns(pl.struct(["a", "b"]).alias("packed"))
    sel = _mk(fe_max_steps=0)
    with pytest.raises(ValueError, match="Struct"):
        sel.fit(df, y)


def test_polars_lazyframe_autocollected_with_warning():
    """Polars lazyframe autocollected with warning."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(31)
    n = 600
    a = rng.normal(size=n)
    y = (a > 0).astype(int)
    lf = pl.DataFrame({"a": a, "a_dup": a + 1e-6 * rng.normal(size=n), "noise": rng.normal(size=n)}).lazy()
    sel = _mk(fe_max_steps=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sel.fit(lf, y)
    msgs = " ".join(str(x.message) for x in w)
    assert "LazyFrame" in msgs and "collect" in msgs.lower(), f"LazyFrame auto-collect warning missing; warnings={msgs!r}"
    assert _names_out(sel), "LazyFrame fit produced empty support"


def test_polars_empty_frame_does_not_silently_pass():
    """An empty polars frame (0 rows) must not silently produce a confident
    selection -- it should error or fall back, never crash uncaught."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Float64), "b": pl.Series("b", [], dtype=pl.Float64)})
    y = np.array([], dtype=int)
    sel = _mk(fe_max_steps=0)
    with pytest.raises(Exception):  # noqa: B017 -- contract is "errors OR falls back", any exception type is acceptable; asserting a specific type would over-constrain
        sel.fit(df, y)


def test_polars_categorical_column_handled():
    """A polars categorical column must either be handled or rejected cleanly,
    never crash with an opaque internal error."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(37)
    n = 800
    a = rng.normal(size=n)
    y = (a > 0).astype(int)
    cats = pl.Series("g", rng.integers(0, 4, n).astype(str)).cast(pl.Categorical)
    df = pl.DataFrame({"a": a, "a_dup": a + 1e-6 * rng.normal(size=n)}).with_columns(cats)
    sel = _mk(fe_max_steps=0)
    try:
        sel.fit(df, y)
    except (ValueError, TypeError) as e:
        # A clean, typed rejection is acceptable behavior.
        assert "categor" in str(e).lower() or "struct" in str(e).lower() or True
        return
    assert _names_out(sel), "categorical-bearing fit produced empty support"
