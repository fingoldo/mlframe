"""Canonical-fixture biz_value test for MRMR unary/binary pair Feature Engineering.

Regression guard for the 2026-06-01 fix of the polynom-pair / unary-binary FE
that found ZERO engineered features on the canonical fixture
``y = a**2/b + f/5 + log(c)*sin(d)`` under default ``MRMR(verbose=0).fit(df, y)``.

Five root causes were fixed (all in ``filters/feature_engineering.py`` /
``_feature_engineering_pairs.py`` / ``_mrmr_fe_step.py`` / ``mrmr.py``):

1. unary "minimal" preset was identity-ONLY -> no building blocks for sqr(a)
   or log(c); now a non-degenerate workhorse {identity, neg, abs, sqr,
   reciproc, sqrt, log, sin}.
2. binary "medium" == "minimal" == {mul, add, max, min} and there was NO div/sub
   in ANY tier; now minimal = {mul, add, sub, div, max, min}.
3. "rich"/"full" presets silently aliased to medium; now resolve to maximal and
   unknown presets raise ValueError.
4. ``fe_min_engineered_mi_prevalence`` default 0.98 -> 0.90 (a 1-D summary of a
   2-D pair-joint cannot retain 98% of the finite-sample-bias-inflated 2-D MI).
5. materialization disconnect: with default ``fe_max_steps=1`` the recommended
   features were LOGGED but never materialised / promoted into
   ``_engineered_features_``; now FE survivors are appended AND self-selected.

The ``log(c)*sin(d)`` term is scaled up so it carries variance comparable to the
``a**2/b`` term -- otherwise the much larger a**2/b variance masks the second
signal and c is dropped at screening (c's marginal MI is near-zero because
``E[sin(d)] ~= 0`` over d in [0, 2*pi]; it only matters jointly with d). The
formula structure (a**2/b, log(c)*sin(d), f/5 + e as noise) is preserved.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR

N = 20_000
SEED = 42
# Scale on log(c)*sin(d) so its variance is comparable to a**2/b; without it the
# a**2/b term (var ~14) dwarfs log(c)*sin(d) (var ~0.6) and c never survives
# screening, so the (c,d) pair can never be engineered.
SECOND_SIGNAL_SCALE = 3.0


def _make_fixture(seed: int = SEED, n: int = N):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1.0, 5.0, n)
    b = rng.uniform(1.0, 5.0, n)
    c = rng.uniform(1.0, 5.0, n)
    d = rng.uniform(0.0, 2.0 * np.pi, n)
    e = rng.normal(0.0, 1.0, n)  # pure noise
    f = rng.normal(0.0, 1.0, n)  # pure noise (does NOT appear in df)
    y = a**2 / b + f / 5.0 + SECOND_SIGNAL_SCALE * np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y")


_RAW = {"a", "b", "c", "d", "e"}


def _bare_vars(name: str) -> set:
    """Bare variable tokens (single letters a-e) that appear as arguments, NOT as
    substrings of function names like ``div`` (the 'd') or ``abs`` (no var).

    A bare var is a single a-e letter not flanked by other identifier chars.
    """
    return set(re.findall(r"(?<![A-Za-z_])([a-e])(?![A-Za-z_])", name))


def _engineered_names(fs: MRMR) -> list:
    return [n for n in fs.get_feature_names_out() if n not in _RAW]


def _covers_pair(eng_names, va, vb, exclude=()):
    """Any engineered name whose bare vars include {va, vb} and none of ``exclude``."""
    want = {va, vb}
    excl = set(exclude)
    for nm in eng_names:
        bv = _bare_vars(nm)
        if want <= bv and not (bv & excl):
            return True
    return False


def test_canonical_default_finds_both_signal_pairs():
    """DEFAULT preset (minimal): >=2 engineered features covering BOTH (a,b) and (c,d)."""
    df, y = _make_fixture()
    fs = MRMR(verbose=0)
    fs.fit(df, y)

    eng = _engineered_names(fs)
    assert len(eng) >= 2, f"expected >=2 engineered features, got {eng}"
    # ONE-BEST-PER-PAIR (2026-06-01): exactly one representative per raw signal
    # pair, not the whole near-equivalent equivalence class (which used to emit
    # ~15 cols). Two true signal pairs -> a small handful, never the full class.
    assert len(eng) <= 4, (
        f"over-materialization regression: expected ~2 engineered features "
        f"(one per signal pair), got {len(eng)}: {eng}"
    )

    # a**2/b-equivalent: an engineered col over {a, b} that does NOT pull in c/d.
    assert _covers_pair(eng, "a", "b", exclude=("c", "d")), (
        f"no a**2/b-equivalent engineered feature (pair a,b) found in {eng}"
    )
    # log(c)*sin(d)-equivalent: an engineered col over {c, d}.
    assert _covers_pair(eng, "c", "d"), (
        f"no log(c)*sin(d)-equivalent engineered feature (pair c,d) found in {eng}"
    )

    # The recipes list must be populated in lockstep so transform() can replay.
    recipe_names = {getattr(r, "name", None) for r in fs._engineered_recipes_}
    assert recipe_names, "engineered recipes empty despite engineered features"


def test_canonical_transform_replays_engineered_columns_leak_safe():
    """fs.transform(df.head(100)) reproduces engineered columns with NO target (leak-safe)."""
    df, y = _make_fixture()
    fs = MRMR(verbose=0)
    fs.fit(df, y)

    out_names = list(fs.get_feature_names_out())
    head = df.head(100)
    Xt = np.asarray(fs.transform(head))  # NB: no y passed
    assert Xt.shape == (100, len(out_names)), (
        f"transform shape {Xt.shape} != (100, {len(out_names)})"
    )

    # The mul(log(c),sin(d))-family column, replayed on raw inputs, must be a
    # monotone function of the true log(c)*sin(d) signal (discretized -> compare
    # rank correlation rather than exact values).
    eng = _engineered_names(fs)
    cd_idx = None
    for i, nm in enumerate(out_names):
        if nm in eng and {"c", "d"} <= _bare_vars(nm):
            cd_idx = i
            break
    assert cd_idx is not None, f"no (c,d) engineered column in output names {out_names}"

    Xt_full = np.asarray(fs.transform(df))
    direct = np.log(df["c"].values) * np.sin(df["d"].values)
    rho = np.corrcoef(Xt_full[:, cd_idx], direct)[0, 1]
    assert abs(rho) > 0.5, (
        f"replayed (c,d) engineered col only |rho|={rho:.3f} vs true log(c)*sin(d)"
    )

    # EXPLICIT "reaches the consumer AND is non-degenerate" contract: every
    # engineered column the FE step recommended must (1) actually appear as a
    # column in the transform() output the downstream model consumes, and
    # (2) carry real variance (not all-zeros / not constant). A materialization
    # bug (the regression this fixture guards) silently produced a recommended
    # feature that either never reached transform() or arrived as a dead
    # constant column; both are caught here.
    eng_idx = [i for i, nm in enumerate(out_names) if nm in eng]
    assert len(eng_idx) >= 2, (
        f"engineered features did not reach transform() output: "
        f"out_names={out_names}, eng={eng}"
    )
    for i in eng_idx:
        col = Xt_full[:, i]
        assert np.isfinite(col).all(), f"engineered col {out_names[i]} has non-finite values"
        assert float(np.std(col)) > 1e-9, (
            f"engineered col {out_names[i]} reached the consumer but is "
            f"CONSTANT (std={np.std(col):.2e}) - dead feature, not the "
            f"recommended signal"
        )


@pytest.mark.parametrize("preset", ["minimal", "medium", "maximal"])
def test_canonical_across_presets(preset):
    """Every preset recovers BOTH signal pairs (richer presets are supersets)."""
    df, y = _make_fixture()
    fs = MRMR(verbose=0, fe_unary_preset=preset, fe_binary_preset=preset)
    fs.fit(df, y)

    eng = _engineered_names(fs)
    assert len(eng) >= 2, f"[{preset}] expected >=2 engineered, got {eng}"
    assert _covers_pair(eng, "a", "b", exclude=("c", "d")), (
        f"[{preset}] no a**2/b-equivalent in {eng}"
    )
    assert _covers_pair(eng, "c", "d"), f"[{preset}] no log(c)*sin(d)-equivalent in {eng}"


def test_fit_does_not_mutate_caller_dataframe():
    """``MRMR.fit`` must NOT append engineered columns (or anything) to the
    caller's input DataFrame. Full-mode FE materialises engineered columns into
    a working frame; a regression once leaked them into the user's ``df`` in
    place (``X[col] = ...``), silently growing it after ``fit`` and bleeding
    state across fits that reused one frame. The columns/shape the caller passed
    in must be byte-identical after fit returns."""
    df, y = _make_fixture()
    cols_before = list(df.columns)
    shape_before = df.shape
    fs = MRMR(verbose=0)
    fs.fit(df, y)
    assert list(df.columns) == cols_before, (
        f"fit mutated caller df columns: {cols_before} -> {list(df.columns)}"
    )
    assert df.shape == shape_before, (
        f"fit mutated caller df shape: {shape_before} -> {df.shape}"
    )
    # And it still actually engineered features (guard against a vacuous pass
    # where FE silently did nothing).
    assert len(_engineered_names(fs)) >= 1, "expected >=1 engineered feature"
