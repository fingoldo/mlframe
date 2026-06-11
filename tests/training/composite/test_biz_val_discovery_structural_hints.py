"""Structural-affinity auto-base hints: unit + biz_value coverage.

``_auto_base`` ranks base candidates primarily by pairwise ``MI(y, x)``.  The
structural-affinity boost (``discovery._structural_hints``) AUGMENTS that
ranking with a bounded nudge so an UNMISTAKABLE structural base surfaces in the
top-k even when a noisier competitor's pairwise MI lands a hair higher:

* a near-affine predictor of ``y`` -> prime ``linear_residual`` base,
* a low-cardinality integer column -> prime ``grouped`` base,
* a monotone / timestamp column -> prime ``time`` base.

The detectors are unit-tested in isolation (they fire on the structure they
target and stay silent on noise); the boost is biz_value-tested end-to-end
through ``_auto_base`` on synthetic data with a known dominant affine base --
auto-detect surfaces it in the top candidates WITHOUT an explicit
``dominant_features_hint``.  A pin proves the boost is a no-op (bit-identical
MI ranking) on structureless data so enabling it by default never perturbs a
clean ranking.

cProfile / bench note: the scorer is a handful of vectorised numpy passes over
the (small) screening matrix -- one centred-dot correlation + closed-form OLS
residual ratio per column, one integer-level count, one monotone-diff sign
check; no per-row Python loop.  On the n=2000 x 6-feature screening shape used
here it runs in well under 1ms, negligible next to the MI sweep + the
permutation-null it feeds.  No actionable hotspot; the bound (``O(n * cols)``,
single pass per column) is already at the numpy floor.  See
``discovery/_structural_hints.py`` module docstring.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training._composite_target_discovery_config import (
    CompositeTargetDiscoveryConfig,
)
from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.discovery._structural_hints import (
    _is_low_card_integer,
    _monotone_fraction,
    _residual_variance_ratio,
    boost_for_features,
    structural_affinity_scores,
)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Unit: individual detectors fire on their structure, silent on noise
# ----------------------------------------------------------------------


def test_residual_variance_ratio_collapses_on_affine() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    y = 3.0 * x + 1.0 + rng.standard_normal(2000) * 0.05  # near-affine
    ratio = _residual_variance_ratio(y, x)
    assert ratio < 0.05, f"affine residual ratio should collapse, got {ratio}"


def test_residual_variance_ratio_full_on_noise() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    y = rng.standard_normal(2000)  # independent
    ratio = _residual_variance_ratio(y, x)
    assert ratio > 0.9, f"independent residual ratio should stay ~1, got {ratio}"


def test_residual_variance_ratio_constant_x_is_one() -> None:
    y = np.arange(100, dtype=np.float64)
    x = np.full(100, 7.0)
    assert _residual_variance_ratio(y, x) == 1.0


def test_is_low_card_integer_true_on_grouping() -> None:
    rng = np.random.default_rng(2)
    col = rng.integers(0, 8, size=2000).astype(np.float64)  # 8 levels
    assert _is_low_card_integer(col)


def test_is_low_card_integer_false_on_continuous() -> None:
    rng = np.random.default_rng(3)
    assert not _is_low_card_integer(rng.standard_normal(2000))


def test_is_low_card_integer_false_on_id_column() -> None:
    # One level per row -> id, not a grouping.
    assert not _is_low_card_integer(np.arange(2000, dtype=np.float64))


def test_monotone_fraction_one_on_strictly_increasing() -> None:
    assert _monotone_fraction(np.arange(500, dtype=np.float64)) == 1.0


def test_monotone_fraction_low_on_noise() -> None:
    rng = np.random.default_rng(4)
    assert _monotone_fraction(rng.standard_normal(2000)) < 0.7


def test_structural_scores_tag_each_kind() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    y = rng.standard_normal(n)
    affine = 2.0 * y + rng.standard_normal(n) * 0.02
    group = rng.integers(0, 6, size=n).astype(np.float64)
    timecol = np.arange(n, dtype=np.float64)
    noise = rng.standard_normal(n)
    X = np.column_stack([affine, group, timecol, noise])
    scores, kinds = structural_affinity_scores(
        X, y, ["affine", "group", "timecol", "noise"],
    )
    assert kinds.get("affine") == "linear_residual"
    assert kinds.get("group") == "grouped"
    assert kinds.get("timecol") == "time"
    assert "noise" not in kinds
    assert scores[3] == 0.0  # noise gets no boost


def test_boost_zero_when_no_structure() -> None:
    rng = np.random.default_rng(6)
    n = 2000
    y = rng.standard_normal(n)
    X = rng.standard_normal((n, 4))  # all continuous noise
    boost, kinds = boost_for_features(X, y, list("abcd"), mi_spread=0.5)
    assert not kinds
    assert np.all(boost == 0.0)


# ----------------------------------------------------------------------
# biz_value: auto-detect surfaces a dominant affine base WITHOUT a hint
# ----------------------------------------------------------------------


def _make_discovery(**overrides) -> CompositeTargetDiscovery:
    params = dict(
        enabled=True,
        base_candidates="auto",
        auto_base_top_k=2,
        mi_sample_n=2000,
        auto_base_null_perms=0,  # keep the test fast + deterministic
        random_state=42,
    )
    params.update(overrides)
    return CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(**params))


def _synthetic_dominant_affine(n: int = 4000, seed: int = 7):
    """A frame where ``base`` is a near-affine predictor of ``y`` plus a couple
    of pure-noise columns.  The affine base must surface as the auto-detected
    top candidate without any explicit hint.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    # y is almost entirely an affine function of base.
    y = 2.5 * base + 0.7 + rng.standard_normal(n) * 0.10
    noise1 = rng.standard_normal(n)
    noise2 = rng.standard_normal(n)
    df = pd.DataFrame({"base": base, "noise1": noise1, "noise2": noise2})
    return df, y.astype(np.float64)


def test_biz_val_auto_base_surfaces_dominant_affine_without_hint() -> None:
    """The near-affine dominant base must be the auto-detected top-1 candidate
    WITHOUT any explicit ``dominant_features_hint``.

    This is the headline biz_value claim: a column that linearly explains
    almost all of ``y`` is the prime ``linear_residual`` base, and auto-detect
    surfaces it from data shape / correlation alone.
    """
    df, y = _synthetic_dominant_affine()
    n = y.shape[0]
    train_idx = np.arange(n)
    disc = _make_discovery(auto_base_top_k=1)
    usable = ["base", "noise1", "noise2"]
    top = disc._auto_base(df, usable, y, train_idx)
    assert top and top[0] == "base", (
        f"dominant affine base should be auto top-1 (no hint): {top}"
    )


def _synthetic_equal_mi_affine_vs_nonaffine(n: int = 8000, seed: int = 7):
    """A controlled MI near-tie.

    ``comp`` is a strictly-monotone (rank-preserving) transform of ``base``, so
    binned ``MI(y, comp)`` equals ``MI(y, base)`` EXACTLY -- a perfect tie.  But
    only ``base`` is an *affine* predictor of ``y`` (``comp`` is the same
    information in a non-affine shape), so ``base`` is the prime
    ``linear_residual`` base.  ``comp`` is listed FIRST so the MI-only ranking
    (stable sort on the tie) puts ``comp`` ahead of ``base``; the structural
    boost must flip that.  ``anchor`` is a near-perfect high-MI predictor that
    widens the MI spread (so the bounded boost has room to act on the tie).
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    y = 2.0 * base + rng.standard_normal(n) * 0.8
    comp = np.sign(base) * np.abs(base) ** 1.7  # strictly monotone in base
    anchor = y + rng.standard_normal(n) * 0.05
    df = pd.DataFrame({"comp": comp, "base": base, "anchor": anchor})
    return df, y.astype(np.float64)


def test_biz_val_structural_boost_breaks_mi_tie_toward_affine_base() -> None:
    """Quantitative win on a controlled MI tie.

    With the structural boost OFF, the MI-only ranking (``comp`` and ``base``
    tie EXACTLY; ``comp`` is listed first) selects ``comp`` over ``base``.  With
    the boost ON, the affine ``base`` is promoted into the selection and
    ``comp`` is dropped -- the boost surfaces the prime ``linear_residual``
    base that pairwise MI alone could not distinguish from its non-affine twin.

    The DELTA is the assertion: ``base`` selected ON, NOT selected OFF.  A
    regression that removes the boost (or makes it too weak to break a tie)
    trips this; one that makes it override real MI gaps trips the noop test.
    """
    df, y = _synthetic_equal_mi_affine_vs_nonaffine()
    n = y.shape[0]
    train_idx = np.arange(n)
    usable = ["comp", "base", "anchor"]

    top_off = _make_discovery(
        auto_base_top_k=2, auto_base_structural_boost=False,
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
    )._auto_base(df, usable, y, train_idx)
    top_on = _make_discovery(
        auto_base_top_k=2, auto_base_structural_boost=True,
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
    )._auto_base(df, usable, y, train_idx)

    assert "base" not in top_off, (
        f"MI-only ranking should pick the non-affine twin on the tie: {top_off}"
    )
    assert "base" in top_on, (
        f"structural boost should surface the affine base on the tie: {top_on}"
    )
    assert "comp" in top_off and "comp" not in top_on, (
        f"the boost should swap comp out for base: OFF={top_off} ON={top_on}"
    )


def test_biz_val_boost_is_noop_on_structureless_ranking() -> None:
    """On a frame with NO structural columns the boost is bit-identical to the
    MI-only ranking -- enabling it by default never perturbs a clean ranking.
    """
    rng = np.random.default_rng(11)
    n = 3000
    X = rng.standard_normal((n, 5))
    # y depends on several continuous columns (none low-card / monotone /
    # near-affine-dominant), so no detector fires.
    y = (0.5 * X[:, 0] + 0.4 * X[:, 1] + 0.3 * X[:, 2]
         + rng.standard_normal(n) * 1.0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    usable = list(df.columns)
    train_idx = np.arange(n)

    top_off = _make_discovery(auto_base_structural_boost=False)._auto_base(
        df, usable, y.astype(np.float64), train_idx,
    )
    top_on = _make_discovery(auto_base_structural_boost=True)._auto_base(
        df, usable, y.astype(np.float64), train_idx,
    )
    assert top_off == top_on, (
        f"boost must be a no-op on structureless data: {top_off} != {top_on}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
