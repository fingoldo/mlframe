"""Regression tests for D13 (composite audit 2026-06-10): unary
(``requires_base=False``) transforms must be scored against the FULL feature
matrix and named base-free, NOT bound to an arbitrary "first" base.

Pre-fix behaviour (the bug these tests pin):

* ``discovery/_fit.py`` deduped each unary transform to the FIRST base in
  ``base_candidates`` order and routed it through THAT base's context, so the
  unary spec was scored against ``x_remaining`` = (all features minus the first
  base) -- a column the unary never reads -- and its ``mi_gain`` shifted when
  the auto-base ranking reordered (an irrelevant degree of freedom).
* ``discovery/_eval.py`` stamped ``base_column=<first base>`` and named the spec
  ``y-cbrtY-<first base>`` (3-segment), claiming a base dependence that does not
  exist.

Post-fix:

* Each unary routes through ONE dedicated context whose ``x_remaining`` is the
  FULL feature matrix (no base dropped) and whose ``base_column`` is the empty
  sentinel -> ``mi_gain`` is invariant to base/feature ordering.
* The spec name is the base-free 2-segment form ``y-cbrtY`` and
  ``base_column == ""``; ``is_composite_target_name`` recognises the 2-segment
  unary form so downstream metric labels still route to MTRESID.

Each test below FAILS on pre-fix code (the 3-segment name / non-empty base /
order-dependent mi_gain) and PASSES post-fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.transforms.naming import is_composite_target_name
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_UNARY_TRANSFORMS = ("cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y")


@pytest.fixture
def synthetic_df() -> tuple[pd.DataFrame, np.ndarray]:
    """y = 0.8*x_base + heavy-tail residual, with several decoy features whose
    MI ranking (hence auto-base order) is deliberately permutable."""
    rng = np.random.default_rng(20260611)
    n = 3000
    df = pd.DataFrame(
        {
            "x_base": rng.normal(100.0, 20.0, n),
            "x_a": rng.normal(0.0, 1.0, n),
            "x_b": rng.normal(0.0, 1.0, n),
            "x_c": rng.normal(0.0, 1.0, n),
            "x_d": rng.normal(0.0, 1.0, n),
        }
    )
    df["y"] = 0.8 * df["x_base"] + 5.0 + rng.laplace(0.0, 3.0, n) + 0.5 * df["x_a"]
    train_idx = np.arange(int(0.8 * n))
    return df, train_idx


def _run_disc(df, feature_cols, train_idx, **cfg_kwargs):
    """Fits CompositeTargetDiscovery with MI-only screening (isolating the MI scoring layer) and returns the fitted object."""
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        mi_sample_n=1500,
        screening="mi",  # isolate the MI scoring layer (no tiny rerank)
        **cfg_kwargs,
    )
    disc = CompositeTargetDiscovery(config=cfg)
    disc.fit(
        df=df,
        target_col="y",
        feature_cols=list(feature_cols),
        train_idx=train_idx,
    )
    return disc


def _report_rows_for(disc, transform_name):
    """Filters a fitted discovery's report_ list to the rows matching the given transform name."""
    return [r for r in disc.report_ if isinstance(r, dict) and r.get("transform_name") == transform_name]


def test_unary_spec_name_has_no_base_segment(synthetic_df):
    """A unary spec is named ``y-cbrtY`` (2-segment) with an EMPTY base_column;
    pre-fix it was ``y-cbrtY-<first base>`` with base_column=<first base>."""
    df, train_idx = synthetic_df
    feats = ["x_base", "x_a", "x_b", "x_c", "x_d"]
    disc = _run_disc(df, feats, train_idx)

    found_any_unary = False
    for unary in _UNARY_TRANSFORMS:
        for row in _report_rows_for(disc, unary):
            if row.get("rejected"):
                continue  # rejected rows synthesise a __name__; skip
            found_any_unary = True
            name = row["name"]
            base_col = row.get("base_column", None)
            # Base-free: no third dash-segment, and base_column is the sentinel.
            assert base_col == "", f"unary {unary!r} spec has base_column={base_col!r}; expected the empty sentinel (no base dependence)"
            # 2-segment form: exactly one dash separates target from alias, and
            # NONE of the feature columns appears as a trailing base segment.
            assert name.count("-") == 1, f"unary {unary!r} name {name!r} is not the 2-segment base-free form ``y-<alias>``"
            for f in feats:
                assert not name.endswith(f"-{f}"), f"unary {unary!r} name {name!r} carries a spurious base segment {f!r}"
            # The base-free name must still be recognised as a composite target
            # so downstream metric labels route to MTRESID, not raw MTTR.
            assert is_composite_target_name(name), f"2-segment unary name {name!r} not recognised as composite"
    assert found_any_unary, "no non-rejected unary spec produced; test fixture cannot exercise D13"


def test_unary_mi_gain_invariant_to_base_ordering(synthetic_df):
    """The unary spec's ``mi_gain`` must be IDENTICAL when the feature columns
    (and hence the auto-base ranking) are reordered -- because it is now scored
    against the FULL feature matrix, not an arbitrary first base's x_remaining.

    Pre-fix the unary was bound to whichever base ranked first, so reordering
    the features changed which base's ``x_remaining`` it was scored against and
    shifted ``mi_gain``.
    """
    df, train_idx = synthetic_df
    order_a = ["x_base", "x_a", "x_b", "x_c", "x_d"]
    order_b = ["x_d", "x_c", "x_b", "x_a", "x_base"]  # reversed -> different base order

    disc_a = _run_disc(df, order_a, train_idx)
    disc_b = _run_disc(df, order_b, train_idx)

    compared_any = False
    for unary in _UNARY_TRANSFORMS:
        rows_a = [r for r in _report_rows_for(disc_a, unary) if not r.get("rejected")]
        rows_b = [r for r in _report_rows_for(disc_b, unary) if not r.get("rejected")]
        if not rows_a or not rows_b:
            continue
        # Exactly one non-rejected unary spec each (dedup -> single context).
        ga = float(rows_a[0]["mi_gain"])
        gb = float(rows_b[0]["mi_gain"])
        if not (np.isfinite(ga) and np.isfinite(gb)):
            continue
        compared_any = True
        assert ga == pytest.approx(gb, abs=1e-9), (
            f"unary {unary!r} mi_gain shifted with feature ordering: "
            f"{ga} (order A) vs {gb} (order B); it should be scored against "
            f"the full feature matrix and be order-invariant"
        )
        # And mi_y (the baseline) is the full-X baseline in both orderings.
        ya = float(rows_a[0]["mi_y"])
        yb = float(rows_b[0]["mi_y"])
        assert ya == pytest.approx(yb, abs=1e-9), f"unary {unary!r} mi_y baseline differs across orderings ({ya} vs {yb}); not scored against the full matrix"
    assert compared_any, "no finite-mi_gain unary spec available in BOTH orderings to compare; fixture cannot exercise the order-invariance guarantee"


def test_unary_dedup_single_context_one_spec_per_unary(synthetic_df):
    """Each unary appears exactly once across the whole report (one dedicated
    context, not once per base)."""
    df, train_idx = synthetic_df
    feats = ["x_base", "x_a", "x_b", "x_c", "x_d"]
    disc = _run_disc(df, feats, train_idx)
    from collections import Counter

    counts = Counter(r.get("transform_name") for r in disc.report_ if isinstance(r, dict))
    for unary in _UNARY_TRANSFORMS:
        assert counts[unary] == 1, f"unary {unary!r} evaluated {counts[unary]} times; expected exactly one (single dedicated full-X context)"


def test_unary_spec_iter_transform_applies_without_base(synthetic_df):
    """A unary spec with an empty base_column must apply cleanly through
    ``iter_transform`` (which previously extracted ``df[base_column]`` and would
    crash on the empty sentinel)."""
    df, train_idx = synthetic_df
    feats = ["x_base", "x_a", "x_b", "x_c", "x_d"]
    # Force the unary to survive the eps gate so it lands in specs_.
    disc = _run_disc(df, feats, train_idx, eps_mi_gain=-1e9, top_k_after_mi=50)
    unary_specs = [s for s in disc.specs_ if s.transform_name in _UNARY_TRANSFORMS]
    if not unary_specs:
        pytest.skip("no unary spec survived to specs_ on this fixture")
    for s in unary_specs:
        assert s.base_column == "", f"kept unary spec {s.name!r} has non-empty base_column {s.base_column!r}"
    # iter_transform must not raise on the empty base_column.
    emitted = dict(disc.iter_transform(df))
    for s in unary_specs:
        assert s.name in emitted, f"iter_transform dropped unary spec {s.name!r}"
        t = emitted[s.name]
        assert t.shape[0] == len(df)
        assert np.isfinite(t).any(), f"unary spec {s.name!r} produced an all-NaN T column"
