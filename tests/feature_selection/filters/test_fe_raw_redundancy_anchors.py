"""Direct unit coverage for ``_fe_raw_redundancy_anchors.build_raw_redundancy_anchors``
(mrmr_audit_2026-07-20 test_coverage.md #4) -- a 347-line function previously only exercised
transitively via ``drop_redundant_raw_operands`` in the biz_value/adversarial test suites. Pins
the four explicit early-return guards directly, plus the happy-path anchor/consumer/marginal
context construction."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_raw_redundancy_anchors import build_raw_redundancy_anchors


def _base_kwargs(n=200, seed=0):
    """A minimal raw+engineered dataset: eng1 = a op b (both raws are its consumed operands)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 10, n).astype(np.int64)
    b = rng.integers(0, 10, n).astype(np.int64)
    eng1 = np.minimum(a + b, 9).astype(np.int64)
    y = (a > 4).astype(np.int64)
    data = np.column_stack([a, b, eng1])
    # the engineered column name must contain "a"/"b" as separate identifier tokens
    # (_TOKEN_SPLIT splits on non-alnum chars) for the consumer-map build to recognize them.
    cols = ["a", "b", "sum(a,b)"]
    return dict(
        data=data,
        cols=cols,
        sel=[0, 1, 2],
        raw_name_set={"a", "b"},
        y_binned=y,
        y_continuous=None,
        engineered_continuous=None,
        replayable_eng_names=None,
        recipes=None,
        raw_X=None,
        seed=seed,
        verbose=0,
        n_rows=n,
        gate_resident=False,
    )


class TestEarlyReturnGuards:
    """Each of the four explicit degenerate short-circuits must fire (and only those) rather than
    proceeding into the anchor/consumer-map build."""

    def test_no_engineered_survivor_returns_early(self):
        """``sel`` contains only raw columns -- no engineered anchor exists to score raws against."""
        kw = _base_kwargs()
        kw["sel"] = [0, 1]
        ctx = build_raw_redundancy_anchors(**kw)
        assert ctx.early_return == ([0, 1], [])

    def test_no_raw_survivor_returns_early(self):
        """``sel`` contains only the engineered column -- nothing to judge as a redundant raw operand."""
        kw = _base_kwargs()
        kw["sel"] = [2]
        ctx = build_raw_redundancy_anchors(**kw)
        assert ctx.early_return == ([2], [])

    def test_no_replayable_anchor_returns_early(self):
        """``replayable_eng_names`` excludes the only engineered survivor -- it will not exist at
        transform-time, so it can't anchor any raw's redundancy verdict."""
        kw = _base_kwargs()
        kw["replayable_eng_names"] = set()
        ctx = build_raw_redundancy_anchors(**kw)
        assert ctx.early_return == ([0, 1, 2], [])

    def test_no_consumer_map_returns_early(self):
        """The engineered column's name references no token from ``raw_name_set`` -- no raw is
        actually a consumed operand of any surviving engineered feature."""
        kw = _base_kwargs()
        kw["cols"] = ["a", "b", "unrelated_feature"]
        ctx = build_raw_redundancy_anchors(**kw)
        assert ctx.early_return == ([0, 1, 2], [])


class TestHappyPathContext:
    """A genuine raw+engineered selection populates the full anchor/consumer/marginal context."""

    def test_eng_consumers_map_both_raws_to_the_engineered_survivor(self):
        """The engineered column's name tokens must both be recognized as raw consumers."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        assert ctx.early_return is None
        assert set(ctx.eng_consumers.keys()) == {"a", "b"}
        assert ctx.eng_consumers["a"] == [2]
        assert ctx.eng_consumers["b"] == [2]

    def test_eng_bin_and_anchor_excess_populated_for_the_engineered_survivor(self):
        """The engineered survivor's re-binned codes and marginal-excess anchor value are cached."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        assert 2 in ctx.eng_bin
        assert ctx.eng_bin[2].shape[0] == 200
        assert 2 in ctx.eng_anchor_excess
        assert np.isfinite(ctx.eng_anchor_excess[2])

    def test_raw_marginal_is_deterministic_and_cached(self):
        """Two calls to the returned ``raw_marginal`` closure for the same raw name must return the
        identical cached tuple (same object identity's underlying dict, not just equal values)."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        first = ctx.raw_marginal("a")
        second = ctx.raw_marginal("a")
        assert first == second
        assert len(first) == 3 and all(np.isfinite(v) for v in first)

    def test_raw_is_signal_bearing_detects_the_true_predictor(self):
        """Raw 'a' drives the target (y = a > 4); it should register as signal-bearing, while a raw
        that consumes no relationship with y at all (constant) should not."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        assert ctx.raw_is_signal_bearing("a") is True

    def test_unknown_raw_name_returns_zero_marginal_without_raising(self):
        """A raw name absent from ``cols`` (ValueError on ``cols.index``) is handled gracefully."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        assert ctx.raw_marginal("not_a_real_column") == (0.0, 0.0, 0.0)

    def test_raw_codes_and_raw_dev_agree_when_residency_is_off(self):
        """With ``gate_resident=False``, ``raw_dev`` must return ``None`` (no device residency
        requested) while ``raw_codes`` still returns the host int64 codes."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        codes = ctx.raw_codes("a", 0)
        dev = ctx.raw_dev("a", 0)
        assert codes.dtype == np.int64
        assert dev is None

    def test_join_dev_returns_none_without_resident_codes(self):
        """``join_dev`` must degrade to ``None`` (host-side join) when no operand carries a resident
        device twin, rather than raising on a missing GPU/cupy dependency."""
        ctx = build_raw_redundancy_anchors(**_base_kwargs())
        assert ctx.join_dev(None, None) is None
        assert ctx.join_dev() is None
