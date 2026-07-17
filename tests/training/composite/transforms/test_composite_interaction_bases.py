"""Tests for ``generate_interaction_bases`` (R10c brainstorm #9; synthetic-base generator).

The helper pairs the top-K candidate base columns under binary ops (mul / div / add / sub) to produce synthetic base columns named ``"<a>__<op>__<b>"`` that downstream transforms (linear_residual etc.) can consume just like raw bases.

Coverage:
- Default ops + top_k produce the expected synthetic set size.
- Custom op list filters output.
- self-pairs skipped by default; explicit ``forbid_self_pairs=False`` allows them.
- Division eps-floor protects against near-zero divisors; returned values are finite.
- Provenance dict carries parent names + op + finite counts + constant flag.
- Biz_value: on a multiplicative-interaction DGP ``y = b1 * b2 + eps``, the synthetic ``b1__mul__b2`` has STRICTLY higher correlation with y than either parent alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import generate_interaction_bases


class TestBasic:
    def test_default_mul_div_top3(self) -> None:
        """3 candidates x 2 ops x (3*2 ordered pairs - 3 self) = 12 synthetics."""
        rng = np.random.default_rng(0)
        n = 100
        candidates = {
            "b1": rng.normal(loc=10, scale=2, size=n),
            "b2": rng.normal(loc=5, scale=1, size=n),
            "b3": rng.normal(loc=2, scale=0.5, size=n),
        }
        synthetics, provenance = generate_interaction_bases(candidates, top_k=3)
        assert len(synthetics) == 12  # 3*2 (ordered, no self) * 2 ops
        # Names follow convention.
        for name in synthetics:
            assert "__" in name
            parts = name.split("__")
            assert len(parts) == 3
            assert parts[1] in ("mul", "div")
        # Provenance entries match.
        assert set(provenance.keys()) == set(synthetics.keys())

    def test_custom_ops(self) -> None:
        """Custom ops."""
        rng = np.random.default_rng(1)
        candidates = {
            "b1": rng.normal(size=50),
            "b2": rng.normal(size=50),
        }
        synthetics, _ = generate_interaction_bases(candidates, ops=("add",), top_k=2)
        # 2 candidates, 2 ordered pairs (a,b) and (b,a), 1 op = 2 synthetics.
        assert len(synthetics) == 2
        assert all("__add__" in name for name in synthetics)

    def test_top_k_below_2_returns_empty(self) -> None:
        """Top k below 2 returns empty."""
        candidates = {"only": np.array([1.0, 2.0, 3.0])}
        synthetics, provenance = generate_interaction_bases(candidates, top_k=1)
        assert synthetics == {}
        assert provenance == {}

    def test_self_pairs_skipped_by_default(self) -> None:
        """Self pairs skipped by default."""
        rng = np.random.default_rng(2)
        candidates = {"b1": rng.normal(size=100), "b2": rng.normal(size=100)}
        synthetics, _ = generate_interaction_bases(candidates, ops=("mul",), top_k=2)
        # 2 candidates, 2 ordered pairs (forbid self), 1 op = 2 synthetics.
        assert len(synthetics) == 2
        # No self-pair names.
        assert not any("b1__mul__b1" in name or "b2__mul__b2" in name for name in synthetics)

    def test_self_pairs_kept_when_requested(self) -> None:
        """Self pairs kept when requested."""
        rng = np.random.default_rng(3)
        candidates = {"b1": rng.normal(size=100), "b2": rng.normal(size=100)}
        synthetics, _ = generate_interaction_bases(
            candidates,
            ops=("mul",),
            top_k=2,
            forbid_self_pairs=False,
        )
        # 2*2 = 4 (a,a), (a,b), (b,a), (b,b).
        assert len(synthetics) == 4
        assert "b1__mul__b1" in synthetics
        assert "b2__mul__b2" in synthetics

    def test_invalid_op_raises(self) -> None:
        """Invalid op raises."""
        candidates = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        with pytest.raises(ValueError, match="unsupported op"):
            generate_interaction_bases(candidates, ops=("nonsense",), top_k=2)


class TestSafety:
    """Groups tests covering safety."""
    def test_div_eps_floor_avoids_inf(self) -> None:
        """Near-zero divisor gets floored so the synthetic is finite."""
        candidates = {
            "a": np.array([1.0, 2.0, 3.0, 4.0]),
            "b": np.array([0.0, 1e-15, 1.0, 2.0]),  # near-zero first two
        }
        synthetics, provenance = generate_interaction_bases(
            candidates,
            ops=("div",),
            top_k=2,
        )
        out = synthetics["a__div__b"]
        assert np.all(np.isfinite(out)), f"div synthetic should be finite via eps-floor; got {out}"
        # Eps floor recorded.
        assert provenance["a__div__b"]["scale_eps_b"] > 0

    def test_constant_synthetic_flagged(self) -> None:
        """``add`` of two columns that sum to a constant has ptp ~ 0; provenance flags it."""
        candidates = {
            "a": np.array([1.0, 2.0, 3.0, 4.0]),
            "b": np.array([4.0, 3.0, 2.0, 1.0]),  # a + b == 5 for every row
        }
        _synthetics, provenance = generate_interaction_bases(
            candidates,
            ops=("add",),
            top_k=2,
        )
        assert provenance["a__add__b"]["constant"] is True
        # And the symmetric one (b + a == 5).
        assert provenance["b__add__a"]["constant"] is True

    def test_provenance_carries_parents_and_op(self) -> None:
        """Provenance carries parents and op."""
        rng = np.random.default_rng(4)
        candidates = {"x": rng.normal(size=50), "y": rng.normal(size=50)}
        _, provenance = generate_interaction_bases(candidates, ops=("mul",), top_k=2)
        info = provenance["x__mul__y"]
        assert info["parents"] == ("x", "y")
        assert info["op"] == "mul"
        assert info["n_finite"] == 50


# ===========================================================================
# Biz_value
# ===========================================================================


class TestBizValueMultiplicativeDGP:
    """On a multiplicative interaction DGP ``y = b1 * b2 + eps``, the synthetic ``b1__mul__b2`` should have HIGHER correlation with y than either parent alone."""

    def test_mul_synthetic_correlates_strongly_with_y(self) -> None:
        """Mul synthetic correlates strongly with y."""
        rng = np.random.default_rng(0)
        n = 2000
        b1 = rng.uniform(low=1.0, high=5.0, size=n)
        b2 = rng.uniform(low=1.0, high=5.0, size=n)
        y = b1 * b2 + rng.normal(scale=0.5, size=n)
        candidates = {"b1": b1, "b2": b2}
        synthetics, _ = generate_interaction_bases(candidates, ops=("mul",), top_k=2)
        corr_synth = abs(float(np.corrcoef(synthetics["b1__mul__b2"], y)[0, 1]))
        corr_b1 = abs(float(np.corrcoef(b1, y)[0, 1]))
        corr_b2 = abs(float(np.corrcoef(b2, y)[0, 1]))
        assert corr_synth > max(corr_b1, corr_b2), (
            f"mul-synthetic must correlate more strongly with y than either parent; got synth={corr_synth:.3f}, b1={corr_b1:.3f}, b2={corr_b2:.3f}"
        )
        # And the corr is high (the DGP IS the synthetic).
        assert corr_synth > 0.95
