"""Layer 73 biz_value: upper/lower tail-dependence coefficient scorer (mrmr_audit_2026-07-20
fe_expansion.md "tail-dependence coefficient").

Validates ``tail_dependence_score`` (``_orthogonal_tail_dependence_fe``): a dedicated co-exceedance
statistic distinct from the Layer 66 copula-MI scorer's full-distribution average.

Contracts pinned
-----------------
* ``TestGumbelLikeTailDependence`` (biz_value): on a synthetic where two columns are independent in
  the BULK (95% of rows) but strongly co-dependent in their joint upper tail (5% of rows), the tail
  score reads strongly positive while the full-distribution copula-MI reads comparatively weak --
  the exact dilution gap the audit names.
* ``TestIndependentPairNearZero``: fully independent uniforms give a near-zero floored score.
* ``TestLowerTail``: the lower-tail variant detects a symmetric joint-minimum co-dependence.
* Degenerate inputs (n<2, non-finite, q<=0 denom) return 0.0, never raise.
"""

from __future__ import annotations

import numpy as np
import pytest

SEEDS = (1, 7, 13, 42, 101)


def _import_tail_dep():
    """Lazily import the Layer-73 tail-dependence primitive."""
    from mlframe.feature_selection.filters._orthogonal_tail_dependence_fe import tail_dependence_score

    return tail_dependence_score


class TestGumbelLikeTailDependence:
    """biz_value: bulk-independent, tail-co-dependent data must read strongly on the tail score
    while the full-distribution copula-MI average is comparatively diluted."""

    def test_tail_score_high_copula_mi_diluted(self):
        """The tail statistic must clearly exceed a near-zero floor and materially exceed the
        full-distribution copula-MI, which averages over the 95% independent bulk."""
        from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import copula_mi

        tail_dependence_score = _import_tail_dep()
        rng = np.random.default_rng(0)
        n = 8000
        u = rng.uniform(0.0, 1.0, n)
        v = rng.uniform(0.0, 1.0, n)
        # Force co-exceedance in the joint upper 5% tail (Gumbel-copula-like structure): wherever u
        # is in its own top 5%, v is ALSO forced into its top 5% -- bulk stays fully independent.
        mask = u > 0.95
        # Force these rows' v into a narrow band right at the top of the value range: since only a
        # negligible fraction of the untouched 95% bulk naturally lands this high (~0.1% of ~7600
        # rows), the forced rows dominate the TRUE top-5%-by-rank band -- a looser band (e.g.
        # [0.951, 0.999]) lets enough natural bulk exceedances above 0.95 crowd the forced rows out
        # of the top-rank band once re-ranked over the WHOLE array, silently diluting the signal.
        v[mask] = np.clip(u[mask] + rng.normal(0.0, 0.0005, int(mask.sum())), 0.999, 0.9999)

        tail_score = tail_dependence_score(u, v, q=0.95, tail="upper", n_perm=200, random_state=0)
        mi = copula_mi(u, v, n_bins=20)

        assert tail_score > 0.5, f"tail-dependence score should clearly detect the joint-tail structure, got {tail_score:.4f}"
        assert tail_score > 3.0 * max(mi, 1e-6), f"tail score ({tail_score:.4f}) should materially exceed the diluted full-distribution copula-MI ({mi:.4f})"


class TestIndependentPairNearZero:
    """Fully independent uniforms must give a near-zero floored tail-dependence score."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_independent_uniforms_near_zero(self, seed):
        """Two fully independent uniform arrays must not read as tail-dependent."""
        tail_dependence_score = _import_tail_dep()
        rng = np.random.default_rng(seed)
        n = 3000
        u = rng.uniform(0.0, 1.0, n)
        v = rng.uniform(0.0, 1.0, n)
        score = tail_dependence_score(u, v, q=0.95, tail="upper", n_perm=100, random_state=seed)
        assert score < 0.5, f"seed={seed}: independent pair should give a near-zero floored score, got {score:.4f}"


class TestLowerTail:
    """The lower-tail variant must symmetrically detect a joint-minimum co-dependence structure."""

    def test_lower_tail_detects_joint_minimum_dependence(self):
        """A joint-minimum co-dependence structure must read strongly on the lower-tail variant."""
        tail_dependence_score = _import_tail_dep()
        rng = np.random.default_rng(2)
        n = 8000
        u = rng.uniform(0.0, 1.0, n)
        v = rng.uniform(0.0, 1.0, n)
        mask = u < 0.05
        v[mask] = np.clip(u[mask] + rng.normal(0.0, 0.0005, int(mask.sum())), 0.0001, 0.001)
        score = tail_dependence_score(u, v, q=0.95, tail="lower", n_perm=200, random_state=2)
        assert score > 0.5, f"lower-tail score should detect the joint-minimum structure, got {score:.4f}"


class TestScorerPoolWiring:
    """tail_dep must be a first-class member of the Layer 68/69 scorer pool, dispatchable
    without raising (mrmr_audit_2026-07-20: wired in the same pass as Xi)."""

    def test_tail_dep_in_scorer_names(self):
        """Layer 68's SCORER_NAMES tuple must include 'tail_dep'."""
        from mlframe.feature_selection.filters._orth_auto_scorer_fe import SCORER_NAMES

        assert "tail_dep" in SCORER_NAMES, f"SCORER_NAMES missing 'tail_dep': {SCORER_NAMES}"

    def test_score_tail_dep_dispatches_without_raising(self):
        """The _score_tail_dep wrapper must return a finite, non-negative value without raising."""
        from mlframe.feature_selection.filters._orth_auto_scorer_fe import _score_tail_dep

        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        y = rng.standard_normal(500)
        val = _score_tail_dep(x, y, random_state=0)
        assert np.isfinite(val)
        assert val >= 0.0


class TestDegenerateInputsReturnZero:
    """n<2, non-finite input, and an invalid tail argument must return 0.0 / raise cleanly."""

    def test_single_row_returns_zero(self):
        """n=1 hits the explicit n<2 early-return guard."""
        tail_dependence_score = _import_tail_dep()
        assert tail_dependence_score(np.array([1.0]), np.array([2.0])) == 0.0

    def test_nan_input_returns_zero(self):
        """A NaN in x hits the explicit finite-check guard rather than propagating into the rank transform."""
        tail_dependence_score = _import_tail_dep()
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert tail_dependence_score(x, y) == 0.0

    def test_invalid_tail_argument_raises(self):
        """An unrecognized tail value must raise ValueError, not silently default to one tail."""
        tail_dependence_score = _import_tail_dep()
        with pytest.raises(ValueError, match="tail must be"):
            tail_dependence_score(np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64), tail="both")

    def test_zero_permutations_skips_null_floor(self):
        """n_perm=0 must not crash and must return the raw (unfloored) rate, clamped at 0."""
        tail_dependence_score = _import_tail_dep()
        rng = np.random.default_rng(5)
        n = 1000
        u = rng.uniform(0.0, 1.0, n)
        v = rng.uniform(0.0, 1.0, n)
        score = tail_dependence_score(u, v, q=0.95, tail="upper", n_perm=0, random_state=5)
        assert score >= 0.0
        assert np.isfinite(score)
