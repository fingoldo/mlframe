"""Unit tests for the realistic-distribution synthetic-data infra.

Covers the three pieces of ``_synthetic_distributions``:
  * the distribution registry (shapes, finiteness, distinguishable marginals),
  * ``with_outliers`` (count, scale, at-least-one-clean invariant),
  * domain-aware sampling (positivity / divisor floors honoured AFTER outliers),
and the two reproducibility guarantees (same seed -> identical data; per-operand
family choice stable and operand-dependent).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection import _synthetic_distributions as sd


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(sd.DISTRIBUTIONS.keys()))
def test_registry_shape_and_finite(name):
    rng = np.random.default_rng(123)
    arr = sd.DISTRIBUTIONS[name](rng, 2000)
    assert arr.shape == (2000,)
    assert arr.dtype == np.float64
    assert np.all(np.isfinite(arr)), f"{name} produced non-finite values"


@pytest.mark.parametrize("name", sd._POSITIVE_FAMILIES)
def test_positive_families_are_positive(name):
    rng = np.random.default_rng(7)
    arr = sd.DISTRIBUTIONS[name](rng, 5000)
    assert arr.min() > 0.0, f"{name} is declared positive but produced <= 0"


def test_marginals_are_distinguishable():
    """Different families must produce genuinely different marginal shapes
    (guards against a registry entry silently aliasing another)."""
    rng = np.random.default_rng(0)
    samples = {nm: fn(np.random.default_rng(0), 20000) for nm, fn in sd.DISTRIBUTIONS.items()}
    # skewness sign / magnitude separates skewed (lognormal, exp, gamma, pareto)
    # from symmetric (normal, t, bimodal) and U-shaped (beta_u).
    from scipy.stats import skew, kurtosis

    sk = {nm: float(skew(a)) for nm, a in samples.items()}
    ku = {nm: float(kurtosis(a)) for nm, a in samples.items()}
    # skewed-positive families are clearly right-skewed
    for nm in ("lognormal", "exponential", "gamma", "pareto"):
        assert sk[nm] > 0.5, f"{nm} expected right-skewed, got skew={sk[nm]:.3f}"
    # student_t(df=3) is heavy-tailed -> large positive excess kurtosis
    assert ku["student_t"] > 3.0, f"student_t expected heavy-tailed, kurtosis={ku['student_t']:.2f}"
    # beta_u (U-shape) has NEGATIVE excess kurtosis (mass at the ends, light center)
    assert ku["beta_u"] < 0.0, f"beta_u expected platykurtic, kurtosis={ku['beta_u']:.2f}"
    # bimodal also platykurtic (two separated modes)
    assert ku["bimodal"] < 0.0, f"bimodal expected platykurtic, kurtosis={ku['bimodal']:.2f}"


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frac", [0.01, 0.03, 0.05])
def test_with_outliers_count_and_clean_point(frac):
    rng = np.random.default_rng(11)
    a = rng.uniform(0, 1, 3000)
    b = sd.with_outliers(rng, a, frac=frac, mag=20.0)
    assert b.shape == a.shape
    changed = int(np.sum(~np.isclose(a, b)))
    expected = max(1, round(frac * a.size))
    # allow +-1 for rounding; never zero, never all
    assert abs(changed - expected) <= 1, f"changed {changed}, expected ~{expected}"
    assert changed < a.size, "must leave at least one clean point"
    # injected outliers dwarf the original range on at least one side
    assert b.max() > a.max() or b.min() < a.min()


def test_with_outliers_scale_tracks_input_spread():
    """A heavy-tailed column gets heavy-tailed-scale outliers; a tight column
    gets tight-scale outliers. The injected extreme tracks the input IQR."""
    rng = np.random.default_rng(3)
    tight = rng.normal(0, 0.01, 4000)
    wide = rng.normal(0, 100.0, 4000)
    bt = sd.with_outliers(rng, tight, frac=0.02, mag=10.0)
    bw = sd.with_outliers(rng, wide, frac=0.02, mag=10.0)
    # The max excursion of the wide column's outliers >> the tight column's.
    assert bw.max() > 100.0 * bt.max()


def test_with_outliers_zero_frac_is_noop():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 100)
    b = sd.with_outliers(rng, a, frac=0.0)
    assert np.array_equal(a, b)


def test_with_outliers_degenerate_spread():
    """Near-constant column (IQR=0) must not raise / produce NaN."""
    rng = np.random.default_rng(2)
    a = np.full(500, 5.0)
    b = sd.with_outliers(rng, a, frac=0.05, mag=10.0)
    assert np.all(np.isfinite(b))
    assert int(np.sum(~np.isclose(a, b))) >= 1


# ---------------------------------------------------------------------------
# Domain awareness
# ---------------------------------------------------------------------------


PROFILES = list(sd.available_profiles())


@pytest.mark.parametrize("profile", PROFILES)
def test_positive_domain_safe_for_log_sqrt(profile):
    """Across EVERY profile, a positive-domain operand stays > 0 (so log/sqrt of
    it is finite) and a divisor-domain operand stays bounded away from 0 (so the
    reciprocal is bounded) -- even with outliers injected."""
    doms = {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "c": sd.DOMAIN_POSITIVE}
    data = sd.sample_operands(seed=5, n=8000, domains=doms, profile=profile)
    c = data["c"]
    b = data["b"]
    assert c.min() > 0.0, f"[{profile}] positive operand hit <= 0 (log would be -inf/nan)"
    assert np.all(np.isfinite(np.log(c))), f"[{profile}] log(c) not finite"
    assert np.all(np.isfinite(np.sqrt(c))), f"[{profile}] sqrt(c) not finite"
    assert b.min() >= sd.DIVISOR_FLOOR - 1e-9, f"[{profile}] divisor below floor -> blow-up risk"
    assert np.all(np.isfinite(1.0 / b)), f"[{profile}] 1/b not finite"


@pytest.mark.parametrize("profile", PROFILES)
def test_formula_well_defined_under_profile(profile):
    """The user's F2 operand domains: a(any), b(divisor), c(positive), d(any).
    y = 0.2*a**2/b + log(c*2)*sin(d/3) must be all-finite under every profile."""
    doms = {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "c": sd.DOMAIN_POSITIVE, "d": sd.DOMAIN_ANY}
    data = sd.sample_operands(seed=9, n=6000, domains=doms, profile=profile)
    a, b, c, d = data["a"], data["b"], data["c"], data["d"]
    y = 0.2 * a**2 / b + np.log(c * 2.0) * np.sin(d / 3.0)
    assert np.all(np.isfinite(y)), f"[{profile}] F2 target produced non-finite values"


# ---------------------------------------------------------------------------
# Reproducibility & per-operand heterogeneity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", PROFILES)
def test_reproducible_same_seed(profile):
    doms = {"a": sd.DOMAIN_ANY, "b": sd.DOMAIN_DIVISOR, "c": sd.DOMAIN_POSITIVE}
    d1 = sd.sample_operands(seed=42, n=3000, domains=doms, profile=profile)
    d2 = sd.sample_operands(seed=42, n=3000, domains=doms, profile=profile)
    for k in doms:
        assert np.array_equal(d1[k], d2[k]), f"[{profile}] operand {k} not reproducible"


def test_different_seed_changes_data():
    doms = {"a": sd.DOMAIN_ANY}
    d1 = sd.sample_operands(seed=1, n=2000, domains=doms, profile="uniform")
    d2 = sd.sample_operands(seed=2, n=2000, domains=doms, profile="uniform")
    assert not np.array_equal(d1["a"], d2["a"])


def test_mixed_profile_assigns_different_families():
    """The 'mixed' profile must give a genuine mix -- not all operands the same
    family. We check several operands draw at least two distinct families."""
    fams = set()
    for op in ["a", "b", "c", "d", "e", "f", "g", "h"]:
        fams.add(sd.family_for_operand(op, seed=42, domain=sd.DOMAIN_ANY, candidates=tuple(sd.DISTRIBUTIONS)))
    assert len(fams) >= 2, f"mixed profile collapsed to one family: {fams}"


def test_family_choice_stable_per_operand():
    f1 = sd.family_for_operand("xyz", seed=3, domain=sd.DOMAIN_ANY, candidates=tuple(sd.DISTRIBUTIONS))
    f2 = sd.family_for_operand("xyz", seed=3, domain=sd.DOMAIN_ANY, candidates=tuple(sd.DISTRIBUTIONS))
    assert f1 == f2


def test_positive_domain_prefers_positive_family():
    """A positive/divisor operand under the mixed profile must land on a genuinely
    positive family whenever the candidate pool contains one (so the shape is real
    lognormal/gamma/etc., not a shifted-symmetric impostor)."""
    for op in ["a", "b", "c", "d", "e", "f"]:
        fam = sd.family_for_operand(op, seed=11, domain=sd.DOMAIN_POSITIVE, candidates=tuple(sd.DISTRIBUTIONS))
        assert fam in sd._POSITIVE_FAMILIES, f"positive operand {op} got non-positive family {fam}"
