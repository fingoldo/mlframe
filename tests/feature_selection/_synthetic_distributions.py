"""Realistic-distribution synthetic data generation for MRMR feature-selection tests.

The historical MRMR test suite drew every operand from ``np.random.rand`` -- i.e.
a clean uniform marginal. Real tabular features are almost never uniform: they are
skewed (income, counts), heavy-tailed (durations, prices), bimodal (mixtures of
sub-populations), or carry a few gross outliers (sensor glitches, data-entry
errors). MI is monotone-invariant, so the GENUINE signal a formula encodes is the
SAME regardless of the marginal shape of its operands -- which makes varied
marginals the perfect stress test: a selector that recovers ``a**2/b`` under
uniform inputs but loses it under lognormal/outlier inputs has a *robustness* bug,
not a data-shape mismatch.

This module provides three composable, fully-seeded pieces:

1. A **distribution REGISTRY** (:data:`DISTRIBUTIONS`): name -> ``(rng, n) -> float64``
   sampler. Families: ``uniform``, ``normal``, ``lognormal``, ``exponential``,
   ``gamma``, ``student_t`` (df=3, heavy-tailed), ``pareto`` (power-law),
   ``beta_u`` (Beta(0.5,0.5), a U-shape that piles mass at both ends), and
   ``bimodal`` (a two-component Gaussian mixture).

2. :func:`with_outliers` -- inject a small fraction (1-5%) of gross outliers into
   any array, scaled to its own spread, for the "dirty real data" profile.

3. **DOMAIN-AWARE** sampling (:func:`sample_operand`, :func:`sample_operands`):
   each operand carries a declarative DOMAIN TAG -- ``"any"``, ``"positive"``
   (a log/sqrt argument or a multiplicative factor that must stay > 0), or
   ``"divisor"`` (a denominator that must stay bounded away from 0). The sampler
   *picks or transforms* the chosen family so the realised values honour the
   domain, so a formula like ``log(c)`` or ``a/b`` is always well defined -- no
   ``log`` of a negative, no division by ~0 blow-up -- WITHOUT any per-formula
   special-casing. The domain rule lives on the operand, not in the formula; that
   declarative separation is the "elegant" bar.

Everything is reproducible: every sampler takes a ``numpy.random.Generator`` and
draws only from it, so a fixed seed yields byte-identical data. Per-operand family
selection for the "mixed" profile is itself derived deterministically from the
operand name + seed (:func:`family_for_operand`), so a profile is a pure function
of ``(profile_name, seed, n, operand_names, domains)``.

Design note -- why transform, not reject-sample: to keep a heavy-tailed *shape*
while guaranteeing positivity we apply a monotone, shape-preserving map (shift to a
positive support, or ``abs``/``exp``). Monotone maps do not change MI, so the
genuine signal the formula encodes is preserved exactly -- the realised operand is
still "lognormal-shaped" (or t-shaped, etc.), just relocated to a legal domain.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Domain tags (declarative per-operand constraints)
# ---------------------------------------------------------------------------
# A formula author tags each operand with the domain its USE imposes:
#   "any"      -- no constraint (additive term, sin/cos argument, ...).
#   "positive" -- must be strictly > 0 (argument of log/sqrt, a multiplicative
#                 magnitude). Realised min is pushed to >= POSITIVE_FLOOR.
#   "divisor"  -- a denominator: must be bounded AWAY from 0 so a/b does not blow
#                 up. Realised min is pushed to >= DIVISOR_FLOOR (> POSITIVE_FLOOR).
DOMAIN_ANY = "any"
DOMAIN_POSITIVE = "positive"
DOMAIN_DIVISOR = "divisor"

# Floors chosen so log/sqrt are well conditioned and a/b has bounded magnitude
# while leaving the heavy tail intact (the tail is on the high side; only the low
# side is clipped/shifted).
POSITIVE_FLOOR = 1e-3
DIVISOR_FLOOR = 0.25


# ---------------------------------------------------------------------------
# Distribution registry: name -> (rng, n) -> float64 array
# ---------------------------------------------------------------------------
# Each sampler returns an unconstrained draw of the named family; domain
# enforcement is layered on top by ``sample_operand`` (so the registry stays a
# clean catalogue of marginal SHAPES, orthogonal to operand domains).


def _uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, n).astype(np.float64)


def _normal(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, n).astype(np.float64)


def _lognormal(rng: np.random.Generator, n: int) -> np.ndarray:
    # sigma=1.0 gives a clearly right-skewed, heavy-ish right tail; always > 0.
    return rng.lognormal(mean=0.0, sigma=1.0, size=n).astype(np.float64)


def _exponential(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.exponential(scale=1.0, size=n).astype(np.float64)


def _gamma(rng: np.random.Generator, n: int) -> np.ndarray:
    # shape=2.0 -> right-skewed, strictly positive, lighter tail than lognormal.
    return rng.gamma(shape=2.0, scale=1.0, size=n).astype(np.float64)


def _student_t(rng: np.random.Generator, n: int) -> np.ndarray:
    # df=3 -> finite mean/variance but heavy tails (kurtosis undefined at df<=4);
    # the canonical "heavy-tailed with extreme excursions" marginal.
    return rng.standard_t(df=3, size=n).astype(np.float64)


def _pareto(rng: np.random.Generator, n: int) -> np.ndarray:
    # Lomax/Pareto-II via numpy's pareto (shape a=2.0). Power-law right tail,
    # strictly positive. Values are (1+X) so support starts at 0.
    return rng.pareto(a=2.0, size=n).astype(np.float64)


def _beta_u(rng: np.random.Generator, n: int) -> np.ndarray:
    # Beta(0.5, 0.5): U-shaped on (0,1) -- mass piles at both 0 and 1, the
    # opposite of a bell. Stresses bin-edge / quantile-discretization paths.
    return rng.beta(a=0.5, b=0.5, size=n).astype(np.float64)


def _bimodal(rng: np.random.Generator, n: int) -> np.ndarray:
    # Two-component Gaussian mixture (modes at -3 and +3): a mixture-of-
    # subpopulations marginal that no single unimodal family captures.
    comp = rng.integers(0, 2, size=n)
    centers = np.where(comp == 0, -3.0, 3.0)
    return (centers + rng.normal(0.0, 1.0, n)).astype(np.float64)


DISTRIBUTIONS: Dict[str, Callable[[np.random.Generator, int], np.ndarray]] = {
    "uniform": _uniform,
    "normal": _normal,
    "lognormal": _lognormal,
    "exponential": _exponential,
    "gamma": _gamma,
    "student_t": _student_t,
    "pareto": _pareto,
    "beta_u": _beta_u,
    "bimodal": _bimodal,
}

# Families that are ALREADY strictly positive -- usable directly for a positive
# operand with NO transform, so the heavy tail is preserved verbatim. (uniform is
# >= 0 but can be ~0, so it needs a floor; it is intentionally NOT listed here.)
_POSITIVE_FAMILIES = ("lognormal", "exponential", "gamma", "pareto", "beta_u")

# Families safe to use for the "heavy-tailed" profile (skewed / fat-tailed shapes).
HEAVY_TAILED_FAMILIES = ("lognormal", "exponential", "gamma", "student_t", "pareto")


# ---------------------------------------------------------------------------
# Outlier injection
# ---------------------------------------------------------------------------


def with_outliers(
    rng: np.random.Generator,
    arr: np.ndarray,
    frac: float = 0.02,
    mag: float = 10.0,
) -> np.ndarray:
    """Return a copy of ``arr`` with ``frac`` of its entries replaced by gross
    outliers, scaled to the array's own robust spread.

    An outlier is placed at ``median +/- mag * IQR`` (sign random), so the
    injected extremes dwarf the bulk of the data the way real sensor glitches /
    data-entry errors do, while staying on the array's own scale (a lognormal
    column gets lognormal-scale outliers, a t column gets t-scale outliers).

    ``frac`` in [0, 1]; typical 0.01-0.05. ``mag`` is the IQR multiple. Always
    leaves at least one non-outlier point. Reproducible: draws only from ``rng``.

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> a = rng.uniform(0, 1, 1000)
    >>> b = with_outliers(rng, a, frac=0.05, mag=20.0)
    >>> b.shape == a.shape
    True
    >>> b.max() > a.max()  # at least one injected high outlier dominates
    True
    """
    arr = np.asarray(arr, dtype=np.float64).copy()
    n = arr.size
    if n == 0 or frac <= 0.0:
        return arr
    k = int(round(frac * n))
    k = max(1, min(k, n - 1))  # at least one outlier, at least one clean point
    q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        # Degenerate spread (near-constant column): fall back to abs-scale.
        iqr = max(float(np.std(arr)), 1.0)
    idx = rng.choice(n, size=k, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=k)
    arr[idx] = med + signs * mag * iqr
    return arr


# ---------------------------------------------------------------------------
# Domain enforcement (monotone, shape-preserving)
# ---------------------------------------------------------------------------


def _enforce_domain(arr: np.ndarray, domain: str) -> np.ndarray:
    """Map ``arr`` into the legal range for ``domain`` with a MONOTONE transform
    (so MI -- hence the genuine signal -- is preserved exactly).

    - ``DOMAIN_ANY``: returned unchanged.
    - ``DOMAIN_POSITIVE``: shifted so the realised minimum sits at
      ``POSITIVE_FLOOR``. For an already-positive family this is a no-op when the
      min already clears the floor; otherwise the whole array is translated up by
      a constant (a monotone shift). For symmetric families (normal, t, bimodal)
      the shift can be large, which is fine -- the SHAPE is unchanged.
    - ``DOMAIN_DIVISOR``: same shift but to the larger ``DIVISOR_FLOOR`` so the
      reciprocal stays bounded.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if domain == DOMAIN_ANY:
        return arr
    floor = DIVISOR_FLOOR if domain == DOMAIN_DIVISOR else POSITIVE_FLOOR
    amin = float(np.nanmin(arr)) if arr.size else 0.0
    if amin < floor:
        arr = arr + (floor - amin)
    return arr


# ---------------------------------------------------------------------------
# Deterministic per-operand family selection (for the "mixed" profile)
# ---------------------------------------------------------------------------


def family_for_operand(
    operand: str,
    seed: int,
    domain: str,
    candidates: Sequence[str],
) -> str:
    """Deterministically choose a family for one operand from ``candidates``.

    The choice is a pure function of ``(operand, seed)`` so the "mixed-per-feature"
    profile is fully reproducible AND assigns DIFFERENT families to different
    operands of the same formula. If the operand has a positivity domain
    (``positive`` / ``divisor``), the candidate pool is restricted to families
    that can host it without destroying the shape (already-positive families, or
    families that survive a monotone shift) -- we still allow symmetric families
    because ``_enforce_domain`` shifts them, but we prefer genuinely-positive ones
    so a "positive lognormal operand" reads as lognormal, not shifted-normal.
    """
    pool = list(candidates)
    if domain in (DOMAIN_POSITIVE, DOMAIN_DIVISOR):
        positive_pool = [f for f in pool if f in _POSITIVE_FAMILIES]
        if positive_pool:
            pool = positive_pool
    # Stable hash from operand name + seed (avoid Python's salted hash()).
    h = abs(hash((operand, int(seed)))) if False else None  # noqa: F841 (documented: do NOT use hash())
    # Use a deterministic mixing of the operand bytes + seed via a private RNG.
    mixer = np.random.default_rng([int(seed), _str_to_int(operand)])
    return pool[int(mixer.integers(0, len(pool)))]


def _str_to_int(s: str) -> int:
    """Deterministic, salt-free int from a string (FNV-1a 64-bit)."""
    h = 1469598103934665603
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------
# A profile is a named recipe for HOW to sample every operand. Each profile maps
# an operand (name + domain) to a family + optional outlier injection. Resolved
# lazily by ``sample_operand`` so a profile is just a small spec dict.

PROFILES: Dict[str, dict] = {
    # Baseline: the historical regime -- every operand uniform[0,1].
    "uniform": {"mode": "fixed", "family": "uniform", "outliers": None},
    # Every operand a heavy-tailed / skewed family (the SAME one across operands),
    # chosen per-operand from the heavy-tailed pool so a formula sees realistic
    # fat tails everywhere.
    "heavy_tailed": {"mode": "per_operand", "candidates": HEAVY_TAILED_FAMILIES, "outliers": None},
    # Each operand a DIFFERENT family (full marginal heterogeneity), drawn from
    # the entire registry deterministically by operand name.
    "mixed": {"mode": "per_operand", "candidates": tuple(DISTRIBUTIONS.keys()), "outliers": None},
    # Uniform marginals but with 1-5% gross outliers injected into every operand.
    "with_outliers": {"mode": "fixed", "family": "uniform", "outliers": {"frac": 0.03, "mag": 15.0}},
    # Heavy-tailed marginals AND outliers -- the dirtiest realistic profile.
    "heavy_tailed_outliers": {
        "mode": "per_operand",
        "candidates": HEAVY_TAILED_FAMILIES,
        "outliers": {"frac": 0.02, "mag": 12.0},
    },
}


def sample_operand(
    rng: np.random.Generator,
    n: int,
    operand: str,
    domain: str = DOMAIN_ANY,
    profile: str = "uniform",
    seed: int = 0,
) -> np.ndarray:
    """Sample one operand of length ``n`` honouring its ``domain`` under ``profile``.

    ``rng`` supplies the actual draw (so consecutive operands consume the stream
    reproducibly); ``seed`` only seeds the deterministic per-operand FAMILY choice
    for ``per_operand`` profiles (kept separate from ``rng`` so adding an operand
    does not reshuffle the others' families).
    """
    spec = PROFILES[profile]
    if spec["mode"] == "fixed":
        family = spec["family"]
    else:
        family = family_for_operand(operand, seed, domain, spec["candidates"])
    arr = DISTRIBUTIONS[family](rng, n)
    arr = _enforce_domain(arr, domain)
    out_spec = spec.get("outliers")
    if out_spec:
        arr = with_outliers(rng, arr, frac=out_spec["frac"], mag=out_spec["mag"])
        # Re-enforce the domain AFTER outliers so a negative low-side outlier on a
        # positive/divisor operand can't reintroduce a log-of-negative / blow-up.
        arr = _enforce_domain(arr, domain)
    return arr


def sample_operands(
    seed: int,
    n: int,
    domains: Mapping[str, str],
    profile: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Sample a whole operand dict ``{name: domain}`` reproducibly under ``profile``.

    Returns ``{name: float64 array}``. A single ``Generator`` seeded by ``seed``
    backs every draw, so the full operand set is a pure function of
    ``(seed, n, domains, profile)``. Operand order follows ``domains`` insertion
    order, so the RNG stream assignment is stable.

    >>> import numpy as np
    >>> doms = {"a": "any", "b": "divisor", "c": "positive"}
    >>> d1 = sample_operands(7, 500, doms, profile="heavy_tailed")
    >>> d2 = sample_operands(7, 500, doms, profile="heavy_tailed")
    >>> all(np.array_equal(d1[k], d2[k]) for k in doms)   # reproducible
    True
    >>> d1["b"].min() >= 0.25 - 1e-9 and d1["c"].min() > 0  # domains honoured
    True
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for name, domain in domains.items():
        out[name] = sample_operand(rng, n, name, domain=domain, profile=profile, seed=seed)
    return out


def available_profiles() -> Iterable[str]:
    """Names of the registered sampling profiles."""
    return tuple(PROFILES.keys())
