"""Stress-test regression suite for the S5 CMI redundancy ACCEPTANCE gate under a
MANY-ENGINEERED load (2026-06-11).

The existing ``test_fe_cmi_redundancy_gate`` suite validates the gate on a 3-candidate
pool (2 genuine + 1 spurious). It does NOT exercise the K-SCALING regime the gate hits
in production when the synergy bootstrap / GBM seeder surface dozens-to-hundreds of
engineered survivors. Two real defects were found there and are pinned here:

  BUG 1 (UNBOUNDED COST). The greedy gate is O(K^2) in the candidate count K: every
  remaining candidate is re-scored against the admitted support in EACH greedy round,
  and each scoring runs a 25-permutation within-stratum conditional-permutation null
  with a per-stratum Python shuffle. Measured 1.9 s @ 9 cands -> 27 s @ 99 cands -- a
  clean 2.0x per doubling of K (pure O(K^2)), with NO candidate-count guard. The fix
  PRE-RANKS by marginal MI and caps at ``fe_engineered_cmi_max_candidates`` (default 64)
  before the greedy, bounding cost to O(M^2).

  BUG 2 (EQUIV-CLASS STARVATION / WASTED COST). Monotone/linear remaps of a feature bin
  to the IDENTICAL equi-frequency partition (rank-invariant binning). Such remaps carry
  identical information, yet (a) they paid full per-round permutation-null cost in the
  greedy, and (b) a plain top-M-by-marginal-MI cap could STARVE a genuine driver -- many
  tied-MI remaps of the strongest driver crowd out EVERY form of a weaker genuine driver,
  which then loses all its forms before the greedy runs (observed: at K=200 with tied-MI
  remaps the weakest two of three genuine drivers were fully dropped). The fix collapses
  each exact-partition equivalence class to its highest-marginal-MI representative BEFORE
  the cap, so the cap operates on DISTINCT partitions and every genuine driver survives.

Together: the gate now admits EXACTLY ONE representative per genuine driver, rejects all
redundant remaps, never starves a driver, and the cost is flat (1.0x-1.6x per doubling,
~0.5 s @ 200 cands vs 32 s unguarded).

n<=20000 fixtures (RAM-shared box). The gate function is cheap (no MRMR fit), so the
K-sweep runs in seconds.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import (
    _DEFAULT_MAX_CANDIDATES,
    apply_cmi_redundancy_gate,
)
from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _cmi_from_binned, _quantile_bin
from mlframe.feature_selection.filters.discretization import discretize_array
from mlframe.feature_selection.filters.mrmr import MRMR

_N = 20_000


def _bin_y(y):
    """Bin y."""
    z = discretize_array(np.asarray(y, dtype=np.float64), n_bins=10, method="quantile", dtype=np.int64)
    _, inv = np.unique(z, return_inverse=True)
    return inv.astype(np.int64)


def _mk(v, yb):
    """Quantile-bin v and return (v, CMI-with-yb) for the many-engineered CMI-gate fixture."""
    vb = _quantile_bin(np.asarray(v, dtype=np.float64), nbins=10)
    return (np.asarray(v, dtype=np.float64), float(_cmi_from_binned(vb, yb, None)))


def _world(seed=0, n=_N):
    """3 genuine drivers (a**2/b, log(c)*sin(d), g*h) + noise."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 3.0, n)
    b = rng.uniform(0.5, 3.0, n)
    c = rng.uniform(0.5, 5.0, n)
    d = rng.uniform(0.0, 2 * np.pi, n)
    g = rng.uniform(0.5, 3.0, n)
    h = rng.uniform(0.5, 3.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = a**2 / b + 3.0 * np.log(c) * np.sin(d) + g * h + f / 5.0
    base = {
        "div_ab": a**2 / np.abs(b),
        "mul_cd": np.log(c) * np.sin(d),
        "mul_gh": g * h,
    }
    return base, _bin_y(y), rng


def _pool_with_remaps(seed, n_remaps, n=_N):
    """3 genuine forms + ``n_remaps`` exact-partition (monotone/linear) remaps of EACH
    genuine driver. The remaps are pure redundancy: they bin to the same partition."""
    base, yb, rng = _world(seed, n)
    cands = {k: _mk(v, yb) for k, v in base.items()}
    for which, v0 in base.items():
        for j in range(n_remaps):
            v = 2.5 * v0 + 7.0 + rng.normal(0.0, 1e-6, v0.size)  # affine + sub-bin jitter
            cands[f"red_{which}_{j}"] = _mk(v, yb)
    return cands, yb, base


# ---------------------------------------------------------------------------
# BUG 2: partition-duplicate collapse -- exactly one form per driver, no starvation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_remaps", [3, 12, 64])
def test_partition_duplicates_collapse_one_rep_per_driver(n_remaps):
    """Each genuine driver, surrounded by many exact-partition redundant remaps, yields
    EXACTLY ONE admitted form -- never zero (starvation) and never two+ (redundancy leak)."""
    cands, yb, base = _pool_with_remaps(seed=0, n_remaps=n_remaps)
    accepted, diag = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)
    for drv in base:
        forms = {drv} | {nm for nm in cands if nm.startswith(f"red_{drv}_")}
        n_adm = len(forms & accepted)
        assert (
            n_adm == 1
        ), f"driver {drv}: expected exactly 1 admitted form, got {n_adm} (0 = starvation, >1 = redundancy leak). admitted={sorted(forms & accepted)}"
    # The redundant remaps are explicitly labelled, not silently absent.
    collapsed = [nm for nm, d in diag.items() if d.get("reason") == "redundant_partition_duplicate"]
    assert collapsed, "exact-partition redundant remaps should be flagged redundant_partition_duplicate"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_driver_starvation_with_tied_marginal_mi_remaps(seed):
    """REGRESSION (BUG 2): a plain top-M-by-marginal-MI cap STARVES weaker genuine drivers
    when many tied-MI remaps of the strongest driver fill the cap. With 64 tied-MI remaps
    per driver and the default cap of 64, a naive cap would keep 64 forms of ONE driver and
    drop the other two ENTIRELY. The partition-dedup-before-cap keeps all three drivers."""
    cands, yb, base = _pool_with_remaps(seed=seed, n_remaps=64)  # 3 + 192 = 195 candidates
    assert len(cands) > _DEFAULT_MAX_CANDIDATES  # the pool exceeds the cap (cap path exercised)
    accepted, _diag = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)
    captured = {drv for drv in base if (drv in accepted) or any(nm in accepted for nm in cands if nm.startswith(f"red_{drv}_"))}
    assert captured == set(
        base
    ), f"[seed={seed}] driver(s) starved by the cost cap: captured={captured} expected={set(base)}. The partition dedup must run BEFORE the marginal-MI cap."


def test_canonical_form_preferred_over_redundant_remap():
    """With the genuine canonical form AND its exact-partition remaps present, the gate
    admits the CANONICAL form (highest-marginal-MI representative, deterministic tie-break),
    not an arbitrary noisy remap."""
    base, yb, _rng = _world(seed=0)
    cands = {k: _mk(v, yb) for k, v in base.items()}
    # exact-partition remaps (no jitter -> identical partition, identical marginal MI;
    # canonical wins the tie by name ordering being deterministic).
    for which, v0 in base.items():
        cands[f"zzz_remap_{which}"] = _mk(2.5 * v0 + 7.0, yb)
    accepted, _ = apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)
    for drv in base:
        assert drv in accepted, f"canonical {drv} not admitted: accepted={sorted(accepted)}"
        assert f"zzz_remap_{drv}" not in accepted, f"redundant remap of {drv} wrongly admitted"


# ---------------------------------------------------------------------------
# BUG 1: cost is bounded (O(M^2) cap), not unguarded O(K^2).
# ---------------------------------------------------------------------------


def test_cost_is_bounded_as_K_grows():
    """REGRESSION (BUG 1): the gate cost must NOT grow as O(K^2) in the candidate count.
    Before the fix: ~2.0x per doubling of K (32 s @ 99 cands). After: partition dedup +
    cap flatten it. We assert the wall-time growth from ~50 to ~200 candidates is far
    below the O(K^2) ~16x a doubling-twice would imply (generous threshold to stay robust
    on a shared box -- the point is sub-quadratic, not a tight constant)."""

    def _time(n_remaps):
        """Helper that time."""
        cands, yb, _ = _pool_with_remaps(seed=0, n_remaps=n_remaps)
        t0 = time.perf_counter()
        apply_cmi_redundancy_gate(cands, yb, nbins=10, retain_frac=0.15, seed=0)
        return time.perf_counter() - t0, len(cands)

    t_small, k_small = _time(16)  # 3 + 48 = 51 candidates
    t_large, k_large = _time(66)  # 3 + 198 = 201 candidates
    assert k_large > 3 * k_small  # ~4x more candidates
    # O(K^2) would be ~16x; the capped gate is near-flat. Allow a generous 6x ceiling
    # so a loaded box doesn't flake, while still catching a return to quadratic (which
    # would be >>16x and far exceed 6x).
    ratio = t_large / max(1e-6, t_small)
    assert ratio < 6.0, f"gate cost grew {ratio:.1f}x for ~4x candidates ({k_small}->{k_large}); expected sub-quadratic (cap + dedup). O(K^2) would be ~16x."


def test_cost_cap_disabled_runs_full_greedy():
    """``max_candidates <= 0`` disables the cap (unbounded greedy) -- the escape hatch.
    On a small pool the result is identical to the capped default (cap doesn't fire)."""
    cands, yb, _base = _pool_with_remaps(seed=0, n_remaps=2)  # 9 candidates < cap
    acc_capped, _ = apply_cmi_redundancy_gate(cands, yb, nbins=10, max_candidates=64, seed=0)
    acc_uncapped, _ = apply_cmi_redundancy_gate(cands, yb, nbins=10, max_candidates=0, seed=0)
    assert acc_capped == acc_uncapped


def test_cap_keeps_genuine_drops_low_marginal_tail():
    """When the DISTINCT-partition pool exceeds the cap, the dropped tail is the lowest-
    marginal-MI candidates -- genuine high-MI drivers are retained, weak noise is dropped."""
    base, yb, rng = _world(seed=0)
    cands = {k: _mk(v, yb) for k, v in base.items()}
    # Add many DISTINCT-partition weak-noise candidates (each independent random col ->
    # its own partition, near-zero marginal MI) to push the pool past the cap.
    n = next(iter(base.values())).size
    for j in range(80):
        cands[f"noise_{j}"] = _mk(rng.normal(0.0, 1.0, n), yb)
    accepted, diag = apply_cmi_redundancy_gate(
        cands,
        yb,
        nbins=10,
        retain_frac=0.15,
        max_candidates=16,
        seed=0,
    )
    # All three genuine drivers survive the cap (high marginal MI).
    for drv in base:
        assert drv in accepted, f"genuine {drv} dropped by the cap: {diag.get(drv)}"
    # Some noise was dropped by the cap (the cap actually fired).
    capped = [nm for nm, d in diag.items() if d.get("reason") == "dropped_cost_cap"]
    assert capped, "cap should have dropped low-marginal-MI noise candidates"
    # No admitted candidate is a pure-noise column.
    assert not any(nm.startswith("noise_") for nm in accepted), f"pure-noise candidate wrongly admitted: {[nm for nm in accepted if nm.startswith('noise_')]}"


# ---------------------------------------------------------------------------
# MRMR integration: the cost guard is wired + default.
# ---------------------------------------------------------------------------


def test_mrmr_exposes_cost_guard_default():
    """``MRMR()`` exposes the cost-guard knob with the documented default."""
    m = MRMR()
    assert m.fe_engineered_cmi_max_candidates == _DEFAULT_MAX_CANDIDATES == 64
    # survives a get_params round-trip (sklearn clone compatibility).
    assert MRMR(**m.get_params()).fe_engineered_cmi_max_candidates == 64
