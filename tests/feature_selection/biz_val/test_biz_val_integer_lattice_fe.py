"""biz_value tests for the EXPERIMENTAL integer-lattice FE prototype (gcd / bitwise).

The prototype (``filters._integer_lattice_fe_proto``) is NOT wired into prod yet -- these tests call it directly to pin
the measured win over the closest existing basis, so a future prod-wiring (or a regression) is caught by the quantitative floor.

Measured (n=2000, seed=0, nbins=12, bench_integer_lattice_fe):
  gcd_shared_factor : proto MI 0.471 vs best-existing 0.068 -> lift 6.97x  (GENUINE EDGE)
  bitwise_and_flag  : proto MI 0.539 vs best-existing 0.439 -> lift 1.23x  (marginal edge)
  bitwise_xor_lowbits: lift 0.09  (REDUNDANT -- modular residue already captures low-bit parity; pinned as a non-edge)
  controls (smooth-monotone, noise): proto emits ZERO hits (specificity).

Floors set ~15% below measured.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters._integer_lattice_fe_proto import (
    apply_integer_lattice,
    scan_integer_lattice_pairs,
)
from mlframe.feature_selection.filters._pairwise_modular_fe import _mi

_NBINS = 12


def _best_existing_mi(a, b, y):
    """Best MI the current pipeline could recover: raw cols, arithmetic combos, modular residues."""
    af, bf = a.astype(np.float64), b.astype(np.float64)
    cands = [af, bf, af * bf, af + bf, af - bf, af / (np.abs(bf) + 1.0)]
    for m in (2, 3, 4, 5, 6, 7, 8, 10, 12, 16):
        for base in (af, bf, af + bf, af - bf, af * bf):
            cands.append(np.mod(np.rint(base).astype(np.int64), m).astype(np.float64))
    return max(_mi(c, y, nbins=_NBINS) for c in cands)


@pytest.fixture(scope="module")
def gcd_target():
    """Gcd target."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(1, 60, n)
    b = rng.integers(1, 60, n)
    y = (np.gcd(a, b) >= 3).astype(np.int64)
    return a, b, y


def test_biz_val_gcd_beats_existing_basis_by_3x(gcd_target):
    """gcd captures shared-factor structure no smooth/arithmetic/modular op can. Measured 6.97x; floor 3.0x."""
    a, b, y = gcd_target
    proto_mi = _mi(apply_integer_lattice(a, b, "gcd"), y, nbins=_NBINS)
    existing = _best_existing_mi(a, b, y)
    assert proto_mi / max(existing, 1e-6) >= 3.0, f"gcd lift {proto_mi / max(existing, 1e-6):.2f} below floor"


def test_biz_val_gcd_fires_on_shared_factor_target(gcd_target):
    """Biz val gcd fires on shared factor target."""
    a, b, y = gcd_target
    hits = scan_integer_lattice_pairs(np.column_stack([a, b]).astype(float), y, ["a", "b"], nbins=_NBINS)
    assert any(h["op"] == "gcd" for h in hits), "gcd scan must emit a hit on the shared-factor target"


def test_biz_val_bitwise_and_marginal_edge():
    """AND-flag co-occurrence: measured 1.23x. Floor 1.05x (a genuine, if small, edge over the existing panel)."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 256, n)
    b = rng.integers(0, 256, n)
    y = ((np.bitwise_and(a, b) & 0x80) > 0).astype(np.int64)
    proto_mi = _mi(apply_integer_lattice(a, b, "and"), y, nbins=_NBINS)
    existing = _best_existing_mi(a, b, y)
    assert proto_mi / max(existing, 1e-6) >= 1.05


def test_biz_val_xor_lowbits_is_redundant_with_modular():
    """Pinned NON-edge: XOR-of-low-bits is already captured by the modular residue operator (measured lift 0.09)."""
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.integers(0, 256, n)
    b = rng.integers(0, 256, n)
    y = ((np.bitwise_xor(a, b) & 0x0F).astype(int) % 5).astype(np.int64)
    proto_mi = _mi(apply_integer_lattice(a, b, "xor"), y, nbins=_NBINS)
    existing = _best_existing_mi(a, b, y)
    assert proto_mi / max(existing, 1e-6) < 0.5, "XOR-lowbits should NOT beat the modular operator (redundant)"


@pytest.mark.parametrize("seed", [0, 1])
def test_biz_val_integer_lattice_specific_on_controls(seed):
    """Specificity: no hit on a smooth-monotone target nor on pure noise."""
    rng = np.random.default_rng(seed)
    n = 2000
    a = rng.integers(0, 100, n)
    b = rng.integers(0, 100, n)
    y_smooth = (a + 0.3 * b > 60).astype(np.int64)
    y_noise = rng.integers(0, 4, n).astype(np.int64)
    for y in (y_smooth, y_noise):
        hits = scan_integer_lattice_pairs(np.column_stack([a, b]).astype(float), y, ["a", "b"], nbins=_NBINS)
        assert hits == [], f"integer-lattice scan must not fire on control (seed={seed}); got {hits}"
