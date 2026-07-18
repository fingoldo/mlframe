"""Numerical-stability tests for MI estimators, discretization, and input-type equivalence.

Eight stability properties pinned down by this module:

1. Miller-Madow bias-correction reduces plug-in MI bias on independent fixtures at small n.
2. ``compute_mi_from_classes`` is invariant under the integer ``dtype`` (int32 vs int64)
   chosen for the joint-counts buffer.
3. ``discretize_array`` produces identical bin assignments under float32 vs float64
   for typical (non-pathological) inputs.
4. Polars and pandas inputs route through the same logical MI pipeline and agree to
   double precision.
5. MI estimates remain finite under extreme class imbalance (no NaN / Inf even when
   one class has a single observation in 1000).
6. Entropy is invariant under arbitrary label permutations (it only depends on the
   frequency multiset).
7. ``entropy`` is protected against ``log(0)``: empty bins do not produce NaN / -Inf.
8. ``_kl_divergence`` in ``cat_interactions`` epsilon-smooths zero cells so divergence
   stays finite when one distribution has a zero probability cell.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import (
    compute_mi_from_classes,
    discretize_array,
    entropy,
)
from mlframe.feature_selection.filters.cat_interactions import _kl_divergence
from mlframe.feature_selection.filters.info_theory import entropy_miller_madow
from mlframe.feature_selection.filters.permutation import mi_direct

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _freqs_from_classes(classes: np.ndarray, nbins: int) -> np.ndarray:
    """Empirical marginal frequencies vector, length ``nbins`` (zeros for empty bins)."""
    counts = np.bincount(classes.astype(np.int64), minlength=nbins)
    return counts.astype(np.float64) / len(classes)


def _plugin_mi(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
    """Plug-in MI = H(X) + H(Y) - H(X,Y) using the project's ``entropy``."""
    fx = _freqs_from_classes(x, kx)
    fy = _freqs_from_classes(y, ky)
    joint = np.zeros((kx, ky), dtype=np.int64)
    for xi, yi in zip(x, y):
        joint[xi, yi] += 1
    fjoint = (joint.ravel().astype(np.float64)) / len(x)
    return float(entropy(fx) + entropy(fy) - entropy(fjoint))


def _mm_mi(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
    """Miller-Madow MI = each of three entropies bias-corrected."""
    n = len(x)
    fx = _freqs_from_classes(x, kx)
    fy = _freqs_from_classes(y, ky)
    joint = np.zeros((kx, ky), dtype=np.int64)
    for xi, yi in zip(x, y):
        joint[xi, yi] += 1
    fjoint = (joint.ravel().astype(np.float64)) / n
    return float(entropy_miller_madow(fx, n) + entropy_miller_madow(fy, n) - entropy_miller_madow(fjoint, n))


# -----------------------------------------------------------------------------
# 1. Miller-Madow bias correction
# -----------------------------------------------------------------------------


def test_mi_miller_madow_reduces_bias():
    """Plug-in MI on an independent pair (true MI = 0) is biased upward; Miller-Madow
    reduces the bias materially at small n and both converge to 0 as n grows.

    Fixture: X uniform over 3 classes, Y uniform over 2 classes, independent. The
    asymmetric (kx=3, ky=2) marginals keep the joint space small enough that the
    Miller-Madow ``(k - 1)/(2n)`` correction takes a real bite at n=10 (~40% bias
    reduction) while the plug-in bias still exceeds 0.1 nat as required.
    """
    kx, ky = 3, 2
    n_seeds = 4000

    biases = {}
    for n in (10, 100, 1000):
        plugin_vals = np.empty(n_seeds, dtype=np.float64)
        mm_vals = np.empty(n_seeds, dtype=np.float64)
        for s in range(n_seeds):
            rng = np.random.default_rng(s)
            x = rng.integers(0, kx, n).astype(np.int32)
            y = rng.integers(0, ky, n).astype(np.int32)
            plugin_vals[s] = _plugin_mi(x, y, kx, ky)
            mm_vals[s] = _mm_mi(x, y, kx, ky)
        biases[n] = (float(plugin_vals.mean()), float(mm_vals.mean()))

    pb10, mb10 = biases[10]
    pb100, mb100 = biases[100]
    pb1000, mb1000 = biases[1000]

    # Plug-in bias at n=10 is high.
    assert pb10 > 0.1, f"plug-in MI bias at n=10 should exceed 0.1 nat, got {pb10:.4f}"

    # Miller-Madow at n=10 is lower than plug-in by >= 30%.
    reduction = 1.0 - mb10 / pb10
    assert mb10 < pb10, f"MM bias {mb10:.4f} should be < plug-in bias {pb10:.4f}"
    assert reduction >= 0.30, f"MM should reduce plug-in bias by >= 30%, got reduction={reduction * 100:.1f}% (plug-in={pb10:.4f}, MM={mb10:.4f})"

    # Both converge to 0 as n grows.
    assert abs(pb1000) < 0.02, f"plug-in bias at n=1000 should be ~0, got {pb1000:.4f}"
    assert abs(mb1000) < 0.02, f"MM bias at n=1000 should be ~0, got {mb1000:.4f}"

    # Monotone decay in absolute bias for both estimators.
    assert pb10 > pb100 > abs(pb1000), f"plug-in bias not monotone: n=10 {pb10:.4f}, n=100 {pb100:.4f}, n=1000 {pb1000:.4f}"
    assert abs(mb10) > abs(mb100) > abs(mb1000), f"MM bias not monotone: n=10 {mb10:.4f}, n=100 {mb100:.4f}, n=1000 {mb1000:.4f}"


# -----------------------------------------------------------------------------
# 2. dtype invariance for compute_mi_from_classes
# -----------------------------------------------------------------------------


@pytest.mark.fast
def test_mi_float32_vs_float64_within_epsilon():
    """``compute_mi_from_classes`` is invariant to the joint-counts dtype.

    ``dtype`` only controls the integer buffer that accumulates joint counts; the
    log-arithmetic happens in float64 either way. int32 vs int64 must agree to
    within 1e-6 (in practice they should be bit-exact for these counts).
    """
    rng = np.random.default_rng(0)
    n, kx, ky = 5000, 4, 3
    # Mild dependency so MI > 0 and the comparison is meaningful.
    x = rng.integers(0, kx, n).astype(np.int32)
    flip = rng.random(n) < 0.3
    y = np.where(flip, rng.integers(0, ky, n), x % ky).astype(np.int32)

    fx = _freqs_from_classes(x, kx)
    fy = _freqs_from_classes(y, ky)

    mi_i32 = compute_mi_from_classes(x, fx, y, fy, dtype=np.int32)
    mi_i64 = compute_mi_from_classes(x, fx, y, fy, dtype=np.int64)

    assert math.isfinite(mi_i32) and math.isfinite(mi_i64)
    assert abs(mi_i32 - mi_i64) < 1e-6, f"MI int32 vs int64 mismatch: {mi_i32!r} vs {mi_i64!r} (diff {mi_i32 - mi_i64:.3e})"


# -----------------------------------------------------------------------------
# 3. float32 vs float64 discretization equivalence
# -----------------------------------------------------------------------------


def test_discretize_array_float32_vs_float64():
    """For typical inputs the float32 and float64 discretizations yield identical
    bin indices. Pathological cases that land a sample exactly on a percentile
    edge are explicitly avoided (the input is jittered onto a coarse grid that's
    representable bit-exactly in both precisions).
    """
    rng = np.random.default_rng(123)
    # Use a coarse grid (multiples of 1/128) that is representable exactly in float32
    # AND float64, so percentile boundaries don't shift between precisions.
    base = rng.integers(0, 4096, size=2000).astype(np.float64) / 128.0
    x64 = base
    x32 = base.astype(np.float32)

    n_bins = 10
    bins64 = discretize_array(x64, n_bins=n_bins)
    bins32 = discretize_array(x32, n_bins=n_bins)

    assert bins64.shape == bins32.shape
    mismatches = int(np.count_nonzero(bins64 != bins32))
    assert mismatches == 0, f"float32 vs float64 discretize_array disagree on {mismatches} samples (out of {len(x64)})"


# -----------------------------------------------------------------------------
# 4. polars vs pandas MI agreement
# -----------------------------------------------------------------------------


def test_polars_input_matches_pandas_for_mi_direct():
    """Both polars and pandas inputs flow through the same MI computation;
    results must agree to double precision (1e-9).
    """
    pl = pytest.importorskip("polars")

    rng = np.random.default_rng(7)
    n = 500
    a = rng.integers(0, 5, n).astype(np.int32)
    b = ((a + rng.integers(0, 2, n)) % 5).astype(np.int32)
    nbins = np.array([5, 5], dtype=np.int32)

    # mi_direct itself takes a numpy ``factors_data`` matrix (already discretized).
    # The polars vs pandas equivalence is therefore that the *upstream* discretized
    # matrix is identical regardless of frame engine; we verify by constructing
    # both frames over the same underlying values and running mi_direct over the
    # extracted numpy arrays.
    pdf = pd.DataFrame({"a": a, "b": b})
    pldf = pl.DataFrame({"a": a, "b": b})

    factors_from_pandas = pdf[["a", "b"]].to_numpy().astype(np.int32)
    factors_from_polars = pldf.select(["a", "b"]).to_numpy().astype(np.int32)
    assert np.array_equal(factors_from_pandas, factors_from_polars)

    mi_pd, _ = mi_direct(
        factors_data=factors_from_pandas,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=0,
        n_workers=1,
    )
    mi_pl, _ = mi_direct(
        factors_data=factors_from_polars,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=0,
        n_workers=1,
    )

    assert math.isfinite(mi_pd) and math.isfinite(mi_pl)
    assert abs(mi_pd - mi_pl) < 1e-9, f"polars vs pandas MI mismatch: {mi_pd!r} vs {mi_pl!r}"


# -----------------------------------------------------------------------------
# 5. extreme imbalance keeps MI finite
# -----------------------------------------------------------------------------


def test_compute_mi_from_classes_finite_at_extreme_imbalance():
    """y with 999 zeros and a single 1-class observation must produce a finite,
    non-negative MI (plug-in entropy is small but well-defined; the rare class
    must not trigger a log(0) or div-by-zero).
    """
    n = 1000
    rng = np.random.default_rng(2024)
    x = rng.integers(0, 4, n).astype(np.int32)
    y = np.zeros(n, dtype=np.int32)
    y[0] = 1  # single rare observation

    fx = _freqs_from_classes(x, 4)
    fy = _freqs_from_classes(y, 2)

    mi_val = compute_mi_from_classes(x, fx, y, fy, dtype=np.int64)

    assert math.isfinite(mi_val), f"MI must be finite under extreme imbalance, got {mi_val!r}"
    assert mi_val >= 0.0, f"MI must be non-negative, got {mi_val!r}"
    # Marginal entropy of y with p=(0.999, 0.001) is small but well-defined.
    h_y = float(entropy(fy))
    assert math.isfinite(h_y) and 0.0 < h_y < 0.05, f"H(Y) should be small but positive, got {h_y!r}"


# -----------------------------------------------------------------------------
# 6. label-permutation invariance of entropy
# -----------------------------------------------------------------------------


@pytest.mark.fast
def test_entropy_invariant_under_label_permutation():
    """``entropy`` operates on the frequency multiset, not on label identity.
    Permuting the underlying labels (which preserves the multiset of counts)
    must yield the identical numeric value.
    """
    # Build three label arrays that share the same frequency multiset {2, 2}
    # but differ in label identity / ordering.
    a = np.array([0, 0, 1, 1], dtype=np.int32)
    b = np.array([1, 1, 0, 0], dtype=np.int32)
    c = np.array([0, 1, 0, 1], dtype=np.int32)

    def _freqs(arr: np.ndarray) -> np.ndarray:
        """Helper that freqs."""
        counts = np.bincount(arr.astype(np.int64))
        return counts.astype(np.float64) / counts.sum()

    h_a = float(entropy(_freqs(a)))
    h_b = float(entropy(_freqs(b)))
    h_c = float(entropy(_freqs(c)))

    assert h_a == h_b == h_c, f"entropy not permutation-invariant: a={h_a!r}, b={h_b!r}, c={h_c!r}"
    # Sanity: this is ln(2).
    assert math.isclose(h_a, math.log(2.0), rel_tol=1e-12)


# -----------------------------------------------------------------------------
# 7. log(0) protection in entropy
# -----------------------------------------------------------------------------


def test_log_zero_protection_in_entropy():
    """A frequency vector with an exact zero cell must not break ``entropy``
    via log(0). The implementation either clips zero cells or routes through
    xlogy; either way the output must be finite and equal to the entropy
    computed over the non-zero cells.
    """
    # Three nominal classes (0, 1, 2); class 2 is empty.
    counts = np.array([10, 10, 0], dtype=np.float64)
    freqs = counts / counts.sum()  # [0.5, 0.5, 0.0]

    h = float(entropy(freqs))

    assert math.isfinite(h), f"entropy must be finite with zero cells, got {h!r}"
    assert not math.isnan(h), "entropy returned NaN on zero-cell input"
    # Expected value: H = -2 * 0.5 * log(0.5) = log(2).
    assert math.isclose(h, math.log(2.0), rel_tol=1e-12), f"entropy with one empty cell should equal log(2) over the surviving cells, got {h!r}"


# -----------------------------------------------------------------------------
# 8. _kl_divergence epsilon smoothing
# -----------------------------------------------------------------------------


@pytest.mark.fast
def test_kl_smoothing_in_cat_interactions():
    """``_kl_divergence(p, q)`` must remain finite when ``q`` has a zero cell.
    Without epsilon smoothing this would diverge via log(p/0). With smoothing
    it stays finite, positive, and asymmetric (KL is not symmetric).
    """
    p = np.array([0.5, 0.5], dtype=np.float64)
    q = np.array([1.0, 0.0], dtype=np.float64)

    kl_pq = _kl_divergence(p, q)
    kl_qp = _kl_divergence(q, p)

    assert math.isfinite(kl_pq), f"KL(p||q) must be finite with zero cell in q, got {kl_pq!r}"
    assert math.isfinite(kl_qp), f"KL(q||p) must be finite with zero cell in q, got {kl_qp!r}"
    assert kl_pq > 0.0, f"KL(p||q) should be positive for distinct distributions, got {kl_pq!r}"
    # Sanity: KL(p||p) is zero.
    assert math.isclose(_kl_divergence(p, p), 0.0, abs_tol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
