"""CPX33 regression: the Kendall-tau prescreen path must use scipy.stats.kendalltau at FULL n (Knight's O(n log n)),
not the prior O(n^2) hand-rolled loop with a silent n>2000 subsample.

Pins two contracts:
  1. scipy's tau-b equals an INDEPENDENT exact full-n O(n^2) tau-b reference within ~1e-9 (correctness of the statistic).
  2. The continuous-target p-value path is full-n: identical inputs differing only past row 2000 must yield different
     p-values (a 2000-row subsample would ignore those rows and return the same p). I.e. no silent subsampling.
"""

import numpy as np
import pytest

scipy_stats = pytest.importorskip("scipy.stats")

from mlframe.feature_selection.wrappers._univariate_ht import _kendall_p_numeric_continuous


def _exact_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Independent exact O(n^2) Kendall tau-b reference over the FULL n.

    tau-b = (P - Q) / sqrt((P+Q+T_x) * (P+Q+T_y)) where P/Q are concordant/discordant pairs, T_x is the number of pairs
    tied ONLY in x (and not in y), T_y tied ONLY in y. A pair tied in BOTH x and y is excluded from all four counts.
    """
    n = x.shape[0]
    P = Q = Tx = Ty = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            tied_x = dx == 0
            tied_y = dy == 0
            if tied_x and tied_y:
                continue
            if tied_x:
                Tx += 1
            elif tied_y:
                Ty += 1
            elif (dx > 0) == (dy > 0):
                P += 1
            else:
                Q += 1
    denom = np.sqrt((P + Q + Tx) * (P + Q + Ty))
    if denom <= 0:
        return 0.0
    return (P - Q) / denom


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_scipy_tau_matches_exact_full_n_reference(seed):
    """Scipy tau matches exact full n reference."""
    rng = np.random.default_rng(seed)
    n = 300  # small enough for the O(n^2) reference, large enough to be non-trivial
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(size=n)
    tau_scipy, _ = scipy_stats.kendalltau(x, y, variant="b")
    tau_ref = _exact_tau_b(x, y)
    assert abs(tau_scipy - tau_ref) < 1e-9, f"scipy tau {tau_scipy} vs exact full-n ref {tau_ref}"


def test_scipy_tau_matches_exact_reference_with_ties():
    """Scipy tau matches exact reference with ties."""
    rng = np.random.default_rng(3)
    n = 250
    x = rng.integers(0, 5, size=n).astype(np.float64)  # heavy ties -> tau-b correction exercised
    y = rng.integers(0, 4, size=n).astype(np.float64)
    tau_scipy, _ = scipy_stats.kendalltau(x, y, variant="b")
    tau_ref = _exact_tau_b(x, y)
    assert abs(tau_scipy - tau_ref) < 1e-9, f"tied: scipy {tau_scipy} vs ref {tau_ref}"


def test_kendall_p_is_full_n_not_subsampled():
    """A change confined to rows beyond 2000 MUST move the p-value. If the path still subsampled to the first/random 2000
    rows it could miss those rows and return an identical p -- the regression this test guards against."""
    rng = np.random.default_rng(11)
    n = 5000
    x = rng.normal(size=n)
    y = rng.normal(size=n)  # base: independent -> high p

    # Strengthen association ONLY in the tail beyond row 2000 by aligning y with x there.
    x2 = x.copy()
    y2 = y.copy()
    y2[2000:] = x2[2000:]

    p_base = _kendall_p_numeric_continuous(x, y, random_state=0)
    p_tail = _kendall_p_numeric_continuous(x2, y2, random_state=0)

    assert p_base != p_tail, "tail-only change did not move p -> path is ignoring rows past 2000 (subsampling regression)"
    assert p_tail < p_base, "injected tail association should lower the p-value at full n"


def test_kendall_p_random_state_has_no_effect():
    """random_state is retained for back-compat only; distinct seeds must give identical p (no subsample draw)."""
    rng = np.random.default_rng(5)
    n = 4000
    x = rng.normal(size=n)
    y = 0.3 * x + rng.normal(size=n)
    p0 = _kendall_p_numeric_continuous(x, y, random_state=0)
    p1 = _kendall_p_numeric_continuous(x, y, random_state=12345)
    assert p0 == p1, "distinct seeds gave distinct p -> a subsample draw is still happening"
