"""CPX27 regression: incremental sorted ACI buffer == per-step ``np.sort``.

The online ACI radius helper was changed from a full ``np.sort`` of the rolling
FIFO window every step to an incrementally-maintained sorted buffer
(``bisect.insort`` on append, ``bisect``-located eviction). The per-step radius
is a plain integer-rank index (no interpolation), so the new path must be
BIT-IDENTICAL to the old one over a long sequence -- including the saturation
branches (alpha_t driven to 0 / 1) and the eviction path once the window fills.

This test pins identity against a vendored copy of the pre-change kernel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.training.composite import conformal_online as co


def _old_radius(residuals, alpha):
    """Vendored pre-CPX27 kernel: full ``np.sort`` per step."""
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    r = r[np.isfinite(r)]
    m = int(r.size)
    if m == 0:
        return float("inf")
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0
    rank = int(math.ceil((m + 1) * (1.0 - alpha)))
    if rank > m:
        return float("inf")
    return float(np.sort(r)[rank - 1])


def _eq(a, b):
    return (a == b) or (math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0))


@pytest.mark.parametrize("buffer_n", [50, 500, 2000])
def test_cpx27_sorted_buffer_radius_bit_identical(buffer_n):
    rng = np.random.default_rng(20260623)
    steps = 8000
    # Heavy-tailed magnitudes + duplicate-prone small ints to stress the
    # bisect-eviction (ties must be removed value-for-value, not positionally).
    residuals = np.concatenate(
        [
            np.abs(rng.normal(size=steps // 2)),
            rng.integers(0, 5, size=steps - steps // 2).astype(float),
        ]
    )
    rng.shuffle(residuals)
    # alpha_t wandering across (0,1) incl. both saturations.
    alphas = np.clip(0.1 + np.cumsum(rng.normal(scale=0.02, size=steps)), 0.0, 1.0)

    old_buf = []
    new_state = co._aci_default_state(0.1, 0.05, buffer_n)

    for ar, alpha in zip(residuals.tolist(), alphas.tolist()):
        # Read radius pre-append (online contract), comparing OLD vs NEW.
        old_r = _old_radius(np.asarray(old_buf, dtype=np.float64), alpha) if old_buf else float("inf")
        new_state["alpha_t"] = alpha
        new_r = co._aci_radius(new_state)
        assert _eq(old_r, new_r), f"radius mismatch buffer_n={buffer_n} alpha={alpha}: {old_r} != {new_r}"

        old_buf.append(ar)
        if len(old_buf) > buffer_n:
            del old_buf[: len(old_buf) - buffer_n]
        co._aci_step(new_state, ar, in_interval=True)
        # mirror buffers must agree in content
        assert sorted(old_buf) == new_state["residuals_sorted"]


def test_cpx27_warmup_seeds_sorted_mirror():
    class _Wrap:
        pass

    w = _Wrap()
    warm = np.array([3.0, -1.0, 2.0, 2.0, 5.0])
    co.init_aci(w, alpha=0.1, gamma=0.05, buffer_n=3, warmup_residuals=warm)
    st = w._aci_state_
    # Only the last buffer_n abs-residuals are kept (FIFO), sorted mirror matches.
    assert st["residuals_sorted"] == sorted(st["residuals"])
    assert len(st["residuals_sorted"]) == 3
    assert co._aci_radius(st) == co._rolling_quantile_radius(np.asarray(st["residuals"]), st["alpha_t"])
