"""``data_banzhaf`` draws its coalition masks in blocks and still returns the estimate one full-size draw would (INP-03)."""

import numpy as np

from mlframe.data_valuation import _mc_sampling as mc


def _utility(idx: np.ndarray) -> float:
    """A deterministic utility: the sum of a fixed per-row weight over the coalition."""
    w = np.linspace(-1.0, 1.0, 40)
    return float(w[idx].sum())


def _reference(n_coalitions: int, n_rows: int, seed: int) -> np.ndarray:
    """The estimate computed from a single (n_coalitions, n_rows) draw, as the function did before blocking."""
    rng = np.random.default_rng(seed)
    masks = rng.random((n_coalitions, n_rows)) < 0.5
    v = np.array([_utility(np.flatnonzero(masks[m]).astype(np.int64)) for m in range(n_coalitions)])
    beta = np.zeros(n_rows)
    for j in range(n_rows):
        inm = masks[:, j]
        if inm.sum() == 0 or (~inm).sum() == 0:
            continue
        beta[j] = v[inm].mean() - v[~inm].mean()
    return beta


def test_blocked_masks_give_the_single_draw_estimate(monkeypatch):
    """A block of 7 coalitions per draw reproduces the one-draw estimate to rounding."""
    monkeypatch.setattr(mc, "_MASK_BLOCK_BYTES", 7 * 40)
    beta, info = mc.data_banzhaf(_utility, 40, n_coalitions=100, rng=np.random.default_rng(3))
    np.testing.assert_allclose(beta, _reference(100, 40, 3), rtol=0, atol=1e-12)
    assert info["n_coalitions"] == 100


def test_mask_block_holds_no_more_than_the_byte_budget(monkeypatch):
    """No random draw is larger than the budget: every block is at most budget // n_rows coalitions."""
    monkeypatch.setattr(mc, "_MASK_BLOCK_BYTES", 5 * 40)
    sizes = []
    real = np.random.Generator.random

    class _Spy:
        """Wraps the generator and records the number of rows of each (rows, cols) draw."""

        def __init__(self, rng):
            """Hold the real generator."""
            self._rng = rng

        def random(self, size):
            """Record the shape and delegate."""
            sizes.append(size[0])
            return real(self._rng, size)

    mc.data_banzhaf(_utility, 40, n_coalitions=23, rng=_Spy(np.random.default_rng(0)))
    assert max(sizes) == 5 and sum(sizes) == 23
