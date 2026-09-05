"""The correspondence-analysis residual must not be shrunk by a pad on a quadratically small denominator.

`S = (P - expected) / sqrt(expected + 1e-12)` where `expected = r @ c` is a product of two marginal
PROBABILITIES, so it is quadratically small for a rare-by-rare pair: two categories each at 1e-6 of the
rows give an expected of 1e-12, the same order as the pad. Measured on a cell occurring three times as
often as independence predicts, the pad shrinks the standardised residual by 29.3% at marginals of 1e-6 and
by 90.0% at 1e-7 -- and a large residual in a rare cell is precisely what correspondence analysis exists to
surface, so this is damping the signal rather than the noise.

A zero here means an empty row or column, where `P` is zero too and the residual is genuinely zero. The
mask says that; the pad approximated it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.cat_cooccurrence_svd import _contingency_svd_row_coords


def _rare_pair_table(mass: float, lift: float = 3.0, n_other: int = 4) -> np.ndarray:
    """A co-occurrence table with one rare-by-rare cell carrying `lift` times the independent expectation."""
    table = np.zeros((n_other + 1, n_other + 1), dtype=np.float64)
    table[:n_other, :n_other] = 1.0 / (n_other * n_other)
    bulk = table.sum()
    table[n_other, n_other] = lift * mass * mass * bulk
    table[n_other, :n_other] = mass * bulk / n_other
    table[:n_other, n_other] = mass * bulk / n_other
    return table


def _standardised_residuals(table: np.ndarray, pad: float | None) -> np.ndarray:
    """The chi-square residual matrix, computed either the masked way or the padded way."""
    total = table.sum()
    P = table / total
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    expected = r @ c
    if pad is None:
        return np.divide(P - expected, np.sqrt(expected), out=np.zeros_like(P), where=expected > 0.0)
    return (P - expected) / np.sqrt(expected + pad)


@pytest.mark.parametrize("mass", [1e-6, 1e-7], ids=["mass_1e-6", "mass_1e-7"])
def test_the_pad_shrinks_a_rare_cell_residual_and_the_mask_does_not(mass: float):
    """Pins both halves: the pad damps the rare cell, and the mask leaves it alone."""
    table = _rare_pair_table(mass)
    exact = _standardised_residuals(table, pad=None)
    padded = _standardised_residuals(table, pad=1e-12)
    rare = (-1, -1)
    assert abs(exact[rare]) > 0.0, "the fixture has no rare-cell residual to lose"
    shrinkage = 1.0 - abs(padded[rare]) / abs(exact[rare])
    assert shrinkage > 0.2, f"the fixture no longer reproduces the damping it was built to show (shrunk {shrinkage:.1%})"


def test_an_empty_category_gives_a_zero_row_rather_than_a_division_error():
    """A zero marginal is what the pad was guarding against; the mask has to handle it without raising."""
    table = np.zeros((4, 4), dtype=np.float64)
    table[:3, :3] = 1.0
    with np.errstate(divide="raise", invalid="raise"):
        out = _contingency_svd_row_coords(table, n_eff=2, normalize="ca")
    assert out.shape == (4, 2)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[3], 0.0), "the unused category should sit at the origin, not at a padded approximation"


def test_an_all_zero_table_still_returns_zeros():
    """The degenerate short-circuit above the residual is unchanged."""
    out = _contingency_svd_row_coords(np.zeros((3, 3)), n_eff=2, normalize="ca")
    assert out.shape == (3, 2)
    assert np.allclose(out, 0.0)


def test_an_ordinary_table_is_unchanged_by_the_rewrite():
    """Away from the degenerate corner the masked form must agree with the padded one it replaces."""
    rng = np.random.default_rng(0)
    table = rng.integers(5, 50, size=(6, 5)).astype(np.float64)
    exact = _standardised_residuals(table, pad=None)
    padded = _standardised_residuals(table, pad=1e-12)
    assert np.allclose(exact, padded, rtol=1e-9)


def test_the_embedding_is_finite_on_a_rare_by_rare_table():
    """End to end: the public entry point must come back finite on the regime that motivated the fix."""
    out = _contingency_svd_row_coords(_rare_pair_table(1e-7), n_eff=2, normalize="ca")
    assert np.all(np.isfinite(out))
