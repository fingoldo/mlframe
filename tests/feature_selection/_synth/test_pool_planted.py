"""The planted-signal generators plant what they claim: a real effect through the intended transform, and (where stated) no marginal binned-MI signal on the raw column."""

import numpy as np
import pytest

from tests.feature_selection._synth.pool_planted import GENERATORS


def _binned_mi(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Plug-in MI (nats) of the 10-bin quantile code of ``x`` and the binary ``y``, minus the Miller-Madow bias."""
    edges = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1])
    c = np.searchsorted(edges, x)
    n = len(x)
    joint = np.zeros((bins, 2))
    np.add.at(joint, (c, y.astype(int)), 1.0)
    pxy = joint / n
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    return mi - (bins - 1) * 1 / (2.0 * n)


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_shape_and_determinism(name):
    """Shape (n, p), planted names listed first, binary target, same output for the same seed."""
    a = GENERATORS[name](n=3000, p=30, seed=4)
    b = GENERATORS[name](n=3000, p=30, seed=4)
    X, y, planted = a
    assert X.shape == (3000, 30) and set(np.unique(y)) <= {0, 1}
    assert list(X.columns[: len(planted)]) == planted and all(c.startswith("noise_") for c in X.columns[len(planted) :])
    assert X.equals(b[0]) and np.array_equal(y, b[1])


def test_hidden_periodic_is_invisible_to_binned_mi_but_not_to_the_sine():
    """At frequency 20 the raw column's 10-bin MI is at the noise level while the sine of it is a strong predictor."""
    X, y, _ = GENERATORS["hidden_periodic"](n=20000, p=5, seed=1, freq=20.0)
    x = X["sig_x"].to_numpy()
    assert _binned_mi(x, y) < 0.01
    assert _binned_mi(np.sin(2 * np.pi * 20.0 * x), y) > 0.1


def test_weak_marginal_u_shape_has_no_linear_signal_and_a_clear_transformed_signal():
    """A U shape of a symmetric column: Pearson correlation about zero, MI of ``x**2`` clearly positive."""
    X, y, _ = GENERATORS["weak_marginal_shape"](n=20000, p=5, seed=2, strength=1.5)
    x = X["sig_x"].to_numpy()
    assert abs(np.corrcoef(x, y)[0, 1]) < 0.03
    assert _binned_mi(x**2, y) > 0.02


def test_grouped_key_has_no_marginal_effect_and_the_centred_value_does():
    """The key alone is uninformative; the value minus its group mean is."""
    import pandas as pd

    X, y, _ = GENERATORS["grouped_by_irrelevant_key"](n=20000, p=5, seed=3)
    assert _binned_mi(X["sig_key"].to_numpy(), y) < 0.01
    centred = X["sig_x"].to_numpy() - pd.Series(X["sig_x"].to_numpy()).groupby(X["sig_key"].to_numpy()).transform("mean").to_numpy()
    assert _binned_mi(centred, y) > 0.05


def test_ratio_partner_is_irrelevant_alone():
    """``sig_b`` has no marginal signal; the ratio does."""
    X, y, _ = GENERATORS["ratio_irrelevant_normaliser"](n=20000, p=5, seed=5)
    assert _binned_mi(X["sig_b"].to_numpy(), y) < 0.01
    assert _binned_mi((X["sig_a"] / X["sig_b"]).to_numpy(), y) > 0.05


def test_conditional_hinge_gate_column_is_irrelevant_alone():
    """``sig_z`` carries no marginal signal; the sign-gated hinge does."""
    X, y, _ = GENERATORS["conditional_hinge"](n=20000, p=5, seed=6)
    z, x = X["sig_z"].to_numpy(), X["sig_x"].to_numpy()
    assert _binned_mi(z, y) < 0.01
    assert _binned_mi(np.where(z > 0, np.maximum(x, 0.0), np.maximum(-x, 0.0)), y) > 0.05
