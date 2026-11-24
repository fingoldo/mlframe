"""The target-transform recommender recognises ``y = U(F(x) + e1) + e2`` for every invertible unary ``U`` of the admissible set: it returns the inverse of ``U`` (or an equally good one) and abstains on a linear target.

``F`` is a linear index of five uniform columns, ``e1`` is noise inside the index (0.1 of its standard deviation) and ``e2`` noise on the target (0.02 of its standard deviation, only for the ``U`` defined on the whole line: a
noise on a target that lives in (0, 1) or on the positive half-line would leave the domain of its inverse). The judge is MAE on the original scale: a pick other than the exact inverse counts only if its held-out MAE is within
3 percent of that of the exact inverse (``log`` and ``log1p`` or ``asinh`` of a large positive target are the same curve to within noise)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._target_transform_ll import DEFAULT_CANDIDATES, build_transform, recommend_target_transform

N = 4000
NOISES = {
    "normal": lambda r, n: r.standard_normal(n),
    "laplace": lambda r, n: r.laplace(0, 1 / np.sqrt(2), n),
    "t3": lambda r, n: r.standard_t(3, n) / np.sqrt(3),
}
WEIGHTS = np.array([1.0, 0.8, 0.6, 0.4, 0.2])


def _standardised(s: np.ndarray) -> np.ndarray:
    """The index with zero mean and unit variance."""
    return (s - s.mean()) / s.std()


#: U -> (index from the weighted sum ``s`` of the columns, U, the pool member that is its inverse, whether noise on the target keeps it inside the domain)
LINKS = {
    "exp": (lambda s: 2.0 * s, np.exp, "log", False),
    "square": (lambda s: 0.5 + s, lambda v: v * v, "signed_sqrt", False),
    "cube": (lambda s: 2.0 * _standardised(s), lambda v: v**3, "cbrt", True),
    "sqrt": (lambda s: 0.2 + 3.0 * s, np.sqrt, "signed_square", False),
    "recip": (lambda s: 0.3 + 2.0 * s, lambda v: 1.0 / v, "reciprocal", False),
    "sigmoid": (lambda s: 4.0 * _standardised(s), lambda v: 1.0 / (1.0 + np.exp(-v)), "logit", False),
    "sinh": (lambda s: 1.5 * _standardised(s), np.sinh, "asinh", True),
    "tanh": (lambda s: 1.5 * _standardised(s), np.tanh, "atanh", False),
    "log": (lambda s: 0.1 + 3.0 * s, np.log, "exp", False),
}


def _make(link: str, noise: str, seed: int, n: int = N):
    """``(X, y)`` of one cell of the grid."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    index_fn, fn, _, outer = LINKS[link]
    index = index_fn(X @ WEIGHTS)
    index = index + 0.1 * float(index.std()) * NOISES[noise](rng, n)
    y = fn(index)
    if outer:
        y = y + 0.02 * float(y.std()) * NOISES[noise](rng, n)
    return X, y


@pytest.mark.parametrize("noise", ["normal", "t3"])
@pytest.mark.parametrize("link", sorted(LINKS))
def test_the_inverse_of_the_link_is_recommended_or_something_as_good(link, noise):
    """The recommended transform is the inverse of U, or its held-out MAE is within 3 percent of the exact inverse's; the MAE falls well below the identity's."""
    expected = LINKS[link][2]
    exact_named = 0
    for seed in range(3):
        X, y = _make(link, noise, seed)
        choice = recommend_target_transform(X, y, seed=seed)
        assert not choice.abstained, (link, noise, seed, choice.reason)
        exact = recommend_target_transform(X, y, seed=seed, candidates=[expected])
        assert not exact.abstained, (link, noise, seed, exact.reason)
        assert choice.err_chosen <= 1.03 * exact.err_chosen, (link, noise, seed, choice.name, choice.err_chosen, exact.err_chosen)
        assert choice.err_chosen < 0.9 * choice.err_identity, (link, noise, seed, choice.name)
        exact_named += choice.name == expected
    assert exact_named >= 2, (link, noise, exact_named)


@pytest.mark.parametrize("noise", sorted(NOISES))
def test_a_linear_target_abstains(noise):
    """The control: ``y = F(x) + noise`` is already linear, whatever the law of the noise; ten datasets, no transform is recommended."""
    chosen = 0
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        X = rng.random((N, 5))
        s = X @ WEIGHTS
        y = s + 0.3 * float(s.std()) * NOISES[noise](rng, N)
        chosen += not recommend_target_transform(X, y, seed=seed).abstained
    assert chosen <= 1, chosen


def test_a_target_independent_of_the_inputs_abstains():
    """Pure noise: nothing is recommended in ten datasets (a family-wise level of 0.05 allows one)."""
    chosen = 0
    for seed in range(10):
        rng = np.random.default_rng(200 + seed)
        chosen += not recommend_target_transform(rng.random((N, 5)), rng.standard_normal(N), seed=seed).abstained
    assert chosen <= 1, chosen


def test_every_fitted_transform_inverts_its_forward_on_the_training_range():
    """``inverse(forward(y))`` returns ``y`` for every candidate family on a target that all of them accept (positive, inside (0, 1)); the clip never bites inside the training range."""
    rng = np.random.default_rng(0)
    y = rng.uniform(0.05, 0.95, 500)
    for name in (*DEFAULT_CANDIDATES, "box_cox"):
        lam = 1.5 if name in ("yeo_johnson", "box_cox") else (-1.0 if name in ("asinh", "sinh") else None)
        fitted = build_transform(name, y, lam)
        back = fitted.inverse(fitted.forward(y))
        np.testing.assert_allclose(back, y, rtol=1e-6, atol=1e-8, err_msg=name)


def test_the_domain_of_a_transform_excludes_it_for_a_target_outside_it():
    """A target with negative values has no ``log``, ``log1p``, ``logit`` or ``reciprocal`` candidate, a constant target and too few rows abstain with a reason, and an unknown name raises."""
    from mlframe.feature_selection.filters._target_transform_ll import analyse_target_transforms

    rng = np.random.default_rng(0)
    X = rng.random((500, 3))
    an = analyse_target_transforms(X, X.sum(1) - 1.5 + 0.1 * rng.standard_normal(500), n_null=19)
    assert not {"log", "log1p", "logit", "reciprocal"} & set(an.names), an.names
    assert recommend_target_transform(X, np.ones(500), n_null=19).abstained
    assert recommend_target_transform(X[:50], rng.standard_normal(50), n_null=19).abstained
    with pytest.raises(ValueError):
        recommend_target_transform(X, rng.standard_normal(500), candidates=["not_a_transform"], n_null=19)
