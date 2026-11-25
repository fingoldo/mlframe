"""The exact link of ``y = f(x1) + x2 + noise`` is named by the credit, over the link function, the law of the input and the law of the noise (the pin of the owner's unary-transform probe).

Each cell asserts that the credited form of ``x1`` is exactly the generating one (not a basis approximation, not a near twin) and that the plain partner ``x2`` keeps the identity. Noise is scaled to 30 percent of the
standard deviation of the signal, so a heavy-tailed noise (t with 3 degrees of freedom) and a Laplace noise are as loud as the normal one."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._unary_credit_stats import credit_columns

N = 5000

#: link name -> (sampler of x1, link function, expected form)
LINKS = {
    "sin_uniform": (lambda r, n: r.uniform(-6, 6, n), np.sin, "sin"),
    "cos_uniform": (lambda r, n: r.uniform(-6, 6, n), np.cos, "cos"),
    "log_lognormal": (lambda r, n: r.lognormal(0, 1, n), np.log, "log"),
    "log_gamma": (lambda r, n: r.gamma(2.0, 1.0, n) + 0.05, np.log, "log"),
    "sqrt_exponential": (lambda r, n: r.exponential(1.0, n), np.sqrt, "sqrt"),
    "sqrt_uniform": (lambda r, n: r.uniform(0, 9, n), np.sqrt, "sqrt"),
    "square_normal": (lambda r, n: r.standard_normal(n), lambda x: x * x, "sq0"),
    "square_uniform": (lambda r, n: r.uniform(-3, 3, n), lambda x: x * x, "sq0"),
    "square_t5": (lambda r, n: r.standard_t(5, n), lambda x: x * x, "sq0"),
    "recip_wide": (lambda r, n: r.uniform(0.12, 3.0, n), lambda x: 1.0 / x, "recip"),
    "cbrt_normal": (lambda r, n: r.standard_normal(n), np.cbrt, "cbrt"),
    "abs_normal": (lambda r, n: r.standard_normal(n), np.abs, "abs"),
}
#: noise name -> unit-variance sampler
NOISES = {
    "normal": lambda r, n: r.standard_normal(n),
    "laplace": lambda r, n: r.laplace(0, 1 / np.sqrt(2), n),
    "t3": lambda r, n: r.standard_t(3, n) / np.sqrt(3),
}


def _fixture(link: str, noise: str, seed: int = 0):
    """``(X with x1 and x2, y)`` of one cell of the grid."""
    rng = np.random.default_rng(seed)
    sampler, fn, _ = LINKS[link]
    x1 = sampler(rng, N)
    x2 = rng.standard_normal(N)
    signal = fn(x1)
    return np.column_stack([x1, x2]), signal + x2 + 0.3 * float(np.std(signal)) * NOISES[noise](rng, N)


@pytest.mark.parametrize("noise", sorted(NOISES))
@pytest.mark.parametrize("link", sorted(LINKS))
def test_the_generating_link_is_named_exactly(link, noise):
    """The credited form of x1 is the generating link and x2 keeps the identity, for every input law and noise law of the grid."""
    X, y = _fixture(link, noise)
    verdicts = credit_columns(X, y, seed=0)
    expected = LINKS[link][2]
    assert verdicts[0].form is not None and verdicts[0].form.name == expected, f"{link}/{noise}: credited {verdicts[0].form and verdicts[0].form.name!r}, expected {expected!r}"
    assert verdicts[1].form is None, f"{link}/{noise}: the partner x2 was credited {verdicts[1].form.name!r}"
    assert verdicts[0].effect > 0.0


@pytest.mark.parametrize("noise", sorted(NOISES))
@pytest.mark.parametrize("sampler", ["uniform", "normal", "lognormal", "t3"])
def test_a_linear_link_credits_nothing_whatever_the_input_and_noise_law(sampler, noise):
    """Controls over the same grid of input and noise laws: a linear link has no form to credit."""
    rng = np.random.default_rng(1)
    x1 = {"uniform": rng.uniform(-1, 1, N), "normal": rng.standard_normal(N), "lognormal": rng.lognormal(0, 1, N), "t3": rng.standard_t(3, N)}[sampler]
    x2 = rng.standard_normal(N)
    y = x1 + x2 + 0.3 * float(np.std(x1)) * NOISES[noise](rng, N)
    assert [v.form for v in credit_columns(np.column_stack([x1, x2]), y, seed=1)] == [None, None]
