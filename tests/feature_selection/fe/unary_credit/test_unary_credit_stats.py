"""``credit_columns``: the linear-usability credit of a unary transform, on the four probe fixtures of ``research_mrmr_unary_probe.md`` and on controls where nothing may be credited.

MRMR's rank-based MI cannot credit these (a function of a selected column has zero conditional MI); the criterion here asks what a linear head gains from ``f(x_j)`` over ``x_j`` given the other columns, with a
max-over-candidates permutation null, a family-wise correction over the columns and replication on held-out rows."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._unary_credit_stats import (
    CANDIDATES,
    adaptive_two_pvalues,
    apply_form,
    credit_columns,
    curvature_basis,
    fit_form,
    oof_replacement_gain,
)

N = 5000


def _fixture(case: str, seed: int = 0):
    """``(X with x1 and x2, y)`` of a probe case."""
    rng = np.random.default_rng(seed)
    x2 = rng.standard_normal(N)
    noise = 0.3 * rng.standard_normal(N)
    if case == "sin_wide":
        x1 = rng.uniform(-6, 6, N)
        y = np.sin(x1) + x2 + noise
    elif case == "sin_narrow":
        x1 = rng.uniform(-1, 1, N)
        y = np.sin(x1) + x2 + noise
    elif case == "log_lognormal":
        x1 = rng.lognormal(0, 1, N)
        y = np.log(x1) + x2 + noise
    elif case == "square":
        x1 = rng.standard_normal(N)
        y = x1**2 + x2 + noise
    elif case == "linear_lognormal":
        x1 = rng.lognormal(0, 1, N)
        y = x1 + x2 + noise
    elif case == "linear_t3":
        x1 = rng.standard_t(3, N)
        y = x1 + x2 + noise
    elif case == "linear_uniform":
        x1 = rng.uniform(0.2, 4.0, N)
        y = x1 + x2 + noise
    else:
        raise ValueError(case)
    return np.column_stack([x1, x2]), y


@pytest.mark.parametrize("case,expected", [("sin_wide", "sin"), ("log_lognormal", "log"), ("square", "sq0")])
def test_the_exact_transform_of_the_probe_fixtures_is_credited(case, expected):
    """Goals 1-3: the exact form is credited (not an approximation), on the transformed column only, never on the noise partner."""
    X, y = _fixture(case)
    verdicts = credit_columns(X, y, seed=0)
    assert verdicts[0].form is not None and verdicts[0].form.name == expected, verdicts[0]
    assert verdicts[0].replicated and verdicts[0].effect > 0.05
    assert verdicts[1].form is None, "the plain partner x2 must keep the identity"


def _sin_narrow(n: int, seed: int):
    """``y = sin(x1) + x2 + noise`` with x1 on [-1, 1], where sin is x minus a cubic term of rms 0.025."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.standard_normal(n)
    return np.column_stack([x1, x2]), np.sin(x1) + x2 + 0.3 * rng.standard_normal(n)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_nearly_linear_sine_is_credited_when_the_rows_resolve_its_cubic_term(seed):
    """The owner's case: the sine on [-1, 1] differs from the identity by a cubic term of 0.4 percent of the variance. An added-variable test of that curvature has t of about 3.6 per 1800 rows, so it is
    credited reliably from about 6000 rows (measured: 12/12 at 6000, 10000 and 20000 with normal noise) and the form it names is ``sin``, the simplest of the forms the data cannot tell apart."""
    X, y = _sin_narrow(20000, seed)
    verdicts = credit_columns(X, y, seed=seed)
    assert verdicts[0].form is not None and verdicts[0].form.name == "sin", verdicts[0]
    assert verdicts[0].effect > 0.0 and verdicts[1].form is None


def test_a_nearly_linear_sine_is_credited_at_6000_rows():
    """The omnibus curvature test (second and third powers of the partialled column) and the out-of-fold replacement filter move the owner's case down from 20000 to 6000 rows: ``sin`` is credited and named in
    11 of 12 seeds with normal noise there (measured; 3 of 12 at 3600 rows for the candidate-only test before the omnibus, 6 of 12 after it). Six seeds here: at least five name ``sin``, none names a wrong form or credits the partner."""
    named = 0
    for seed in range(6):
        X, y = _sin_narrow(6000, seed)
        verdicts = credit_columns(X, y, seed=seed)
        assert verdicts[1].form is None
        assert verdicts[0].form is None or verdicts[0].form.name == "sin" or "sin" in verdicts[0].tied, verdicts[0]
        named += verdicts[0].form is not None and verdicts[0].form.name == "sin"
    assert named >= 5, named


def test_a_replacement_that_throws_the_linear_signal_away_is_not_a_candidate():
    """``cbrt`` of a sine on [-1, 1] explains some of the leftover curvature in an added-variable score and loses 8 percent of the held-out MAE as a replacement: the out-of-fold check refuses it, ``sin`` passes."""
    X, y = _sin_narrow(6000, 0)
    others = [X[:, 1]]
    assert oof_replacement_gain(y, others, X[:, 0], fit_form("sin", X[:, 0])) > 0.0
    assert oof_replacement_gain(y, others, X[:, 0], fit_form("cbrt", X[:, 0])) < 0.0


def test_the_curvature_basis_is_orthonormal_to_the_design_and_empty_for_a_constant_column():
    """The basis holds at most the second and third powers, orthogonal to the design that already contains the column; a constant column has no curvature to test."""
    rng = np.random.default_rng(0)
    x = rng.standard_t(3, 500)
    q, _ = np.linalg.qr(np.column_stack([np.ones(500), x, rng.standard_normal(500)]))
    basis = curvature_basis(x, q)
    assert basis.shape == (500, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-8)
    assert np.abs(q.T @ basis).max() < 1e-8
    assert curvature_basis(np.full(500, 3.0), q[:, :1]).shape[1] == 0


def test_two_statistics_share_the_draws_and_stop_when_neither_can_reach_the_level():
    """The p-values are ``(1 + exceedances) / (draws + 1)`` of the same draws; a strong statistic is resolved at the floor, a hopeless pair stops after the first batch."""
    rng = np.random.default_rng(0)
    calls = []

    def draws(k):
        """``k`` null values of two independent statistics; the batch sizes are recorded."""
        calls.append(k)
        return rng.standard_normal(k), rng.standard_normal(k)

    p_a, p_b = adaptive_two_pvalues((8.0, -8.0), draws, 0.05)
    assert p_a < 0.05 < p_b
    assert sum(calls) >= 40
    calls.clear()
    p_a, p_b = adaptive_two_pvalues((-8.0, -8.0), draws, 0.05)
    assert min(p_a, p_b) > 0.5 and len(calls) == 1


def test_a_nearly_linear_sine_is_not_credited_when_the_rows_cannot_resolve_it():
    """With 600 rows the cubic term is below the noise: nothing is credited (no false credit in 6 seeds), which is the honest answer and not a missed detection."""
    for seed in range(6):
        X, y = _sin_narrow(600, seed)
        assert [v.form for v in credit_columns(X, y, seed=seed)] == [None, None]


@pytest.mark.parametrize("case", ["linear_lognormal", "linear_t3", "linear_uniform"])
def test_a_linear_target_credits_nothing_whatever_the_feature_law(case):
    """Controls: the identity is right for a lognormal, a heavy-tailed and a uniform feature alike (the equal-width MI proposer picked signed_sqrt here 7 of 7)."""
    X, y = _fixture(case)
    assert [v.form for v in credit_columns(X, y, seed=1)] == [None, None]


def test_the_global_null_credits_nothing_in_almost_every_dataset():
    """Twenty pure-noise datasets with eight columns: the family-wise level is 0.05, so at most one dataset may credit anything."""
    credited = 0
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        X = rng.standard_normal((1500, 8))
        y = rng.standard_normal(1500)
        credited += any(v.form is not None for v in credit_columns(X, y, seed=seed, passes=1))
    assert credited <= 1


def test_a_correlated_partner_does_not_steal_the_credit():
    """x2 is nearly a copy of x1 and carries a square: only one of the two copies may be credited, and a linear target on a correlated pair is not credited at all."""
    rng = np.random.default_rng(3)
    x1 = rng.standard_normal(N)
    x2 = x1 + 0.1 * rng.standard_normal(N)
    y_linear = x1 + x2 + 0.3 * rng.standard_normal(N)
    assert [v.form for v in credit_columns(np.column_stack([x1, x2]), y_linear, seed=0)] == [None, None]
    y_square = x1**2 + 0.3 * rng.standard_normal(N)
    credited = [v.form.name for v in credit_columns(np.column_stack([x1, x2]), y_square, seed=0) if v.form is not None]
    assert len(credited) <= 1, credited


def test_small_or_degenerate_inputs_credit_nothing():
    """Fewer than 400 rows, a constant target and non-finite columns are no-ops, not errors."""
    X, y = _fixture("square")
    assert all(v.form is None for v in credit_columns(X[:300], y[:300]))
    assert all(v.form is None for v in credit_columns(X, np.zeros(N)))
    broken = X.copy()
    broken[0, 0] = np.nan
    assert credit_columns(broken, y)[0].form is None


@pytest.mark.parametrize("name", CANDIDATES)
def test_forms_replay_on_new_rows_with_the_fitted_constants(name):
    """A form fitted on one sample replays on another one without NaN and gives the same values for the same inputs (the contract of a recipe)."""
    rng = np.random.default_rng(5)
    fit_rows = rng.lognormal(0, 1, 800) + 0.5
    form = fit_form(name, fit_rows)
    assert form is not None
    new_rows = rng.lognormal(0, 1, 300) + 0.5
    values = apply_form(form, new_rows)
    assert values.shape == new_rows.shape and np.isfinite(values).all()
    np.testing.assert_array_equal(values, apply_form(form, new_rows))


@pytest.mark.parametrize("name", ["log", "recip", "sqrt"])
def test_domain_restricted_forms_are_refused_outside_their_domain(name):
    """log and reciprocal need a positive column and sqrt a non-negative one; the form is not even offered otherwise."""
    assert fit_form(name, np.array([-1.0] + [1.0] * 50 + [2.0] * 50)) is None
