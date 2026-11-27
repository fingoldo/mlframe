"""Which unary transform u was applied to each input, when the target is ``F(u_i(x_i) + e_i) + e`` and F folds the factors into one number: a sum, a product, or a sum of pairwise products.

The inputs carry the generating transform each; the transformed value gets its own noise (``0.1`` of its standard deviation, the law of the grid), F folds them, and the target gets noise of ``0.3`` of the standard
deviation of F. The detection must name the generating closed form of EVERY input, leave the three pure-noise columns alone, and (for products) accept exactly the products that exist. The grid covers every admissible unary
form the credit can name except ``sq`` and ``rankgauss``: ``sq`` is ``sq0`` plus a linear term and so identical for a linear head with the identity in the design, ``rankgauss`` is the fallback for any monotone shape.

Two honest limits. (1) Near twins: ``tanh``, ``asinh`` and ``sin`` of a bell-shaped column are different curves that predict equally well within the noise; the credit then names the simplest of the forms the data cannot
tell apart and lists the others in ``ColumnVerdict.tied`` (``unary_forms_tied_`` in MRMR), so a generating form counts as detected when it is the credited form or in that list. (2) Power: ``asinh`` of a column
with t3 tails (most of its variance sits in a few extreme rows) is a factor of a product that the credit names in about half of the seeds, otherwise it credits nothing or a twin; it is measured in the last test and kept out of
the product grid."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._unary_credit_interaction import credit_interactions, select_products
from mlframe.feature_selection.filters._unary_credit_stats import credit_columns

#: form -> (sampler of the input, the transform)
FORMS = {
    "sin": (lambda r, n: r.uniform(-6, 6, n), np.sin),
    "cos": (lambda r, n: r.uniform(-6, 6, n), np.cos),
    "log": (lambda r, n: r.lognormal(0, 1, n), np.log),
    "sqrt": (lambda r, n: r.exponential(1.0, n), np.sqrt),
    "recip": (lambda r, n: r.uniform(0.3, 3.0, n), lambda x: 1.0 / x),
    "sq0": (lambda r, n: r.standard_normal(n), lambda x: x * x),
    "abs": (lambda r, n: r.standard_normal(n), np.abs),
    "tanh": (lambda r, n: r.standard_normal(n), np.tanh),
    "cbrt": (lambda r, n: r.standard_normal(n), np.cbrt),
    "asinh": (lambda r, n: 3.0 * r.standard_normal(n), np.arcsinh),
}
NOISES = {
    "normal": lambda r, n: r.standard_normal(n),
    "laplace": lambda r, n: r.laplace(0, 1 / np.sqrt(2), n),
    "t3": lambda r, n: r.standard_t(3, n) / np.sqrt(3),
}
NOISE_COLUMNS = 3


def _build(fold: str, forms: "list[str]", n: int, seed: int, noise: str):
    """``(X, y)``: one column per generating form, then three pure-noise columns; ``y = F(u_i(x_i) + e_i) + e``."""
    rng = np.random.default_rng(seed)
    columns, factors = [], []
    for name in forms:
        sampler, fn = FORMS[name]
        x = sampler(rng, n)
        columns.append(x)
        v = fn(x)
        factors.append(v + 0.1 * float(v.std()) * NOISES[noise](rng, n))
    columns += [rng.standard_normal(n) for _ in range(NOISE_COLUMNS)]
    if fold == "sum":
        total = sum(factors)
    elif fold == "product":
        total = np.prod(factors, axis=0)
    else:
        total = sum(factors[i] * factors[j] for i in range(len(factors)) for j in range(i + 1, len(factors)))
    return np.column_stack(columns), total + 0.3 * float(total.std()) * NOISES[noise](rng, n)


def _names(verdicts) -> "list[str | None]":
    """Form names of a list of verdicts."""
    return [v.form.name if v.form else None for v in verdicts]


def _detected(verdicts, forms: "list[str]") -> "tuple[int, int]":
    """``(exact, covered)``: how many generating forms are the credited form, and how many are the credited form or one of its indistinguishable twins; the columns after the signal ones must credit nothing."""
    assert all(v.form is None for v in verdicts[len(forms) :]), _names(verdicts)
    exact = sum(1 for v, f in zip(verdicts, forms) if v.form is not None and v.form.name == f)
    covered = sum(1 for v, f in zip(verdicts, forms) if v.form is not None and (v.form.name == f or f in v.tied))
    return exact, covered


@pytest.mark.parametrize("noise", sorted(NOISES))
def test_a_sum_of_ten_transformed_inputs_names_every_form(noise):
    """F = sum over ten inputs, each with its own admissible transform: the credit names all ten and nothing among the noise columns, whatever the law of the noise."""
    forms = list(FORMS)
    X, y = _build("sum", forms, 8000, 0, noise)
    exact, covered = _detected(credit_columns(X, y, seed=0), forms)
    assert covered == len(forms) and exact >= len(forms) - 1, (exact, covered)


PAIRS = [("sin", "log"), ("sq0", "sqrt"), ("recip", "abs"), ("cos", "tanh"), ("cbrt", "cos")]


@pytest.mark.parametrize("pair", PAIRS, ids=["-".join(p) for p in PAIRS])
def test_a_product_of_two_transformed_inputs_names_both_forms_and_accepts_the_product(pair):
    """F = product: neither factor has a linear effect, so the single-column test is blind; the interaction credit names both forms and the product is accepted."""
    forms = list(pair)
    X, y = _build("product", forms, 6000, 0, "normal")
    verdicts, _ = credit_interactions(X, y, seed=0)
    assert _detected(verdicts, forms) == (2, 2), _names(verdicts)
    found = [v.form for v in verdicts[:2]]
    assert select_products(X[:, :2], y, found, [(0, 1)], seed=0) == [(0, 1)]


TRIPLES = [("sin", "log", "sq0"), ("sqrt", "recip", "abs"), ("cos", "tanh", "cbrt")]


@pytest.mark.parametrize("triple", TRIPLES, ids=["-".join(t) for t in TRIPLES])
def test_a_sum_of_pairwise_products_names_all_three_forms_and_accepts_all_three_products(triple):
    """F = u1 u2 + u1 u3 + u2 u3: every input's form is named through its products and the three products are accepted."""
    forms = list(triple)
    X, y = _build("pairs", forms, 6000, 0, "normal")
    verdicts, _ = credit_interactions(X, y, seed=0)
    exact, covered = _detected(verdicts, forms)
    assert covered == 3 and exact >= 2, _names(verdicts)
    found = [v.form for v in verdicts[:3]]
    accepted = select_products(X[:, :3], y, found, [(0, 1), (0, 2), (1, 2)], seed=0)
    assert sorted(accepted) == [(0, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize("noise", ["laplace", "t3"])
def test_a_product_keeps_its_forms_under_heavy_tailed_noise(noise):
    """Laplace and t3 noise (inside the factors and on the target) leave the forms of a product intact."""
    X, y = _build("product", ["sin", "log"], 6000, 0, noise)
    verdicts, _ = credit_interactions(X, y, seed=0)
    assert _names(verdicts) == ["sin", "log"] + [None] * NOISE_COLUMNS


def test_additive_factors_do_not_make_a_product():
    """The controls of the product search: two transformed factors that only ADD are credited as forms and their product is rejected; a product of two raw columns has no form to credit; pure noise credits nothing."""
    X, y = _build("sum", ["sin", "log"], 6000, 0, "normal")
    verdicts, _ = credit_interactions(X, y, seed=0)
    found = [v.form for v in verdicts[:2]]
    assert found and all(f is not None for f in found), _names(verdicts)
    assert select_products(X[:, :2], y, found, [(0, 1)], seed=0) == []
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((6000, 4))
    assert _names(credit_interactions(raw, raw[:, 0] * raw[:, 1] + 0.3 * rng.standard_normal(6000), seed=1)[0]) == [None] * 4
    assert _names(credit_interactions(raw, rng.standard_normal(6000), seed=2)[0]) == [None] * 4


def test_the_limit_of_a_heavy_tailed_factor_is_measured_not_hidden():
    """``asinh`` of a t3 column inside a product: the credited form is ``asinh``, a twin of it or nothing (named in about half of the seeds at 10000 rows; the cause is the leverage of a few extreme rows)."""
    outcomes = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        x1, x2 = rng.standard_normal(10000), rng.standard_t(3, 10000)
        y = np.cbrt(x1) * np.arcsinh(x2)
        y = y + 0.3 * float(y.std()) * rng.standard_normal(10000)
        verdicts, _ = credit_interactions(np.column_stack([x1, x2, rng.standard_normal((10000, 3))]), y, seed=seed)
        assert _names(verdicts)[0] == "cbrt" and _names(verdicts)[2:] == [None] * 3
        outcomes.append(_names(verdicts)[1])
    assert set(outcomes) <= {"asinh", "tanh", "rankgauss", "sin", None}, outcomes


@pytest.mark.slow
@pytest.mark.parametrize(
    "fold,forms",
    [("sum", ["sin", "log", "sq0"]), ("product", ["sq0", "sqrt"]), ("product", ["cos", "tanh"]), ("pairs", ["sin", "log", "sq0"]), ("pairs", ["sqrt", "recip", "abs"])],
    ids=lambda v: v if isinstance(v, str) else "-".join(v),
)
def test_mrmr_reports_the_generating_form_of_every_input_and_a_compact_output(fold, forms):
    """End to end: ``unary_forms_`` holds the generating form of every signal input, the output names them (``mul(sin(x1),log(x2))``) in at most one column per term of F, and a ridge head on the output is within 5 percent of the oracle MAE."""
    from sklearn.linear_model import RidgeCV

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _build(fold, forms, 6000, 0, "normal")
    frame = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(len(forms))] + [f"n{k}" for k in range(NOISE_COLUMNS)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MRMR(verbose=0, random_state=0, n_jobs=1, fe_max_steps=1).fit(frame, pd.Series(y))
    names = list(m.get_feature_names_out())
    assert {k: v for k, v in m.unary_forms_.items() if k.startswith("x")} == {f"x{i + 1}": f for i, f in enumerate(forms)}, (m.unary_forms_, names)
    assert not [k for k in m.unary_forms_ if k.startswith("n")], m.unary_forms_
    terms = {"sum": len(forms), "product": 1, "pairs": len(forms) * (len(forms) - 1) // 2}[fold]
    assert len(names) <= terms, names
    for i, f in enumerate(forms):
        assert any(f"{f}(x{i + 1})" in nm for nm in names), (f, names)
    truth = [FORMS[f][1](X[:, i]) for i, f in enumerate(forms)]
    oracle = {
        "sum": truth,
        "product": [truth[0] * truth[1]],
        "pairs": [truth[i] * truth[j] for i in range(len(forms)) for j in range(i + 1, len(forms))] + truth,
    }[fold]

    def head_mae(design: np.ndarray) -> float:
        """In-sample MAE of a ridge head (both heads see the same rows)."""
        return float(np.abs(y - RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(design, y).predict(design)).mean())

    assert head_mae(np.nan_to_num(np.asarray(m.transform(frame), dtype=np.float64))) <= 1.05 * head_mae(np.column_stack(oracle))
