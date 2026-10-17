"""biz_value: conjunction with weak partners, proxies and noise, under any column order.

The target is y ~ Bernoulli(sigmoid(-1.5 + 2.5*A + W*A*B*C)). A is strong; B and C act only through their joint AND with A.
Each operand has 50 proxies (standardised copies of the operand with correlation 0.7), and there are 100 pure noise columns.

Contract: default MRMR selects EXACTLY the raw A and one composite naming A, B and C, and nothing else. A carries the main effect and the
composite carries the conjunction, which together explain the target completely, so a proxy, a noise column, a lone B or C, a partial
composite or a second variant of the composite is a redundant selection. The outcome does not depend on the order of the columns.
"""

import re
import warnings
from functools import cache

import numpy as np
import pandas as pd
import pytest

N_SAMPLES = 50_000  # at n=4000 the 150 proxies make the weak partners statistically marginal; the recovery is tested at 50k
W = 3.0
N_PROXY = 50
N_NOISE = 100
RHO = 0.7
OPERANDS = ("A", "B", "C")
_BAD_TOKENS = {"decA", "decB", "decC", "noise"}


def _make_frame(seed, perm_seed=None, n=N_SAMPLES):
    """Build the conjunction frame: A, B, C operands, their proxies and noise, optionally with shuffled columns."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, n)
    B = rng.integers(0, 2, n)
    C = rng.integers(0, 2, n)
    logit = -1.5 + 2.5 * A + W * (A & B & C)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.int64)
    cols = {"A": A.astype(float), "B": B.astype(float), "C": C.astype(float)}
    for prefix, src in (("decA", A), ("decB", B), ("decC", C)):
        z = (src - src.mean()) / src.std()
        for k in range(N_PROXY):
            cols[f"{prefix}_{k}"] = RHO * z + np.sqrt(1.0 - RHO**2) * rng.standard_normal(n)
    for k in range(N_NOISE):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    names = list(cols)
    if perm_seed is not None:
        order = np.random.default_rng(perm_seed).permutation(len(names))
        names = [names[i] for i in order]
    return pd.DataFrame({c: cols[c] for c in names}), pd.Series(y)


def test_proxy_correlation_matches_rho():
    """Proxies of each operand correlate with it at the configured rho."""
    X, _ = _make_frame(seed=0, n=20_000)
    for operand in OPERANDS:
        proxy = X[f"dec{operand}_0"].to_numpy()
        corr = np.corrcoef(proxy, X[operand].to_numpy())[0, 1]
        assert abs(corr - RHO) < 0.02, f"proxies of {operand} correlate {corr:.3f}, expected {RHO}"


def _tokens(name):
    """Split a column name into alphanumeric tokens."""
    return set(re.split(r"[^A-Za-z0-9]+", str(name)))


def _operands_named(names):
    """Operands (A, B, C) named by the selected features, raw or inside a composite."""
    return {op for op in OPERANDS if any(op in _tokens(nm) for nm in names)}


def _bad_names(names):
    """Selected features that name a proxy or a noise column."""
    return [nm for nm in names if _tokens(nm) & _BAD_TOKENS]


@cache
def _fit_selection(perm_seed):
    """Fit default MRMR on the frame with the given column permutation (cached per permutation)."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _make_frame(seed=0, perm_seed=perm_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    return tuple(sel.get_feature_names_out())


def _support_problems(names) -> list[str]:
    """What is wrong with a selected support: it must be the raw A plus exactly one composite of A, B and C."""
    names = [str(nm) for nm in names]
    composites = [nm for nm in names if set(OPERANDS) <= _tokens(nm)]
    problems = []
    if "A" not in names:
        problems.append("the raw A is not selected")
    if not composites:
        problems.append("no composite names A, B and C")
    elif len(composites) > 1:
        problems.append(f"more than one composite of A, B and C: {composites}")
    for nm in names:
        if nm == "A" or nm in composites:
            continue
        problems.append(f"extra selection {nm!r}" + (" (proxy or noise)" if _tokens(nm) & _BAD_TOKENS else " (redundant operand fragment)"))
    return problems


def _assert_contract(names):
    """Assert the support is exactly the raw A plus one composite of A, B and C."""
    assert not _support_problems(names), f"{_support_problems(names)}; support={list(names)}"


@pytest.mark.slow
def test_conjunction_recovers_operands_native_order():
    """Default MRMR names every operand in the native column order."""
    _assert_contract(_fit_selection(None))


@pytest.mark.slow
@pytest.mark.parametrize("perm_seed", [1, 2])
def test_conjunction_recovers_operands_under_column_permutation(perm_seed):
    """Default MRMR names every operand under a column permutation."""
    _assert_contract(_fit_selection(perm_seed))


@pytest.mark.slow
def test_conjunction_operand_set_is_column_order_invariant():
    """The named operand set is the same in native and permuted column order."""
    assert _operands_named(_fit_selection(None)) == _operands_named(_fit_selection(1))


@pytest.mark.slow
def test_triplet_stage_pool_is_column_order_invariant():
    """The triplet stage winners are the same in every column order and include a triplet over A, B, C."""
    from mlframe.feature_selection.filters._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe

    results = []
    for perm_seed in (None, 1, 2):
        X, y = _make_frame(seed=0, perm_seed=perm_seed)
        _, _, tri = hybrid_orth_mi_triplet_fe(X, y.to_numpy(), cols=list(X.columns), nbins=10, top_k=0)
        results.append(set(tri["engineered_col"]) if len(tri) else set())
    assert results[0] == results[1] == results[2], f"triplet winners depend on column order: {results}"
    assert any(
        {"A", "B", "C"} <= set(c.split("__", 1)[0].replace("*", " ").split()) for c in results[0]
    ), f"no triplet over A, B, C among winners: {results[0]}"
