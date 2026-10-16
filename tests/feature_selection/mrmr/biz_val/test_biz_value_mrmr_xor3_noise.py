"""biz_value: MRMR recovers a 3-way XOR hidden among 100 pure-noise columns.

Target is a 3-way interaction with zero marginal and zero pairwise signal, so no single
column or pair of columns predicts it. Two variants:

* binary:     ``y = x1 XOR x2 XOR x3`` with ``x_i`` in {0, 1}, classification target.
* continuous: ``y = x1 * x2 * x3`` with ``x_i ~ U(-1, 1)``, regression target.

Contract: default MRMR (no tuning) selects EXACTLY ``['x1*x2*x3__T1_T1_T1']``. The target is fully explained by that
composite, so anything next to it (a noise column, a raw operand, a partial composite, another variant of the composite)
is a redundant selection and fails the contract.

Baseline (not run in CI, kept for reference): boosted trees on the same data.
They fail to find the signal once the noise columns are present. Measured with CatBoost,
XGBoost and LightGBM at n=4000 and seed 0: binary accuracy about 0.49 to 0.52 with 100
noise columns (1.000 without noise). Continuous CatBoost R2 about 0.60.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

N_SAMPLES = 4000
N_NOISE = 100
SEED = 0
SEEDS = (0, 1, 2)
OPERANDS = ("x1", "x2", "x3")


def _operands_named(names) -> set[str]:
    """Operands that a selected feature names, raw or inside a composite.

    A composite such as ``x1*x2*x3__T1_T1_T1`` names all three operands as delimiter-bounded
    tokens, so ``x1`` does not match ``x12``.
    """
    found: set[str] = set()
    for nm in names:
        tokens = set(re.split(r"[^A-Za-z0-9]+", str(nm)))
        found.update(op for op in OPERANDS if op in tokens)
    return found


EXPECTED_SUPPORT = ["x1*x2*x3__T1_T1_T1"]


def _support_problems(names) -> list[str]:
    """What is wrong with a selected support: it must be exactly the one composite of the three operands, with nothing next to it."""
    names = [str(nm) for nm in names]
    if names == EXPECTED_SUPPORT:
        return []
    problems = [] if EXPECTED_SUPPORT[0] in names else [f"the composite {EXPECTED_SUPPORT[0]!r} is not selected"]
    for nm in names:
        if nm == EXPECTED_SUPPORT[0]:
            continue
        tokens = set(re.split(r"[^A-Za-z0-9]+", nm))
        problems.append(f"extra selection {nm!r}" + (" (noise)" if "noise" in tokens else " (redundant operand or composite variant)"))
    return problems


def _make_frame(y_kind: str, seed: int = SEED, perm_seed: int | None = None):
    """Frame with a three-way XOR among noise columns."""
    rng = np.random.default_rng(seed)
    if y_kind == "binary":
        info = rng.integers(0, 2, size=(N_SAMPLES, 3)).astype(float)
        y = info[:, 0].astype(np.int64) ^ info[:, 1].astype(np.int64) ^ info[:, 2].astype(np.int64)
    else:
        info = rng.uniform(-1, 1, size=(N_SAMPLES, 3))
        y = info[:, 0] * info[:, 1] * info[:, 2]
    noise = rng.standard_normal(size=(N_SAMPLES, N_NOISE))
    columns = list(OPERANDS) + [f"noise_{i}" for i in range(N_NOISE)]
    X = pd.DataFrame(np.hstack([info, noise]), columns=columns)
    if perm_seed is not None:
        # Same data, columns in a shuffled order: the recovered operands must not change.
        X = X[[columns[i] for i in np.random.default_rng(perm_seed).permutation(len(columns))]]
    return X, pd.Series(y)


def _fit_names(y_kind, seed, perm_seed=None):
    """Fit MRMR and return the selected column names."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _make_frame(y_kind, seed, perm_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    return list(sel.get_feature_names_out())


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("y_kind", ["binary", "continuous"])
def test_mrmr_recovers_3way_xor_among_noise(y_kind, seed):
    """MRMR recovers the three-way XOR among noise columns."""
    names = _fit_names(y_kind, seed)
    assert not _support_problems(names), f"{y_kind} seed={seed} 3-way XOR: {_support_problems(names)}; support={names}"


@pytest.mark.slow
@pytest.mark.parametrize("perm_seed", [1, 2])
@pytest.mark.parametrize("y_kind", ["binary", "continuous"])
def test_mrmr_recovers_3way_xor_under_column_permutation(y_kind, perm_seed):
    """The XOR recovery holds when the columns are permuted."""
    names = _fit_names(y_kind, SEED, perm_seed)
    assert not _support_problems(names), f"{y_kind} perm={perm_seed} 3-way XOR: {_support_problems(names)}; support={names}"


# Baseline comparison, kept commented out: it fits three boosters per variant and takes minutes.
#
# import xgboost as xgb
# import lightgbm as lgb
# from catboost import CatBoostClassifier, CatBoostRegressor
# from sklearn.metrics import accuracy_score, r2_score
# from sklearn.model_selection import train_test_split
#
# X, y = _make_frame("binary")
# Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
# for name, model in [
#     ("catboost", CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0, random_seed=SEED)),
#     ("xgboost", xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.1, random_state=SEED)),
#     ("lightgbm", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.1, random_state=SEED, verbose=-1)),
# ]:
#     model.fit(Xtr, ytr)
#     print(name, "binary acc", accuracy_score(yte, model.predict(Xte)))  # about 0.5 with noise
#
# X, y = _make_frame("continuous")
# Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
# cb = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0, random_seed=SEED)
# cb.fit(Xtr, ytr)
# print("catboost continuous R2", r2_score(yte, cb.predict(Xte)))  # about 0.60
