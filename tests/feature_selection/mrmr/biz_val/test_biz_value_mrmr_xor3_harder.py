"""biz_value: harder variants of the 3-way XOR hidden among noise.

Companion to ``test_biz_value_mrmr_xor3_noise.py``, which uses a pure ``x1 XOR x2 XOR x3``
and ``x1*x2*x3``. Those are too easy: MRMR's FE builds the exact composite. The cases here
break that shortcut:

* ``noise_leak``:   the target also carries a linear term from a noise column.
* ``mixed_types``:  one binary operand, two continuous operands thresholded at zero.
* ``exp_transform``: ``y = x1 * x2 * exp(x3)``, so no polynomial composite is exact.
* ``correlated_decoys``: 30 columns, each a noisy copy of one operand, sit next to 100 noise columns.

Contract for every case and every seed: default MRMR must select all three operands, raw or
named inside a composite. Decoy-named features do not count. Feature names avoid
underscore-separated operand tokens so ``_operands_named`` cannot match a decoy by accident.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

N_SAMPLES = 4000
N_NOISE = 100
SEEDS = (0, 1, 2)
OPERANDS = ("x1", "x2", "x3")


def _operands_named(names) -> set[str]:
    """Operands named by a selected feature, raw or inside a composite (delimiter-bounded tokens)."""
    found: set[str] = set()
    for nm in names:
        tokens = set(re.split(r"[^A-Za-z0-9]+", str(nm)))
        found.update(op for op in OPERANDS if op in tokens)
    return found


def _noise(rng, n_cols=N_NOISE):
    """Independent noise columns for the harder XOR cases."""
    return rng.standard_normal(size=(N_SAMPLES, n_cols))


def _frame(info, noise, extra=None):
    """Build the frame for one harder XOR case from its target and noise columns."""
    cols = {f"x{i + 1}": info[:, i] for i in range(3)}
    for j in range(noise.shape[1]):
        cols[f"noise_{j}"] = noise[:, j]
    if extra:
        cols.update(extra)
    return pd.DataFrame(cols)


def _case_noise_leak(seed):
    """XOR case where noise columns must not leak into the selection."""
    rng = np.random.default_rng(seed)
    info = rng.uniform(-1, 1, size=(N_SAMPLES, 3))
    noise = _noise(rng)
    y = info[:, 0] * info[:, 1] * info[:, 2] + 0.1 * noise[:, 0]
    return _frame(info, noise), pd.Series(y)


def _case_mixed_types(seed):
    """XOR case mixing numeric and categorical inputs."""
    rng = np.random.default_rng(seed)
    x1 = rng.integers(0, 2, size=N_SAMPLES).astype(float)
    x2 = rng.uniform(-1, 1, size=N_SAMPLES)
    x3 = rng.uniform(-1, 1, size=N_SAMPLES)
    info = np.column_stack([x1, x2, x3])
    y = x1.astype(np.int64) ^ (x2 > 0).astype(np.int64) ^ (x3 > 0).astype(np.int64)
    return _frame(info, _noise(rng)), pd.Series(y)


def _case_exp_transform(seed):
    """XOR case where the informative signal is passed through an exponential."""
    rng = np.random.default_rng(seed)
    info = rng.uniform(-1, 1, size=(N_SAMPLES, 3))
    y = info[:, 0] * info[:, 1] * np.exp(info[:, 2])
    return _frame(info, _noise(rng)), pd.Series(y)


def _case_correlated_decoys(seed):
    """XOR case with correlated decoy columns next to the true pair."""
    rng = np.random.default_rng(seed)
    info = rng.uniform(-1, 1, size=(N_SAMPLES, 3))
    y = info[:, 0] * info[:, 1] * info[:, 2]
    # Decoy names carry no underscore-separated operand token, e.g. "dx1q0" not "dx1_0".
    decoys = {}
    for i in range(3):
        for k in range(10):
            decoys[f"dx{i + 1}q{k}"] = info[:, i] + 0.5 * rng.standard_normal(N_SAMPLES)
    return _frame(info, _noise(rng), extra=decoys), pd.Series(y)


CASES = {
    "noise_leak": _case_noise_leak,
    "mixed_types": _case_mixed_types,
    "exp_transform": _case_exp_transform,
    "correlated_decoys": _case_correlated_decoys,
}


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", list(CASES))
def test_mrmr_recovers_harder_3way_xor(case, seed):
    """MRMR recovers the three XOR inputs on the harder cases."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = CASES[case](seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    names = list(sel.get_feature_names_out())
    operands = _operands_named(names)
    assert operands == set(OPERANDS), f"{case} seed={seed}: recovered operands {sorted(operands)} of {OPERANDS}; support={names}"


@pytest.mark.slow
@pytest.mark.parametrize("case", ["mixed_types", "correlated_decoys"])
def test_mrmr_recovers_harder_3way_xor_under_column_permutation(case):
    """Same contract as above, with the columns shuffled: the recovered operands must not depend on column order."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = CASES[case](SEEDS[0])
    order = np.random.default_rng(1).permutation(X.shape[1])
    X = X.iloc[:, order]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = MRMR(verbose=0).fit(X, y)
    names = list(sel.get_feature_names_out())
    operands = _operands_named(names)
    assert set(OPERANDS) <= operands, f"{case} perm: recovered operands {sorted(operands)} of {OPERANDS}; support={names}"
