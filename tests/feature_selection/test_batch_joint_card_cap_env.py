"""MLFRAME_BATCH_JOINT_CARD_CAP is read per call and reaches the njit kernels."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._batch_kernels import (
    MAX_JOINT_CARDINALITY,
    batch_pair_mi_prange,
    check_joint_cardinality,
    joint_cardinality_cap,
)


def test_the_env_var_sets_the_cap_after_import(monkeypatch):
    """It was documented as an operator override and read nowhere."""
    monkeypatch.setenv("MLFRAME_BATCH_JOINT_CARD_CAP", "1_000")
    assert joint_cardinality_cap() == 1000
    with pytest.raises(ValueError, match="exceeds cap 1000"):
        check_joint_cardinality(40, 40)


def test_an_unparseable_value_falls_back_instead_of_breaking(monkeypatch):
    monkeypatch.setenv("MLFRAME_BATCH_JOINT_CARD_CAP", "lots")
    assert joint_cardinality_cap() == MAX_JOINT_CARDINALITY


def test_the_kernel_skips_a_pair_over_the_passed_cap():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 30, size=(500, 2)).astype(np.int32)
    y = (data[:, 0] % 2).astype(np.int32)
    args = (data, np.array([0]), np.array([1]), np.array([30, 30], dtype=np.int64), y, np.array([0.5, 0.5]))
    free = batch_pair_mi_prange(*args)
    capped = batch_pair_mi_prange(*args, 100)
    assert free[0] > 0.0 and capped[0] == 0.0, "30*30*2 cells exceed a cap of 100, so the pair must score the no-information 0.0"
