"""Audit 2026-11-20 finding 05-F6: the data-valuation benchmark modules import and their fixtures build at tiny sizes.

Nothing else under tests/ imports the _benchmarks package, so a signature drift in the library call a script makes would
surface only when someone ran the script by hand. This imports each script and builds its fixture at a tiny size.
"""

import importlib

import numpy as np
import pytest

SCRIPTS = (
    "mlframe.data_valuation._benchmarks.profile_data_valuation",
    "mlframe.data_valuation._benchmarks.profile_training_weight_adapter",
    "mlframe.data_valuation._benchmarks.profile_adversarial_reweighting",
)


@pytest.mark.parametrize("module_name", SCRIPTS)
def test_benchmark_script_imports(module_name):
    """Each script imports without running its benchmark (the entry point is under __main__)."""
    assert importlib.import_module(module_name) is not None


def test_shared_fixture_builds_the_documented_shapes():
    """The shared blob fixture returns train and validation blocks of the requested shape with binary labels."""
    from mlframe.data_valuation._benchmarks._dv_bench_shared import make_blob_fixture

    X_train, y_train, X_val, y_val = make_blob_fixture(40, 10, 5, seed=0)
    assert X_train.shape == (40, 5) and X_val.shape == (10, 5)
    assert set(np.unique(y_train)) <= {0, 1} and set(np.unique(y_val)) <= {0, 1}


def test_adversarial_fixture_helper_builds_at_a_tiny_size():
    """The adversarial script's fixture helper returns a feature matrix and binary labels of the requested length."""
    module = importlib.import_module("mlframe.data_valuation._benchmarks.profile_adversarial_reweighting")
    X, y = module._make_fixture(30, n_features=4, seed=1)
    assert X.shape == (30, 4) and y.shape == (30,)
