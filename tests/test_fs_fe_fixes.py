"""Regression tests for fix-agent #9 (feature_engineering / feature_selection)."""

import pytest
import numpy as np


# ----------------------------------------------------------------------------
# boruta_shap
# ----------------------------------------------------------------------------


def test_boruta_shap_imports_without_binom_test_error():
    """Module must import cleanly even on SciPy >=1.12 where binom_test was removed."""
    pytest.importorskip("scipy")
    pytest.importorskip("shap")
    import mlframe.boruta_shap as bs

    # binom_test wrapper must be callable and return a float p-value.
    p = bs.binom_test(3, 10, 0.5, alternative="greater")
    assert 0.0 <= float(p) <= 1.0


def test_boruta_zscore_no_divzero_on_equal_values():
    pytest.importorskip("shap")
    from mlframe.boruta_shap import BorutaShap

    out = BorutaShap.calculate_Zscore(np.array([1.0, 1.0, 1.0, 1.0]))
    arr = np.asarray(out, dtype=np.float64)
    assert np.all(np.isfinite(arr))


# ----------------------------------------------------------------------------
# mi.py — 3 kernels in agreement
# ----------------------------------------------------------------------------


def test_three_mi_kernels_agree_within_tolerance():
    pytest.importorskip("numba")
    from mlframe.feature_selection.mi import (
        grok_compute_mutual_information,
        chatgpt_compute_mutual_information,
        deepseek_compute_mutual_information,
    )

    rng = np.random.default_rng(42)
    data = rng.integers(0, 15, size=(5_000, 20), dtype=np.int8)
    targets = np.array([0, 5, 10], dtype=np.int64)

    mi_g = grok_compute_mutual_information(data=data, target_indices=targets)
    mi_c = chatgpt_compute_mutual_information(data=data, target_indices=targets)
    mi_d = deepseek_compute_mutual_information(data=data, target_indices=targets)

    assert np.allclose(mi_g, mi_c, atol=1e-4)
    assert np.allclose(mi_c, mi_d, atol=1e-4)


# ----------------------------------------------------------------------------
# filters.entropy
# ----------------------------------------------------------------------------


def test_filters_entropy_handles_zero_probabilities():
    pytest.importorskip("numba")
    from mlframe.feature_selection.filters import entropy

    val = entropy(np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(val)
    assert float(val) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------
# general.py — early exit
# ----------------------------------------------------------------------------


def test_general_early_exit_no_indexerror():
    pl = pytest.importorskip("polars")
    pytest.importorskip("numba")
    from mlframe.feature_selection.general import estimate_features_relevancy

    rng = np.random.default_rng(0)
    data = rng.integers(0, 15, size=(1000, 5), dtype=np.int8)
    df = pl.DataFrame({f"c{i}": data[:, i] for i in range(5)})
    # Must not crash with IndexError when max_runtime_mins forces early exit.
    try:
        estimate_features_relevancy(
            bins=df,
            target_columns=["c0"],
            max_runtime_mins=0.0001,
            min_randomized_permutations=1,
            benchmark_mi_algorithms=False,
            verbose=0,
        )
    except IndexError as e:
        pytest.fail(f"IndexError on early-exit path: {e}")
    except Exception:
        # other exceptions (e.g., empty stack) are allowed — we only care about IndexError
        pass


# ----------------------------------------------------------------------------
# timeseries — accumulated_amount NameError regression
# ----------------------------------------------------------------------------


def test_timeseries_window_var_empty_no_nameerror():
    pd = pytest.importorskip("pandas")
    from mlframe.feature_engineering.timeseries import create_and_process_windows

    df = pd.DataFrame({"x": np.arange(20, dtype=np.float64)})

    def apply_fcn(df, row_features, targets, features_names, dataset_name):
        row_features.append(float(df["x"].sum()))
        if not features_names or dataset_name not in features_names:
            features_names.append(dataset_name)

    row_features = []
    features_names = []
    try:
        create_and_process_windows(
            df=df,
            base_point=5,
            apply_fcn=apply_fcn,
            windows={"": [3, 5]},
            window_features_names=features_names,
            window_features=row_features,
            targets=[],
            forward_direction=True,
        )
    except NameError as e:
        pytest.fail(f"accumulated_amount NameError: {e}")
