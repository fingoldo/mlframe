"""Env knobs that change selection results are read per call, and a bad value warns instead of breaking an import."""

import importlib
import subprocess
import sys

import pytest

from mlframe.utils.env_flags import env_float, env_int


def test_bad_numeric_value_falls_back(monkeypatch):
    """A value that is not a number, is NaN or is below the minimum falls back to the default."""
    monkeypatch.setenv("MLFRAME_TEST_KNOB", "auto")
    assert env_int("MLFRAME_TEST_KNOB", 7) == 7
    assert env_float("MLFRAME_TEST_KNOB", 0.5) == 0.5
    monkeypatch.setenv("MLFRAME_TEST_KNOB", "nan")
    assert env_float("MLFRAME_TEST_KNOB", 0.5) == 0.5
    monkeypatch.setenv("MLFRAME_TEST_KNOB", "1_000")
    assert env_int("MLFRAME_TEST_KNOB", 7, minimum=2) == 1000
    monkeypatch.setenv("MLFRAME_TEST_KNOB", "1")
    assert env_int("MLFRAME_TEST_KNOB", 7, minimum=2) == 7


@pytest.mark.parametrize(
    "module, accessor, name, value, expected",
    [
        ("mlframe.feature_selection.filters.evaluation", "mrmr_null_signif_alpha", "MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.01", 0.01),
        ("mlframe.feature_selection.filters.evaluation", "jmim_exponent_discount_only", "MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY", "1", True),
        ("mlframe.feature_selection.filters.permutation", "null_mean_min_perms", "MLFRAME_MRMR_NULL_PERMS", "64", 64),
        ("mlframe.feature_selection.filters._ksg", "ksg_gpu_threshold", "MLFRAME_KSG_GPU_N", "1000", 1000),
    ],
)
def test_value_set_after_import_takes_effect(monkeypatch, module, accessor, name, value, expected):
    """A knob set after the module was imported is still honoured: it is read on every call."""
    fn = getattr(importlib.import_module(module), accessor)
    monkeypatch.delenv(name, raising=False)
    default = fn()
    monkeypatch.setenv(name, value)
    assert fn() == expected != default


def test_unparseable_value_does_not_break_the_import():
    """A bad knob value at import time does not stop the module importing; the default is used."""
    code = "import mlframe.feature_selection.filters._ksg as k; import mlframe.feature_selection.filters.evaluation as e; print(k.ksg_gpu_threshold(), e.mrmr_null_signif_alpha())"
    import os

    import mlframe

    src = os.path.dirname(os.path.dirname(mlframe.__file__))  # the tree under test, not whatever copy is installed
    env = {"MLFRAME_KSG_GPU_N": "auto", "MLFRAME_MRMR_NULL_SIGNIF_ALPHA": "x", "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}

    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env}, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["50000", "0.05"]


_NUMERIC_KNOB_MODULES = {
    "mlframe.core.robust_location": "MLFRAME_ROBUST_MEAN_PARALLEL_MIN_N",
    "mlframe.feature_engineering.basic": "MLFRAME_CYCLICAL_PAR_THRESHOLD",
    "mlframe.feature_engineering.grouped": "MLFRAME_GROUPED_COUNT_VECTORIZE_MAX_AVG",
    "mlframe.feature_engineering.nadaraya_watson": "MLFRAME_NW_PARALLEL_MIN_QUERIES",
    "mlframe.feature_engineering.recency_density": "MLFRAME_RECENCY_KDE_PARALLEL_MIN_GROUPS",
    "mlframe.feature_engineering.transformer._utils": "MLFRAME_NONFINITE_PAR_THRESHOLD",
    "mlframe.feature_selection.boruta_shap._auto_dispatch": "MLFRAME_BORUTA_AUTO_NP_RATIO",
    "mlframe.feature_selection.filters.discretization._discretization_dataset": "MLFRAME_DISCRETIZE_COL_CACHE_MAX_BYTES",
    "mlframe.feature_selection.filters.discretization": "MLFRAME_DISCRETIZE_UNIFORM_PAR_THRESHOLD",
    "mlframe.feature_selection.filters.hermite_fe._hermite_oracle": "MLFRAME_POLYEVAL_PAR_THRESHOLD",
    "mlframe.feature_selection.filters.hermite_fe._hermite_robust": "MLFRAME_DETECT_HEAVY_TAIL_NJIT_MAX_N",
    "mlframe.feature_selection.filters._conditional_gate_fe": "MLFRAME_GATE_BUILD_NJIT_MIN_N",
    "mlframe.feature_selection.filters._fe_gpu_batch._devices": "MLFRAME_FE_VRAM_CUSHION_FRAC",
    "mlframe.feature_selection.filters._fe_gpu_batch._executor": "MLFRAME_FE_VRAM_BLOCKS_PER_DEVICE",
    "mlframe.feature_selection.filters._fe_gpu_batch._packer": "MLFRAME_FE_VRAM_CPSAT_TIME_LIMIT_S",
    "mlframe.feature_selection.filters._orthogonal_three_gate_mi_fe": "MLFRAME_OOF_BATCH_BINNING_MAX_TRAIN_ROWS",
    "mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi": "MLFRAME_FE_IMBALANCE_PRIOR",
    "mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup": "MLFRAME_FE_DEDUP_MAX_CORR_ROWS",
    "mlframe.metrics.calibration._calibration_plot": "MLFRAME_CALIB_BINNING_PRANGE_THRESHOLD",
    "mlframe.metrics.regression._regression_metrics": "MLFRAME_MAX_ERROR_PAR_THRESHOLD",
    "mlframe.metrics._core_auc_brier": "MLFRAME_METRICS_ARGSORT_GPU_MIN_N",
    "mlframe.training.core._phase_helpers": "MLFRAME_FORCED_GC_MIN_DF_MB",
    "mlframe.training.neural._recurrent_wrapper_base": "MLFRAME_RECURRENT_PREDICTION_CACHE_MAX",
}


def test_no_module_fails_to_import_on_a_bad_numeric_knob():
    """Every module with a numeric knob still imports when all of them hold junk."""
    import os

    import mlframe

    src = os.path.dirname(os.path.dirname(mlframe.__file__))
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.update({name: "not-a-number" for name in _NUMERIC_KNOB_MODULES.values()})
    code = "import importlib\nfor m in %r:\n    importlib.import_module(m)\nprint('ok')" % sorted(_NUMERIC_KNOB_MODULES)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-2000:]


def test_boruta_auto_thresholds_are_read_per_call(monkeypatch):
    """BorutaShap's auto-dispatch thresholds follow the environment at call time."""
    import numpy as np

    from mlframe.feature_selection.boruta_shap._auto_dispatch import resolve_auto_importance_measure

    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(400, 4)), rng.integers(0, 2, 400)
    monkeypatch.setenv("MLFRAME_BORUTA_AUTO_NP_RATIO", "1e6")
    _, diag = resolve_auto_importance_measure(X, y, classification=True, random_state=0)
    assert diag["thresholds"]["np_ratio"] == 1e6 and diag["resolved_measure"] == "permutation"
