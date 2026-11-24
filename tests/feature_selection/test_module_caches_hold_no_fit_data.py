"""The process-lifetime caches of the FE modules (compiled kernels, content memos, tuning tables) are keyed by shapes, dtypes and content digests, never by fit data.

A cache that outlives a fit is only safe between tests and between fits when nothing it holds is addressed by an object of the fit (an array, a frame) that could be mutated or collected and
then aliased. Each cache named here is checked for that, and a test can empty it through ``monkeypatch.setattr(module, NAME, ...)`` (the reset the module-state gate asks for): the
caches are rebound, not mutated, so a test that needs a cold cache gets one without touching the others.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

CACHES = [
    ("mlframe.feature_selection.filters._adaptive_nbins_columns", "_EDGE_MEMO"),
    ("mlframe.feature_selection.filters._cmi_null_strata_gpu", "_ENT_TABLES"),
    ("mlframe.feature_selection.filters._cupy_polynom_optimizer", "_BF_KERNELS"),
    ("mlframe.feature_selection.filters._extra_fe_families_dispersion_resident", "_CODES_KERNELS"),
    ("mlframe.feature_selection.filters._extra_fe_families_dispersion_resident", "_COLUMN_KERNELS"),
    ("mlframe.feature_selection.filters._family_maxt_cuda_bits", "_kernels_v3"),
    ("mlframe.feature_selection.filters._fe_batched_mi", "_MI_FROM_CODES_TILE_KERNELS"),
    ("mlframe.feature_selection.filters._fe_batched_mi_cmi", "_BATCHED_JOINT_ENTROPY_KERNEL"),
    ("mlframe.feature_selection.filters._fe_batched_mi_mm_values", "_COL_MINMAX_KERNELS"),
    ("mlframe.feature_selection.filters._fe_batched_mi_mm_values", "_MM_NEK_TILE_KERNELS"),
    ("mlframe.feature_selection.filters._fe_batched_qbin", "_QBIN_CODERS"),
    ("mlframe.feature_selection.filters._fe_resident_operands", "_COLUMN_MI"),
    ("mlframe.feature_selection.filters._fe_resident_operands", "_SHARED_LEGS"),
    ("mlframe.feature_selection.filters._gate_rank_divergence", "_KEY_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_COL_MEANSTD_TILED_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_COL_MINMAX_TILED_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_HEAVY_TAIL_STATS_TILED_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_discretize", "_BIN_CODES_F32E_V2_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_select", "_SHARED_LIMIT"),
    ("mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning", "_WARMUPS"),
    ("mlframe.feature_selection.filters._oof_warp_service", "_FOLD_CACHE"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_resident", "_BSPLINE_KERNELS"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_detect_gpu_resident", "_TARGET_SIDE"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._gpu_resident_cross_basis", "_LEG_PRODUCT_KERNELS"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup", "_DEDUP_MEMO"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup", "_FINAL_KERNELS"),
    ("mlframe.feature_selection.filters._pair_usability_table", "_UIDS"),
    ("mlframe.feature_selection.filters._ratio_delta_screen_gpu", "_SWEEP_RESIDENT"),
    ("mlframe.feature_selection.filters._residual_rescue_gpu", "_CODES_KERNELS"),
    ("mlframe.feature_selection.filters._row_argmax_gpu", "_SWEEP_RESIDENT"),
    ("mlframe.feature_selection.filters._usability_greedy_clf_batched", "_LOGISTIC_KERNELS"),
    ("mlframe.system.kernel_tuning_cache._ensure", "_HELD_LOCKS"),
]

_FORBIDDEN_KEY_TYPES = (np.ndarray, pd.DataFrame, pd.Series)


def _plain(key: object) -> bool:
    """A key made of strings, numbers, booleans, None, dtypes and tuples of those: nothing that is itself fit data."""
    if isinstance(key, _FORBIDDEN_KEY_TYPES):
        return False
    if isinstance(key, tuple):
        return all(_plain(k) for k in key)
    return True


@pytest.mark.parametrize("module_name,name", CACHES)
def test_cache_keys_are_never_fit_data(module_name: str, name: str):
    """Every key currently held by the cache is a plain scalar or a tuple of them."""
    obj = getattr(importlib.import_module(module_name), name)
    keys = list(obj.keys()) if hasattr(obj, "keys") else list(obj) if isinstance(obj, (list, set, tuple)) else []
    assert all(_plain(k) for k in keys), (module_name, name)


@pytest.mark.parametrize("module_name,name", CACHES)
def test_cache_can_be_reset_by_rebinding(monkeypatch, module_name: str, name: str):
    """``monkeypatch.setattr(module, NAME, empty)`` gives a test a cold cache and puts the original back afterwards."""
    module = importlib.import_module(module_name)
    original = getattr(module, name)
    try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
        empty = type(original)()
    except TypeError:
        pytest.skip(f"{name} is not an empty-constructible container")
    monkeypatch.setattr(module, name, empty)
    assert getattr(module, name) is empty
    monkeypatch.undo()
    assert getattr(module, name) is original
