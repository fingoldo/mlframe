"""Per-test snapshot and restore of module-level mutable state that production code fills from functions.

The autouse fixture ``_reset_module_state`` (imported into ``tests/conftest.py``) records the contents of every
name in ``_MODULE_STATE`` before a test and restores it afterwards: containers are restored in place (same object,
same contents), other values are rebound. Names are listed here so a process-lifetime cache cannot leak one test's
entries into the next. Benchmark scripts are only read when already imported, so importing them never runs their
bodies.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator, MutableMapping, MutableSequence, MutableSet
from typing import Any

import pytest

# ``_fe_device_scheduler._SLOT_POOLS`` is deliberately absent: the stage slots' memory pools live for the whole process because device arrays allocated in them are held by
# process-global caches (radix interpolation entries, operand tables); restoring the dict dropped the pools a test created, cupy freed their in-use chunks, and the next test's
# cached arrays dangled (cudaErrorIllegalAddress in the next kernel that read one).
_MODULE_STATE = (
    ("mlframe", "_lazy_toplevel_cache"),
    ("mlframe._numba_cache_deps", "_DEPENDENCY_FILES"),
    ("mlframe._numba_cache_deps", "_FACTS"),
    ("mlframe.feature_selection._benchmarks.kernel_arg_dtype_audit", "_FOUND"),
    ("mlframe.feature_selection._benchmarks.target_lens.exp_target_mix", "ARMS_ON"),
    ("mlframe.feature_selection.filters._adaptive_nbins", "_RETIRED_EDGE_POOLS"),
    ("mlframe.feature_selection.filters._analytic_mi_null", "_min_n_overrides"),
    ("mlframe.feature_selection.filters._dcor_gpu", "_KERNEL"),
    ("mlframe.feature_selection.filters._family_maxt_cuda_bits", "_kernels_v2"),
    ("mlframe.feature_selection.filters._batch_mi_noise_gate_bits", "_OBSERVED"),
    ("mlframe.feature_selection.filters._binned_numeric_agg_resident", "_OOF_GATHER_INTO_KERNELS"),
    ("mlframe.feature_selection.filters._device_sort_safe", "_POOL_LOCKS"),
    ("mlframe.feature_selection.filters._edge_pool", "_ACTIVE_STAGES"),
    ("mlframe.feature_selection.filters._fe_resident_operands", "_QBIN_HOST_CODES"),
    ("mlframe.feature_selection.filters._gpu_resident_operand_build", "_BATCH_UNARY_KERNELS"),
    ("mlframe.feature_selection.filters._optimal_joint_edges", "_FOLD_MASKS"),
    ("mlframe.feature_selection.filters._family_maxt_gate", "_label_bits_cache"),
    ("mlframe.feature_selection.filters._fe_batched_mi", "_DEDUP_EDGES_KERNELS"),
    ("mlframe.feature_selection.filters._fe_batched_mi_cmi", "_BATCHED_JOINT_ENTROPY_G_KERNEL"),
    ("mlframe.feature_selection.filters._fe_batched_mi_cmi_lut", "_XLOGX_TABLES"),
    ("mlframe.feature_selection.filters._fe_cmi_redundancy_gate", "_Y_DENSE_MEMO"),
    ("mlframe.feature_selection.filters._fe_content_memo", "_STORES"),
    ("mlframe.feature_selection.filters._fe_family_timing", "_FE_FAMILY_WALL"),
    ("mlframe.feature_selection.filters._fe_resident_operands", "_FE_RESIDENT_QBIN_CODES"),
    ("mlframe.feature_selection.filters._fe_sufficient_summary_pairs", "_FUSED_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_ABS_CORR_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_BASIS_PREPROCESS_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_BASIS_PREPROCESS_NR_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_CLENSHAW_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_COL_MEANSTD_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_COL_MINMAX_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_HEAVY_TAIL_STATS_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_basis", "_ROBUST_SCALE_COMBINE_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_discretize", "_BIN_CODES_F32E_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_discretize", "_BIN_CODES_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_fe", "_PREWARP_TRANSFORM_KERNELS"),
    ("mlframe.feature_selection.filters._gpu_resident_shared", "_COMBO_IDX_CACHE"),
    ("mlframe.feature_selection.filters._gpu_resident_shared", "_QLEVELS_CACHE"),
    ("mlframe.feature_selection.filters._hermite_fe_mi", "_CUDA_KERNELS_BY_REAL"),
    ("mlframe.feature_selection.filters._internals", "_NJIT_DISPATCHER_CACHE"),
    ("mlframe.feature_selection.filters._mi_greedy_cmi_fe_gpu", "_YZ_CARD_CACHE"),
    ("mlframe.feature_selection.filters._neural_mi", "_INFONET_MODEL_CACHE"),
    ("mlframe.feature_selection.filters._neural_mi", "_INFONET_Y_PREP_CACHE"),
    ("mlframe.feature_selection.filters._neural_mi", "_MIST_CALIBRATION_CACHE"),
    ("mlframe.feature_selection.filters._neural_mi", "_Y_KIND_CACHE"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_detect_gpu_resident", "_SPLIT_MASK_CACHE"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._fourier_detect_gpu_resident", "_SUBSAMPLE_IDX_CACHE"),
    ("mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_helpers", "_ORDER_CACHE_BYTES"),
    ("mlframe.feature_selection.filters._qbin_block_gpu", "_COUNT_KERNELS"),
    ("mlframe.feature_selection.filters._resident_candidate_mi", "_BUILD_OPCANDS_CM_KERNELS"),
    ("mlframe.feature_selection.filters._resident_candidate_mi", "_BUILD_OPCANDS_KERNELS"),
    ("mlframe.feature_selection.filters._resident_candidate_mi", "_TAU_KERNELS"),
    ("mlframe.feature_selection.filters._row_argmax_gpu", "_FINITE"),
    ("mlframe.feature_selection.filters._unified_fe_gate", "_COERCE_Y_MEMO"),
    ("mlframe.feature_selection.filters._wavelet_basis_fe", "_Y_CODES_MEMO"),
    ("mlframe.feature_selection.filters.batch_pair_usability_corr_gpu", "_CUDA_KERNELS_WARP"),
    ("mlframe.feature_selection.filters.feature_engineering", "_REGISTRY_CACHE"),
    ("mlframe.feature_selection.filters.hermite_fe._als_kernels_gpu", "_UNPACK"),
    ("mlframe.feature_selection.filters.info_theory._cmi_cuda", "_cmi_from_joint_kernels"),
    ("mlframe.feature_selection.filters.info_theory._cmi_cuda", "_xlogx_kernels"),
    ("mlframe.feature_selection.filters.mrmr._mrmr_class_config", "_CTOR_DEFAULTS_CACHE"),
    ("mlframe.feature_selection.filters.mrmr._mrmr_class_config", "_DEFAULT_SCREEN_SUBSAMPLE_N_CACHE"),
    ("mlframe.feature_selection.filters.mrmr._mrmr_class_config", "_FAST_SEARCH_SUBSAMPLE_N_CACHE"),
    ("mlframe.feature_selection.filters.mrmr._mrmr_class_config", "_FE_ENABLE_ATTR_NAMES_CACHE"),
    ("mlframe.feature_selection.filters.supervised_binning", "_ENCODE_MEMO"),
    ("mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu", "_SUBSET_LOSS_KERNELS"),
    ("mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu", "_TREESHAP_KERNELS"),
    ("mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu", "_INTERACTION_KERNELS"),
    ("mlframe.metrics.regression._regression_extras", "_MAPE_MEAN_ZERO_WARN_SEEN"),
    ("mlframe.metrics.regression._regression_extras", "_RMSLE_NEG_WARN_SEEN"),
    ("mlframe.models.ensembling._benchmarks.bench_caruana_vs_nnls", "y_test_cache"),
    ("mlframe.models.ensembling.predict", "_gate_cache_order"),
    ("mlframe.reporting._shared_arrays", "_ATTACHED"),
    ("mlframe.reporting.diagnostics_dispatch", "_PSI_CACHE"),
    ("mlframe.reporting.renderers._render_timings", "_TIMINGS"),
    ("mlframe.signal.dtw", "_CUPY_BANDED_KERNEL_CACHE"),
    ("mlframe.signal.dtw", "_CUPY_KERNEL_CACHE"),
    ("mlframe.training._dataset_build_stats", "_BUILDS"),
    ("mlframe.training.cb._cb_pool_budget", "_CACHE_SIZES"),
    ("mlframe.training.cb._cb_pool_budget", "_UNMEASURABLE_TYPES_SEEN"),
    ("mlframe.training.core._phase_train_one_target", "_DTYPE_PAIRS_MEMO_WEAKREFS"),
    ("mlframe.training.core._phase_train_one_target_mlp_helpers", "_PER_GROUP_PATTERN_CACHE"),
    ("mlframe.training.core._predict_pre_pipeline", "EXTENSIONS_SOFT_FAIL_TAINT"),
    ("mlframe.training.feature_handling.compat", "_MODEL_AXIS_SUPPORT"),
    ("mlframe.training.io", "_LIB_VERSIONS_CACHE"),
    ("mlframe.training.io", "_LOAD_MODEL_CACHE"),
    ("mlframe.training.mlp_runtime_defaults", "_PROBE_PRECISION_CACHE"),
    ("mlframe.training.neural._base_tensor_helpers", "_CUDA_PROBE_CACHE"),
    ("mlframe.training.phases", "_ALL_PHASE_STACKS"),
    ("mlframe.training.reporting._reporting_regression._sensor_ledger", "_TRIPS"),
    ("mlframe.utils.warning_filters", "_INSTALLED"),
)


def _load(dotted: str) -> Any:
    """The imported module, or None when it cannot be imported (an optional GPU or extra dependency)."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    if "._benchmarks." in dotted:
        return None
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None


def _snapshot() -> list[tuple[Any, str, Any, Any]]:
    """Record the current contents of every listed name that resolves to a module attribute."""
    out: list[tuple[Any, str, Any, Any]] = []
    for dotted, name in _MODULE_STATE:
        mod = _load(dotted)
        if mod is None or not hasattr(mod, name):
            continue
        value = getattr(mod, name)
        if isinstance(value, (MutableMapping, MutableSet)):
            out.append((mod, name, value, dict(value) if isinstance(value, MutableMapping) else set(value)))
        elif isinstance(value, MutableSequence):
            out.append((mod, name, value, list(value)))
        else:
            out.append((mod, name, None, value))
    return out


def _restore(snapshot: list[tuple[Any, str, Any, Any]]) -> None:
    """Put back the contents recorded by ``_snapshot``, in place for containers and by rebinding for scalars."""
    for mod, name, container, saved in snapshot:
        if container is None:
            setattr(mod, name, saved)
        elif isinstance(container, MutableMapping):
            container.clear()
            container.update(saved)
        elif isinstance(container, MutableSet):
            container.clear()
            container.update(saved)
        else:
            container.clear()
            container.extend(saved)


@pytest.fixture(autouse=True)
def _reset_module_state() -> Iterator[None]:
    """Restore the listed module-level state after each test, even when the test fails."""
    snapshot = _snapshot()
    try:
        yield
    finally:
        _restore(snapshot)
