"""``check_prospective_fe_pairs`` carved out of
``mlframe.feature_selection.filters.feature_engineering``.

Re-imported at the parent's module bottom so historical
``from mlframe.feature_selection.filters.feature_engineering import check_prospective_fe_pairs``
resolves transparently. The pair-search orchestrates MRMR's FE candidate
generation (unary+binary ops), batched MI + permutation noise-gate,
prewarp / median-gate pseudo-unaries, chunked materialise, and the per-host
kernel-tuning dispatch; the body + supporting kernels are split across the
submodules below and re-exported here so every historical import path
(public ``check_prospective_fe_pairs`` and the private helpers the tests +
sibling modules pull) resolves from the package surface.
"""
from __future__ import annotations

from ._pairs_common import _TIMES_SPENT_LOCK, _module_logger
from ._pairs_dispatch import (
    _BATCH_MI_NOISE_GATE_CODE_VERSION,
    _batch_mi_with_noise_gate_gpu,
    _dispatch_batch_mi_with_noise_gate,
)
from ._pairs_gates import (
    _FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_MIN_RATIO,
    _FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO,
    _FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT,
    _FE_REJECTION_RESULT_KEY,
    _GATE_MED_SPECS_RESULT_KEY,
    _GATE_MED_UNARY,
    _PREWARP_SPECS_RESULT_KEY,
    _PREWARP_UNARY,
    _gate_med_apply,
    _neg_name_key,
    _select_single_best,
)
from ._pairs_materialise import (
    _FE_PARALLELISM_SALT,
    _FE_PARALLELISM_SPEC,
    _FE_PARALLELISM_SWEEP_COLS,
    _NJIT_BINARY_OP_CODES,
    _fe_parallelism_fallback_choice,
    _fe_use_parallel_kernels,
    _make_fe_parallelism_inputs,
    _materialise_chunk_njit,
    _materialise_chunk_njit_parallel,
    _materialise_extval_njit,
    _narrow_code_dtype,
    _njit_binary_op_codes,
    _run_fe_parallelism_sweep,
)
from ._pairs_chunks import (
    _FE_BUFFER_RAM_BUDGET_RATIO,
    _FE_CHUNK_MAX_COLS_HARD_CAP,
    _compute_one_fe_chunk,
    _plan_fe_chunks,
)
from ._pairs_core import FE_DEFAULT_SUBSAMPLE_N, check_prospective_fe_pairs

__all__ = [
    "check_prospective_fe_pairs",
    "FE_DEFAULT_SUBSAMPLE_N",
    "_gate_med_apply",
    "_select_single_best",
    "_neg_name_key",
    "_dispatch_batch_mi_with_noise_gate",
    "_batch_mi_with_noise_gate_gpu",
    "_BATCH_MI_NOISE_GATE_CODE_VERSION",
    "_materialise_chunk_njit",
    "_materialise_chunk_njit_parallel",
    "_materialise_extval_njit",
    "_njit_binary_op_codes",
    "_narrow_code_dtype",
    "_fe_use_parallel_kernels",
    "_fe_parallelism_fallback_choice",
    "_make_fe_parallelism_inputs",
    "_run_fe_parallelism_sweep",
    "_FE_PARALLELISM_SPEC",
    "_FE_PARALLELISM_SWEEP_COLS",
    "_FE_PARALLELISM_SALT",
    "_NJIT_BINARY_OP_CODES",
    "_plan_fe_chunks",
    "_compute_one_fe_chunk",
    "_FE_BUFFER_RAM_BUDGET_RATIO",
    "_FE_CHUNK_MAX_COLS_HARD_CAP",
    "_PREWARP_UNARY",
    "_PREWARP_SPECS_RESULT_KEY",
    "_FE_REJECTION_RESULT_KEY",
    "_GATE_MED_UNARY",
    "_GATE_MED_SPECS_RESULT_KEY",
    "_FE_MARGINAL_UPLIFT_MIN_RATIO",
    "_FE_MARGINAL_UPLIFT_MIN_JOINT_RATIO",
    "_FE_MARGINAL_UPLIFT_STRICT_JOINT_RATIO",
    "_FE_MARGINAL_UPLIFT_SYNERGY_UPLIFT",
    "_TIMES_SPENT_LOCK",
    "_module_logger",
]
