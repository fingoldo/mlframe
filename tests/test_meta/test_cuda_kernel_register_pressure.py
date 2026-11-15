"""Register-pressure ratchet for fused RawKernels that expose a ``kernel(cp, real) -> CompiledFunction`` factory.

CUDA.md (section 3, `choose-launch-configuration`) says to take the register count from ``ptxas -v`` or the profiler, never
from a hand estimate, and that a local load or store (a spill) costs 30 to 800 cycles depending on which cache level
answers it. cupy's compiled ``RawKernel``/``RawModule`` functions report this directly: ``num_regs`` and
``local_size_bytes`` are read off the cubin, which is the ``ptxas -v`` number without needing the compiler's text
output. This test compiles every kernel in the registry below for every dtype it supports and asserts there is no
spill (``local_size_bytes == 0``), skipping the whole module honestly when no CUDA device is present rather than
asserting on host-only numbers.

The registry starts with the kernels this session wrote or touched; CUDA.md kernels are added here as they are
written or revisited, the same incremental-ratchet shape as ``_fe_gpu_contract.json``. A kernel enters
``_SPILL_EXEMPT`` only with the reason a spill is accepted for it (e.g. a kernel with too many live values to avoid
one even after trying), never silently.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

# name -> (module path, factory attribute name, dtypes to compile)
_REGISTRY: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "fe_accuracy_gate_kernel.gate_logistic_auc": (
        "mlframe.feature_selection.filters._fe_accuracy_gate_kernel",
        "kernel",
        ("float32", "float64"),
    ),
    "lof_gpu.lof_scores": (
        "mlframe.feature_selection.filters._lof_gpu",
        "kernel",
        ("float32", "float64"),
    ),
    "nonpoly_cells_gpu.nonpoly_cells": (
        "mlframe.feature_selection.filters.hermite_fe._nonpoly_cells_gpu",
        "kernel",
        ("float32", "float64"),
    ),
    "family_maxt_cuda_bits.family_null_bits_v3": (
        "mlframe.feature_selection.filters._family_maxt_cuda_bits",
        "kernel_v3",
        ("float32", "float64"),
    ),
    "cmi_null_strata_gpu.strata_xy_entropy": (
        "mlframe.feature_selection.filters._cmi_null_strata_gpu",
        "kernel",
        ("float32", "float64"),
    ),
    "strata_shuffle_gpu.within_stratum_shuffle": (
        "mlframe.feature_selection.filters._strata_shuffle_gpu",
        "kernel",
        ("float32", "float64"),
    ),
    "cmi_null_strata_gpu.strata_xy_entropy_rows": (
        "mlframe.feature_selection.filters._cmi_null_strata_gpu",
        "kernel_rows",
        ("float32", "float64"),
    ),
    "extra_fe_families_dispersion_resident.disp_columns": (
        "mlframe.feature_selection.filters._extra_fe_families_dispersion_resident",
        "kernel",
        ("float32", "float64"),
    ),
    "fe_pure_form_retention_gpu_resident.pair_gram_blocks": (
        "mlframe.feature_selection.filters._fe_pure_form_retention_gpu_resident",
        "kernel",
        ("float64",),
    ),
    "extra_fe_families_dispersion_resident.disp_bin_codes": (
        "mlframe.feature_selection.filters._extra_fe_families_dispersion_resident",
        "kernel_codes",
        ("float32", "float64"),
    ),
    "extra_basis_resident.bspline_block": (
        "mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_resident",
        "kernel_spline",
        ("float32", "float64"),
    ),
    "gpu_pairs.pair_joint_mi": (
        "mlframe.feature_selection.filters._gpu_pairs",
        "kernel",
        ("float64",),
    ),
    "rolling_window_gpu.rolling_stat_past_only": (
        "mlframe.feature_selection.filters._rolling_window_gpu",
        "kernel",
        ("int64", "float64"),
    ),
    "order_stats_gpu.segment_order_stats": (
        "mlframe.feature_selection.filters._order_stats_gpu",
        "kernel",
        ("float64",),
    ),
    "cupy_polynom_optimizer.bf_apply": (
        "mlframe.feature_selection.filters._cupy_polynom_optimizer",
        "kernel",
        ("float32", "float64"),
    ),
    "fe_batched_qbin.qbin_code": (
        "mlframe.feature_selection.filters._fe_batched_qbin",
        "kernel",
        ("float32", "float64"),
    ),
    "fe_batched_mi_cmi_hist.rows_ent_nnz": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel",
        ("float64",),
    ),
    "fe_batched_mi_cmi_hist.rows_ent_nnz_i32": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_i32",
        ("float64",),
    ),
    "fe_batched_mi_cmi_hist.rows_ent_nnz_i32_split": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_split",
        ("float64",),
    ),
    "fe_batched_mi_cmi_hist.rows_ent_nnz_finish": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_finish",
        ("float64",),
    ),
    "fe_batched_mi_cmi_hist.joint_hist1": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_joint_hist1",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.joint_hist2": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_joint_hist2",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.joint_hist3": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_joint_hist3",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist2": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist2",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist1": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist1",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist2_i32": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist2_i32",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist2_i32_smem": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist2_i32_smem",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist1_i32": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist1_i32",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.joint_hist_nnz1": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_joint_hist_nnz1",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.joint_hist_nnz2": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_joint_hist_nnz2",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.joint_hist_nnz3": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_joint_hist_nnz3",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist2_i32_u8": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist2_i32_u8",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist2_i32_smem_u8": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist2_i32_smem_u8",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist2_i32_smem_u8_v2": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist2_i32_smem_u8_v2",
        ("int64",),
    ),
    "fe_batched_mi_cmi_hist.batched_joint_hist1_i32_u8": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi_hist",
        "kernel_batched_joint_hist1_i32_u8",
        ("int64",),
    ),
    "fe_batched_mi_mm_values.mi_mm_from_values": (
        "mlframe.feature_selection.filters._fe_batched_mi_mm_values",
        "kernel",
        ("float64",),
    ),
    "fe_batched_mi_mm_values.mi_mm_from_values_nek": (
        "mlframe.feature_selection.filters._fe_batched_mi_mm_values",
        "kernel_nek",
        ("float32", "float64"),
    ),
    "fe_batched_mi_mm_values.mm_nek_tile": (
        "mlframe.feature_selection.filters._fe_batched_mi_mm_values",
        "kernel_tile",
        ("float32", "float64"),
    ),
    "fe_batched_mi_mm_values.col_minmax": (
        "mlframe.feature_selection.filters._fe_batched_mi_mm_values",
        "kernel_minmax",
        ("float32", "float64"),
    ),
    "fe_batched_mi_mm_values.dedup_njit_edges": (
        "mlframe.feature_selection.filters._fe_batched_mi_mm_values",
        "kernel_dedup",
        ("float32", "float64"),
    ),
    "fe_batched_mi_cmi.batched_joint_entropy2": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_entropy",
        ("uint8", "int64"),
    ),
    "fe_batched_mi_cmi.batched_joint_entropy2_g": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_entropy_g",
        ("uint8", "int64"),
    ),
    "fe_batched_mi_cmi.batched_joint_entropy2_g_wide": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_entropy_g_wide",
        ("uint8", "int64"),
    ),
    "fe_batched_mi_cmi.ent_nnz_1d": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_ent_nnz",
        ("int64",),
    ),
    "fe_batched_mi_cmi.joint_entropy1": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_joint1",
        ("int64",),
    ),
    "fe_batched_mi_cmi.joint_entropy2": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_joint2",
        ("int64",),
    ),
    "fe_batched_mi_cmi.joint_entropy3": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_joint3",
        ("int64",),
    ),
    "fe_batched_mi_cmi.cmi_joint_entropies": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_cmi_joint",
        ("int64",),
    ),
    "fe_batched_mi_cmi.marginal_mi_entropies": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_marginal",
        ("int64",),
    ),
    "fe_batched_mi_cmi.cmi_assemble": (
        "mlframe.feature_selection.filters._fe_batched_mi_cmi",
        "kernel_assemble",
        ("int64",),
    ),
    "gpu_resident_select_kernels.radix_select_f64": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_radix_f64",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_select_f32": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_radix_f32",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_select_f32_bsearch": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_radix_f32_bsearch",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_select_f32_v3": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_radix_f32_v3",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_select_interp_f64": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_interp_f64",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_select_interp_f64_v2": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_interp_f64_v2",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_select_interp_f64_v3": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_interp_f64_v3",
        ("float64",),
    ),
    "gpu_resident_select_kernels.transpose_f32": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_transpose_f32",
        ("float64",),
    ),
    "gpu_resident_select_kernels.transpose_f64": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_transpose_f64",
        ("float64",),
    ),
    "gpu_resident_select_kernels.transpose_i8": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_transpose_i8",
        ("float64",),
    ),
    "gpu_resident_select_kernels.transpose_i16": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_transpose_i16",
        ("float64",),
    ),
    "gpu_resident_select_kernels.radix_interp": (
        "mlframe.feature_selection.filters._gpu_resident_select_factories",
        "kernel_interp",
        ("float64",),
    ),
    "gpu_resident_shared.fused_gen": (
        "mlframe.feature_selection.filters._gpu_resident_shared",
        "kernel",
        ("float64",),
    ),
    "gpu_resident_shared.fused_gen_cm": (
        "mlframe.feature_selection.filters._gpu_resident_shared",
        "kernel_cm",
        ("float32", "float64"),
    ),
    "gpu_resident_shared.fused_gen_bin_hist": (
        "mlframe.feature_selection.filters._gpu_resident_shared",
        "kernel_bin_hist",
        ("float64",),
    ),
    "gpu_resident_basis.clenshaw_basis": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_clenshaw",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.robust_scale_combine": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_robust_scale",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.heavy_tail_stats_tiled": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_heavy_tail",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.heavy_tail_finish": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_heavy_tail_finish",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.basis_preprocess": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_preprocess",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.basis_preprocess_nr": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_preprocess_nr",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.col_meanstd_tiled": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_meanstd",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.col_minmax_tiled": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_minmax",
        ("float32", "float64"),
    ),
    "gpu_resident_basis.abs_corr": (
        "mlframe.feature_selection.filters._gpu_resident_basis_factories",
        "kernel_abs_corr",
        ("float32", "float64"),
    ),
    "gpu.compute_joint_hist_cuda": (
        "mlframe.feature_selection.filters.gpu",
        "kernel_joint_hist",
        ("int64",),
    ),
    "gpu.compute_joint_hist_batched_cuda": (
        "mlframe.feature_selection.filters.gpu",
        "kernel_joint_hist_batched",
        ("int64",),
    ),
    "gpu.compute_joint_hist_batched_shared_cuda": (
        "mlframe.feature_selection.filters.gpu",
        "kernel_joint_hist_batched_shared",
        ("int64",),
    ),
    "gpu.compute_joint_hist_multi_pair_cuda": (
        "mlframe.feature_selection.filters.gpu",
        "kernel_multi_pair",
        ("int64",),
    ),
    "gpu.compute_joint_hist_multi_pair_shared_cuda": (
        "mlframe.feature_selection.filters.gpu",
        "kernel_multi_pair_shared",
        ("int64",),
    ),
    "gpu.compute_mi_from_classes_cuda": (
        "mlframe.feature_selection.filters.gpu",
        "kernel_mi_from_classes",
        ("int64",),
    ),
    "pairs_analytic_gpu.minmax_i8": (
        "mlframe.feature_selection.filters._feature_engineering_pairs._pairs_analytic_gpu",
        "kernel_minmax",
        ("float64",),
    ),
    "pairs_analytic_gpu.analytic_obs_mi_bins_i8": (
        "mlframe.feature_selection.filters._feature_engineering_pairs._pairs_analytic_gpu",
        "kernel",
        ("float64",),
    ),
    "pairs_analytic_gpu.analytic_mi_from_hist": (
        "mlframe.feature_selection.filters._feature_engineering_pairs._pairs_analytic_gpu",
        "kernel_finish",
        ("float64",),
    ),
    "honest_null_k4.root_split_stats_k4f": (
        "mlframe.feature_selection.filters._honest_null_k4",
        "kernel_k4f",
        ("float64",),
    ),
    "honest_null_k4.root_split_stats_k5": (
        "mlframe.feature_selection.filters._honest_null_k4",
        "kernel_k5",
        ("float64",),
    ),
    "honest_null_k4.root_split_stats_k4": (
        "mlframe.feature_selection.filters._honest_null_k4",
        "kernel_k4",
        ("float64",),
    ),
    "honest_null_device.root_split_stats": (
        "mlframe.feature_selection.filters._honest_null_device",
        "kernel_v1",
        ("float64",),
    ),
    "honest_null_device.root_split_stats_v2": (
        "mlframe.feature_selection.filters._honest_null_device",
        "kernel_v2",
        ("float64",),
    ),
    "honest_null_device.root_split_stats_k2": (
        "mlframe.feature_selection.filters._honest_null_device",
        "kernel_k2",
        ("float64",),
    ),
    "gpu_resident_select_f32_compact.radix_select_f32_v4": (
        "mlframe.feature_selection.filters._gpu_resident_select_f32_compact",
        "kernel_v4",
        ("float64",),
    ),
    "gpu_resident_select_f32_compact.radix_select_f32_v5": (
        "mlframe.feature_selection.filters._gpu_resident_select_f32_compact",
        "kernel_v5",
        ("float64",),
    ),
    "gpu_resident_select_f32_compact.radix_select_f32_v5_winmap": (
        "mlframe.feature_selection.filters._gpu_resident_select_f32_compact",
        "kernel_v5_winmap",
        ("float64",),
    ),
    "gpu_resident_materialise.fe_materialise": (
        "mlframe.feature_selection.filters._gpu_resident_materialise",
        "kernel",
        ("float64",),
    ),
    "gpu_resident_materialise.fe_materialise_cm": (
        "mlframe.feature_selection.filters._gpu_resident_materialise",
        "kernel_cm",
        ("float64",),
    ),
    "gpu_resident_materialise.fe_materialise_cm_tiled": (
        "mlframe.feature_selection.filters._gpu_resident_materialise",
        "kernel_cm_tiled",
        ("float64",),
    ),
    "gpu_resident_discretize.bin_codes": (
        "mlframe.feature_selection.filters._gpu_resident_discretize",
        "kernel_bin",
        ("float32", "float64"),
    ),
    "gpu_resident_discretize.bin_codes_f32e": (
        "mlframe.feature_selection.filters._gpu_resident_discretize",
        "kernel_f32e",
        ("int8", "int16", "int32"),
    ),
    "gpu_resident_discretize.bin_codes_f32e_v2": (
        "mlframe.feature_selection.filters._gpu_resident_discretize",
        "kernel_f32e_v2",
        ("int8", "int16", "int32"),
    ),
    "shap_proxy_treeshap_gpu.treeshap": (
        "mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu",
        "kernel",
        ("float64",),
    ),
    "shap_proxy_treeshap_interactions_gpu.treeshap_interactions": (
        "mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu",
        "kernel",
        ("float64",),
    ),
    "null_max_perm_gpu.null_mi_perms": (
        "mlframe.feature_selection.filters._null_max_perm_gpu",
        "kernel_counts",
        ("float32", "float64"),
    ),
    "null_max_perm_gpu.null_mi_perms_bits": (
        "mlframe.feature_selection.filters._null_max_perm_gpu",
        "kernel_bits",
        ("float32", "float64"),
    ),
    "offset_product_gpu.scan_offset_products": (
        "mlframe.feature_selection.filters._offset_product_gpu",
        "kernel",
        ("float64",),
    ),
    "perm_labels_device.fy_permutations": (
        "mlframe.feature_selection.filters._perm_labels_device",
        "kernel_fy",
        ("int64",),
    ),
    "perm_labels_device.pack_class_bits": (
        "mlframe.feature_selection.filters._perm_labels_device",
        "kernel_pack",
        ("int64",),
    ),
    "extra_basis_fourier_cuda.fourier_block": (
        "mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_fourier_cuda",
        "kernel",
        ("float32", "float64"),
    ),
    "relaxmrmr_pair_gpu.relaxmrmr_pair": (
        "mlframe.feature_selection.filters._relaxmrmr_pair_gpu",
        "kernel",
        ("uint8", "int32"),
    ),
    "synergy_combo_gpu.synergy_combo": (
        "mlframe.feature_selection.filters._synergy_combo_gpu",
        "kernel",
        ("uint8", "int32"),
    ),
    "fe_batched_mi.mi_from_codes_tile": (
        "mlframe.feature_selection.filters._fe_batched_mi",
        "kernel",
        ("uint8", "int64"),
    ),
}

# kernel name -> reason a register spill is accepted for it. Empty is the default: fix the kernel instead of exempting
# it. Pack any thread-private triangular-matrix array (Hessian, Cholesky factor) before reaching for this dict - that
# cut this kernel's spill from 1512 to 936 bytes/thread for free (see the reason string for what remains).
_SPILL_EXEMPT: dict[str, str] = {
    "offset_product_gpu.scan_offset_products": (
        "160 bytes/thread = `{ACC} M[4][5]` (20 doubles): the augmented matrix of the 4 x 4 normal equations the kernel solves per task by Gaussian elimination, indexed by the run-time pivot row, "
        "so it is a stack-frame array, not a spill (register spill is zero, `--fmad=false` build). The same system solved with the matrix in registers would need the elimination fully unrolled "
        "for every pivot order; the kernel is measured at 0.1 percent of a 100000-row fit (the tiny-cupy row for `scan_offset_products_device`), so the frame is not worth a rewrite."
    ),
    "perm_labels_device.pack_class_bits": (
        "128 bytes/thread = `unsigned long long acc[16]`, one 64-bit word of the class bit set per class (at most 16 classes), indexed by the run-time class label of the permuted row, so it "
        "lives in the stack frame; no register spill. The array is the output accumulator of the pack."
    ),
    "extra_basis_fourier_cuda.fourier_block": (
        "32 bytes/thread (float) and 40 (double) with no thread-private array in the source: the frame is the one libdevice reserves for the slow path of `sin` / `cos` (a large-argument "
        "reduction), the same effect as the exempted `nonpoly_cells` (40 bytes for its trig calls); no register spill, and the kernel is arithmetic-bound, not occupancy-bound."
    ),
    "null_max_perm_gpu.null_mi_perms_bits": (
        "512 bytes/thread = the three per-thread arrays `int cnt[PMAX_LANE][NBMAX]` (2 x 32 x 4 = 256), `int tl[NBMAX]` (128) and `unsigned mb[NBMAX]` (128), indexed by the "
        "run-time bin, so they sit in the stack frame (L1-resident local memory) next to 39 registers and no register spill. Holding the 256 lane counters in registers is not possible "
        "(the unrolled form would need over 200 registers and spill for real); the alternative that moves them to shared atomics is the layout the ballot / popcount design replaces (UC11-05 "
        "measured the coalescing rewrite of this kernel and rejected it). The comment of the kernel used to say the counts stay in registers; it now says where they are."
    ),
    "shap_proxy_treeshap_gpu.treeshap": (
        "2688 bytes/thread at the reference depth 6 (MAXW 8, MAXSP 28) = the per-thread TreeSHAP path scratch: the extend / unwind weight arrays indexed by the unique-path depth and "
        "the explicit DFS stack that replaces recursion, all dynamically indexed, so they live in the stack frame (54 registers, no register spill). The module header states the design; the "
        "frame grows with the square of the depth, which is why `_ensure_kernel` sizes it by the ensemble's real depth and not the cap (at the cap of 24 it reserved gigabytes of device "
        "local memory and starved later GPU paths). Removing the frame means a different algorithm (the shared-memory tile of the tree), not a register fix."
    ),
    "shap_proxy_treeshap_interactions_gpu.treeshap_interactions": (
        "3136 bytes/thread at depth 6 with 16 features (MAXW 8, MAXLVL 9, MAXP 16) = the same per-thread path scratch as treeshap plus the conditioned-pass accumulators (one level deeper, "
        "one slot per feature), dynamically indexed so they sit in the stack frame (64 registers, no register spill). Sized by the ensemble's real depth and feature count in "
        "`_ensure_kernel` (the caps would reserve about 2.6 GB of local memory on a 4 GB card); the frame is the algorithm's working set, not live-value pressure."
    ),
    "honest_null_device.root_split_stats": (
        "256 bytes/thread = the per-thread class-count arrays `long long cl[16], cr[16]` (2 x 16 x 8), indexed by the runtime label `yi`: a dynamically indexed array lives in the "
        "stack frame (local memory, L1-resident), it is not a spill of live values (48 registers, far under the cap). The alternative layout that keeps the counters elsewhere "
        "(version 3, shared/bit-packed) was measured 1.4 to 1.8 times SLOWER than this one (see the header of _honest_null_k3.py and FM11-02), so the frame is the faster design. "
        "The kernel is the retained v1 of the --fmad=false root-split family."
    ),
    "honest_null_device.root_split_stats_v2": (
        "256 bytes/thread = the same `long long cl[16], cr[16]` class-count arrays as root_split_stats (runtime-indexed by the label, so they sit in the stack frame, 48 registers, no "
        "register spill); v2 only reads the next row's label one step ahead. Version 3, which moves the counters out of local memory, measured 1.4 to 1.8 times slower (_honest_null_k3.py "
        "header, FM11-02). v2 is the default of the family."
    ),
    "honest_null_device.root_split_stats_k2": (
        "40 bytes/thread = `int ybuf[TILE + 1]` (9 ints rounded up), the tile of look-ahead labels, indexed by the loop variable `t` of the scoring loop whose trip count depends on the rows "
        "left (so it cannot be fully unrolled); 56 registers, no register spill. The tile is what makes the label loads independent of the entropy chain (the measured gain of k2 over v2 at the "
        "binary-target shapes); an unrolled copy would add a second code path for the tail."
    ),
    "nonpoly_cells_gpu.nonpoly_cells": (
        "the float32 build reserves a 40-byte stack frame (local_size_bytes 40; ptxas: 40 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads) for the "
        "libdevice trig slow path that its sin and cos calls use; cosf on a float argument reserves 32 bytes the same way, and the double build reserves none. "
        "The frame is not register pressure (28 registers, no spill), so occupancy is unchanged. Measured in the register-pressure registry test (2026-11-12) and "
        "audits/2026-10-22/profiles/hermite_fe.md; the cell kernel is FP64-bound regardless of this frame."
    ),
    "fe_accuracy_gate_kernel.gate_logistic_auc": (
        "936 bytes/thread remain (down from 1512 after packing the per-thread Hessian/Cholesky arrays into their "
        "lower triangle) in the per-row gradient accumulation loop's g[MAXP]/x[MAXP]/h[MAXTRI] work vectors. k is a "
        "runtime kernel argument (0..MAXK), so the compiler cannot unroll the loop that indexes them and keep them "
        "in registers. Removing the spill would need a second pass over the design columns (recompute z without "
        "storing x[]) at MAXK<=8, doubling the global reads this kernel does per Newton iteration, for a kernel "
        "that already had one silent-divergence incident from a tolerance mismatch (see "
        "audits/2026-10-22/profiles/_fe_accuracy_gate_kernel.md); not worth the risk for this occupancy-limited, "
        "already compute-bound kernel per CUDA.md section 3 (a spill is not worth a higher occupancy)."
    ),
}


def _cupy_or_skip() -> Any:
    """Import cupy or skip the test; skip again if the device present cannot actually compile/launch a kernel."""
    cp = pytest.importorskip("cupy")
    # Skip only when there is no device; a real failure of the device query must surface, not turn into a skip.
    if not cp.cuda.is_available():
        pytest.skip("no usable CUDA device")
    return cp


@pytest.mark.parametrize("name", sorted(_REGISTRY))
def test_registered_kernel_has_no_register_spill(name: str):
    """Every dtype a registered kernel compiles for reports zero local_size_bytes, unless listed in _SPILL_EXEMPT."""
    cp = _cupy_or_skip()
    module_path, factory_name, dtypes = _REGISTRY[name]
    module = importlib.import_module(module_path)
    factory = getattr(module, factory_name)
    for dtype in dtypes:
        real = getattr(cp, dtype)
        compiled = factory(cp, real)
        compiled.compile()
        local_bytes = int(compiled.local_size_bytes)
        if name in _SPILL_EXEMPT:
            continue
        assert local_bytes == 0, (
            f"{name} ({dtype}) spills {local_bytes} bytes/thread to local memory (num_regs={compiled.num_regs}); "
            f"either reduce live values per CUDA.md section 3 register-pressure-warp-occupancy, or add an entry to "
            f"_SPILL_EXEMPT with the reason."
        )


@pytest.mark.parametrize("name,reason", sorted(_SPILL_EXEMPT.items()) or [pytest.param(None, None, marks=pytest.mark.skip(reason="no spill exemptions registered"))])
def test_every_exempted_kernel_states_its_reason(name, reason):
    """Every entry in _SPILL_EXEMPT names a registered kernel and carries a non-empty reason (an empty registry is a skip, not a silent pass)."""
    assert name in _REGISTRY, f"{name} is exempted but not registered"
    assert reason.strip(), f"{name} has an empty exemption reason"


_OFFLINE_ARCHS = (52, 61, 70, 75, 86, 89, 90, 120)
_OFFLINE_LOF_SHAPES = ((20, 3), (32, 16), (33, 3), (64, 16))


def _ptxas_or_skip() -> str:
    """Path of the CUDA toolkit's ptxas, or skip: the offline check needs the assembler, not a device."""
    import os
    import shutil

    exe = "ptxas.exe" if os.name == "nt" else "ptxas"
    candidates = [shutil.which("ptxas")]
    if os.environ.get("CUDA_PATH"):
        candidates.append(os.path.join(os.environ["CUDA_PATH"], "bin", exe))
    for path in candidates:
        if path and os.path.exists(path):
            return path
    pytest.skip("no ptxas on PATH or under CUDA_PATH")


@pytest.mark.parametrize("arch", _OFFLINE_ARCHS)
@pytest.mark.parametrize("k, p", _OFFLINE_LOF_SHAPES)
@pytest.mark.parametrize("real", ("float", "double"))
def test_lof_scores_has_no_spill_on_every_arch(arch: int, k: int, p: int, real: str, tmp_path):
    """lof_scores assembles without a spill for every supported compute capability (offline: nvrtc PTX, then ptxas -v)."""
    import re
    import subprocess

    ptxas = _ptxas_or_skip()
    nvrtc = pytest.importorskip("cupy_backends.cuda.libs.nvrtc")
    from mlframe.feature_selection.filters import _lof_gpu

    src = _lof_gpu._SRC.replace("{REAL}", real).replace("{ACC}", "double").replace("{K}", str(k)).replace("{P}", str(p)).replace("{SINGLE_FILTER}", "1" if real == "float" else "0")
    prog = nvrtc.createProgram(src, "lof.cu", [], [])
    try:
        nvrtc.compileProgram(prog, ["-std=c++14", f"-arch=compute_{arch}"])
        ptx = nvrtc.getPTX(prog)
    finally:
        nvrtc.destroyProgram(prog)
    # the NVRTC shipped with cupy can emit a newer PTX ISA than the toolkit's ptxas accepts; the instructions used here are the same
    ptx = re.sub(rb"\.version \d+\.\d+", b".version 8.7", ptx)
    path = tmp_path / f"lof_{real}_k{k}_p{p}_sm{arch}.ptx"
    path.write_bytes(ptx)
    res = subprocess.run([ptxas, "-v", f"-arch=sm_{arch}", "-o", str(path) + ".cubin", str(path)], capture_output=True, text=True)
    report = res.stdout + res.stderr
    assert res.returncode == 0, report[-400:]
    spill = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", report)
    assert spill is not None, report[-400:]
    assert spill.group(2) == "0" and spill.group(3) == "0", f"sm_{arch} k={k} p={p} {real}: {spill.group(0)}"


def _registered() -> set:
    """``{(module, factory)}`` of the registry."""
    return {(module, factory) for module, factory, _dtypes in _REGISTRY.values()}


def _src_root():
    """The package source directory the factory scan reads."""
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "src" / "mlframe"


def test_every_kernel_factory_in_the_package_is_registered():
    """A ``kernel(cp, real)``-style factory that is not in ``_REGISTRY`` is a kernel whose spills nobody checks on a device: register it (or rename it if it is not a kernel factory). The scan is py-ci-shared's ``cuda_kernel_factories`` gate."""
    from py_ci_shared.cuda_kernel_factories import assert_cuda_kernel_factories

    assert_cuda_kernel_factories(_src_root(), registered=_registered(), min_files=500)


def test_the_factory_scan_sees_the_known_factories():
    """The scan finds at least the factories registered today (a changed convention would make the completeness test vacuous)."""
    from py_ci_shared.cuda_kernel_factories import find_cuda_kernel_factories

    root = _src_root()
    seen = set()
    for finding in find_cuda_kernel_factories(root, registered=(), min_files=500):
        module = ".".join(("mlframe", *finding.path.replace("\\", "/")[:-3].split("/")))
        name = finding.message.split("kernel factory ")[1].split("(")[0]
        seen.add((module, name))
    assert _registered() <= seen, sorted(_registered() - seen)
