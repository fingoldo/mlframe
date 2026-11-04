# Fleuret confidence kernel: in-fit numba compiles

Warm 100k x 100 every-family mdlp fit (stage_sample3b, warm-up fit first, GTX 1050 Ti): 93.0 s, of which 2.4-2.8 s were llvmlite
(`ffi.__call__` under `fleuret_conditional_recheck`): `get_fleuret_criteria_confidence` (njit, cache=True) compiled new signatures inside
the measured fit. Its compacted inputs had three: a tuple `x` (typed `UniTuple(int64, k)`, a new type per arity), int8 vs int32 codes,
float32 vs float64 gain. Normalised in `_compact_confidence_inputs` / its caller (int64 array `x`, int32 codes, float gain): one signature
over two fits, same verdicts (`test_fleuret_confidence_compact.py`). Wall 93.0 s -> 90.3 s (tuple only) -> 85.4 s (all three); llvmlite
gone from the profile.

Inventory, numba dispatchers of mlframe with more than one signature after two fits (20k x 30, 30k x 30), for later if one shows up
in a profile (each extra signature is a disk-cache load per process, a compile only when it first appears):
merge_vars 8; feature_engineering lambdas 6 and 5; _abs_pearson_njit 3; _bin_sum_count_masked_njit 3; 2 each: _minmax_neg1_1_njit,
_apply_minmax_njit, _hermeval_njit, _chebval_njit, _is_int_as_cat_njit, _row_center_kernel, _digit_extract_njit, group_mean_std_count,
quantile_bin_block, _relevance_mi_1var_fused, evaluate_gain, arr2str, masked_abs_spearman_presorted, _abs_corr_finite_njit,
label_bitsets_from_index, column_marginals, column_bin_bitsets, _zscore_from_bins_njit, _batch_joint_entropy_pairs,
_composite_residuals_kernel.
