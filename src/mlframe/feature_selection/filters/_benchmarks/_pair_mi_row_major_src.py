"""The row-major ``(n, F)`` read of ``pair_mi_shared_fused`` that production used before the referenced columns were gathered column-major, for the A/B benchmarks."""

from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused as shared

ROW_MAJOR_SRC = shared._PAIR_MI_SHARED_FUSED_SRC.replace("factors_data[a * n_samples + i]", "factors_data[i * (long long)n_features + a]").replace(
    "factors_data[b * n_samples + i]", "factors_data[i * (long long)n_features + b]"
)
assert ROW_MAJOR_SRC != shared._PAIR_MI_SHARED_FUSED_SRC, "kernel text changed; update the benchmark source"
