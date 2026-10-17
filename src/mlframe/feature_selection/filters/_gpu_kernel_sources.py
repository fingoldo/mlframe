"""CUDA C source strings for the ``RawKernel`` objects built by ``mlframe.feature_selection.filters.gpu.init_kernels``."""

COMPUTE_JOINT_HIST_CUDA_SRC: str = r"""
    extern "C" __global__
    void compute_joint_hist_cuda(const int *classes_x, const int *classes_y, int *joint_counts, int n, int nbins_y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid<n){
            atomicAdd(&joint_counts[classes_x[tid]*nbins_y+classes_y[tid]],1);
        }
    }
    """

COMPUTE_JOINT_HIST_BATCHED_CUDA_SRC: str = r"""
    extern "C" __global__
    void compute_joint_hist_batched_cuda(
        const int *classes_x,           // (n,)
        const int *perms_y,             // (batch, n) row-major
        int *joint_counts_batch,        // (batch, nbins_x * nbins_y) row-major
        int n,
        int nbins_x,
        int nbins_y
    ) {
        int batch_id = blockIdx.y;
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
            int cy = perms_y[(long long)batch_id * n + tid];
            int cx = classes_x[tid];
            atomicAdd(&joint_counts_batch[(long long)batch_id * (nbins_x * nbins_y) + cx * nbins_y + cy], 1);
        }
    }
    """

COMPUTE_JOINT_HIST_BATCHED_SHARED_CUDA_SRC: str = r"""
    extern "C" __global__
    void compute_joint_hist_batched_shared_cuda(
        const int *classes_x,           // (n,)
        const int *perms_y,             // (batch, n) row-major
        int *joint_counts_batch,        // (batch, nbins_x * nbins_y) row-major
        int n,
        int nbins_x,
        int nbins_y
    ) {
        extern __shared__ int sm_hist[];  // size: nbins_x * nbins_y * sizeof(int)
        int batch_id = blockIdx.y;
        int joint_size = nbins_x * nbins_y;
        int tid = threadIdx.x;
        int nthreads = blockDim.x;
        long long gid_offset = (long long)blockIdx.x * nthreads;

        // Zero the shared histogram.
        for (int i = tid; i < joint_size; i += nthreads) {
            sm_hist[i] = 0;
        }
        __syncthreads();

        // Grid-strided read of n_samples (only the block's slice).
        long long my_row = gid_offset + tid;
        if (my_row < n) {
            int cy = perms_y[(long long)batch_id * n + my_row];
            int cx = classes_x[my_row];
            atomicAdd(&sm_hist[cx * nbins_y + cy], 1);
        }
        __syncthreads();

        // Reduce: each thread merges its slice of sm_hist into the per-batch
        // global output. Block-level atomicAdd into global is still atomic
        // (many blocks may have this batch_id), so we use atomicAdd here too.
        for (int i = tid; i < joint_size; i += nthreads) {
            atomicAdd(&joint_counts_batch[(long long)batch_id * joint_size + i], sm_hist[i]);
        }
    }
    """

COMPUTE_JOINT_HIST_MULTI_PAIR_CUDA_SRC: str = r"""
    extern "C" __global__
    void compute_joint_hist_multi_pair_cuda(
        const int *factors_data_T,    // (n_cols, n)
        const int *classes_y,          // (n,)  - already merged target
        const int *pairs_a,            // (n_pairs,)
        const int *pairs_b,            // (n_pairs,)
        const int *nbins_a,            // (n_pairs,)
        const int *joint_offsets,      // (n_pairs + 1,) prefix sum into joint_counts_flat
        int *joint_counts_flat,        // sum(nbins_a[p]*nbins_b[p]*nbins_y)
        int n_rows,
        int n_pairs,
        int nbins_y
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int pid = blockIdx.y;
        if (pid >= n_pairs) return;
        if (row >= n_rows) return;

        int col_a = pairs_a[pid];
        int col_b = pairs_b[pid];
        int nba = nbins_a[pid];
        int va = factors_data_T[(long long)col_a * n_rows + row];
        int vb = factors_data_T[(long long)col_b * n_rows + row];
        int cy = classes_y[row];
        // Merged (a, b) code = va + vb * nba. Joint cell = merged * nbins_y + cy.
        int merged = va + vb * nba;
        atomicAdd(
            &joint_counts_flat[joint_offsets[pid] + merged * nbins_y + cy],
            1
        );
    }
    """

COMPUTE_JOINT_HIST_MULTI_PAIR_SHARED_CUDA_SRC: str = r"""
    extern "C" __global__
    void compute_joint_hist_multi_pair_shared_cuda(
        const int *factors_data_T,    // (n_cols, n)
        const int *classes_y,          // (n,)
        const int *pairs_a,            // (n_pairs,)
        const int *pairs_b,            // (n_pairs,)
        const int *nbins_a,            // (n_pairs,)
        const int *joint_offsets,      // (n_pairs + 1,) prefix sum
        int *joint_counts_flat,        // sum(nbins_a[p]*nbins_b[p]*nbins_y)
        int n_rows,
        int n_pairs,
        int nbins_y,
        int max_joint_size_y           // shared-mem allocation size in cells
    ) {
        extern __shared__ int sm_hist[];  // size: max_joint_size_y * sizeof(int)
        int pid = blockIdx.y;
        if (pid >= n_pairs) return;

        int pair_size = joint_offsets[pid + 1] - joint_offsets[pid];
        int tid = threadIdx.x;
        int nthreads = blockDim.x;

        // Zero only the pair's slice of shared (rest is unused / garbage).
        for (int i = tid; i < pair_size; i += nthreads) {
            sm_hist[i] = 0;
        }
        __syncthreads();

        int col_a = pairs_a[pid];
        int col_b = pairs_b[pid];
        int nba = nbins_a[pid];

        long long row = (long long)blockIdx.x * nthreads + tid;
        if (row < n_rows) {
            int va = factors_data_T[(long long)col_a * n_rows + row];
            int vb = factors_data_T[(long long)col_b * n_rows + row];
            int cy = classes_y[row];
            int merged = va + vb * nba;
            atomicAdd(&sm_hist[merged * nbins_y + cy], 1);
        }
        __syncthreads();

        // Reduce shared -> per-pair global slot.
        int g_off = joint_offsets[pid];
        for (int i = tid; i < pair_size; i += nthreads) {
            atomicAdd(&joint_counts_flat[g_off + i], sm_hist[i]);
        }
    }
    """

COMPUTE_MI_FROM_CLASSES_CUDA_SRC: str = r"""
    extern "C" __global__
    void compute_mi_from_classes_cuda(const int *classes_x, const double *freqs_x,const int *classes_y, const double *freqs_y, int *joint_counts, double *totals, int n,int nbins_x,int nbins_y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid==0){
            // joint_counts is a (nbins_x, nbins_y) row-major histogram written by
            // compute_joint_hist_cuda with stride nbins_y (``cx*nbins_y+cy``). The
            // reduction MUST read it with the SAME stride. A pre-2026-06-11 bug
            // hardcoded ``i*2+j`` (stride 2), which only matched a binary target
            // (nbins_y==2); for any nbins_y!=2 (multi-class y, or quantile-binned
            // regression y with nbins_y up to 10) it read the WRONG cells, yielding
            // garbage permutation-null MIs that systematically over-rejected genuine
            // candidates - silently diverging the GPU selection from the CPU path.
            // Use double throughout (no float32 narrowing) to match the bit-exact
            // CPU njit compute_mi_from_classes.
            double total = 0.0;
            for (int i=0;i<nbins_x;++i){
                double prob_x = freqs_x[i];
                for (int j=0;j<nbins_y;++j){
                    int jc = joint_counts[i*nbins_y+j];
                    if (jc>0){
                        double prob_y = freqs_y[j];
                        double jf=(double)jc/ (double)n;
                        total += jf* log(jf / (prob_x * prob_y));
                    }
                }
            }
            totals[0]=total;
        }
    }
    """
