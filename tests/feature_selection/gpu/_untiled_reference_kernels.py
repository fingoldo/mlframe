"""Untiled one-block-per-column reference kernels (the pre-coalescing versions), kept as oracles for the tiled production kernels."""

from mlframe.feature_selection.filters._kernel_types import render

_HEAVY_TAIL_STATS_SRC = render(r"""
extern "C" __global__
void heavy_tail_stats(const {REAL}* __restrict__ M, const {REAL}* __restrict__ med,
                      const {REAL}* __restrict__ scale, const {ACC} outerK, const {ACC} gapthr,
                      const {ACC} maxfrac, const long long n, const int K,
                      signed char* __restrict__ out_v) {
    int col = blockIdx.x;
    if (col >= K) return;
    {ACC} mc = ({ACC})med[col];
    {ACC} sc = ({ACC})scale[col];
    {ACC} thr = outerK * sc;
    int tid = threadIdx.x;
    long long nloc = 0;
    {ACC} bulk = -1e308, omin = 1e308;
    for (long long i = tid; i < n; i += blockDim.x) {
        {ACC} dev = fabs(({ACC})M[i * (long long)K + col] - mc);
        if (dev > thr) { nloc += 1; if (dev < omin) omin = dev; }
        else if (dev > bulk) bulk = dev;
    }
    __shared__ long long sh_n[256];
    __shared__ {ACC} sh_b[256];
    __shared__ {ACC} sh_o[256];
    sh_n[tid] = nloc; sh_b[tid] = bulk; sh_o[tid] = omin;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sh_n[tid] += sh_n[tid + s];
            if (sh_b[tid + s] > sh_b[tid]) sh_b[tid] = sh_b[tid + s];
            if (sh_o[tid + s] < sh_o[tid]) sh_o[tid] = sh_o[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        long long no = sh_n[0];
        {ACC} bk = sh_b[0] > 1e-12 ? sh_b[0] : 1e-12;
        {ACC} gap = sh_o[0] / bk;
        out_v[col] = ((sc > 1e-12) && (no > 0) && (({ACC})no <= maxfrac * ({ACC})n) && (gap >= gapthr)) ? 1 : 0;
    }
}
""")

_COL_MEANSTD_SRC = render(r"""
extern "C" __global__
void col_meanstd(const {REAL}* __restrict__ M, const long long n, const int K,
                 {REAL}* __restrict__ out_mean, {REAL}* __restrict__ out_std) {
    int col = blockIdx.x;
    if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ {ACC} sh[256];
    {ACC} s = 0.0;
    for (long long i = tid; i < n; i += nt) s += ({ACC})M[i * (long long)K + col];
    sh[tid] = s; __syncthreads();
    for (int j = nt >> 1; j > 0; j >>= 1) { if (tid < j) sh[tid] += sh[tid + j]; __syncthreads(); }
    {ACC} mean = sh[0] / ({ACC})n;
    __syncthreads();
    {ACC} d2 = 0.0;
    for (long long i = tid; i < n; i += nt) { {ACC} dv = ({ACC})M[i * (long long)K + col] - mean; d2 += dv * dv; }
    sh[tid] = d2; __syncthreads();
    for (int j = nt >> 1; j > 0; j >>= 1) { if (tid < j) sh[tid] += sh[tid + j]; __syncthreads(); }
    if (tid == 0) { out_mean[col] = ({REAL})mean; out_std[col] = ({REAL})sqrt(sh[0] / ({ACC})n); }
}
""")

_COL_MINMAX_SRC = render(r"""
extern "C" __global__
void col_minmax(const {REAL}* __restrict__ M, const long long n, const int K,
                {REAL}* __restrict__ out_min, {REAL}* __restrict__ out_max) {
    int col = blockIdx.x;
    if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    {ACC} lmin = 1e308, lmax = -1e308;
    for (long long i = tid; i < n; i += nt) { {ACC} v = M[i * (long long)K + col]; if (v < lmin) lmin = v; if (v > lmax) lmax = v; }
    __shared__ {ACC} sh_min[256];
    __shared__ {ACC} sh_max[256];
    sh_min[tid] = lmin; sh_max[tid] = lmax;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) { if (sh_min[tid + s] < sh_min[tid]) sh_min[tid] = sh_min[tid + s]; if (sh_max[tid + s] > sh_max[tid]) sh_max[tid] = sh_max[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_min[col] = sh_min[0]; out_max[col] = sh_max[0]; }
}
""")


def get_heavy_tail_stats_kernel(cp, real_t: str = "double"):
    """The untiled ``heavy_tail_stats`` RawKernel for the element type ``real_t``."""
    return cp.RawKernel(_HEAVY_TAIL_STATS_SRC.replace("{REAL}", real_t), "heavy_tail_stats")
