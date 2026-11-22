"""GPU CUDA kernel sources of the resident basis module, carved out of ``_gpu_resident_basis.py`` to bring that file under the 1k-LOC ceiling.

Each constant is a ``render`` template consumed by the kernel factories; names are re-exported by the parent so existing references keep resolving.
"""
from ._kernel_types import render

_CLENSHAW_SRC = render(r"""
__device__ {ACC} eval_basis(int basis, int d, {ACC} x) {
    if (d == 0) return 1.0;
    if (d == 1) return basis == 3 ? (1.0 - x) : x;
    if (basis == 3) {                                  // laguerre forward L_d
        {ACC} pp = 1.0, pc = 1.0 - x, pn;
        for (int k = 2; k <= d; ++k) { pn = ((2.0*k - 1.0 - x) * pc - (k - 1) * pp) / k; pp = pc; pc = pn; }
        return pc;
    }
    if (basis == 0) {                                  // chebyshev Clenshaw one-hot
        {ACC} x2 = 2.0 * x, c0 = 0.0, c1 = 1.0;
        for (int i = 3; i <= d + 1; ++i) { {ACC} tmp = c0; c0 = -c1; c1 = tmp + c1 * x2; }
        return c0 + c1 * x;
    }
    if (basis == 1) {                                  // legendre Clenshaw one-hot
        int nd = d + 1; {ACC} c0 = 0.0, c1 = 1.0;
        for (int i = 3; i <= d + 1; ++i) { {ACC} tmp = c0; nd -= 1; c0 = -(c1 * (nd - 1)) / nd; c1 = tmp + (c1 * x * (2*nd - 1)) / nd; }
        return c0 + c1 * x;
    }
    // basis == 2: hermite(He) Clenshaw one-hot
    int nd = d + 1; {ACC} c0 = 0.0, c1 = 1.0;
    for (int i = 3; i <= d + 1; ++i) { {ACC} tmp = c0; nd -= 1; c0 = -c1 * (nd - 1); c1 = tmp + c1 * x; }
    return c0 + c1 * x;
}
extern "C" __global__
void clenshaw_basis(const {REAL}* __restrict__ Z, const int* __restrict__ degs, const long long n,
                    const int m, const int nd, const int basis, {REAL}* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)m * nd;
    if (t >= total) return;
    int j = (int)(t % (long long)m);
    long long r = t / (long long)m;
    int k = (int)(r % (long long)nd);
    long long i = r / (long long)nd;
    {ACC} z = ({ACC})Z[i * (long long)m + j];
    out[i * ((long long)m * nd) + (long long)k * m + j] = ({REAL})eval_basis(basis, degs[k], z);
}
""")


_CLENSHAW_FWD_SRC = render(r"""
extern "C" __global__
void clenshaw_basis_fwd(const {REAL}* __restrict__ Z, const int* __restrict__ degs, const long long n,
                        const int m, const int nd, const int basis, {REAL}* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (long long)m) return;
    int j = (int)(t % (long long)m);
    long long i = t / (long long)m;
    {ACC} x = ({ACC})Z[t];
    {REAL}* o = out + i * ((long long)m * nd) + j;
    int top = degs[nd - 1];
    {ACC} pp = 1.0, pc = (basis == 3) ? (1.0 - x) : x;       // P_0, P_1
    int q = 0;
    for (int d = 0; d <= top; ++d) {
        {ACC} v;
        if (d == 0) v = 1.0;
        else if (d == 1) v = pc;
        else {
            {ACC} pn;
            if (basis == 0) pn = 2.0 * x * pc - pp;
            else if (basis == 1) pn = ((2.0 * (d - 1) + 1.0) * x * pc - (d - 1) * pp) / d;
            else if (basis == 2) pn = x * pc - (d - 1) * pp;
            else pn = ((2.0 * d - 1.0 - x) * pc - (d - 1) * pp) / d;
            pp = pc; pc = pn; v = pn;
        }
        if (d == degs[q]) { o[(long long)q * m] = ({REAL})v; ++q; }
    }
}
""")


_ROBUST_SCALE_COMBINE_SRC = render(r"""
extern "C" __global__
void robust_scale_combine(const {REAL}* __restrict__ mad, const {REAL}* __restrict__ q25,
                          const {REAL}* __restrict__ q75, const int K, {REAL}* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    {ACC} sc = 1.4826 * mad[i];
    {ACC} iqr = (q75[i] - q25[i]) / 1.349;
    out[i] = sc > 1e-12 ? sc : (iqr > 1e-12 ? iqr : 0.0);
}
""")


_HEAVY_TAIL_STATS_TILED_SRC = render(r"""
extern "C" __global__
void heavy_tail_stats_tiled(const {REAL}* __restrict__ M, const {REAL}* __restrict__ med,
                            const {REAL}* __restrict__ scale, const {ACC} outerK,
                            const long long n, const int K,
                            long long* __restrict__ part_n, {ACC}* __restrict__ part_b, {ACC}* __restrict__ part_o) {
    // One 32-column x 8-row tile per block, and blockIdx.y owns a share of the rows (grid fill: 8 columns used to be a single block): the partial outer count, the largest bulk deviation and
    // the smallest outer deviation of the row share go to (blockIdx.y, column). The three reductions are order independent (an exact integer sum, a max, a min), so the combine in
    // ``heavy_tail_finish`` gives the single-block values whatever the number of shares.
    __shared__ long long sh_n[8][33];
    __shared__ {ACC} sh_b[8][33];
    __shared__ {ACC} sh_o[8][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * 32 + tx;
    const bool ok = col < K;
    {ACC} mc = ok ? ({ACC})med[col] : 0.0;
    {ACC} sc = ok ? ({ACC})scale[col] : 0.0;
    {ACC} thr = outerK * sc;
    long long nloc = 0;
    {ACC} bulk = -1e308, omin = 1e308;
    if (ok) for (long long i = (long long)blockIdx.y * 8 + ty; i < n; i += 8LL * gridDim.y) {
        {ACC} dev = fabs(({ACC})M[i * (long long)K + col] - mc);
        if (dev > thr) { nloc += 1; if (dev < omin) omin = dev; }
        else if (dev > bulk) bulk = dev;
    }
    sh_n[ty][tx] = nloc; sh_b[ty][tx] = bulk; sh_o[ty][tx] = omin;
    __syncthreads();
    if (ty == 0 && ok) {
        long long no = 0;
        {ACC} bmax = -1e308, omn = 1e308;
        for (int r = 0; r < 8; ++r) {
            no += sh_n[r][tx];
            if (sh_b[r][tx] > bmax) bmax = sh_b[r][tx];
            if (sh_o[r][tx] < omn) omn = sh_o[r][tx];
        }
        part_n[(long long)blockIdx.y * K + col] = no;
        part_b[(long long)blockIdx.y * K + col] = bmax;
        part_o[(long long)blockIdx.y * K + col] = omn;
    }
}

extern "C" __global__
void heavy_tail_finish(const {REAL}* __restrict__ scale, const long long* __restrict__ part_n, const {ACC}* __restrict__ part_b, const {ACC}* __restrict__ part_o,
                       const int shares, const {ACC} gapthr, const {ACC} maxfrac, const long long n, const int K, signed char* __restrict__ out_v) {
    // The verdict of column `col` from the partials of its row shares: sc > 1e-12, some outer rows, at most maxfrac of the rows outer, and the outer minimum at least gapthr times the bulk maximum.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= K) return;
    long long no = 0;
    {ACC} bmax = -1e308, omn = 1e308;
    for (int s = 0; s < shares; ++s) {
        no += part_n[(long long)s * K + col];
        if (part_b[(long long)s * K + col] > bmax) bmax = part_b[(long long)s * K + col];
        if (part_o[(long long)s * K + col] < omn) omn = part_o[(long long)s * K + col];
    }
    const {ACC} sc = ({ACC})scale[col];
    {ACC} bk = bmax > 1e-12 ? bmax : 1e-12;
    {ACC} gap = omn / bk;
    out_v[col] = ((sc > 1e-12) && (no > 0) && (({ACC})no <= maxfrac * ({ACC})n) && (gap >= gapthr)) ? 1 : 0;
}
""")


_BASIS_PREPROCESS_SRC = render(r"""
#define REL_TOL 7.105427357601002e-15   // 32 * DBL_EPSILON, the host _safe_scale._REL_TOL
extern "C" __global__
void basis_preprocess(const {REAL}* __restrict__ M, const {REAL}* __restrict__ med,
                      const {REAL}* __restrict__ scale, const {REAL}* __restrict__ mn,
                      const {REAL}* __restrict__ mx, const {REAL}* __restrict__ stdcol,
                      const int basis_code, const {ACC} Kc, const long long n, const int K,
                      {REAL}* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    {ACC} x = M[row * (long long)K + col];
    {ACC} mc = med[col], sc = scale[col];
    bool deg = sc <= 1e-12;
    {ACC} lo = deg ? mn[col] : mc - Kc * sc;
    {ACC} hi = deg ? mx[col] : mc + Kc * sc;
    {ACC} z;
    if (basis_code == 0) {                                   // hermite robust z-score
        {ACC} std = (hi - lo) / 6.0;
        if (!(std > 1e-12)) std = stdcol[col];               // cp.where(std>1e-12, std, guarded M.std)
        if (!(std > REL_TOL * fabs(mc))) std = 1.0;
        z = (x - mc) / std;
        z = z < -6.0 ? -6.0 : (z > 6.0 ? 6.0 : z);
    } else if (basis_code == 1) {                            // legendre/chebyshev min-max -> [-1,1]
        {ACC} mag = fabs(lo) > fabs(hi) ? fabs(lo) : fabs(hi);
        {ACC} span = hi - lo;
        if (!(span > REL_TOL * mag)) span = 1.0;
        z = 2.0 * (x - lo) / span - 1.0;
        z = z < -1.0 ? -1.0 : (z > 1.0 ? 1.0 : z);
    } else {                                                 // laguerre shift -> >= 0
        {ACC} hic = (hi - lo) + 1e-9;
        z = x - lo + 1e-9;
        z = z < 0.0 ? 0.0 : (z > hic ? hic : z);
    }
    out[row * (long long)K + col] = z;
}
""")


_BASIS_PREPROCESS_NR_SRC = render(r"""
extern "C" __global__
void basis_preprocess_nr(const {REAL}* __restrict__ M, const {REAL}* __restrict__ p0,
                         const {REAL}* __restrict__ p1, const int basis_code,
                         const long long n, const int K, {REAL}* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    {ACC} x = M[row * (long long)K + col];
    {ACC} a = p0[col], b = p1[col];
    {ACC} z;
    if (basis_code == 0)       z = (x - a) / b;            // hermite z-score (b = std + 1e-12)
    else if (basis_code == 1)  z = 2.0 * (x - a) / b - 1.0; // legendre/cheb min-max (a = lo, b = span)
    else                       z = x - a + 1e-9;            // laguerre shift (a = lo)
    out[row * (long long)K + col] = z;
}
""")


_COL_MEANSTD_TILED_SRC = render(r"""
extern "C" __global__
void col_meanstd_tiled(const {REAL}* __restrict__ M, const long long n, const int K,
                       {REAL}* __restrict__ out_mean, {REAL}* __restrict__ out_std) {
    __shared__ {ACC} sh[8][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * 32 + tx;
    const bool ok = col < K;
    {ACC} s = 0.0;
    if (ok) for (long long i = ty; i < n; i += 8) s += ({ACC})M[i * (long long)K + col];
    sh[ty][tx] = s; __syncthreads();
    if (ty == 0) { {ACC} t = 0.0; for (int r = 0; r < 8; ++r) t += sh[r][tx]; sh[0][tx] = t; }
    __syncthreads();
    const {ACC} mean = sh[0][tx] / ({ACC})n;
    __syncthreads();
    {ACC} d2 = 0.0;
    if (ok) for (long long i = ty; i < n; i += 8) { {ACC} dv = ({ACC})M[i * (long long)K + col] - mean; d2 += dv * dv; }
    sh[ty][tx] = d2; __syncthreads();
    if (ty == 0) { {ACC} t = 0.0; for (int r = 0; r < 8; ++r) t += sh[r][tx]; sh[0][tx] = t; }
    __syncthreads();
    if (ty == 0 && ok) { out_mean[col] = ({REAL})mean; out_std[col] = ({REAL})sqrt(sh[0][tx] / ({ACC})n); }
}
""")


_COL_MINMAX_TILED_SRC = render(r"""
extern "C" __global__
void col_minmax_tiled(const {REAL}* __restrict__ M, const long long n, const int K,
                      {REAL}* __restrict__ out_min, {REAL}* __restrict__ out_max) {
    __shared__ {ACC} sh_min[8][33];
    __shared__ {ACC} sh_max[8][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int col = blockIdx.x * 32 + tx;
    const bool ok = col < K;
    {ACC} lmin = 1e308, lmax = -1e308;
    if (ok) for (long long i = ty; i < n; i += 8) { {ACC} v = M[i * (long long)K + col]; if (v < lmin) lmin = v; if (v > lmax) lmax = v; }
    sh_min[ty][tx] = lmin; sh_max[ty][tx] = lmax; __syncthreads();
    if (ty == 0) {
        {ACC} mn = sh_min[0][tx], mx = sh_max[0][tx];
        for (int r = 1; r < 8; ++r) { if (sh_min[r][tx] < mn) mn = sh_min[r][tx]; if (sh_max[r][tx] > mx) mx = sh_max[r][tx]; }
        if (ok) { out_min[col] = mn; out_max[col] = mx; }
    }
}
""")
