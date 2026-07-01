"""GPU-resident FE: residency-buffer + radix-select edge kernels (Tier E carve).

Carved VERBATIM out of ``_gpu_resident_fe.py`` (sibling re-export pattern) to bring the parent under the
1k-LOC ceiling. Holds the residency-buffer + radix-select edge block: the rank-EXACT sort-free
``radix_select_*`` RawKernels + their wrappers, the fused ``bin_codes`` kernel, the resident discretize
path, the FE-materialise kernel, the pinned-D2H staging buffer, the operand-table residency cache/build
helpers, and the GPU materialise/discretize codes host fast paths.

The gate helpers (``_cuda_present`` / ``_env_gpu_default_on`` / ``fe_gpu_*_enabled``) and the candidate-
grid primitives stay in the PARENT and are imported below; the parent re-exports every public/used name
moved here so all ``from .._gpu_resident_fe import X`` paths still resolve byte-for-byte. The few
cross-sibling references (``_gpu_apply_prewarp`` in ``_gpu_resident_basis``) are LAZY-imported inside the
function bodies to avoid an import cycle. No kernel-source, dispatch-threshold, residency, or selection
behavior changed.
"""
from __future__ import annotations

import os

import numpy as np


# RANK-EXACT SORT-FREE QUANTILE EDGES via RADIX-SELECT (roadmap #2, 2026-06-20). cp.percentile bins each
# column with a FULL O(n log n) sort (profiled n=100k/79s: the cp.percentile SORT in this function = 12.9s,
# the #1 production GPU cost) but it only needs the nbins-1 INTERIOR quantile EDGES -- i.e. the ~2*(nbins-1)
# bracketing ORDER STATISTICS per column, not a full ordering. This kernel extracts exactly those order
# statistics with a byte-digit RADIX-SELECT: one block per column, R<=2*(nbins-1) target ranks resolved
# TOGETHER in 8 (float64) / 4 (float32) histogram passes over the column. Each pass reads the column once,
# bins each row's current byte-digit into its rank-window's 256-bucket SHARED-MEM histogram (a row maps to
# exactly ONE active window, found by matching its fixed high-byte prefix), then advances every rank's
# prefix to the bucket holding its target rank. After all passes each rank's exact order-statistic VALUE
# is recovered from the converged key. The float key is the standard order-preserving IEEE transform
# (flip sign bit for positives, all bits for negatives) so the byte order == the float order EXACTLY ->
# the recovered values are BIT-IDENTICAL to the sorted column at those ranks (verified maxdiff 0 on the
# order stats; the codes through cp.searchsorted match cp.percentile maxdiff 0 across all columns).
#
# WHY THIS WINS (the prior estimate said ~8-11 passes ~= sort bandwidth so it may NOT win -- DISPROVEN for
# THIS cupy: cp.percentile uses a comparison MERGE-sort over (value,index) zip-iterators, NOT a radix sort;
# nvprof n=300k K=384: DeviceMergeSort{Merge,BlockSort,Partition} = 65.6% of binning, far above the linear
# bandwidth floor. The 8-pass radix-select read floor measured 16-17x faster than that sort.) MEASURED
# GTX 1050 Ti, R=38, heavy-tailed a**2/b candidates, CUDA-event A/B vs cp.percentile, BIT-IDENTICAL codes
# (maxdiff 0 all columns): f64  100k 1.17x / 300k 1.19x / 1M 2.06x;  f32  100k 2.38x / 300k 2.30x / 1M
# 3.67x. The win GROWS with n (O(n) select vs O(n log n) sort) -- exactly the large-n*K regime the GPU
# binning engages (the auto-router keeps small n on the CPU). The per-row inner window-match loop (R<=40)
# keeps the real time above the bare bandwidth floor; the sorted-prefix BINARY SEARCH variant (Lever C,
# 2026-06-23, ``radix_select_f32_bsearch``) cuts that divergence -- isolated radix kernel A/B at the
# production shape (n=100k, K=583, R=38, threads=1024) measured 65.05ms -> 48.63ms = 1.337x, BIT-IDENTICAL
# codes (maxdiff 0); it is now the per-host KTC default for the f32 path (base linear scan = fallback).
#
# EXACTNESS / fallback: the order statistics are exact; the cupy 'linear' interpolation is reproduced in
# float64 EXACTLY (idx=q*(N-1); w=idx-floor(idx); w<0.5 ? below+diff*w : above-diff*(1-w); diff in f64 over
# the (f32-promoted-to-f64 or native-f64) order stats) so the edges -> codes equal cp.percentile bit-for-
# bit. cp.percentile stays the gated exact fallback: MLFRAME_FE_GPU_RADIX_EDGES=0 forces it, and ANY kernel
# failure (compile / launch / shared-mem overflow) falls back to cp.percentile inside this function. The
# shared histogram is R*256 uint32 (R<=40 -> <=40KB < the 48KB default) plus a few small per-rank arrays;
# the host gates the radix path off (-> cp.percentile) if that ever exceeds the device limit.
_RADIX_SELECT_F64_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f64(const double* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, double* __restrict__ out){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];          // W*256 histogram (counts <= n < 2^31 -> uint32 ok)
    __shared__ unsigned long long prefix[MAXR];   // per-rank running key prefix (high bytes fixed)
    __shared__ unsigned long long below[MAXR];    // count strictly below each rank's window
    __shared__ unsigned long long wpref[MAXR];    // distinct active window prefixes (masked)
    __shared__ int rank2w[MAXR];                  // rank -> window index
    __shared__ int W;
    if(tid<R){prefix[tid]=0ULL;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=7;byte>=0;--byte){
        int shift=byte*8;
        unsigned long long hmask=(byte==7)?0ULL:(0xFFFFFFFFFFFFFFFFULL<<((byte+1)*8));
        if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned long long p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;}
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            double d=data[(long long)col*n+i];unsigned long long u;memcpy(&u,&d,8);  // COLUMN-MAJOR: coalesced
            u=(u&0x8000000000000000ULL)?~u:(u|0x8000000000000000ULL);
            unsigned long long pm=u&hmask;int win=-1;
            for(int q=0;q<Wl;++q)if(wpref[q]==pm){win=q;break;}
            if(win>=0){int dig=(int)((u>>shift)&0xFFULL);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        if(tid==0){for(int r=0;r<R;++r){int w=rank2w[r];unsigned long long acc=below[r];int chosen=0;long long want=ranks[r];
            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[r]=acc;prefix[r]=(prefix[r]&hmask)|((unsigned long long)chosen<<shift);}}
        __syncthreads();
    }
    if(tid<R){unsigned long long u=prefix[tid];u=(u&0x8000000000000000ULL)?(u&0x7FFFFFFFFFFFFFFFULL):~u;
        double d;memcpy(&d,&u,8);out[tid*K+col]=d;}
}
"""
_RADIX_SELECT_F32_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f32(const float* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, float* __restrict__ out){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned int prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned int wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ int W;
    if(tid<R){prefix[tid]=0u;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=3;byte>=0;--byte){
        int shift=byte*8;
        unsigned int hmask=(byte==3)?0u:(0xFFFFFFFFu<<((byte+1)*8));
        if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned int p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;}
        __syncthreads();
        int Wl=W;
        // bench-attempt-rejected (2026-06-21, elevated nvprof): the per-window stride 256 is bank-aligned
        // (shared_store_transactions_per_request ~6.1), but padding to 257 to de-conflict was SLOWER
        // (264->324ms) -- the kernel is WARP-DIVERGENCE-bound (warp_execution_efficiency ~42% from the
        // per-thread window search), not bank-conflict-bound, and the extra shared bytes cut occupancy.
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            float d=data[(long long)col*n+i];unsigned int u;memcpy(&u,&d,4);  // COLUMN-MAJOR: coalesced
            u=(u&0x80000000u)?~u:(u|0x80000000u);
            unsigned int pm=u&hmask;int win=-1;
            for(int q=0;q<Wl;++q)if(wpref[q]==pm){win=q;break;}
            if(win>=0){int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        if(tid==0){for(int r=0;r<R;++r){int w=rank2w[r];unsigned long long acc=below[r];int chosen=0;long long want=ranks[r];
            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[r]=acc;prefix[r]=(prefix[r]&hmask)|((unsigned int)chosen<<shift);}}
        __syncthreads();
    }
    if(tid<R){unsigned int u=prefix[tid];u=(u&0x80000000u)?(u&0x7FFFFFFFu):~u;float d;memcpy(&d,&u,4);out[tid*K+col]=d;}
}
"""
# BINARY-SEARCH WINDOW-MATCH VARIANT (2026-06-23, Lever C). The base ``radix_select_f32`` is WARP-DIVERGENCE
# bound (~42% warp_execution_efficiency, prior nvprof): the dominant per-row inner loop linearly scans up to
# Wl (<=R<=40) active window prefixes to find the ONE matching ``pm = u & hmask`` (lines 139-140). Each row
# does O(Wl) compares and threads in a warp diverge on WHICH q matches / WHEN they break -> low SIMD eff.
#
# This variant replaces that linear scan with a BRANCHLESS BINARY SEARCH (O(log Wl), uniform iteration count
# across the warp -> no early-break divergence). The ``tid==0`` discovery loop already builds ``wpref`` /
# ``rank2w`` in first-appearance order; we ADD a SORTED view (``wsort`` = ascending distinct prefixes, built
# once per byte-pass by tid==0 via insertion sort over W<=40 entries -- trivially cheap, serial in tid==0
# alongside the existing serial discovery) plus ``wsort2w`` mapping sorted-position -> the ORIGINAL window
# index. Each data thread lower-bounds ``pm`` in ``wsort`` with a fixed-trip-count loop, then confirms an
# EXACT match (rows whose ``pm`` is not an active window prefix get win=-1 and are skipped, EXACTLY as the
# linear scan did) and maps to the original window index via ``wsort2w`` -- so ``rank2w`` and the per-window
# histogram slot ``sh[win*256+..]`` are byte-for-byte the SAME layout as the base kernel. The order
# statistics are therefore BIT-IDENTICAL; only HOW each row finds its window changes (verified maxdiff 0).
# A direct prefix->window LUT was rejected: ``pm`` spans the full 32-bit transformed-key space (high bytes of
# an arbitrary IEEE key), so a dense LUT is infeasible; the prefixes are few (W<=40) but sparse -> binary
# search over the sorted small array is the correct + cheap structure. Kept as a SEPARATE kernel (the base
# stays the exact fallback / dispatch alternative); selected by the KTC variant sweep, default = whichever is
# measured faster per host (the base remains the fallback on any compile/launch failure).
_RADIX_SELECT_F32_BSEARCH_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f32_bsearch(const float* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, float* __restrict__ out){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned int prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned int wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ unsigned int wsort[MAXR];   // wpref sorted ascending (distinct active window prefixes)
    __shared__ int wsort2w[MAXR];          // sorted position -> ORIGINAL window index (into wpref / sh)
    __shared__ int W;
    if(tid<R){prefix[tid]=0u;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=3;byte>=0;--byte){
        int shift=byte*8;
        unsigned int hmask=(byte==3)?0u:(0xFFFFFFFFu<<((byte+1)*8));
        if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned int p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;
            // insertion-sort the W<=40 distinct prefixes ascending, carrying the original window index.
            for(int q=0;q<w;++q){wsort[q]=wpref[q];wsort2w[q]=q;}
            for(int a=1;a<w;++a){unsigned int kv=wsort[a];int kw=wsort2w[a];int b=a-1;
                while(b>=0&&wsort[b]>kv){wsort[b+1]=wsort[b];wsort2w[b+1]=wsort2w[b];--b;}
                wsort[b+1]=kv;wsort2w[b+1]=kw;}}
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            float d=data[(long long)col*n+i];unsigned int u;memcpy(&u,&d,4);  // COLUMN-MAJOR: coalesced
            u=(u&0x80000000u)?~u:(u|0x80000000u);
            unsigned int pm=u&hmask;
            // branchless lower_bound over wsort[0..Wl): fixed-trip-count, no early break -> low warp divergence
            int lo=0,hi=Wl;
            while(lo<hi){int mid=(lo+hi)>>1;if(wsort[mid]<pm)lo=mid+1;else hi=mid;}
            if(lo<Wl&&wsort[lo]==pm){int win=wsort2w[lo];
                int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        if(tid==0){for(int r=0;r<R;++r){int w=rank2w[r];unsigned long long acc=below[r];int chosen=0;long long want=ranks[r];
            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[r]=acc;prefix[r]=(prefix[r]&hmask)|((unsigned int)chosen<<shift);}}
        __syncthreads();
    }
    if(tid<R){unsigned int u=prefix[tid];u=(u&0x80000000u)?(u&0x7FFFFFFFu):~u;float d;memcpy(&d,&u,4);out[tid*K+col]=d;}
}
"""
# PARALLEL PER-RANK SCAN VARIANT (2026-06-23, Lever 4 -> the real win). The per-byte-pass histogram is
# followed by a CUMULATIVE-SCAN + prefix-advance that the base/bsearch kernels run ENTIRELY in ``tid==0``:
# a serial loop over the R<=40 ranks, each doing its own up-to-256-bucket cumulative scan while the other
# 1023 threads idle at the barrier. CUDA-event A/B at the production shape (n=100k, K=583, R=38, f32,
# threads=1024, GTX 1050 Ti, interleaved-min, 2x-confirmed) measured this serial scan -- NOT the n-read
# bandwidth -- is the dominant cost: the kernel achieved only ~19-20 GB/s vs the card's ~96 GB/s effective
# read bandwidth (a single n-pass sum runs at 96.5 GB/s -> a 4-pass bandwidth floor is ~10ms, but the
# bsearch kernel takes ~46-49ms = ~4.7x above the floor). EARLY-TERMINATION (Lever 1) was probed and is
# DEAD: continuous f32 quantile edges need ~3-4 byte passes to resolve all R ranks (instrumented: all
# windows collapse to width<=1 only at pass 3/4 for uniform/normal/heavy), so no full n-read can be skipped.
#
# THIS variant parallelises that scan: thread ``r`` (r<R) owns rank ``r`` and runs its window's cumulative
# 256-scan + prefix advance IN PARALLEL with the other ranks (R parallel scans instead of one serial loop
# over R). Each rank reads its OWN window's histogram slice ``sh[w*256+..]`` and writes its OWN
# ``below[r]``/``prefix[r]`` -- disjoint slots, no cross-rank dependency -> the result is IDENTICAL to the
# serial loop (verified maxdiff 0 across the full n x K x {uniform,normal,heavy,ties,all-equal} grid). The
# n-read + binary-search window-match (carried over from the bsearch variant) are UNCHANGED. MEASURED
# (CUDA-event A/B vs the bsearch kernel, interleaved-min >=9 reps, 2x-confirmed): uniform 48.6->22.1ms =
# 2.20x; normal 45.6->22.0ms = 2.07x; heavy 45.8->22.1ms = 2.07x; BIT-IDENTICAL (maxdiff 0). Kept as a
# SEPARATE kernel (bsearch/linear stay the exact fallbacks + dispatch alternatives); selected by the KTC
# variant sweep, default = whichever is measured faster per host (bsearch remains the fallback on any
# compile/launch failure of this variant).
#
# bench-attempt-rejected (2026-06-23): radix_select_f32 v4 -- v3 IS AT ITS PRACTICAL FLOOR on the GTX 1050 Ti.
# Full CUDA-event decomposition at the production shape (n=100k, K=583, R=38, f32, threads=1024, interleaved-
# min >=11 reps, 2x-confirmed): pure 4-pass column read = 9.66ms @ 96.5 GB/s (== the card's read-bandwidth
# CEILING); + the per-row binary-search window-match (atomics stubbed out) = ~19ms (match adds ~9.3ms); +
# shared atomicAdd = the full v3 28.5ms (atomics add ~9.6ms). The atomic cost is INTRINSIC Pascal shared-
# atomic INSTRUCTION latency, NOT contention: single-pass A/B shows the worst-contention high-byte pass
# (Wl=1, all 1024 threads -> one 256-bucket histogram) = 10.8ms vs the well-spread byte=0 pass (Wl=31) =
# 9.4ms, i.e. contention is only ~1.4ms of the 9.6ms; the rest is paid even when atomics are fully spread.
# Levers tried, all BIT-IDENTICAL (maxdiff 0, uniform/normal/heavy/ties/all-equal) but NO production win:
#   L1 replicated/privatized sub-histograms (NREP=R/Wl copies, merge before scan): 0.97-1.00x -- the per-
#      bucket merge + tid%rep dispatch eats the ~1.4ms contention saving; contention was never the bottleneck.
#   warp-aggregated atomics (emulated __match_any, Pascal has no HW match): 0.35-0.46x (O(32) shfl/row).
#   warp-aggregate FAST PATH (one add iff all active lanes agree on slot): real distros 0.88-0.92x (ballot/
#      shfl per row, agreement rare since digits spread); only all-equal wins (1.37x) -- not a production shape.
#   Wl==1 special-case compare + [w0,wlast] range-cull: 0.99-1.10x (cull adds divergence; net flat).
#   per-thread 4/8-elem unroll + local same-slot combine: 0.65-0.69x (per-elem bsearch dominates, combine
#      rarely hits, register pressure). L2 occupancy is already MAXED (2 blocks/SM x 1024 = 2048 = the SM
#      thread cap; 38KB shared/block leaves 2 blocks/SM regardless). L3 (split n across blocks/column) only
#      helps K < ~SMs*blocks_per_SM (=12); production K>=50 already fills all 6 SMs -> N/A. EARLY-TERMINATION
#      was already proven DEAD (continuous edges need all 3-4 passes). VERDICT: the GPU radix-select branch is
#      optimized for this hardware; the three ~9.6ms components (read floor / match / shared-atomic latency)
#      are each architecturally irreducible on Pascal. Next real win would need a different card (faster shared
#      atomics / HW __match_any -> warp-aggregation flips positive) or a non-histogram select algorithm.
_RADIX_SELECT_F32_V3_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f32_v3(const float* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, float* __restrict__ out){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned int prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned int wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ unsigned int wsort[MAXR];   // wpref sorted ascending (distinct active window prefixes)
    __shared__ int wsort2w[MAXR];          // sorted position -> ORIGINAL window index (into wpref / sh)
    __shared__ int W;
    if(tid<R){prefix[tid]=0u;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=3;byte>=0;--byte){
        int shift=byte*8;
        unsigned int hmask=(byte==3)?0u:(0xFFFFFFFFu<<((byte+1)*8));
        if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned int p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;
            for(int q=0;q<w;++q){wsort[q]=wpref[q];wsort2w[q]=q;}
            for(int a=1;a<w;++a){unsigned int kv=wsort[a];int kw=wsort2w[a];int b=a-1;
                while(b>=0&&wsort[b]>kv){wsort[b+1]=wsort[b];wsort2w[b+1]=wsort2w[b];--b;}
                wsort[b+1]=kv;wsort2w[b+1]=kw;}}
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            float d=data[(long long)col*n+i];unsigned int u;memcpy(&u,&d,4);  // COLUMN-MAJOR: coalesced
            u=(u&0x80000000u)?~u:(u|0x80000000u);
            unsigned int pm=u&hmask;
            int lo=0,hi=Wl;
            while(lo<hi){int mid=(lo+hi)>>1;if(wsort[mid]<pm)lo=mid+1;else hi=mid;}
            if(lo<Wl&&wsort[lo]==pm){int win=wsort2w[lo];
                int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        // PARALLEL per-rank cumulative scan + prefix advance: thread r owns rank r (disjoint below/prefix
        // slots, own window histogram slice) -> identical result to the serial tid==0 loop, R-way parallel.
        if(tid<R){int w=rank2w[tid];unsigned long long acc=below[tid];int chosen=0;long long want=ranks[tid];
            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[tid]=acc;prefix[tid]=(prefix[tid]&hmask)|((unsigned int)chosen<<shift);}
        __syncthreads();
    }
    if(tid<R){unsigned int u=prefix[tid];u=(u&0x80000000u)?(u&0x7FFFFFFFu):~u;float d;memcpy(&d,&u,4);out[tid*K+col]=d;}
}
"""
_RADIX_SELECT_F32_V3_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)
_RADIX_SELECT_F32_BSEARCH_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)
_RADIX_SELECT_F64_KERNEL = None  # module-level singletons (lazy-compiled; never on an instance -> pickle-safe)
_RADIX_SELECT_F32_KERNEL = None
_RADIX_SELECT_THREADS = 512  # historical default + KTC fallback (see _gpu_resident_radix_ktc / Lever B)
_RADIX_SELECT_MAXR = 64      # must match MAXR in the kernel sources
# STATIC __shared__ footprint of the radix-select kernels, in bytes (the host shared-mem gate must subtract
# this from the device per-block limit before testing the DYNAMIC histogram ``shmem = R*256*4`` -- otherwise
# dynamic+static can exceed the limit and cuLaunchKernel returns CUDA_ERROR_INVALID_VALUE). Worst case is the
# fused v2 kernel ``radix_select_interp_f64_v2``: prefix/below/wpref/wsort (4 x MAXR x 8B u64) + rank2w/wsort2w
# (2 x MAXR x 4B i32) + W (4B i32) + osv_sh (MAXR x 8B f64) = 2048+512+4+512 = 3076B at MAXR=64. The other
# (f64-original, f32 linear/bsearch/v3) variants use <= this, so gating on the worst case keeps every kernel
# the path may launch within budget. Bit-identical: gating only changes WHEN the radix path returns None (then
# the caller takes the cp.percentile fallback, same edges); a passing R still launches unchanged.
_RADIX_STATIC_SHARED_BYTES = 4 * _RADIX_SELECT_MAXR * 8 + 2 * _RADIX_SELECT_MAXR * 4 + 4 + _RADIX_SELECT_MAXR * 8
# Lever B (2026-06-23): threads/block is now KTC-TUNED per host (radix_select_f32 was the biggest kernel,
# 512 under-utilised the card -- 512->1024 = 1.20x at n=100k/K=583 on the GTX 1050 Ti). The sweep probe
# forces a specific count via this override (set/reset by _gpu_resident_radix_ktc._radix_edges_with_threads);
# the production launch reads it when set, else looks the tuned count up from the kernel_tuning_cache. Block
# size NEVER changes the order statistics (sum-reduction over the same values) -> edges/codes bit-identical.
_RADIX_THREADS_OVERRIDE = None  # int set by the KTC sweep probe; None -> use the per-host KTC lookup


def _resolve_radix_threads(n: int) -> int:
    """Threads/block for the radix-select launch: the sweep override when one is active (during the KTC
    timing probe), else the per-host tuned count from the kernel_tuning_cache (falls back to 512)."""
    if _RADIX_THREADS_OVERRIDE is not None:
        return int(_RADIX_THREADS_OVERRIDE)
    try:
        from ._gpu_resident_radix_ktc import radix_select_threads
        return int(radix_select_threads(int(n)))
    except Exception:
        return _RADIX_SELECT_THREADS


# COALESCED TILED TRANSPOSE (2026-06-23, nsys-driven Lever A). The radix-select kernel reads its input
# COLUMN-MAJOR (data[col*n+i], coalesced over the dominant n-loop -- see _radix_select_interior_edges),
# so the (n,K) row-major candidate buffer was transposed to (K,n) C-order via cp.ascontiguousarray(cand.T).
# nsys (isolated KTC, F2 100k, one fit): that transpose-copy = cupy_copy__float32_float32 28 calls /
# 1889ms = 99% of ALL f32->f32 copy time and ~18% of the whole fit's GPU kernel time -- and it ran at only
# ~3.5GB/s because cupy's generic strided-copy for a transposed view is uncoalesced on BOTH read+write.
# This shared-memory TILED transpose (32x32 tiles, +1 pad column to kill the shared bank conflict on the
# transposed write) reads the (n,K) buffer coalesced and writes the (K,n) buffer coalesced. MEASURED at the
# production size (n=100k, K=583, 233MB): cp.ascontiguousarray(x.T) 42.6ms -> this kernel 6.2ms = 6.9x,
# BIT-IDENTICAL (maxdiff 0). It MOVES the same bytes into the same (K,n) C-order layout the radix kernel
# already consumes -> the order statistics, edges, and codes are byte-for-byte unchanged. Falls back to
# cp.ascontiguousarray(cand.T) on any kernel failure (compile/launch) -- bit-identical either way.
_TRANSPOSE_F32_SRC = r"""
extern "C" __global__
void transpose_f32(const float* __restrict__ in, float* __restrict__ out,
                   const long long n, const int K) {
    __shared__ float tile[32][33];               // +1 pad column: no shared-bank conflict on the write
    long long row = (long long)blockIdx.y * 32 + threadIdx.y;   // index into n (rows of the (n,K) input)
    int col = blockIdx.x * 32 + threadIdx.x;                    // index into K
    if (row < n && col < K) tile[threadIdx.y][threadIdx.x] = in[row * (long long)K + col];
    __syncthreads();
    int orow = blockIdx.x * 32 + threadIdx.y;                   // index into K (rows of the (K,n) output)
    long long ocol = (long long)blockIdx.y * 32 + threadIdx.x;  // index into n
    if (orow < K && ocol < n) out[(long long)orow * n + ocol] = tile[threadIdx.x][threadIdx.y];
}
"""
_TRANSPOSE_F32_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)


def _get_transpose_f32_kernel():
    global _TRANSPOSE_F32_KERNEL
    if _TRANSPOSE_F32_KERNEL is None:
        import cupy as cp
        _TRANSPOSE_F32_KERNEL = cp.RawKernel(_TRANSPOSE_F32_SRC, "transpose_f32")
    return _TRANSPOSE_F32_KERNEL


def _transpose_to_cm(cand_gpu):
    """Return a (K, n) C-contiguous (== column-major over the original (n,K)) copy of the row-major f32
    ``cand_gpu`` via the coalesced tiled-transpose kernel (6.9x faster than cp.ascontiguousarray(cand.T)
    at the production size; bit-identical bytes). Used ONLY for f32 C-contiguous input; falls back to
    cp.ascontiguousarray(cand.T) for any other dtype/layout or on any kernel failure (bit-identical)."""
    import cupy as cp

    n, K = cand_gpu.shape
    if cand_gpu.dtype != cp.float32 or not cand_gpu.flags.c_contiguous:
        return cp.ascontiguousarray(cand_gpu.T)
    try:
        out = cp.empty((K, n), dtype=cp.float32)
        grid = ((K + 31) // 32, int((n + 31) // 32))
        _get_transpose_f32_kernel()((grid[0], grid[1]), (32, 32),
                                    (cand_gpu, out, np.int64(n), np.int32(K)))
        return out
    except Exception:
        import logging
        logging.getLogger(__name__).debug("tiled transpose kernel failed; cp.ascontiguousarray(.T) fallback", exc_info=True)
        return cp.ascontiguousarray(cand_gpu.T)


def _transpose_cm_to_rm(cm_gpu):
    """Inverse of ``_transpose_to_cm``: return an (n, K) row-major C-contiguous copy of the (K, n) C-order
    ``cm_gpu`` via the coalesced tiled-transpose kernel (the kernel transposes any (R, C) C-order input to
    (C, R) C-order -- here R=K, C=n). f32 only; falls back to ``cp.ascontiguousarray(cm_gpu.T)`` (bit-
    identical) for non-f32 / any kernel failure. Used to bring the column-major fe_materialise result back to
    the (n, K) layout the downstream binning + float D2H expect."""
    import cupy as cp

    Kr, nc = cm_gpu.shape  # (K, n)
    if cm_gpu.dtype != cp.float32 or not cm_gpu.flags.c_contiguous:
        return cp.ascontiguousarray(cm_gpu.T)
    try:
        out = cp.empty((nc, Kr), dtype=cp.float32)             # (n, K)
        # The kernel treats the input as (n_rows=Kr, K_cols=nc); grid.x spans the K_cols (=nc), grid.y the
        # n_rows (=Kr) -- mirror _transpose_to_cm's ((K+31)//32, (n+31)//32) with n:=Kr, K:=nc.
        grid = ((nc + 31) // 32, int((Kr + 31) // 32))
        _get_transpose_f32_kernel()((grid[0], grid[1]), (32, 32),
                                    (cm_gpu, out, np.int64(Kr), np.int32(nc)))
        return out
    except Exception:
        import logging
        logging.getLogger(__name__).debug("tiled cm->rm transpose kernel failed; cp.ascontiguousarray(.T) fallback", exc_info=True)
        return cp.ascontiguousarray(cm_gpu.T)


# COALESCED TILED TRANSPOSE for the int8/int16 DISC CODES (2026-06-24, nsys-driven int8-copy lever). The
# resident-codes noise gate's column-major hist kernel needs the (n,K) int codes as a (K,n) C-order buffer
# and built it with ``cp.ascontiguousarray(d_disc.T)`` -- exactly the uncoalesced strided-copy pattern the
# f32 transpose above replaced. nsys (isolated KTC, F2 100k, one fit) charged that int8 transpose-copy as
# cupy_copy__int8_int8 = 46 calls / 1776ms = 29.5% of ALL GPU-kernel time, run at only ~3-4 GB/s (cupy's
# generic strided copy is uncoalesced on both read+write for a transposed view). This shared-memory TILED
# transpose (32x32 tiles, +1 pad column to kill the shared-bank conflict on the transposed write) reads the
# (n,K) buffer coalesced and writes the (K,n) buffer coalesced -- BIT-IDENTICAL bytes (same values, same
# (K,n) C-order layout the hist kernel already consumes -> counts byte-for-byte unchanged). Falls back to
# cp.ascontiguousarray(disc.T) on any failure. The char body is dtype-agnostic for any 1-byte code (int8);
# a separate int16 kernel covers nbins>128 (the gate accepts itemsize<=2 narrow codes).
_TRANSPOSE_I8_SRC = r"""
extern "C" __global__
void transpose_i8(const signed char* __restrict__ in, signed char* __restrict__ out,
                  const long long n, const int K) {
    __shared__ signed char tile[32][33];           // +1 pad column: no shared-bank conflict on the write
    long long row = (long long)blockIdx.y * 32 + threadIdx.y;   // index into n (rows of the (n,K) input)
    int col = blockIdx.x * 32 + threadIdx.x;                    // index into K
    if (row < n && col < K) tile[threadIdx.y][threadIdx.x] = in[row * (long long)K + col];
    __syncthreads();
    int orow = blockIdx.x * 32 + threadIdx.y;                   // index into K (rows of the (K,n) output)
    long long ocol = (long long)blockIdx.y * 32 + threadIdx.x;  // index into n
    if (orow < K && ocol < n) out[(long long)orow * n + ocol] = tile[threadIdx.x][threadIdx.y];
}
"""
_TRANSPOSE_I16_SRC = _TRANSPOSE_I8_SRC.replace("signed char", "short").replace("transpose_i8", "transpose_i16")
_TRANSPOSE_I8_KERNEL = None   # module-level singletons (lazy-compiled; pickle-safe)
_TRANSPOSE_I16_KERNEL = None


def _get_transpose_int_kernel(itemsize: int):
    global _TRANSPOSE_I8_KERNEL, _TRANSPOSE_I16_KERNEL
    import cupy as cp
    if itemsize == 1:
        if _TRANSPOSE_I8_KERNEL is None:
            _TRANSPOSE_I8_KERNEL = cp.RawKernel(_TRANSPOSE_I8_SRC, "transpose_i8")
        return _TRANSPOSE_I8_KERNEL
    if _TRANSPOSE_I16_KERNEL is None:
        _TRANSPOSE_I16_KERNEL = cp.RawKernel(_TRANSPOSE_I16_SRC, "transpose_i16")
    return _TRANSPOSE_I16_KERNEL


def transpose_codes_to_cm(disc_gpu: object) -> object:
    """Return a (K, n) C-contiguous (== column-major over the original (n,K)) copy of the row-major narrow
    int ``disc_gpu`` (int8 / int16 bin codes) via the coalesced tiled-transpose kernel -- the int analogue
    of ``_transpose_to_cm``, replacing ``cp.ascontiguousarray(disc.T)`` (uncoalesced) for the resident
    noise-gate's column-major hist load. BIT-IDENTICAL bytes. Falls back to ``cp.ascontiguousarray(disc.T)``
    for any non-1/2-byte int / non-C-contiguous input or on any kernel failure (bit-identical either way)."""
    import cupy as cp

    n, K = disc_gpu.shape
    itemsize = disc_gpu.dtype.itemsize
    if itemsize not in (1, 2) or not disc_gpu.flags.c_contiguous:
        return cp.ascontiguousarray(disc_gpu.T)
    try:
        out = cp.empty((K, n), dtype=disc_gpu.dtype)
        grid = ((K + 31) // 32, int((n + 31) // 32))
        _get_transpose_int_kernel(itemsize)((grid[0], grid[1]), (32, 32),
                                            (disc_gpu, out, np.int64(n), np.int32(K)))
        return out
    except Exception:
        import logging
        logging.getLogger(__name__).debug("tiled int-codes transpose kernel failed; cp.ascontiguousarray(.T) fallback", exc_info=True)
        return cp.ascontiguousarray(disc_gpu.T)


def _get_radix_select_kernel(is_f32: bool):
    global _RADIX_SELECT_F64_KERNEL, _RADIX_SELECT_F32_KERNEL
    import cupy as cp
    if is_f32:
        if _RADIX_SELECT_F32_KERNEL is None:
            _RADIX_SELECT_F32_KERNEL = cp.RawKernel(_RADIX_SELECT_F32_SRC, "radix_select_f32")
        return _RADIX_SELECT_F32_KERNEL
    if _RADIX_SELECT_F64_KERNEL is None:
        _RADIX_SELECT_F64_KERNEL = cp.RawKernel(_RADIX_SELECT_F64_SRC, "radix_select_f64")
    return _RADIX_SELECT_F64_KERNEL


# FUSED select+interp for the f64 path (launch-reduction, 2026-06-25): the radix order-statistic select and
# the cupy-'linear' interpolation were two kernels (osv written to global, then radix_interp read it back).
# This combined kernel keeps the R order statistics in shared memory and emits the (ne, K) interior edges
# DIRECTLY -- one launch, no osv global, no second launch. The radix body is byte-for-byte the f64 select; the
# tail reproduces _RADIX_INTERP_SRC exactly (same f64 formula) -> BIT-IDENTICAL edges. Used only by the f64
# _radix_select_interior_edges path; the f32 variants + the cp.percentile fallback keep the two-kernel route.
_RADIX_SELECT_INTERP_F64_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_interp_f64(const double* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R,
                      const long long* __restrict__ bi, const long long* __restrict__ ai,
                      const double* __restrict__ w, const int ne, double* __restrict__ edges){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned long long prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned long long wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ int W;
    __shared__ double osv_sh[MAXR];
    if(tid<R){prefix[tid]=0ULL;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=7;byte>=0;--byte){
        int shift=byte*8;
        unsigned long long hmask=(byte==7)?0ULL:(0xFFFFFFFFFFFFFFFFULL<<((byte+1)*8));
        if(tid==0){int w_=0;for(int r=0;r<R;++r){unsigned long long p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w_;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w_]=p;rank2w[r]=w_;w_++;}else rank2w[r]=f;} W=w_;}
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            double d=data[(long long)col*n+i];unsigned long long u;memcpy(&u,&d,8);
            u=(u&0x8000000000000000ULL)?~u:(u|0x8000000000000000ULL);
            unsigned long long pm=u&hmask;int win=-1;
            for(int q=0;q<Wl;++q)if(wpref[q]==pm){win=q;break;}
            if(win>=0){int dig=(int)((u>>shift)&0xFFULL);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        if(tid==0){for(int r=0;r<R;++r){int w2=rank2w[r];unsigned long long acc=below[r];int chosen=0;long long want=ranks[r];
            for(int b=0;b<256;++b){unsigned long long c=sh[w2*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[r]=acc;prefix[r]=(prefix[r]&hmask)|((unsigned long long)chosen<<shift);}}
        __syncthreads();
    }
    if(tid<R){unsigned long long u=prefix[tid];u=(u&0x8000000000000000ULL)?(u&0x7FFFFFFFFFFFFFFFULL):~u;
        double d;memcpy(&d,&u,8);osv_sh[tid]=d;}
    __syncthreads();
    for(int e=tid;e<ne;e+=nt){
        double ab=osv_sh[bi[e]], aa=osv_sh[ai[e]], ww=w[e]; double diff=aa-ab;
        edges[(long long)e*K+col] = ww<0.5 ? (ab+diff*ww) : (aa-diff*(1.0-ww));
    }
}
"""
_RADIX_SELECT_INTERP_F64_KERNEL = None


# V2 FUSED select+interp f64 (2026-06-27, nvprof-driven). The original radix_select_interp_f64 still runs the
# per-pass window dedup AND the per-rank cumulative scan SERIALLY in tid==0 while the other 1023 threads idle
# (8 byte-passes x R ranks x up-to-256 bins each in one thread). This v2 ports the two parallelisations the
# f32 ``v3`` kernel already proved bit-identical: (a) the data-loop window match becomes a BINARY SEARCH over
# the sorted active-window prefixes (was a divergent linear scan), and (b) the per-rank cumulative-scan +
# prefix advance runs PARALLEL -- thread r owns rank r (disjoint below/prefix slots, own histogram slice) --
# identical result to the serial tid==0 loop. radix_select_interp_f64 was the #1 GPU kernel (33% of GPU time);
# it is on the critical path (each resident twin syncs to read edges). The radix body is byte-for-byte the
# original select (same f64 bit-flip, same histogram, same chosen-digit rule) and the interp tail is identical
# -> BIT-IDENTICAL interior edges. Original kept as the automatic fallback on any compile/launch failure.
_RADIX_SELECT_INTERP_F64_V2_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_interp_f64_v2(const double* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R,
                      const long long* __restrict__ bi, const long long* __restrict__ ai,
                      const double* __restrict__ w, const int ne, double* __restrict__ edges){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned long long prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned long long wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ unsigned long long wsort[MAXR];   // wpref sorted ascending (distinct active window prefixes)
    __shared__ int wsort2w[MAXR];                // sorted position -> ORIGINAL window index (into wpref / sh)
    __shared__ int W;
    __shared__ double osv_sh[MAXR];
    if(tid<R){prefix[tid]=0ULL;below[tid]=0ULL;}
    __syncthreads();
    for(int byte=7;byte>=0;--byte){
        int shift=byte*8;
        unsigned long long hmask=(byte==7)?0ULL:(0xFFFFFFFFFFFFFFFFULL<<((byte+1)*8));
        if(tid==0){int w_=0;for(int r=0;r<R;++r){unsigned long long p=prefix[r]&hmask;int f=-1;
            for(int q=0;q<w_;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w_]=p;rank2w[r]=w_;w_++;}else rank2w[r]=f;} W=w_;
            for(int q=0;q<w_;++q){wsort[q]=wpref[q];wsort2w[q]=q;}
            for(int a=1;a<w_;++a){unsigned long long kv=wsort[a];int kw=wsort2w[a];int b=a-1;
                while(b>=0&&wsort[b]>kv){wsort[b+1]=wsort[b];wsort2w[b+1]=wsort2w[b];--b;}
                wsort[b+1]=kv;wsort2w[b+1]=kw;}}
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        for(long long i=tid;i<n;i+=nt){
            double d=data[(long long)col*n+i];unsigned long long u;memcpy(&u,&d,8);
            u=(u&0x8000000000000000ULL)?~u:(u|0x8000000000000000ULL);
            unsigned long long pm=u&hmask;
            int lo=0,hi=Wl;
            while(lo<hi){int mid=(lo+hi)>>1;if(wsort[mid]<pm)lo=mid+1;else hi=mid;}
            if(lo<Wl&&wsort[lo]==pm){int win=wsort2w[lo];
                int dig=(int)((u>>shift)&0xFFULL);atomicAdd(&sh[win*256+dig],1u);}
        }
        __syncthreads();
        // PARALLEL per-rank cumulative scan + prefix advance (identical result to the serial tid==0 loop).
        if(tid<R){int w2=rank2w[tid];unsigned long long acc=below[tid];int chosen=0;long long want=ranks[tid];
            for(int b=0;b<256;++b){unsigned long long c=sh[w2*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}
            below[tid]=acc;prefix[tid]=(prefix[tid]&hmask)|((unsigned long long)chosen<<shift);}
        __syncthreads();
    }
    if(tid<R){unsigned long long u=prefix[tid];u=(u&0x8000000000000000ULL)?(u&0x7FFFFFFFFFFFFFFFULL):~u;
        double d;memcpy(&d,&u,8);osv_sh[tid]=d;}
    __syncthreads();
    for(int e=tid;e<ne;e+=nt){
        double ab=osv_sh[bi[e]], aa=osv_sh[ai[e]], ww=w[e]; double diff=aa-ab;
        edges[(long long)e*K+col] = ww<0.5 ? (ab+diff*ww) : (aa-diff*(1.0-ww));
    }
}
"""
_RADIX_SELECT_INTERP_F64_V2_KERNEL = None


def _get_radix_select_interp_f64_kernel():
    """The fused f64 select+interp kernel. Returns the parallel-per-rank-scan + binary-search v2 (the measured
    win) when it compiles, else the original serial-tid0 kernel. Both produce BIT-IDENTICAL interior edges."""
    global _RADIX_SELECT_INTERP_F64_KERNEL, _RADIX_SELECT_INTERP_F64_V2_KERNEL
    import cupy as cp
    if os.environ.get("MLFRAME_RADIX_F64_V2", "1").strip().lower() in ("1", "true", "on", "yes"):
        if _RADIX_SELECT_INTERP_F64_V2_KERNEL is None:
            try:
                _RADIX_SELECT_INTERP_F64_V2_KERNEL = cp.RawKernel(
                    _RADIX_SELECT_INTERP_F64_V2_SRC, "radix_select_interp_f64_v2")
            except Exception:
                import logging
                logging.getLogger(__name__).debug("f64 v2 fused kernel compile failed; original", exc_info=True)
                _RADIX_SELECT_INTERP_F64_V2_KERNEL = False
        if _RADIX_SELECT_INTERP_F64_V2_KERNEL:
            return _RADIX_SELECT_INTERP_F64_V2_KERNEL
    if _RADIX_SELECT_INTERP_F64_KERNEL is None:
        _RADIX_SELECT_INTERP_F64_KERNEL = cp.RawKernel(_RADIX_SELECT_INTERP_F64_SRC, "radix_select_interp_f64")
    return _RADIX_SELECT_INTERP_F64_KERNEL


def _get_radix_select_f32_v3_kernel():
    """Lazy-compiled (pickle-safe, module-level singleton) parallel-per-rank-scan f32 variant (the real
    Lever-4 win). Bit-identical order statistics to ``radix_select_f32`` / ``radix_select_f32_bsearch``;
    parallelises the per-pass cumulative scan across the R ranks (was serial in tid==0) -> ~2x at the
    production shape (n=100k, K=583, R=38). Carries the bsearch binary-search window-match."""
    global _RADIX_SELECT_F32_V3_KERNEL
    import cupy as cp
    if _RADIX_SELECT_F32_V3_KERNEL is None:
        _RADIX_SELECT_F32_V3_KERNEL = cp.RawKernel(
            _RADIX_SELECT_F32_V3_SRC, "radix_select_f32_v3")
    return _RADIX_SELECT_F32_V3_KERNEL


def _get_radix_select_f32_bsearch_kernel():
    """Lazy-compiled (pickle-safe, module-level singleton) binary-search window-match f32 variant
    (Lever C). Bit-identical order statistics to ``radix_select_f32``; replaces the divergent linear
    per-row window scan with a branchless binary search over the sorted active-window prefixes."""
    global _RADIX_SELECT_F32_BSEARCH_KERNEL
    import cupy as cp
    if _RADIX_SELECT_F32_BSEARCH_KERNEL is None:
        _RADIX_SELECT_F32_BSEARCH_KERNEL = cp.RawKernel(
            _RADIX_SELECT_F32_BSEARCH_SRC, "radix_select_f32_bsearch")
    return _RADIX_SELECT_F32_BSEARCH_KERNEL


# Lever C variant select (2026-06-23): which f32 radix-select kernel to launch -- the base linear-scan
# ``radix_select_f32`` or the binary-search ``radix_select_f32_bsearch``. KTC-tuned per host (the sweep
# probe forces a choice via this override; the production launch reads the per-host tuned variant). Both
# produce BIT-IDENTICAL order statistics (the binary search only changes HOW a row finds its window) -> the
# sweep ranks by WALL only and the base stays the fallback on any compile/launch failure. f64 is unaffected.
_RADIX_F32_VARIANT_OVERRIDE = None  # "linear" / "bsearch" set by the KTC sweep probe; None -> per-host KTC


def _resolve_radix_f32_variant(n: int) -> str:
    """Which f32 radix-select variant to launch: the sweep override when active, else the per-host KTC-tuned
    choice ("linear" / "bsearch" / "v3"; falls back to "v3" -- the measured-fastest default -- on lookup
    failure). All three are bit-identical in the produced order statistics."""
    if _RADIX_F32_VARIANT_OVERRIDE is not None:
        return str(_RADIX_F32_VARIANT_OVERRIDE)
    try:
        from ._gpu_resident_radix_ktc import radix_select_f32_variant
        return str(radix_select_f32_variant(int(n)))
    except Exception:
        return "v3"


def _get_radix_select_f32_dispatch(n: int):
    """The f32 radix-select kernel for this size/host (Lever C/4 dispatch). Returns the parallel-per-rank-scan
    ``v3`` variant (the measured-fastest, ~2x) when selected and it compiles, else the binary-search variant,
    else the base linear-scan kernel; each falls back to the previous on a compile failure -- all three
    produce bit-identical order statistics."""
    variant = _resolve_radix_f32_variant(n)
    if variant == "v3":
        try:
            return _get_radix_select_f32_v3_kernel()
        except Exception:
            import logging
            logging.getLogger(__name__).debug("v3 radix kernel compile failed; bsearch fallback", exc_info=True)
            variant = "bsearch"
    if variant == "bsearch":
        try:
            return _get_radix_select_f32_bsearch_kernel()
        except Exception:
            import logging
            logging.getLogger(__name__).debug("bsearch radix kernel compile failed; linear fallback", exc_info=True)
    return _get_radix_select_kernel(True)


def fe_gpu_radix_edges_enabled() -> bool:
    """Whether the rank-EXACT sort-free radix-select quantile edges replace cp.percentile's full sort.
    ON unless ``MLFRAME_FE_GPU_RADIX_EDGES`` is explicitly falsy (it is bit-identical to cp.percentile in
    the produced codes -- verified maxdiff 0 -- and faster, the win growing with n; cp.percentile stays
    the gated exact fallback one env flip away and the automatic fallback on any kernel failure)."""
    return os.environ.get("MLFRAME_FE_GPU_RADIX_EDGES", "1").strip().lower() in ("1", "true", "on", "yes")


# (n, nbins) fully determine the radix interp gather-indices (bi/ai) and weight (w) -- they are derived
# only from np.linspace(0,100,nbins+1) and n, NOT from the candidate data -- so they are identical for every
# chunk/pair of a fit. Cache the (tiny, (nbins-1,)) device vectors keyed on (n, nbins) to drop the per-chunk
# tiny-H2D allocs (the cupy._core.core.array hotspot). Module-level dict -> not part of the MRMR instance
# pickle (mirrors the other resident-kernel singletons in this module). (n, nbins) take <=2-3 values per fit.
_RADIX_INTERP_CACHE: dict = {}


_RADIX_INTERP_SRC = r"""
extern "C" __global__
void radix_interp(const double* __restrict__ osv, const long long* __restrict__ bi,
                  const long long* __restrict__ ai, const double* __restrict__ w,
                  const int K, const int ne, double* __restrict__ edges) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)ne * K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    int e = (int)(t / (long long)K);
    double ab = osv[bi[e] * (long long)K + col];
    double aa = osv[ai[e] * (long long)K + col];
    double ww = w[e];
    double diff = aa - ab;
    edges[t] = ww < 0.5 ? (ab + diff * ww) : (aa - diff * (1.0 - ww));
}
"""
_RADIX_INTERP_KERNEL = None


def _get_radix_interp_kernel():
    global _RADIX_INTERP_KERNEL
    if _RADIX_INTERP_KERNEL is None:
        import cupy as cp
        _RADIX_INTERP_KERNEL = cp.RawKernel(_RADIX_INTERP_SRC, "radix_interp")
    return _RADIX_INTERP_KERNEL


def _radix_select_interior_edges(cand_gpu, nbins: int, cm_hint=None):
    """Return the (nbins-1, K) INTERIOR quantile edges of the resident (n, K) cupy ``cand_gpu`` via the
    sort-free radix-select kernel + cupy's exact 'linear' interpolation (reproduced in float64). The edges
    are BIT-IDENTICAL (in the resulting codes) to ``cp.percentile(cand, linspace(0,100,nbins+1))[1:-1]``.
    Returns ``None`` if the radix path is inapplicable (R over the kernel cap, shared-mem over the device
    limit) so the caller uses the cp.percentile fallback. ``cand_gpu`` must be C-contiguous (n, K).

    ``cm_hint`` (LAUNCH-FUSION 2026-06-27): an OPTIONAL (K, n) C-order column-major view of the SAME data as
    ``cand_gpu`` (e.g. the materialise kernel's pre-transpose cm buffer). When supplied, the internal
    ``_transpose_to_cm(cand_gpu)`` is SKIPPED -- the chunk path already produced this exact buffer (the
    materialise kernel wrote cm then transposed it to the rm ``cand_gpu``), so the round-trip transpose pair
    (rm->cm here + cm->rm in materialise) collapses to ONE. The order statistics read the same values ->
    BIT-IDENTICAL edges. Validated shape (K, n) == (cand_gpu.shape[1], cand_gpu.shape[0]); any mismatch
    ignores the hint and transposes (safe)."""
    import cupy as cp

    n, K = cand_gpu.shape
    is_f32 = cand_gpu.dtype == cp.float32
    # ALL of the order-statistic geometry (ranks, R, shared-mem gate, the cupy 'linear' interp gathers bi/ai/w,
    # the device rank vector ranks_g) depends ONLY on (n, nbins) -- NOT the candidate data -- so compute it ONCE
    # per (n, nbins) and cache it. Was recomputed on EVERY per-candidate call (~11.6k x on the FE pair scan): the
    # np.unique host sort (cProfile 2026-07-01: 12,149 calls / 4.05s = the top host-CPU cost on this path), the
    # linspace/floor, the per-call device-attribute lookup, AND the ranks H2D -- all data-independent. A cached
    # None records the "radix inapplicable" verdict (R over the kernel cap / shared-mem over the device limit) so
    # the caller's cp.percentile fallback is taken without recomputing. Selection-IDENTICAL: same ranks -> same
    # order statistics -> bit-identical edges.
    _ik = (int(n), int(nbins))
    if _ik not in _RADIX_INTERP_CACHE:
        # cupy 'linear' positions for the nbins-1 interior quantiles (q in (0,1)), float64 throughout.
        qfr = np.linspace(0.0, 100.0, int(nbins) + 1)[1:-1] / 100.0   # (nbins-1,) fractions
        idx = qfr * (n - 1)
        bel = np.floor(idx).astype(np.int64)
        abv = np.minimum(bel + 1, n - 1)
        uniq = np.unique(np.concatenate([bel, abv]))                   # the order-statistic ranks to extract
        R = int(uniq.size)
        # shared-mem budget: R*256 uint32 histogram (host gate vs the device's per-block shared limit). The gate
        # must reserve the kernels' STATIC __shared__ footprint too -- dynamic ``shmem`` ALONE near the limit, plus
        # the static arrays, overflows the per-block budget and cuLaunchKernel raises CUDA_ERROR_INVALID_VALUE.
        shmem = R * 256 * 4
        try:
            dev = cp.cuda.Device()
            sh_limit = int(dev.attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        except Exception:
            sh_limit = 48 * 1024
        if R > _RADIX_SELECT_MAXR or shmem > sh_limit - _RADIX_STATIC_SHARED_BYTES:
            _RADIX_INTERP_CACHE[_ik] = None                           # radix path inapplicable for this (n, nbins)
        else:
            # cupy 'linear' interp gather indices + weight (bi/ai/w) -- needed BEFORE the kernel for the fused f64
            # path, which interpolates in-kernel.
            ranks_g = cp.asarray(uniq, dtype=cp.int64)
            pos = {int(r): i for i, r in enumerate(uniq)}
            bi = cp.asarray(np.asarray([pos[int(b)] for b in bel], dtype=np.int64))
            ai = cp.asarray(np.asarray([pos[int(a)] for a in abv], dtype=np.int64))
            w = cp.asarray(np.ascontiguousarray(idx - bel))           # float64 weight_above = idx - floor(idx)
            _RADIX_INTERP_CACHE[_ik] = (bi, ai, w, ranks_g, int(R), int(shmem))
    _ic = _RADIX_INTERP_CACHE[_ik]
    if _ic is None:
        return None                                                   # cached: radix inapplicable -> cp.percentile
    bi, ai, w, ranks_g, R, shmem = _ic
    # COLUMN-MAJOR input (nvprof-driven, 2026-06-20): one block/column previously read data[i*K+col] from
    # the (n,K) row-major buffer -> stride-K, gld_efficiency 12.5% (1/8 coalesced) on the dominant n-loop
    # (4 byte-passes x n reads). Transpose to (K,n) C-order so consecutive threads read consecutive memory
    # (data[col*n+i]) -- one transpose pass buys ~8x coalescing across the 4 passes. Values unchanged ->
    # bit-identical order statistics. (The bin_codes step still uses the original (n,K) cand_gpu.)
    # Reuse the materialise kernel's pre-transpose (K, n) cm buffer when handed in (launch-fusion: skip the
    # rm->cm transpose that exactly inverts materialise's cm->rm). Validate shape/contiguity/dtype; else transpose.
    if (cm_hint is not None and cm_hint.shape == (K, n) and cm_hint.flags.c_contiguous
            and cm_hint.dtype == cand_gpu.dtype):
        data_cm = cm_hint
    else:
        data_cm = _transpose_to_cm(cand_gpu)   # (K, n) C-order = column-major (coalesced tiled-transpose kernel)
    threads = _resolve_radix_threads(n)    # Lever B: per-host KTC-tuned block size (bit-identical edges)
    ne_rows = int(bi.shape[0])
    if not is_f32:
        # FUSED select+interp (launch-reduction): the f64 radix select keeps its R order statistics in shared
        # memory and emits the (ne, K) interior edges directly -- ONE launch, no osv global, no separate
        # radix_interp launch. Bit-identical to the two-kernel f64 path (same select body + same f64 interp).
        try:
            edges = cp.empty((ne_rows, K), dtype=cp.float64)
            _get_radix_select_interp_f64_kernel()((K,), (threads,),
                (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R),
                 bi, ai, cp.ascontiguousarray(w), np.int32(ne_rows), edges), shared_mem=shmem)
            return edges                  # (nbins-1, K) float64
        except Exception:
            import logging
            logging.getLogger(__name__).debug("fused f64 select+interp failed; two-kernel path", exc_info=True)
    osv = cp.empty((R, K), dtype=cand_gpu.dtype)
    # Lever C: dispatch the f32 path to the binary-search window-match variant where the per-host KTC selects
    # it (bit-identical order statistics, less warp divergence; base linear-scan = fallback). f64 unchanged.
    ker = _get_radix_select_f32_dispatch(n) if is_f32 else _get_radix_select_kernel(is_f32)
    ker((K,), (threads,),
        (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), osv), shared_mem=shmem)
    # FUSED interpolation (launch-reduction, 2026-06-25): the two fancy-index gathers + diff + cp.where
    # linear-interp collapse into ONE RawKernel that reads the two order-statistic rows from ``osv`` and writes
    # the (nbins-1, K) interior edges directly. Bit-identical to the cupy 'linear' interp (same f64 formula).
    osv64 = osv if osv.dtype == cp.float64 else osv.astype(cp.float64)
    edges = cp.empty((ne_rows, K), dtype=cp.float64)
    _ker_interp = _get_radix_interp_kernel()
    _threads = 256
    _total = ne_rows * K
    _ker_interp(((_total + _threads - 1) // _threads,), (_threads,),
                (osv64, bi, ai, cp.ascontiguousarray(w), np.int32(K), np.int32(ne_rows), edges))
    return edges                          # (nbins-1, K) float64


def _radix_quantiles(cand_gpu, q_fracs):
    """Per-column quantiles at ``q_fracs`` (fractions in [0,1]) over the resident (n, K) cupy ``cand_gpu``
    via the SAME sort-free radix-select kernel + cupy 'linear' interpolation as ``_radix_select_interior_edges``.
    Returns ``(len(q_fracs), K)`` float64, BIT-IDENTICAL to ``cp.percentile(cand, [q*100 ...], axis=0)``
    (and to ``cp.median`` at q=0.5 -- even-n linear interp == mean of the two middle order statistics).
    Returns ``None`` when the radix path is inapplicable (R over the kernel cap / shared-mem over the device
    limit) so the caller takes the cp.percentile fallback. ``cand_gpu`` must be a finite (n, K) cupy array.

    This generalises ``_radix_select_interior_edges`` (which is locked to ``linspace(0,100,nbins+1)`` binning
    edges) to ARBITRARY quantiles so the robust-scale / heavy-tail path (median, MAD-median, q25/q75) reuses
    the radix machinery instead of cp.median/cp.percentile's full per-call sort -- one select+interp launch
    pair per quantile-set vs the sort's ~6-8."""
    import cupy as cp

    cand_gpu = cp.ascontiguousarray(cand_gpu)
    n, K = cand_gpu.shape
    is_f32 = cand_gpu.dtype == cp.float32
    qfr = np.asarray(q_fracs, dtype=np.float64)
    idx = qfr * (n - 1)
    bel = np.floor(idx).astype(np.int64)
    abv = np.minimum(bel + 1, n - 1)
    uniq = np.unique(np.concatenate([bel, abv]))                    # order-statistic ranks to extract
    R = int(uniq.size)
    if R > _RADIX_SELECT_MAXR:
        return None
    shmem = R * 256 * 4
    try:
        dev = cp.cuda.Device()
        sh_limit = int(dev.attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
    except Exception:
        sh_limit = 48 * 1024
    # Reserve the kernels' STATIC __shared__ footprint (see _RADIX_STATIC_SHARED_BYTES): dynamic + static must
    # fit the per-block limit, else cuLaunchKernel raises CUDA_ERROR_INVALID_VALUE.
    if shmem > sh_limit - _RADIX_STATIC_SHARED_BYTES:
        return None
    ranks_g = cp.asarray(uniq, dtype=cp.int64)
    data_cm = _transpose_to_cm(cand_gpu)                            # (K, n) coalesced (same as edges path)
    threads = _resolve_radix_threads(n)
    pos = {int(r): i for i, r in enumerate(uniq)}
    bi = cp.asarray(np.asarray([pos[int(b)] for b in bel], dtype=np.int64))
    ai = cp.asarray(np.asarray([pos[int(a)] for a in abv], dtype=np.int64))
    w = cp.ascontiguousarray(cp.asarray(idx - bel))                 # float64 weight_above
    nq = int(qfr.size)
    out = cp.empty((nq, K), dtype=cp.float64)
    if not is_f32:
        # FUSED select+interp (launch-reduction): the f64 radix select keeps its order statistics in shared
        # memory and emits the nq interior quantiles directly -- ONE launch, no osv global, no separate interp.
        try:
            _get_radix_select_interp_f64_kernel()((K,), (threads,),
                (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), bi, ai, w, np.int32(nq), out),
                shared_mem=shmem)
            return out
        except Exception:
            import logging
            logging.getLogger(__name__).debug("fused f64 radix_quantiles failed; two-kernel path", exc_info=True)
    osv = cp.empty((R, K), dtype=cand_gpu.dtype)
    ker = _get_radix_select_f32_dispatch(n) if is_f32 else _get_radix_select_kernel(is_f32)
    ker((K,), (threads,),
        (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), osv), shared_mem=shmem)
    osv64 = osv if osv.dtype == cp.float64 else osv.astype(cp.float64)
    _ker = _get_radix_interp_kernel()
    t = 256
    total = nq * K
    _ker(((total + t - 1) // t,), (t,), (osv64, bi, ai, w, np.int32(K), np.int32(nq), out))
    return out                             # (len(q_fracs), K) float64


# --- Carve re-exports (sibling pattern): the fused-binning / resident-discretize block ->
# _gpu_resident_discretize.py, the materialise / operand-table / host-fast-path block ->
# _gpu_resident_materialise.py (both carved VERBATIM under the 1k ceiling). Rebind EVERY moved name
# (public AND underscore-private) into THIS namespace so every existing ``from ._gpu_resident_select import X``
# and ``_gpu_resident_select.X`` path still resolves byte-for-byte. At the BOTTOM so the siblings' top-level
# back-imports (from ._gpu_resident_fe import ...) resolve during the partial-init import chain.
from . import _gpu_resident_discretize as _grd  # noqa: E402
from . import _gpu_resident_materialise as _grm  # noqa: E402
for _m in (_grd, _grm):
    for _n in dir(_m):
        if not _n.startswith("__") and _n not in globals():
            globals()[_n] = getattr(_m, _n)
del _m, _n
