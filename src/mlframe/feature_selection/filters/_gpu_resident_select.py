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
import threading
from collections import OrderedDict

import numpy as np

# Parent-defined names this block consumes. Imported at module top: the PARENT does
# ``from ._gpu_resident_select import *`` at its BOTTOM (after all these names are defined), so re-entering
# the partially-initialised parent here always finds them -- no circular-import hazard.
from ._gpu_resident_fe import (
    _gpu_k_chunk,
    _quantile_levels_dev,
    _stash_deferred_host_fill,
    _stash_resident_codes,
    _unary_apply,
    clear_resident_codes_handoff,
    fe_gpu_defer_host_codes_enabled,
    fe_gpu_resident_codes_enabled,
)


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


def _radix_select_interior_edges(cand_gpu, nbins: int):
    """Return the (nbins-1, K) INTERIOR quantile edges of the resident (n, K) cupy ``cand_gpu`` via the
    sort-free radix-select kernel + cupy's exact 'linear' interpolation (reproduced in float64). The edges
    are BIT-IDENTICAL (in the resulting codes) to ``cp.percentile(cand, linspace(0,100,nbins+1))[1:-1]``.
    Returns ``None`` if the radix path is inapplicable (R over the kernel cap, shared-mem over the device
    limit) so the caller uses the cp.percentile fallback. ``cand_gpu`` must be C-contiguous (n, K)."""
    import cupy as cp

    n, K = cand_gpu.shape
    is_f32 = cand_gpu.dtype == cp.float32
    # cupy 'linear' positions for the nbins-1 interior quantiles (q in (0,1)), float64 throughout.
    qfr = np.linspace(0.0, 100.0, int(nbins) + 1)[1:-1] / 100.0   # (nbins-1,) fractions
    idx = qfr * (n - 1)
    bel = np.floor(idx).astype(np.int64)
    abv = np.minimum(bel + 1, n - 1)
    uniq = np.unique(np.concatenate([bel, abv]))                   # the order-statistic ranks to extract
    R = int(uniq.size)
    if R > _RADIX_SELECT_MAXR:
        return None
    # shared-mem budget: R*256 uint32 histogram (host gate vs the device's per-block shared limit).
    shmem = R * 256 * 4
    try:
        dev = cp.cuda.Device()
        sh_limit = int(dev.attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
    except Exception:
        sh_limit = 48 * 1024
    if shmem > sh_limit:
        return None
    ranks_g = cp.asarray(uniq, dtype=cp.int64)
    # COLUMN-MAJOR input (nvprof-driven, 2026-06-20): one block/column previously read data[i*K+col] from
    # the (n,K) row-major buffer -> stride-K, gld_efficiency 12.5% (1/8 coalesced) on the dominant n-loop
    # (4 byte-passes x n reads). Transpose to (K,n) C-order so consecutive threads read consecutive memory
    # (data[col*n+i]) -- one transpose pass buys ~8x coalescing across the 4 passes. Values unchanged ->
    # bit-identical order statistics. (The bin_codes step still uses the original (n,K) cand_gpu.)
    data_cm = _transpose_to_cm(cand_gpu)   # (K, n) C-order = column-major (coalesced tiled-transpose kernel)
    threads = _resolve_radix_threads(n)    # Lever B: per-host KTC-tuned block size (bit-identical edges)
    # cupy 'linear' interp gather indices + weight (bi/ai/w), cached per (n, nbins) -- needed BEFORE the kernel
    # for the fused f64 path, which interpolates in-kernel.
    _ik = (int(n), int(nbins))
    _ic = _RADIX_INTERP_CACHE.get(_ik)
    if _ic is None:
        pos = {int(r): i for i, r in enumerate(uniq)}
        bi = cp.asarray(np.asarray([pos[int(b)] for b in bel], dtype=np.int64))
        ai = cp.asarray(np.asarray([pos[int(a)] for a in abv], dtype=np.int64))
        w = cp.asarray(np.ascontiguousarray(idx - bel))   # float64 weight_above = idx - floor(idx)
        _RADIX_INTERP_CACHE[_ik] = (bi, ai, w)
    else:
        bi, ai, w = _ic
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
    if shmem > sh_limit:
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


# FUSED PER-COLUMN BINNING (2026-06-20, nvprof-driven). The per-column ``for j in range(K): out[:,j] =
# cp.searchsorted(edges[:,j], col, 'right')`` loop fired K separate searchsorted launches PLUS K int64->
# int32 cast-copies (searchsorted returns int64, ``out`` is int32). nvprof on the n=100k/300k binning path:
# cupy_copy__int64_int32 = 19.2% of GPU time (2304 calls) + cupy_searchsorted_kernel = 11.7% (2304 calls)
# -- ~31% of GPU time in launch overhead + a needless dtype cast. This ONE kernel bins the whole (n,K)
# matrix: each thread takes one element, binary-searches its column's nbins-1 interior edges (upper_bound
# = count of edges <= value = EXACTLY cp.searchsorted(.., side='right')), and writes the int32 code
# directly -- coalesced cand/out (row-major) + coalesced strided edge reads (consecutive threads = adjacent
# columns). BIT-IDENTICAL to the per-column searchsorted; one launch, no int64 intermediate.
#
# coalescing-audit (2026-06-23): ALREADY COALESCED -- at floor. CUDA-event A/B at the production shape
# (n=100k, K=583, nbins=10, GTX 1050 Ti): 11.96ms (thread ``idx`` reads cand[idx] coalesced, col=idx%K, and
# writes out[idx] coalesced). A (K,n) column-major-input variant was tried (to mirror the materialise/radix
# wins): 70.3ms = 5.9x SLOWER -- the (n,K) row-major OUTPUT write out[row*K+col] becomes stride-K
# uncoalesced when threads are laid out column-major. The row-major layout is the coalesced one for BOTH the
# value read and the code write here; do not re-audit.
# Edges are ALWAYS float64 (cp.percentile and the radix-select both produce f64 edges). The value is
# promoted to double for the compare -- EXACTLY what cp.searchsorted(f64_edges, f32_value) does (it
# upcasts the value to the edges' dtype). Comparing in the value's f32 instead would 1-off at boundaries
# (a downcast of the f64 edge loses precision) -- the bug that broke bit-identity on the first cut.
#
# bench-attempt-rejected (2026-06-27): nsys re-profile (iter 2) made bin_codes_f32 the #3 GPU kernel (171ms/
# 12 inst); nvprof showed gld_efficiency=51.7% (the divergent f64 edge reads drag the f32 cand read down)
# while gst=100%, achieved_occupancy=0.92, DRAM ~38GB/s (40% of the card's ~96GB/s). Two BIT-IDENTICAL
# (maxdiff 0) alternatives to lift gld, both SLOWER at the production shape (n=100k, K=583, nbins=10, GTX
# 1050 Ti, interleaved-min >=20 reps): (a) unrolled LINEAR scan of all ne edges (uniform coalesced row order,
# no branch divergence) 11.6->16.1ms = SLOWER (ne=9 extra edge reads outweigh the divergence saving); (b)
# stage the whole (ne,K) edge table (~40KB) into SHARED so divergent edge reads hit shared not global
# 11.6->82.7ms = 7x SLOWER (40KB shared caps the block to 1/SM, occupancy collapses). The gld inefficiency is
# the intrinsic f64-vs-f32 mixed-width access of per-element binning; lifting it costs more than it saves on
# Pascal. VERDICT: at its practical floor for this card.
_BIN_CODES_SRC = r"""
extern "C" __global__
void bin_codes_TYPENAME(const TYPE* __restrict__ cand, const double* __restrict__ edges,
                        const long long n, const int K, const int ne, int* __restrict__ out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (idx >= total) return;
    int col = (int)(idx % (long long)K);
    double v = (double)cand[idx];
    int lo = 0, hi = ne;                       // upper_bound over this column's interior edges
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (edges[(long long)mid * K + col] <= v) lo = mid + 1; else hi = mid;
    }
    out[idx] = lo;                             // = #(edges <= v) = searchsorted(.., 'right')
}
"""

_BIN_CODES_KERNELS: dict = {}


def _get_bin_codes_kernel(dtype):
    """Lazy-compiled (pickle-safe, module-level cache) fused binning RawKernel for f32 / f64."""
    import cupy as cp

    key = "f64" if dtype == cp.float64 else "f32"
    k = _BIN_CODES_KERNELS.get(key)
    if k is None:
        ctype = "double" if key == "f64" else "float"
        src = _BIN_CODES_SRC.replace("TYPENAME", key).replace("TYPE", ctype)
        k = cp.RawKernel(src, "bin_codes_" + key)
        _BIN_CODES_KERNELS[key] = k
    return k


def _searchsorted_codes(cand_gpu, interior_edges):
    """Bin (n,K) ``cand_gpu`` against per-column ascending ``interior_edges`` (ne,K) -> int32 (n,K) codes,
    code = #(interior edges <= value) (== per-column cp.searchsorted side='right'). One fused kernel
    launch (no K searchsorted launches, no int64->int32 cast). Falls back to the per-column loop on any
    kernel failure -- bit-identical either way."""
    import cupy as cp

    n, K = cand_gpu.shape
    try:
        # cand_gpu is already C-contiguous f32 on the production path (RawKernel cp.empty output / cp.asarray
        # of a C-contiguous host slice); cp.ascontiguousarray would still alloc+copy the whole (n,K) matrix
        # (the nvprof cupy_copy__float32_float32 hotspot, 19.7%). The kernel only needs C-order memory -> reuse
        # the buffer when already contiguous (bit-identical bytes); a strided view still gets the safety copy.
        cand_c = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
        edges_c = cp.ascontiguousarray(interior_edges, dtype=cp.float64)  # edges f64 (match cp.searchsorted promotion)
        ne = int(edges_c.shape[0])
        out = cp.empty((n, K), dtype=cp.int32)
        total = n * K
        threads = 256
        blocks = (total + threads - 1) // threads
        _get_bin_codes_kernel(cand_c.dtype)(
            (blocks,), (threads,),
            (cand_c, edges_c, np.int64(n), np.int32(K), np.int32(ne), out),
        )
        return out
    except Exception:
        import logging
        logging.getLogger(__name__).debug("fused bin-codes kernel failed; per-column searchsorted fallback", exc_info=True)
        out = cp.empty((n, K), dtype=cp.int32)
        ec = cp.ascontiguousarray(interior_edges)
        for j in range(K):
            out[:, j] = cp.searchsorted(ec[:, j], cand_gpu[:, j], side="right")
        return out


def _gpu_resident_discretize_codes(cand_gpu, nbins: int):
    """Quantile-bin a RESIDENT (n, K) cupy candidate matrix to ordinal codes ON the GPU. Mirrors
    ``discretize_2d_array_cuda`` -- ``cp.percentile(.., linspace(0,100,nbins+1), axis=0)`` for per-column
    edges + per-column ``cp.searchsorted(edges[1:-1], col, side='right')`` -- but keeps the input + output
    on-device (no H2D of the big candidate matrix, no D2H of codes here), so it chains gen -> discretize ->
    noise-gate without round-trips. Returns a cupy int32 (n, K) codes array (resident).

    DTYPE: the percentile + searchsorted run in the INPUT's native dtype by default -- so the float32 FE
    candidate buffer stays float32 (no up-cast; float32 halves the bandwidth of the dominant full sort
    cp.percentile does and preserves the FE selection, the acceptance bar) while the float64 grand-fused
    MI path stays float64 (bit-identical). ``MLFRAME_FE_GPU_BINNING_DTYPE=float64`` forces the exact f64
    path host-wide (bit-identical to the CPU ``discretize_2d_quantile_batch``, whose ``np.percentile``
    upcasts float32 to float64); ``=float32`` forces f32."""
    import cupy as cp

    # Bin in the input's NATIVE dtype by default (the float32 FE candidate buffer stays float32 -- no
    # up-cast, half the sort bandwidth; the float64 grand-fused MI path stays float64 -- bit-identical).
    # MLFRAME_FE_GPU_BINNING_DTYPE forces a specific working dtype (float64 = the exact CPU-parity fallback).
    forced = os.environ.get("MLFRAME_FE_GPU_BINNING_DTYPE", "").strip().lower()
    if forced in ("float64", "f64", "double"):
        work = cp.float64
    elif forced in ("float32", "f32", "single"):
        work = cp.float32
    else:
        work = cand_gpu.dtype
    if cand_gpu.dtype != work:
        cand_gpu = cand_gpu.astype(work, copy=False)
    n, K = cand_gpu.shape

    # RANK-EXACT SORT-FREE EDGES (roadmap #2): extract just the nbins-1 interior quantile edges via the
    # radix-select kernel instead of cp.percentile's full sort. Bit-identical codes (verified maxdiff 0),
    # faster (win grows with n). Returns None -> cp.percentile fallback (R over cap / shared-mem over the
    # device limit); any kernel exception also falls back. cp.percentile's interior edges are bin_edges[1:-1].
    if fe_gpu_radix_edges_enabled() and n > 0:
        try:
            # Already C-contiguous here (see _searchsorted_codes note); _radix_select_interior_edges does its
            # OWN coalescing transpose internally and only needs C-order input, so skip the redundant full
            # (n,K) f32 copy when contiguous (bit-identical edges -> codes). The KEEP transpose stays inside it.
            _cand_c = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
            interior = _radix_select_interior_edges(_cand_c, int(nbins))
        except Exception:
            import logging
            logging.getLogger(__name__).debug("radix-select edges failed; cp.percentile fallback", exc_info=True)
            interior = None
        if interior is not None:
            return _searchsorted_codes(cand_gpu, interior)

    qs = _quantile_levels_dev(cp, nbins, work)
    if K == 1:
        # CUPY BUG GUARD: cp.percentile(X, axis=0) returns WRONG edges for a single-column (n, 1) array
        # (verified maxdiff ~23 vs numpy; multi-column is exact). A K==1 chunk occurs whenever the last
        # candidate block holds one column, which would silently corrupt that column's codes (breaking the
        # discretize bit-identity). Ravel to 1D where cp.percentile is correct, then restore the shape.
        bin_edges = cp.percentile(cand_gpu.ravel(), qs).reshape(-1, 1)  # (nbins+1, 1)
    else:
        bin_edges = cp.percentile(cand_gpu, qs, axis=0)  # (nbins+1, K)
    return _searchsorted_codes(cand_gpu, bin_edges[1:-1])


# CHUNK-MATERIALISE CUDA RawKernel (2026-06-20). The FE chunk path's #1 CPU hotspot is
# ``_materialise_chunk_njit`` -- it builds the (n, K) float32 candidate matrix by gathering strided
# operand columns ``tv[r, ai]`` / ``tv[r, bi]`` out of a row-major operand table and applying the
# binary op-code table (mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise
# ._NJIT_BINARY_OP_CODES). It is MEMORY-BANDWIDTH bound on those gathers, not compute. This kernel does
# the IDENTICAL work on the GPU: each thread owns one (row, candidate) cell, gathers its two operand
# columns by op-code index, applies the binary op, scrubs non-finite -> 0, and writes float32 row-major.
#
# BIT-IDENTICAL to ``_materialise_chunk_njit``: operands are read as float32 (the ``tv`` dtype); mul/add/
# sub/abs_diff are plain float32 ops; max/min/signed propagate NaN exactly (``a+b`` when either is NaN);
# div (op 3) and ratio_abs (op 8) are FLOAT64-PROMOTED then cast back to float32 (matching the njit
# kernel's ``np.float32(np.float64(a)/...)`` -- numba/numpy promote the float64 ``1e-9`` / ``1.0``
# literals); the final nan_to_num(nan=0, +-inf=0) is the same predicate. The op-code numbering is the
# njit table: 0=mul 1=add 2=sub 3=div 4=max 5=min 6=abs_diff 7=signed 8=ratio_abs. ``tv`` is the
# (n, n_operands) row-major float32 operand table; the kernel addresses operand column ``c`` of row
# ``i`` via ``tv[i*n_operands + c]`` (so NO transpose is needed -- it mirrors the njit ``tv[r, ai]``).
_FE_MATERIALISE_SRC = r"""
extern "C" __global__
void fe_materialise(const float* __restrict__ tv,
                    const long long* __restrict__ a_cols,
                    const long long* __restrict__ b_cols,
                    const signed char* __restrict__ ops,
                    const long long n, const long long n_operands, const int K,
                    float* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    int k = (int)(tid % (long long)K);
    long long i = tid / (long long)K;
    long long ai = a_cols[k];
    long long bi = b_cols[k];
    float a = tv[i * n_operands + ai];
    float b = tv[i * n_operands + bi];
    int op = (int)ops[k];
    float v;
    if (op == 0) {            // mul
        v = a * b;
    } else if (op == 1) {     // add
        v = a + b;
    } else if (op == 2) {     // sub
        v = a - b;
    } else if (op == 3) {     // div = _safe_div (2026-06-13 form): exact x/y for y!=0, eps floor only on exact-zero
        v = (float)((double)a / ((b == 0.0f) ? 1e-9 : (double)b));
    } else if (op == 4) {     // max = np.maximum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a > b) ? a : b;
    } else if (op == 5) {     // min = np.minimum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a < b) ? a : b;
    } else if (op == 6) {     // abs_diff = |a - b|
        v = fabsf(a - b);
    } else if (op == 7) {     // signed = sign(a)*|b| (nan-propagating)
        if (a != a || b != b) {
            v = a + b;
        } else {
            float sgn = (a == 0.0f) ? 0.0f : ((a > 0.0f) ? 1.0f : -1.0f);
            v = sgn * fabsf(b);
        }
    } else {                  // op == 8: ratio_abs = float64-promoted a/(|b|+1)
        v = (float)((double)a / ((double)fabsf(b) + 1.0));
    }
    // np.nan_to_num(nan=0, posinf=0, neginf=0)
    if (isnan(v) || isinf(v)) v = 0.0f;
    out[i * (long long)K + k] = v;
}
"""
_FE_MATERIALISE_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)


def _get_fe_materialise_kernel():
    global _FE_MATERIALISE_KERNEL
    if _FE_MATERIALISE_KERNEL is None:
        import cupy as cp
        _FE_MATERIALISE_KERNEL = cp.RawKernel(_FE_MATERIALISE_SRC, "fe_materialise")
    return _FE_MATERIALISE_KERNEL


# COALESCED COLUMN-MAJOR fe_materialise (2026-06-23, coalescing audit -- the SAME stride-uncoalesced-read
# lever that won 5.59x on the noise-gate hist kernel, applied to the materialise). CUDA-event decomposition
# at the production block shape (n=100k, K=1200, n_operands=64, GTX 1050 Ti) showed the row-major kernel
# above runs at only ~20 GB/s vs the card's ~94 GB/s coalesced floor (write-only baseline 5.1ms @ 94 GB/s
# vs the full kernel 72ms): with thread ``tid -> (i = tid//K, k = tid%K)`` consecutive threads share row
# ``i`` and read ``tv[i*n_operands + a_cols[k]]`` -- a SCATTERED per-candidate operand-column gather (each
# warp touches 32 unrelated operand columns of the same row) -> ~1/5 effective bandwidth.
#
# This variant flips the thread mapping to ``tid -> (k = tid//n, i = tid%n)`` and reads from a COLUMN-MAJOR
# operand table ``tv_cm`` (n_operands, n): consecutive threads (consecutive ``i`` within a fixed candidate
# ``k``) read ``tv_cm[ai*n + i]`` = CONSECUTIVE memory -> fully coalesced operand loads, and write the
# (K, n) column-major output ``out[k*n + i]`` coalesced too. The per-element math (op-code table, float64-
# promoted div/ratio_abs, NaN/inf scrub) is BYTE-FOR-BYTE the row-major kernel's (verified array_equal vs
# fe_materialise across n in {3k,10k,40k} x K in {1,50,257,583} incl. zeros/negatives/+-inf). Caller
# transposes the (n, K) operand table to (K=n_operands, n) ONCE per step (cached, ~0.7ms at n=100k) and
# transposes the (K, n) result back to the (n, K) row-major layout the downstream bin/D2H expect via the
# coalesced tiled-transpose kernel. NET (interleaved-min CUDA-event A/B, 2x-confirmed, GTX 1050 Ti):
# n=100k K=583 36.1->18.4ms = 1.96x; n=100k K=1200 73.5->36.1ms = 2.03x (incl. the tv-transpose + the
# result transpose-back). Gated ON; ``MLFRAME_FE_GPU_MATERIALISE_CM=0`` forces the row-major kernel; any
# transpose/compile/launch failure falls back to it -> CPU / no-CUDA path byte-unchanged.
_FE_MATERIALISE_CM_SRC = r"""
extern "C" __global__
void fe_materialise_cm(const float* __restrict__ tv_cm,
                       const long long* __restrict__ a_cols,
                       const long long* __restrict__ b_cols,
                       const signed char* __restrict__ ops,
                       const long long n, const long long n_operands, const int K,
                       float* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    long long k = tid / n;                 // candidate index (consecutive threads share k)
    long long i = tid - k * n;             // row index (consecutive -> coalesced over n)
    long long ai = a_cols[k];
    long long bi = b_cols[k];
    float a = tv_cm[ai * n + i];           // COLUMN-MAJOR: consecutive threads read consecutive memory
    float b = tv_cm[bi * n + i];
    int op = (int)ops[k];
    float v;
    if (op == 0) {            // mul
        v = a * b;
    } else if (op == 1) {     // add
        v = a + b;
    } else if (op == 2) {     // sub
        v = a - b;
    } else if (op == 3) {     // div = _safe_div (exact x/y for y!=0, eps floor only on exact-zero)
        v = (float)((double)a / ((b == 0.0f) ? 1e-9 : (double)b));
    } else if (op == 4) {     // max = np.maximum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a > b) ? a : b;
    } else if (op == 5) {     // min = np.minimum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a < b) ? a : b;
    } else if (op == 6) {     // abs_diff = |a - b|
        v = fabsf(a - b);
    } else if (op == 7) {     // signed = sign(a)*|b| (nan-propagating)
        if (a != a || b != b) {
            v = a + b;
        } else {
            float sgn = (a == 0.0f) ? 0.0f : ((a > 0.0f) ? 1.0f : -1.0f);
            v = sgn * fabsf(b);
        }
    } else {                  // op == 8: ratio_abs = float64-promoted a/(|b|+1)
        v = (float)((double)a / ((double)fabsf(b) + 1.0));
    }
    if (isnan(v) || isinf(v)) v = 0.0f;
    out[k * n + i] = v;                     // COLUMN-MAJOR (K, n) output -> coalesced
}
"""
_FE_MATERIALISE_CM_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)

# tv -> (n_operands, n) column-major copy cache (weakref-identity, mirrors _OPERAND_TABLE_CACHE): the
# operand table is the SAME device array across a step's blocks/chunks, so transpose it ONCE per step.
_OPERAND_TABLE_CM_CACHE: dict = {"ref": None, "cm": None}


def fe_gpu_materialise_cm_enabled() -> bool:
    """Whether the COALESCED column-major fe_materialise (coalescing audit, ~2x net) is active. DEFAULT ON
    (opt-out ``MLFRAME_FE_GPU_MATERIALISE_CM=0``). Bit-identical (array_equal) to the row-major kernel; the
    row-major kernel stays the fallback on any transpose/compile/launch failure (CPU / no-CUDA unchanged)."""
    return os.environ.get("MLFRAME_FE_GPU_MATERIALISE_CM", "1").strip().lower() not in ("0", "false", "no", "off")


def _get_fe_materialise_cm_kernel():
    global _FE_MATERIALISE_CM_KERNEL
    if _FE_MATERIALISE_CM_KERNEL is None:
        import cupy as cp
        _FE_MATERIALISE_CM_KERNEL = cp.RawKernel(_FE_MATERIALISE_CM_SRC, "fe_materialise_cm")
    return _FE_MATERIALISE_CM_KERNEL


def _operand_table_cm(cp, tv_gpu):
    """(n_operands, n) column-major (= C-contiguous transpose of the (n, n_operands) row-major ``tv_gpu``)
    copy, cached by weakref identity of ``tv_gpu`` so the transpose is paid ONCE per step (the operand table
    is the same device object across the step's materialise blocks). Uses the coalesced tiled-transpose
    kernel; falls back to ``cp.ascontiguousarray(tv_gpu.T)`` (bit-identical) for non-f32 / non-contiguous /
    any kernel failure."""
    import weakref
    c = _OPERAND_TABLE_CM_CACHE
    ref = c["ref"]
    if ref is not None and ref() is tv_gpu and c["cm"] is not None:
        return c["cm"]
    cm = _transpose_to_cm(tv_gpu)  # (n_operands, n) C-order
    try:
        c["ref"] = weakref.ref(tv_gpu)
        c["cm"] = cm
    except TypeError:
        c["ref"] = None
        c["cm"] = None
    return cm


def _fe_materialise_block_gpu(tv_gpu, a_cols_block, b_cols_block, ops_block):
    """Generate the (n, len(ops_block)) float32 candidate matrix for the given column blocks in ONE kernel
    launch, RESIDENT on the GPU. ``tv_gpu`` is the (n, n_operands) row-major float32 operand table already
    on the device. ``a_cols_block`` / ``b_cols_block`` (int64) / ``ops_block`` (int8) are host or device
    arrays of length K. Returns a row-major (n, K) cupy float32 matrix, BIT-EQUAL to
    ``_materialise_chunk_njit`` (same float32 ops, same float64-promoted div/ratio_abs, same nan_to_num)."""
    import cupy as cp

    n = int(tv_gpu.shape[0])
    n_operands = int(tv_gpu.shape[1])
    K = int(len(ops_block))
    a_g = cp.asarray(a_cols_block, dtype=cp.int64)
    b_g = cp.asarray(b_cols_block, dtype=cp.int64)
    ops_g = cp.asarray(ops_block, dtype=cp.int8)
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads

    # COALESCED column-major path (coalescing audit, ~2x net): materialise into a (K, n) column-major buffer
    # from the (n_operands, n) column-major operand table (coalesced operand gathers + coalesced write), then
    # transpose the result back to the (n, K) row-major layout the downstream bin/D2H expect. Bit-identical
    # (array_equal) to the row-major kernel; falls back to it on any transpose/compile/launch failure.
    if fe_gpu_materialise_cm_enabled() and n > 0 and K > 0:
        try:
            tv_cm = _operand_table_cm(cp, tv_gpu)                 # (n_operands, n) C-order, once/step (cached)
            cm_out = cp.empty((K, n), dtype=cp.float32)
            _get_fe_materialise_cm_kernel()(
                (blocks,), (threads,),
                (tv_cm, a_g, b_g, ops_g, np.int64(n), np.int64(n_operands), np.int32(K), cm_out),
            )
            return _transpose_cm_to_rm(cm_out)                   # (K, n) -> (n, K) row-major (coalesced)
        except Exception:
            import logging
            logging.getLogger(__name__).debug("column-major fe_materialise failed; row-major fallback", exc_info=True)

    out = cp.empty((n, K), dtype=cp.float32)
    _get_fe_materialise_kernel()(
        (blocks,), (threads,),
        (tv_gpu, a_g, b_g, ops_g, np.int64(n), np.int64(n_operands), np.int32(K), out),
    )
    return out


# PINNED D2H STAGING for the out_cand float buffer (2026-06-21, nvprof+paired-microbench driven).
# The downstream survivor/usability reads need the (n,K) float candidate matrix on host, so out_cand is
# unavoidable -- but ``cp.asnumpy(cand)`` copies into the caller's PAGEABLE buffer, which makes cupy stage
# the D2H through an internal pinned bounce buffer at PAGEABLE PCIe bandwidth (the #1 production wall:
# cProfile cupy.get = 9.07s, 321 blocking syncs). DMA'ing the chunk into a PERSISTENT PINNED host buffer
# first, then a plain host->host memcpy into the caller's pageable slice, runs the device transfer at full
# pinned bandwidth. MEASURED GTX 1050 Ti, (100k, blk=1200) f32 = 480MB: the device D2H 143ms->75ms (1.9x);
# end-to-end into a pageable slice incl. the added host memcpy 209ms->130ms (1.6x); the whole materialise+
# bin+codes call (K=1200) 696ms->~575ms with the float path on. The buffer is a module-level singleton
# (never on an instance -> pickle-safe), grown on demand and reused across the 15 canonical chunks.
# bench-attempt-rejected (2026-06-21, prior): DEFERRING the float D2H entirely (out_cand=None + downstream
# recompute) was a 0.98x fit-level WASH because removing an overlapped transfer cuts no wall -- but here we
# do NOT remove it, we make the SAME bytes move faster (pinned DMA), which DOES cut the blocking-sync wall.
# Thread-local so two concurrent GPU callers never share (and clobber each other's) the same pinned DMA
# staging buffer: each thread DMAs into its OWN pinned allocation. A single module-level singleton would
# have two threads' ``get(out=view)`` writing the same host bytes, corrupting both transfers.
_PINNED_D2H_TLS = threading.local()


def clear_pinned_d2h() -> bool:
    """Release the calling thread's pinned D2H staging buffer so page-locked host memory is freed (e.g. at fit completion).

    The staging buffer is thread-local; this clears only the current thread's allocation (the only one it can safely
    reach). Returns True if a buffer was present and dropped, False otherwise.
    """
    had = getattr(_PINNED_D2H_TLS, "buf", None) is not None
    _PINNED_D2H_TLS.buf = None
    return had


def _pinned_view(n_bytes: int, shape, dtype):
    """A pinned-host numpy view of at least ``n_bytes``, reshaped to ``shape`` (``dtype``). Reuses a
    THREAD-LOCAL pinned allocation, growing it on demand. Lets ``cupy.ndarray.get(out=...)`` DMA at full
    pinned PCIe bandwidth instead of cp.asnumpy's pageable bounce-buffer path. Thread-local (not a shared
    singleton) so concurrent GPU callers don't clobber each other's staging; module-level (not on an
    estimator instance) -> never reachable from pickled state."""
    import cupy as cp

    buf = getattr(_PINNED_D2H_TLS, "buf", None)
    if buf is None or buf.mem.size < n_bytes:
        buf = cp.cuda.alloc_pinned_memory(int(n_bytes))
        _PINNED_D2H_TLS.buf = buf
    count = int(np.prod(shape))
    return np.frombuffer(buf, dtype=dtype, count=count).reshape(shape)


# Operand-table H2D cache (2026-06-21): the FE step's operand table ``transformed_vars`` is the SAME
# array object across all ~15 chunks of a step, but was re-uploaded to the GPU per chunk (and again per
# survivor re-materialise). Cache the device copy by WEAKREF IDENTITY of the host array: reuse while the
# same object is alive (across the step's chunks), re-upload when the step swaps in a new operand table
# (the weakref breaks). NOT keyed on id() -- id reuse after free would false-hit on a different table.
# Pickle-safe (module-global, never on an instance). The data is identical -> candidates/codes/MI/
# selection bit-identical; this only moves the H2D from per-chunk to once-per-step.
# Per-host-object device cache (was a single slot, which two interleaved steps clobbered: step B's upload
# overwrote step A's device table, so A's still-running chunks read B's bytes). Keyed by id() but each entry
# carries a WEAKREF to the host array, so an id-recycle after free can never false-hit (the weakref must
# resolve to the SAME live object). Bounded FIFO so distinct operand tables across a long fit don't grow it.
_OPERAND_TABLE_CACHE: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (weakref(host), gpu)
_OPERAND_TABLE_CACHE_MAX = 8


# GPU-RESIDENT OPERAND TABLE (2026-06-21, phase 1 of the 100%-GPU-resident MRMR FE rewrite, gated).
# The operand table ``transformed_vars`` (n, n_operands) float32 is built on the CPU in
# ``check_prospective_fe_pairs`` (one column per (var, unary)), then ``_resident_operand_table`` H2Ds it to
# the device ONCE per step. Phase 1 removes even that single H2D by building the device mirror's columns ON
# the GPU directly from the resident raw operand inputs (via ``_unary_apply`` -- the same math as the CPU
# ``unary_transformations``), so the materialise consumes a DEVICE array with NO host->device transfer of
# the bulk operand bytes. The CPU ``transformed_vars`` is STILL built (the pair-search inner loops /
# discretize read it on the host -- those move to the GPU in later phases); phase 1 only kills the
# materialise H2D. Operand transforms that are NOT plain GPU unaries (prewarp / gate_med / hermite-poly --
# fitted/special, no straightforward cupy form) are built on the CPU and copied into the resident mirror (a
# few columns); the bulk plain-unary columns are GPU-built. The PREBUILT mirror is registered here by
# weakref-identity of the host ``transformed_vars`` so ``_resident_operand_table`` returns it WITHOUT the
# H2D. Module-global -> never reachable from pickled estimator state. Gated OFF by default
# (``MLFRAME_FE_GPU_RESIDENT_OPERANDS``) until proven 11-green; the CPU / no-CUDA path is unchanged.
# Per-host-object prebuilt-mirror registry (was a single slot, clobbered when two concurrent steps each
# registered their own GPU-resident mirror). Keyed by id() with a co-validating weakref so an id-recycle
# can't return a stale/wrong-table mirror; bounded FIFO. ``device_table=None`` clears the entry for that host.
_PREBUILT_OPERAND_TABLE: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (weakref(host), gpu)
_PREBUILT_OPERAND_TABLE_MAX = 8


def fe_gpu_resident_operands_enabled() -> bool:
    """Whether the GPU-RESIDENT operand-table build (phase 1) is active. DEFAULT ON (opt-out
    ``MLFRAME_FE_GPU_RESIDENT_OPERANDS=0``). When on (and CUDA present -- the caller guards this and
    falls back on any failure) the operand table's bulk plain-unary columns are produced ON the GPU and
    the materialise consumes the device array with no H2D re-upload; the CPU / no-CUDA path is byte-for-
    byte unchanged (operand table H2D'd as before)."""
    return os.environ.get("MLFRAME_FE_GPU_RESIDENT_OPERANDS", "1").strip().lower() not in ("0", "false", "no", "off")


def register_prebuilt_operand_table(transformed_vars, device_table) -> None:
    """Register a GPU-RESIDENT device mirror ``device_table`` for the host operand table ``transformed_vars``
    (keyed on the host array's weakref identity). ``_resident_operand_table`` then returns ``device_table``
    for that exact host object WITHOUT re-uploading. Pass ``device_table=None`` to clear. The device array
    MUST be a row-major (n, n_operands) C-contiguous float32 cupy array matching ``transformed_vars``'s
    shape (the layout ``_fe_materialise_block_gpu``'s kernel addresses); a mismatch is ignored at lookup."""
    import weakref
    c = _PREBUILT_OPERAND_TABLE
    key = id(transformed_vars)
    if device_table is None:
        c.pop(key, None)
        return
    try:
        c[key] = (weakref.ref(transformed_vars), device_table)
        c.move_to_end(key)
        while len(c) > _PREBUILT_OPERAND_TABLE_MAX:
            c.popitem(last=False)
    except TypeError:
        c.pop(key, None)


def _prebuilt_operand_table(transformed_vars):
    """The registered GPU-resident device mirror for ``transformed_vars`` iff it matches the host array by
    weakref identity AND shape (n, n_operands); else None. Shape guard so a stale/mismatched mirror can
    never feed the materialise kernel a wrong-width table (out-of-bounds operand-column reads)."""
    c = _PREBUILT_OPERAND_TABLE
    hit = c.get(id(transformed_vars))
    if hit is None:
        return None
    ref, g = hit
    if g is None or ref() is not transformed_vars:
        return None
    if tuple(g.shape) != tuple(transformed_vars.shape):
        return None
    return g


def _resident_operand_table(cp, transformed_vars):
    """Device (n, n_operands) float32 copy of ``transformed_vars``. When a GPU-RESIDENT mirror was prebuilt
    for this exact host object (phase 1, ``register_prebuilt_operand_table``) it is returned WITH NO H2D --
    the bulk operand bytes were produced on the device. Otherwise the host array is uploaded once per
    distinct object (weakref-identity cache) and reused across a step's chunks; falls back to a plain
    upload if the array is not weakref-able."""
    import weakref
    pre = _prebuilt_operand_table(transformed_vars)
    if pre is not None:
        return pre
    c = _OPERAND_TABLE_CACHE
    key = id(transformed_vars)
    hit = c.get(key)
    if hit is not None:
        ref, g = hit
        if ref() is transformed_vars and g is not None:
            c.move_to_end(key)
            return g
        c.pop(key, None)  # weakref dead (id recycled onto a different object) -> drop the stale entry
    g = cp.asarray(np.ascontiguousarray(transformed_vars, dtype=np.float32))
    try:
        c[key] = (weakref.ref(transformed_vars), g)
        c.move_to_end(key)
        while len(c) > _OPERAND_TABLE_CACHE_MAX:
            c.popitem(last=False)
    except TypeError:
        c.pop(key, None)
    return g


# BATCHED plain-unary op-code kernel (launch-reduction, 2026-06-25). build_resident_operand_table applied
# each GPU-built operand column with a separate _unary_apply (a cupy elementwise op) + .astype(f32) + strided
# slice-assign (~3 cuLaunchKernel/col) -- the measured #2 launch source (282). For the columns that share one
# float64 raw-operand group and use a pure-elementwise unary, ONE kernel now applies the per-column op code
# (libdevice math = the SAME functions cupy's elementwise calls -> bit-identical) and writes f32 straight into
# the operand table. log (smart_log full-column shift), erf/gammaln (special), prewarp, mixed-dtype groups,
# and any per-op parity miss stay on the per-column path. _BATCH_UNARY_OPS maps the bit-verified ops to codes.
# sinc (code 16) is intentionally EXCLUDED: its sin(pi x)/(pi x) form has a sub-ulp mismatch vs cupy's
# xp.sinc, so it stays on the bit-exact per-column path. Every op below is verified maxdiff-0 vs _unary_apply.
_BATCH_UNARY_OPS = {
    "identity": 0, "neg": 1, "abs": 2, "sqr": 3, "reciproc": 4, "sqrt": 5, "sin": 6, "sign": 7, "rint": 8,
    "qubed": 9, "invsquared": 10, "invqubed": 11, "cbrt": 12, "invcbrt": 13, "invsqrt": 14, "exp": 15,
    "cos": 17, "tan": 18, "arcsin": 19, "arccos": 20, "arctan": 21, "sinh": 22, "cosh": 23,
    "tanh": 24, "arcsinh": 25, "arccosh": 26, "arctanh": 27,
}
_BATCH_UNARY_SRC = r"""
extern "C" __global__
void batch_unary(const double* __restrict__ G, const long long* __restrict__ slot,
                 const int* __restrict__ opc, const long long* __restrict__ out_col,
                 const long long n, const int m, const int ncols, const int n_operands,
                 float* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)ncols;
    if (t >= total) return;
    int c = (int)(t % (long long)ncols);
    long long row = t / (long long)ncols;
    double x = G[row * (long long)m + slot[c]];
    double r;
    switch (opc[c]) {
        case 0:  r = x; break;
        case 1:  r = -x; break;
        case 2:  r = fabs(x); break;
        case 3:  r = x * x; break;
        case 4:  r = pow(x, -1.0); break;
        case 5:  r = sqrt(fabs(x)); break;
        case 6:  r = sin(x); break;
        case 7:  r = isnan(x) ? x : (x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)); break;
        case 8:  r = rint(x); break;
        case 9:  r = x * x * x; break;
        case 10: r = 1.0 / (x * x); break;
        case 11: r = 1.0 / (x * x * x); break;
        case 12: r = cbrt(x); break;
        case 13: r = pow(x, -1.0 / 3.0); break;
        case 14: r = pow(x, -1.0 / 2.0); break;
        case 15: r = exp(x); break;
        case 16: { double pix = 3.141592653589793 * x; r = (x == 0.0) ? 1.0 : sin(pix) / pix; break; }
        case 17: r = cos(x); break;
        case 18: r = tan(x); break;
        case 19: r = asin(x); break;
        case 20: r = acos(x); break;
        case 21: r = atan(x); break;
        case 22: r = sinh(x); break;
        case 23: r = cosh(x); break;
        case 24: r = tanh(x); break;
        case 25: r = asinh(x); break;
        case 26: r = acosh(x); break;
        case 27: r = atanh(x); break;
        default: r = x; break;
    }
    out[row * (long long)n_operands + out_col[c]] = (float)r;
}
"""
_BATCH_UNARY_KERNEL = None


def _get_batch_unary_kernel():
    global _BATCH_UNARY_KERNEL
    if _BATCH_UNARY_KERNEL is None:
        import cupy as cp
        _BATCH_UNARY_KERNEL = cp.RawKernel(_BATCH_UNARY_SRC, "batch_unary")
    return _BATCH_UNARY_KERNEL


def build_resident_operand_table(transformed_vars, col_specs, *, fallback_unaries=()):
    """Build a GPU-RESIDENT (n, n_operands) row-major float32 cupy mirror of the host operand table
    ``transformed_vars``, producing the bulk PLAIN-UNARY columns ON the GPU (via ``_unary_apply`` -- the
    same math the CPU ``unary_transformations`` applied) and COPYING the rest (prewarp / gate_med /
    hermite-poly / any name in ``fallback_unaries`` / any GPU-unbuildable column) from the host array.

    ``col_specs`` is a list aligned with the operand-table columns: each entry is ``(col_idx, raw_vals,
    unary_name)`` where ``raw_vals`` is the host float64 raw operand input the CPU applied ``unary_name`` to
    (or ``None`` for a column with no GPU recipe -> copied from the host). A column is GPU-built iff
    ``raw_vals is not None``, ``unary_name`` is a known plain unary (``_unary_apply`` accepts it, not in
    ``fallback_unaries``). The unary is applied on the GPU in float64 (the dtype the CPU ``tr_func``
    received) then cast to float32 (mirroring the CPU's compute-in-f64-then-store-f32) so the GPU column
    matches the host column to fp round-off. Any per-column GPU failure falls that column back to the host
    copy (never a correctness regression). Returns the device array (already row-major C-contiguous f32);
    the caller registers it via ``register_prebuilt_operand_table``."""
    import cupy as cp

    from ._gpu_resident_fe import _gpu_apply_prewarp  # lazy: parent-defined, avoids import cycle

    n, n_operands = transformed_vars.shape
    fb = set(fallback_unaries)
    # Allocate the device mirror WITHOUT uploading the host table: the residency win is precisely that the
    # bulk operand bytes never make the host->device trip. We H2D ONLY the small per-operand RAW inputs (n
    # floats each, cached so each distinct raw operand is uploaded ONCE -- they recur across a var's unaries)
    # and GPU-build the plain columns from them; the FEW non-plain / failed columns are copied from the host
    # one column at a time. Columns with no spec (the unused tail, if any) are zero-filled -- they are never
    # read by the materialise (operand indices are always < the used width), so their content is irrelevant.
    g = cp.zeros((n, n_operands), dtype=cp.float32)
    # ONE-TRANSFER (phase R0, 2026-06-21): batch the DISTINCT raw operands referenced by the GPU-buildable
    # specs into per-dtype host matrices and upload each in ONE H2D, instead of one cp.asarray per distinct
    # raw. Each raw keeps its NATIVE float dtype (we group BY dtype) so the unary still applies in the exact
    # dtype the CPU ``tr_func`` saw -> the GPU column matches the host column to fp round-off (the invariant
    # the per-operand path enforced). Values are byte-identical; only the H2D packaging changes. Per-dtype
    # grouping means uniform-dtype fits (the common case: all-pandas f64 -> 14 distinct raws) collapse to ONE
    # upload. The device column is a strided VIEW into the group matrix -- _unary_apply is elementwise, so the
    # result equals the contiguous-input result bit-for-bit. Any group/build failure falls that column back to
    # the host copy below (never a correctness regression).
    _raw_slot: dict = {}   # id(raw_vals) -> (dtype_key, slot_in_group)
    _groups: dict = {}     # dtype_key -> list[host column in native float dtype]
    for _spec_t in col_specs:
        col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        if raw_vals is not None and unary_name not in fb:
            _rk = id(raw_vals)
            if _rk not in _raw_slot:
                _rv = np.ascontiguousarray(raw_vals)
                if not np.issubdtype(_rv.dtype, np.floating):
                    _rv = _rv.astype(np.float64)  # CPU tr_func on a non-float would also promote
                _dk = _rv.dtype.str
                grp = _groups.setdefault(_dk, [])
                _raw_slot[_rk] = (_dk, len(grp))
                grp.append(_rv)
    _dev_groups: dict = {}  # dtype_key -> device (n, m) array (ONE H2D per dtype group)
    for _dk, cols in _groups.items():
        try:
            _host = (np.ascontiguousarray(np.column_stack(cols)) if len(cols) > 1
                     else np.ascontiguousarray(cols[0]).reshape(-1, 1))
            _dev_groups[_dk] = cp.asarray(_host)
        except Exception:
            _dev_groups[_dk] = None
    n_gpu = 0
    n_cpu = 0
    # BATCHED PRE-PASS (launch-reduction): collect the GPU-buildable plain-unary columns per dtype-group and
    # apply them with ONE batch_unary kernel each (libdevice math = bit-identical to per-column _unary_apply),
    # writing f32 straight into g -- replacing ~3 cuLaunchKernel/col (_unary_apply + astype + slice-assign).
    # Only ops in _BATCH_UNARY_OPS and non-prewarp specs whose group loaded qualify; everything else stays on
    # the exact per-column path below (which skips the already-batched col_idx).
    _batched: set = set()
    _f64_key = np.dtype(np.float64).str   # batch ONLY the float64 group: the kernel computes in f64, matching
    try:                                  # the CPU tr_func's f64 math; an f32 group must compute in f32 -> per-col
        _bg: dict = {}   # dtype_key -> (slots[], opcs[], out_cols[])
        for _spec_t in col_specs:
            col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
            _payload = _spec_t[3] if len(_spec_t) > 3 else None
            if raw_vals is None or unary_name in fb or unary_name not in _BATCH_UNARY_OPS:
                continue
            if _payload is not None and _payload.get("kind") == "prewarp":
                continue
            _rk = id(raw_vals)
            if _rk not in _raw_slot:
                continue
            _dk, _slot = _raw_slot[_rk]
            if _dk != _f64_key or _dev_groups.get(_dk) is None:
                continue
            s, o, oc = _bg.setdefault(_dk, ([], [], []))
            s.append(_slot); o.append(_BATCH_UNARY_OPS[unary_name]); oc.append(col_idx)
        if _bg:
            _ker = _get_batch_unary_kernel()
            for _dk, (s, o, oc) in _bg.items():
                _dev = _dev_groups[_dk]
                m = int(_dev.shape[1]) if _dev.ndim > 1 else 1
                G = cp.ascontiguousarray(_dev.astype(cp.float64, copy=False))
                slot = cp.asarray(np.asarray(s, dtype=np.int64))
                opc = cp.asarray(np.asarray(o, dtype=np.int32))
                out_col = cp.asarray(np.asarray(oc, dtype=np.int64))
                ncols = len(s)
                total = n * ncols
                threads = 256
                _ker(((total + threads - 1) // threads,), (threads,),
                     (G, slot, opc, out_col, np.int64(n), np.int32(m), np.int32(ncols),
                      np.int32(n_operands), g))
                _batched.update(oc)
            n_gpu += len(_batched)
    except Exception:
        _batched = set()   # any batch failure -> every column rebuilt by the exact per-column path below
    for _spec_t in col_specs:
        col_idx, raw_vals, unary_name = _spec_t[0], _spec_t[1], _spec_t[2]
        if col_idx in _batched:
            continue
        _payload = _spec_t[3] if len(_spec_t) > 3 else None  # R1: prewarp GPU-apply payload (or None)
        gpu_built = False
        if raw_vals is not None and unary_name not in fb:
            try:
                _dk, _slot = _raw_slot[id(raw_vals)]
                _dev = _dev_groups.get(_dk)
                if _dev is not None:
                    x = _dev[:, _slot]   # native-dtype device view of this raw operand (no per-operand H2D)
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        if _payload is not None and _payload.get("kind") == "prewarp":
                            # R1: APPLY the prewarp on the device (preprocess + Clenshaw) from the raw + spec,
                            # mirroring hermite_fe.apply_operand_prewarp -- no host-column H2D. _gpu_apply_prewarp
                            # raises for any unported basis -> falls to the host copy below (bit-exact).
                            col = _gpu_apply_prewarp(cp, x, _payload["spec"])
                        else:
                            col = _unary_apply(cp, unary_name, x)
                    # nan_to_num is NOT applied here: the CPU operand table stores the raw unary output
                    # (un-scrubbed) too -- the materialise kernel scrubs NaN/inf inline -> bit-equal.
                    g[:, col_idx] = col.astype(cp.float32)
                    gpu_built = True
            except Exception:
                gpu_built = False
        if not gpu_built:
            # Non-plain (prewarp / gate_med / poly) or failed: copy just THIS column from the host (a single
            # (n,) f32 H2D, not the whole table) so the device column equals the CPU bytes exactly.
            g[:, col_idx] = cp.asarray(np.ascontiguousarray(transformed_vars[:, col_idx], dtype=np.float32))
            n_cpu += 1
        else:
            n_gpu += 1
    return cp.ascontiguousarray(g), n_gpu, n_cpu


def gpu_materialise_discretize_codes_host(
    transformed_vars: np.ndarray, a_cols: np.ndarray, b_cols: np.ndarray, op_codes: np.ndarray,
    nbins: int, *, dtype=np.int8, out_cand: np.ndarray | None = None,
) -> np.ndarray:
    """GPU fast path for the FE chunk's MATERIALISE + BINNING. Uploads the operand table
    ``transformed_vars`` (n, n_operands) float32 ONCE, then for each VRAM-bounded column block: generates
    the float32 candidate matrix on the GPU (``_fe_materialise_block_gpu`` -- bit-equal to
    ``_materialise_chunk_njit``) and quantile-bins it RESIDENT (``_gpu_resident_discretize_codes``,
    bit-equal to ``discretize_2d_quantile_batch``). Returns the (n, K) ``dtype`` codes (BIT-IDENTICAL to
    the CPU njit-materialise -> ``gpu_discretize_codes_host`` pipeline, verified maxdiff 0).

    The candidate matrix is generated + binned RESIDENT (the int codes are the only mandatory D2H). But the
    downstream FE survivor / usability / ext-val stages read the CONTINUOUS candidate columns out of the
    chunk buffer, so the caller passes ``out_cand`` (the ``chunk_buffer[:, :K]`` float32 view) to receive
    the materialised float candidate matrix as well -- this replaces the CPU njit materialise with the GPU
    one (the bandwidth-bound strided-gather op the GPU is good at) while keeping the buffer the rest of the
    pipeline expects. Pass ``out_cand=None`` to skip the float D2H (codes-only, when no downstream
    continuous read is needed). Inputs are finite by construction (the kernel scrubs NaN/inf inline)."""
    import cupy as cp

    # Drop any stale handoff/deferred-fill from a PRIOR chunk before producing this one (releases that
    # chunk's pinned device codes; each chunk's dispatch should already have consumed/cleared its own).
    clear_resident_codes_handoff()
    tv = np.ascontiguousarray(transformed_vars, dtype=np.float32)
    a_cols = np.ascontiguousarray(a_cols, dtype=np.int64)
    b_cols = np.ascontiguousarray(b_cols, dtype=np.int64)
    op_codes = np.ascontiguousarray(op_codes, dtype=np.int8)
    n = int(tv.shape[0])
    K = int(a_cols.shape[0])
    # Operand table H2D cached per-step by weakref identity (same transformed_vars across the step's
    # chunks -> uploaded ONCE, not per chunk). Pass the ORIGINAL array so the weakref tracks it.
    tv_gpu = _resident_operand_table(cp, transformed_vars)
    out = np.empty((n, K), dtype=dtype)
    # RESIDENT-CODES HANDOFF (gated, default OFF): keep the on-device int codes in ONE (n, K) resident
    # cupy array so the noise-gate's resident-CUDA path can consume them DIRECTLY -- skipping the codes'
    # GPU->host (here) ->GPU (the gate's H2D) round-trip. The host ``out`` is STILL filled (the CPU /
    # analytic / opt-out / SU / any-failure dispatch branches need it and it is the safe fallback), so this
    # only ADDS a resident copy when the gate is on; the round-trip is skipped only when the resident gate
    # is the actual consumer (it matches ``out`` by identity via the module handoff).
    _resident_codes_on = fe_gpu_resident_codes_enabled()
    dev_codes = cp.empty((n, K), dtype=cp.dtype(np.dtype(dtype))) if _resident_codes_on else None
    # DEFER the host-codes D2H when the resident handoff is on: the host ``out`` is filled LAZILY (only if a
    # host consumer reads it -- see ensure_host_codes_filled) instead of eagerly per block. This skips the
    # (n, K) codes D2H (the canonical fit's single largest D2H) whenever the resident gate consumes the
    # device codes. Needs dev_codes (the resident copy) to fill from, so it is only active with it.
    _defer_host_codes = bool(dev_codes is not None and fe_gpu_defer_host_codes_enabled())
    # CODES path footprint is f32 (cand + transpose + int32 codes + narrow out), ~4B x ~4 working copies --
    # NOT the f64 MI prototype's 8x5. Budget for that so the VRAM sub-chunk is ~3x wider -> ~3x fewer
    # radix/bin/materialise launches (cuts the launch+sync+GPU-idle overhead). working_multiple=6 keeps a
    # safe margin over the honest ~4 on the 4GB card; still 0.25*free VRAM-governed; per-column-independent
    # so codes are bit-identical regardless of chunk boundary.
    k_chunk = _gpu_k_chunk(n, bytes_per_elem=4, working_multiple=6, max_cols=K)
    for start in range(0, K, k_chunk):
        stop = min(start + k_chunk, K)
        cand = _fe_materialise_block_gpu(
            tv_gpu, a_cols[start:stop], b_cols[start:stop], op_codes[start:stop]
        )  # resident (n, blk) float32 -- bit-equal to _materialise_chunk_njit
        if out_cand is not None:
            # Float candidate D2H for the downstream survivor/usability reads. Stage through a PERSISTENT
            # PINNED host buffer (full PCIe bandwidth) then host->host memcpy into the caller's pageable
            # slice -- 1.6x faster than cp.asnumpy's pageable bounce-buffer path even WITH the added memcpy
            # (see _pinned_view note). Bit-identical bytes. Falls back to cp.asnumpy on any pinned-alloc
            # failure (e.g. host pinned-memory exhaustion) so it can never regress correctness.
            try:
                hv = _pinned_view(cand.nbytes, cand.shape, cand.dtype)
                cand.get(out=hv)
                out_cand[:, start:stop] = hv
            except Exception:
                import logging
                logging.getLogger(__name__).debug("pinned D2H staging failed; cp.asnumpy fallback", exc_info=True)
                out_cand[:, start:stop] = cp.asnumpy(cand)
        # Bin the candidate RESIDENT at its native float32 (the FE buffer dtype) -- no f64 up-cast: the
        # cand already IS float32 (bit-equal to _materialise_chunk_njit), so binning in f32 removes a needless
        # cast AND halves the bandwidth-bound percentile sort, while preserving the FE selection. The exact
        # f64 fallback (bit-identical to the CPU pipeline) is one env flip away (MLFRAME_FE_GPU_BINNING_DTYPE
        # =float64). _gpu_resident_discretize_codes applies the working dtype internally.
        codes_gpu = _gpu_resident_discretize_codes(cand, int(nbins))
        # Cast int32 codes -> target narrow ``dtype`` (int8/int16) ON the GPU before the D2H so the
        # transfer moves 1/4 (int8) the bytes of the int32 codes AND skips the host-side astype copy.
        # bench (GTX 1050 Ti, n=100k K=384): int32-D2H+host-cast 170ms -> gpu-cast+D2H 25ms = 6.7x on
        # the codes export, BIT-IDENTICAL. The narrow dtype is the FE code dtype (nbins<=255 -> int8),
        # so the on-device cast cannot overflow.
        _cd = np.dtype(dtype)
        codes_out = codes_gpu.astype(cp.dtype(_cd), copy=False) if codes_gpu.dtype != _cd else codes_gpu
        if dev_codes is not None:
            # Keep this block's narrow codes RESIDENT (the EXACT bytes we D2H below). Bit-identical to the
            # host ``out`` slice -> selection-equivalent when the resident gate consumes the device copy.
            dev_codes[:, start:stop] = codes_out
        if not _defer_host_codes:
            # Eager host fill (deferral off, or no resident copy): D2H this block's codes into ``out`` now.
            out[:, start:stop] = cp.asnumpy(codes_out)
        del cand, codes_gpu, codes_out
    if dev_codes is not None:
        # Stash by the returned host array's identity so the dispatch can pick the device codes up without
        # the chunk path threading a new argument (see _RESIDENT_CODES_HANDOFF). Any consumer that is NOT
        # the resident CUDA gate simply ignores it + reads ``out`` (host) as before.
        _stash_resident_codes(out, dev_codes)
    if _defer_host_codes:
        # ``out`` is UNFILLED -- register the lazy device->host fill so a host-reading consumer (analytic /
        # CPU / non-resident GPU) can materialise it on demand via ensure_host_codes_filled. The eager
        # per-block D2H above was skipped; the resident gate reads the device codes directly (no host read).
        _stash_deferred_host_fill(out, dev_codes)
    return out


def gpu_discretize_codes_host(cand: np.ndarray, nbins: int, *, dtype=np.int8, defer_host_fill: bool = False) -> np.ndarray:
    """Quantile-bin a host (n, K) float candidate matrix to ordinal codes via the GPU, returning a host
    ``(n, K)`` array of ``dtype``. The FE candidate buffer is ALREADY float32, so the matrix is kept at
    its native dtype (NO f64 up-cast) and binned in float32 (the input's native dtype) -- removing a
    needless cast AND halving the bandwidth-bound cp.percentile sort, while preserving the FE selection
    (the acceptance bar; f32-vs-f64 codes agree ~100%). Set ``MLFRAME_FE_GPU_BINNING_DTYPE=float64`` for
    the bit-identical fallback matching the CPU ``discretize_2d_quantile_batch`` (np.percentile upcasts
    float32 to float64). Feeding the result into the UNCHANGED ``_dispatch_batch_mi_with_noise_gate``
    keeps the FE selection equivalent -- this only moves the binning (CPU partition+searchsorted, the
    dominant per-pair cost at large n) onto the GPU. Inputs are assumed finite (caller scrubs NaN/inf).

    VRAM-chunked over columns so a wide candidate block never over-allocates device memory."""
    import cupy as cp

    cand = np.ascontiguousarray(cand)  # keep native dtype (float32 FE buffer) -- no f64 up-cast
    n, K = cand.shape
    out = np.empty((n, K), dtype=dtype)
    clear_resident_codes_handoff()  # drop any stale prior-chunk handoff before producing this one
    # RESIDENT-CODES HANDOFF (gated, default ON when CUDA present): this is the SECOND codes leg -- the
    # binning-only path the canonical FE chunk takes when the candidate buffer is materialised on the CPU
    # (the default minimal preset's numpy-fallback materialise) then binned on the GPU. It produces the
    # SAME on-device int codes as the fused materialise path, so keep them RESIDENT (one (n, K) cupy array
    # in the narrow code dtype) and stash them by the returned host array's identity -- the noise-gate
    # dispatch then consumes the device codes IN PLACE, skipping the codes' GPU->host (here) ->GPU (the
    # gate's H2D) round-trip. The host ``out`` is STILL filled (the CPU / analytic / opt-out / any-failure
    # branches read it, and it is the safe fallback), so this only ADDS a resident copy when the gate is on;
    # the round-trip is skipped only when the resident CUDA gate is the actual consumer (it matches ``out``
    # by identity). Bit-identical to the host codes -> selection unchanged.
    _resident_codes_on = fe_gpu_resident_codes_enabled()
    dev_codes = cp.empty((n, K), dtype=cp.dtype(np.dtype(dtype))) if _resident_codes_on else None
    # HOST-CODES D2H DEFERRAL (2026-06-27). Direct callers (gpu_pairs_fe_mi's analytic path + the bit-identity
    # tests) read the returned host array IMMEDIATELY, so this leg's DEFAULT stays eager (defer_host_fill=False)
    # to keep that contract. But the per-pair FE-score leg (_pairs_score._score_one_pair) hands the return
    # straight to ``_dispatch_batch_mi_with_noise_gate``, whose resident-CUDA gate consumes the DEVICE codes in
    # place (via take_resident_codes) and never reads host ``disc_2d`` -- so under the strict-resident path the
    # eager (n, K) codes D2H here is the single largest D2H of the fit and is PURE WASTE (measured n=100k: 24/24
    # dispatches hit the resident handoff; the host buffer was filled but never read). When that caller opts in
    # (defer_host_fill=True) AND the deferral is enabled, return an UNFILLED host buffer + register a lazy
    # device->host fill (ensure_host_codes_filled) keyed on the buffer id, exactly like the fused
    # gpu_materialise_discretize_codes_host leg. Bit-identical: the host buffer, if any consumer ever reads it,
    # is device_codes.get() -- the exact bytes the eager D2H produced -> FE selection unchanged.
    # f32 codes-path footprint (see gpu_materialise_discretize_codes_host) -> wider VRAM sub-chunk, ~3x
    # fewer bin/edge launches; per-column-independent -> bit-identical codes.
    _defer_host_codes = bool(defer_host_fill and dev_codes is not None and fe_gpu_defer_host_codes_enabled())
    k_chunk = _gpu_k_chunk(n, bytes_per_elem=4, working_multiple=6, max_cols=K)
    for start in range(0, K, k_chunk):
        block = cand[:, start:start + k_chunk]
        stop = start + block.shape[1]
        codes_gpu = _gpu_resident_discretize_codes(cp.asarray(block), int(nbins))
        # Narrow int32->dtype ON the GPU before D2H (1/4 the bytes for int8, no host astype copy) --
        # same 6.7x codes-export win as gpu_materialise_discretize_codes_host, BIT-IDENTICAL.
        _cd = np.dtype(dtype)
        codes_out = codes_gpu.astype(cp.dtype(_cd), copy=False) if codes_gpu.dtype != _cd else codes_gpu
        if dev_codes is not None:
            # Keep this block's narrow codes RESIDENT (the EXACT bytes we D2H below) for the gate consumer.
            dev_codes[:, start:stop] = codes_out
        if not _defer_host_codes:
            out[:, start:stop] = cp.asnumpy(codes_out)
        del codes_gpu, codes_out
    if dev_codes is not None:
        _stash_resident_codes(out, dev_codes)
    if _defer_host_codes:
        # ``out`` is UNFILLED -- register the lazy device->host fill so a host-reading consumer (analytic /
        # CPU / non-resident GPU) can materialise it on demand via ensure_host_codes_filled. The eager
        # per-block D2H above was skipped; the resident gate reads the device codes directly (no host read).
        _stash_deferred_host_fill(out, dev_codes)
    return out
