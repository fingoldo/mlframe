"""Float32 radix select with candidate compaction (v4): the source and lazy getter, a sibling of ``_gpu_resident_select_kernels`` (which holds v1-v3)."""

from __future__ import annotations

import logging
import os

from mlframe.utils.log_throttle import log_throttle

# V4 = v3 + CANDIDATE COMPACTION (the f64 v3 idea on 32-bit keys). After bytes 3-2 fix the top 16 bits, only elements whose 16-bit prefix is an active
# window can still move a rank; the byte-1 full scan appends their keys to a per-column scratch (K, cap) buffer and the byte-0 pass scans only that.
# Histograms are order-independent sums over the same element sets -> bit-identical order statistics. A column whose candidates overflow ``cap``
# keeps full scans (exact). Compacting at byte 2 (8-bit prefix: sign + 7 exponent bits) is not tried: real columns fill 1-2 such buckets
# (the f64 record above). Off switch MLFRAME_RADIX_F32_V4=0.
_RADIX_SELECT_F32_V4_SRC = r"""
#define MAXR 64
extern "C" __global__
void radix_select_f32_v4(const float* __restrict__ data, const long long n, const int K,
                      const long long* __restrict__ ranks, const int R, float* __restrict__ out,
                      unsigned int* __restrict__ cand, const long long cap){
    int col=blockIdx.x, tid=threadIdx.x, nt=blockDim.x;
    extern __shared__ unsigned int sh[];
    __shared__ unsigned int prefix[MAXR];
    __shared__ unsigned long long below[MAXR];
    __shared__ unsigned int wpref[MAXR];
    __shared__ int rank2w[MAXR];
    __shared__ unsigned int wsort[MAXR];   // wpref sorted ascending (distinct active window prefixes)
    __shared__ int wsort2w[MAXR];          // sorted position -> ORIGINAL window index (into wpref / sh)
    __shared__ int W;
    __shared__ int win_bad;
    __shared__ int wsum[2];
    __shared__ unsigned int scount;      // 32-bit: a 64-bit shared atomic is a compare-and-swap loop on this architecture (the candidate buffer never holds 2^32 entries)
    __shared__ int have_cand;    // the candidate buffer holds every element the remaining passes can count
    __shared__ int overflowed;
    if(tid<R){prefix[tid]=0u;below[tid]=0ULL;}
    if(tid==0){scount=0u;have_cand=0;overflowed=0;}
    __syncthreads();
    for(int byte=3;byte>=0;--byte){
        int shift=byte*8;
        unsigned int hmask=(byte==3)?0u:(0xFFFFFFFFu<<((byte+1)*8));
        // WINDOWS: the ranks are distinct and ascending (every caller builds them with np.unique), so their prefixes never decrease; the windows are the runs of equal prefixes and the
        // window of a rank is the running count of run starts. One flag per rank and a warp scan (two warps cover 64 ranks) replace the serial walk that compared every rank with every
        // window (O(R^2), 125k of the 205k cycles of a block at 38 ranks: the serial part took 61 percent of the time). A decreasing prefix (never produced by the callers), or fewer than
        // 64 threads with more than 32 ranks, takes the serial dedup below, unchanged.
        if(tid==0)win_bad=0;
        __syncthreads();
        const bool par_ok=(R<=32)||(nt>=64);
        {   unsigned int p=0u;int f=0;
            if(tid<R){p=prefix[tid]&hmask;unsigned int pp=(tid>0)?(prefix[tid-1]&hmask):p;f=(tid==0)||(p!=pp);if(tid>0&&p<pp)atomicOr(&win_bad,1);}
            int x=f;const int lane=tid&31;
            #pragma unroll
            for(int o=1;o<32;o<<=1){int y=__shfl_up_sync(0xffffffffu,x,o);if(lane>=o)x+=y;}
            if((tid>>5)<2&&lane==31)wsum[tid>>5]=x;
            __syncthreads();
            if(par_ok&&!win_bad){
                if((tid>>5)==1)x+=wsum[0];
                if(tid<R){int widx=x-1;rank2w[tid]=widx;if(f){wpref[widx]=p;wsort[widx]=p;wsort2w[widx]=widx;}if(tid==R-1)W=x;}
            } else {
            if(tid==0){int w=0;for(int r=0;r<R;++r){unsigned int p=prefix[r]&hmask;int f=-1;
                for(int q=0;q<w;++q)if(wpref[q]==p){f=q;break;} if(f<0){wpref[w]=p;rank2w[r]=w;w++;}else rank2w[r]=f;} W=w;
                for(int q=0;q<w;++q){wsort[q]=wpref[q];wsort2w[q]=q;}
                for(int a=1;a<w;++a){unsigned int kv=wsort[a];int kw=wsort2w[a];int b=a-1;
                    while(b>=0&&wsort[b]>kv){wsort[b+1]=wsort[b];wsort2w[b+1]=wsort2w[b];--b;}
                    wsort[b+1]=kv;wsort2w[b+1]=kw;}}

            }
        }
        __syncthreads();
        int Wl=W;
        for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;
        __syncthreads();
        const int compact=(!have_cand)&&(cap>0)&&(byte==1);
        if(!have_cand){
            for(long long i=tid;i<n;i+=nt){
                float d=data[(long long)col*n+i];unsigned int u;memcpy(&u,&d,4);  // COLUMN-MAJOR: coalesced
                u=((u&0x7FFFFFFFu)>0x7F800000u)?0xFFFFFFFFu:((u&0x80000000u)?~u:(u|0x80000000u));
                unsigned int pm=u&hmask;
                int lo=0,hi=Wl;
                while(lo<hi){int mid=(lo+hi)>>1;if(wsort[mid]<pm)lo=mid+1;else hi=mid;}
                if(lo<Wl&&wsort[lo]==pm){int win=wsort2w[lo];
                    int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);
                    if(compact){unsigned int slot=atomicAdd(&scount,1u);
                        if((long long)slot<cap)cand[(long long)col*cap+(long long)slot]=u;else overflowed=1;}}
            }
        } else {
            long long m=(long long)scount;
            for(long long i=tid;i<m;i+=nt){
                unsigned int u=cand[(long long)col*cap+i];
                unsigned int pm=u&hmask;
                int lo=0,hi=Wl;
                while(lo<hi){int mid=(lo+hi)>>1;if(wsort[mid]<pm)lo=mid+1;else hi=mid;}
                if(lo<Wl&&wsort[lo]==pm){int win=wsort2w[lo];
                    int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);}
            }
        }
        __syncthreads();
        if(tid==0&&compact&&!overflowed)have_cand=1;
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
_RADIX_SELECT_F32_V4_KERNEL = None


def _get_radix_select_f32_v4_kernel():
    """The compaction v4 f32 radix select (``None`` when ``MLFRAME_RADIX_F32_V4=0`` or it does not compile); bit-identical order statistics to v3."""
    global _RADIX_SELECT_F32_V4_KERNEL
    import cupy as cp

    if os.environ.get("MLFRAME_RADIX_F32_V4", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    if _RADIX_SELECT_F32_V4_KERNEL is None:
        try:
            _RADIX_SELECT_F32_V4_KERNEL = cp.RawKernel(_RADIX_SELECT_F32_V4_SRC, "radix_select_f32_v4")
        except Exception:
            log_throttle(logging.getLogger(__name__), "gpu_select.f32_v4_compile", logging.WARNING, "f32 v4 compaction kernel compile failed; v3", exc_info=True)
            _RADIX_SELECT_F32_V4_KERNEL = False
    return _RADIX_SELECT_F32_V4_KERNEL or None


# V5 = v4 with four independent global loads in flight per thread in the full-scan loop. nvprof on v4: 31 percent of the issue stalls wait for data, memory
# utilisation 4 to 5 of 10, because each thread issues one dependent load per iteration; here a thread loads four elements (strided by the block width, so the warp
# reads stay coalesced) before it bins any of them. The bins receive the same elements, so the order statistics are bit-identical to v4. Off switch MLFRAME_RADIX_F32_V5=0.
_V5_HEAD_OLD = """            for(long long i=tid;i<n;i+=nt){
                float d=data[(long long)col*n+i];unsigned int u;memcpy(&u,&d,4);  // COLUMN-MAJOR: coalesced
"""
_V5_HEAD_NEW = """            for(long long i0=tid;i0<n;i0+=4LL*nt){
              float dv[4];
              #pragma unroll
              for(int k=0;k<4;++k){long long ii=i0+(long long)k*nt;dv[k]=(ii<n)?data[(long long)col*n+ii]:0.0f;}  // COLUMN-MAJOR: coalesced, four loads in flight
              #pragma unroll
              for(int k=0;k<4;++k){
                long long i=i0+(long long)k*nt;
                if(i<n){
                float d=dv[k];unsigned int u;memcpy(&u,&d,4);
"""
_V5_TAIL_OLD = """                        if((long long)slot<cap)cand[(long long)col*cap+(long long)slot]=u;else overflowed=1;}}
            }
        } else {"""
_V5_TAIL_NEW = """                        if((long long)slot<cap)cand[(long long)col*cap+(long long)slot]=u;else overflowed=1;}}
                }
              }
            }
        } else {"""
if not (_RADIX_SELECT_F32_V4_SRC.count(_V5_HEAD_OLD) == 1 and _RADIX_SELECT_F32_V4_SRC.count(_V5_TAIL_OLD) == 1):
    raise AssertionError('_RADIX_SELECT_F32_V4_SRC.count(_V5_HEAD_OLD) == 1 and _RADIX_SELECT_F32_V4_SRC.count(_V5_TAIL_OLD) == 1')
_RADIX_SELECT_F32_V5_SRC = _RADIX_SELECT_F32_V4_SRC.replace("radix_select_f32_v4", "radix_select_f32_v5").replace(_V5_HEAD_OLD, _V5_HEAD_NEW, 1).replace(_V5_TAIL_OLD, _V5_TAIL_NEW, 1)
_RADIX_SELECT_F32_V5_KERNEL = None


def _get_radix_select_f32_v5_kernel():
    """The four-loads-in-flight f32 radix select (``None`` when ``MLFRAME_RADIX_F32_V5=0`` or it does not compile); bit-identical order statistics to v4."""
    global _RADIX_SELECT_F32_V5_KERNEL
    import cupy as cp

    if os.environ.get("MLFRAME_RADIX_F32_V5", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    if _RADIX_SELECT_F32_V5_KERNEL is None:
        try:
            _RADIX_SELECT_F32_V5_KERNEL = cp.RawKernel(_RADIX_SELECT_F32_V5_SRC, "radix_select_f32_v5")
        except Exception:
            log_throttle(logging.getLogger(__name__), "gpu_select.f32_v5_compile", logging.WARNING, "f32 v5 four-loads kernel compile failed; v4", exc_info=True)
            _RADIX_SELECT_F32_V5_KERNEL = False
    return _RADIX_SELECT_F32_V5_KERNEL or None
