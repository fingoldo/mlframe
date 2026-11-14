"""Float32 radix select with a 13-bit first digit (v6): the source and lazy getter, a sibling of ``_gpu_resident_select_f32_compact`` (v4/v5)."""

from __future__ import annotations

import logging
import os

from mlframe.utils.log_throttle import log_throttle

from ._gpu_resident_select_f32_compact import _RADIX_SELECT_F32_V5_SRC

# V6 = v5 with the digit schedule 13 / 8 / 8 / 3 bits instead of 8 / 8 / 8 / 8. The first pass has one window (every rank shares the empty prefix), so a 8192-bin shared
# histogram costs the same one scan as a 256-bin one; the 13-bit prefix (sign, 8 exponent bits, 4 mantissa bits: a 1/16 octave bucket) is then narrow enough for the
# candidate compaction to happen already in the SECOND scan, where v4/v5 compact in the third (after a 16-bit prefix). The kernel makes two full scans of the column
# instead of three; the remaining passes read the compacted candidates. The histograms are sums over the same element sets, so the order statistics are bit-identical to
# v5; a column whose candidates overflow ``cap`` keeps full scans (exact). REJECTED AS DEFAULT (kept, opt-in MLFRAME_RADIX_F32_V6=1): GTX 1050 Ti, n=30000/100000, K=100..400, 9/19 ranks, first digit 11, 12 and 13 bits, five column shapes (normal, lognormal, ties, binary, Cauchy): equal to v5 everywhere, but 0.5 to 1.1 times v5 speed (the 13-bit version 0.4 to 1.1; 11 bits 0.6 to 1.0; 12 bits 0.6 to 1.07). A wider first prefix lets more elements through the compaction (lognormal, ties) and the larger histogram costs shared memory and occupancy, which eats the saved scan. Needs 32 KB of dynamic shared memory (the 8192-bin first histogram).
FIRST_DIGIT_BITS = int(os.environ.get("MLFRAME_RADIX_F32_V6_BITS", "13"))  # 11 to 13: the width of the first digit; the later digits are 8, 8 and the rest
_CH = (1 << FIRST_DIGIT_BITS) // 256  # bins per chunk sum of the first pass
_NEW_SCAN = (
    "if(pass==0){   // one window: chunk sums of 32 bins in parallel, then each rank walks the 256 chunk sums and one chunk\n"
    "            for(int c=tid;c<256;c+=nt){unsigned int s=0u;for(int b=0;b<32;++b)s+=sh[c*32+b];csum[c]=s;}\n"
    "            __syncthreads();\n"
    "            if(tid<R){unsigned long long acc=below[tid];int chosen=0;long long want=ranks[tid];\n"
    "                for(int c=0;c<256;++c){unsigned long long cc=csum[c];if(acc+cc>(unsigned long long)want){\n"
    "                    for(int b=c*32;b<c*32+32;++b){unsigned long long x=sh[b];if(acc+x>(unsigned long long)want){chosen=b;break;}acc+=x;}break;}acc+=cc;}\n"
    "                below[tid]=acc;prefix[tid]=(prefix[tid]&hmask)|((unsigned int)chosen<<shift);}\n"
    "        } else\n"
    "        if(tid<R){int w=rank2w[tid];unsigned long long acc=below[tid];int chosen=0;long long want=ranks[tid];\n"
    "            for(int b=0;b<nb;++b){unsigned long long c=sh[w*nb+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}\n"
)
_NEW_SCAN = _NEW_SCAN.replace("c*32+b", "c*%d+b" % _CH).replace("b<32;", "b<%d;" % _CH).replace("b=c*32;b<c*32+32", "b=c*%d;b<c*%d+%d" % (_CH, _CH, _CH))
_OLD_SCAN = (
    "if(tid<R){int w=rank2w[tid];unsigned long long acc=below[tid];int chosen=0;long long want=ranks[tid];\n"
    "            for(int b=0;b<256;++b){unsigned long long c=sh[w*256+b];if(acc+c>(unsigned long long)want){chosen=b;break;}acc+=c;}\n"
)
_V6_REPLACEMENTS = (
    (
        "for(int byte=3;byte>=0;--byte){\n        int shift=byte*8;\n        unsigned int hmask=(byte==3)?0u:(0xFFFFFFFFu<<((byte+1)*8));",
        "for(int pass=0;pass<4;++pass){\n        const int shift=(pass==0)?%d:(pass==1)?%d:(pass==2)?%d:0;\n"
        "        const unsigned int hmask=(pass==0)?0u:(0xFFFFFFFFu<<((pass==1)?%d:(pass==2)?%d:%d));\n"
        "        const int nb=(pass==0)?%d:256;const unsigned int dmask=(unsigned int)(nb-1);"
        % (
            32 - FIRST_DIGIT_BITS,
            24 - FIRST_DIGIT_BITS,
            16 - FIRST_DIGIT_BITS,
            32 - FIRST_DIGIT_BITS,
            24 - FIRST_DIGIT_BITS,
            16 - FIRST_DIGIT_BITS,
            1 << FIRST_DIGIT_BITS,
        ),
    ),
    ("for(int s=tid;s<Wl*256;s+=nt)sh[s]=0u;", "for(int s=tid;s<Wl*nb;s+=nt)sh[s]=0u;"),
    ("(!have_cand)&&(cap>0)&&(byte==1)", "(!have_cand)&&(cap>0)&&(pass==1)"),
    (_OLD_SCAN, _NEW_SCAN),
    ("    __shared__ int overflowed;\n", "    __shared__ int overflowed;\n    __shared__ unsigned int csum[256];\n"),
)
_B = FIRST_DIGIT_BITS
_src = _RADIX_SELECT_F32_V5_SRC
for _old, _new in _V6_REPLACEMENTS:
    if _src.count(_old) != 1:
        raise AssertionError(_old)
    _src = _src.replace(_old, _new, 1)
if _src.count("int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);") != 2:
    raise AssertionError("histogram update sites")
_src = _src.replace("int dig=(int)((u>>shift)&0xFFu);atomicAdd(&sh[win*256+dig],1u);", "int dig=(int)((u>>shift)&dmask);atomicAdd(&sh[win*nb+dig],1u);")
_RADIX_SELECT_F32_V6_SRC = _src.replace("radix_select_f32_v5", "radix_select_f32_v6")
FIRST_DIGIT_SHARED_BYTES = (1 << FIRST_DIGIT_BITS) * 4  # the dynamic shared memory the 13-bit first histogram needs, whatever R is
_RADIX_SELECT_F32_V6_KERNEL = None


def _get_radix_select_f32_v6_kernel():
    """The 13-bit-first-digit f32 radix select (``None`` unless ``MLFRAME_RADIX_F32_V6=1``, or when it does not compile); bit-identical order statistics to v5."""
    global _RADIX_SELECT_F32_V6_KERNEL
    import cupy as cp

    if os.environ.get("MLFRAME_RADIX_F32_V6", "0").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    if _RADIX_SELECT_F32_V6_KERNEL is None:
        try:
            _RADIX_SELECT_F32_V6_KERNEL = cp.RawKernel(_RADIX_SELECT_F32_V6_SRC, "radix_select_f32_v6")
        except Exception:
            log_throttle(
                logging.getLogger(__name__), "gpu_select.f32_v6_compile", logging.WARNING, "f32 v6 wide-digit kernel compile failed; v5", exc_info=True
            )
            _RADIX_SELECT_F32_V6_KERNEL = False
    return _RADIX_SELECT_F32_V6_KERNEL or None
