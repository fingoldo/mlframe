"""Time of the honest-null accept block (``_continuous_accept_block``: sorted order and ties on the host, the ``root_split_accept`` kernel on the device) for ROWS x COLUMNS x PERMUTATIONS.

Usage: ``python bench_honest_accept_block.py ROWS COLUMNS PERMUTATIONS [REPS]``. For the kernel alone run it under ``nvprof --print-gpu-trace`` (``--kernels root_split_accept --metrics achieved_occupancy,sm_efficiency,
issue_slot_utilization,stall_sync`` with the CUPTI directory on PATH; this host is Pascal, ``ncu`` refuses it) and read the ``root_split_accept`` lines.
"""

import sys,time,warnings
warnings.simplefilter("ignore")
import numpy as np, cupy as cp
from mlframe.feature_selection.filters._honest_null_accept import _continuous_accept_block
from mlframe.feature_selection.filters._family_maxt_null_v2_njit import xlogx_table
n=int(sys.argv[1]); kc=int(sys.argv[2]); n_perm=int(sys.argv[3]); reps=int(sys.argv[4]) if len(sys.argv)>4 else 3
rng=np.random.default_rng(0)
raw=rng.standard_normal((n,kc)).astype(np.float64)
y=(rng.random(n)<0.5).astype(np.int8)
yperm=np.stack([rng.permutation(y) for _ in range(n_perm)]).astype(np.int8)
acc=np.zeros((n_perm,0),dtype=np.bool_); order=np.zeros((0,n),dtype=np.int32); tie=np.zeros((0,n),dtype=np.uint8)
def run():
    cp.cuda.Device().synchronize(); t=time.perf_counter()
    r=_continuous_accept_block(kc,raw,yperm,2,n,xlogx_table,cp,n_perm,max(10,int(0.001*n)),0.05,acc,order,tie)
    cp.cuda.Device().synchronize(); return time.perf_counter()-t
run()
ts=[run() for _ in range(reps)]
print("n=%d kc=%d perms=%d  block %.0f ms (best of %d)"%(n,kc,n_perm,1e3*min(ts),reps))
