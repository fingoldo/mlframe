"""Run-to-run determinism probe of the per-cell moment kernel: 20 calls on the same device data, how many differ bitwise from the first and by how much (the double atomicAdds of the kernel add in thread-scheduling order)."""

import numpy as np, cupy as cp
from mlframe.feature_selection.filters._cell_moments_kernel import masked_cell_moments
rng=np.random.default_rng(0); n=100000; nc=64
codes=cp.asarray(rng.integers(0,nc,n).astype(np.int64)); v=cp.asarray(rng.standard_normal(n)); w=cp.asarray(rng.random(n))
outs=[masked_cell_moments(cp,codes,v,w,nc) for _ in range(20)]
print(type(outs[0]), None if outs[0] is None else len(outs[0]))
if outs[0] is not None:
    for j in range(len(outs[0])):
        arrs = [cp.asnumpy(o[j]) for o in outs if o is not None]
        print(j, sum(not np.array_equal(arrs[0], a) for a in arrs[1:]), max(float(np.max(np.abs(arrs[0] - a))) for a in arrs[1:]))
