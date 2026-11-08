"""Time of ``scan_engineered_duplicates`` on 120 engineered columns (a third with NaN, near-duplicates of 12 base columns) and the kept / dropped sets, to compare two versions of the scan.

Usage: ``python bench_eng_dedup_scan.py ROWS TAG``; the kept and dropped sets are written to ``dedup_result_<ROWS>_<TAG>.json`` next to the working directory so that two tags can be compared.
Measured 2026-11-09 at 400000 rows, best of 3: 61.8 s before the pairs with NaN skipped the Python pre-check, 59.6 s after, the same 84 kept and 36 dropped columns.
"""

import sys,time,warnings,json
warnings.simplefilter("ignore")
import numpy as np,pandas as pd
from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_scan import scan_engineered_duplicates
n=int(sys.argv[1]); r=np.random.default_rng(0)
base=r.standard_normal((n,12)).astype(np.float32)
cols={}
for j in range(120):
    b=base[:,j%12]
    kind=j%6
    if kind==0: v=b*(1+0.01*r.standard_normal())+0.0001*r.standard_normal(n)       # near duplicate of a base column
    elif kind==1: v=np.sin(b*(1+j/50))
    elif kind==2: v=b**2+r.standard_normal(n)*0.5
    elif kind==3: v=base[:,(j+1)%12]*base[:,j%12]
    elif kind==4: v=np.abs(b)+0.1*r.standard_normal(n)
    else: v=np.tanh(b+r.standard_normal(n)*0.3)
    v=v.astype(np.float32)
    if j%3==0: v[r.random(n)<0.05]=np.nan                                      # a third of the columns carry NaN
    cols[f"e{j}"]=v
X=pd.DataFrame(cols)
prefer=lambda cand,kept: False
def run():
    t=time.perf_counter(); keep,drop,_,_=scan_engineered_duplicates(X,list(cols),set(),prefer); return time.perf_counter()-t,keep,drop
run()
res=[run() for _ in range(3)]
print("n=%d  scan %.2fs (best of 3)  kept %d dropped %d"%(n,min(r_[0] for r_ in res),len(res[0][1]),len(res[0][2])))
json.dump({"keep":res[0][1],"drop":sorted(res[0][2])},open(f"dedup_result_{n}_{sys.argv[2]}.json","w"))
