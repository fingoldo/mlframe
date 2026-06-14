"""iter101 @10M: rare-value merge in analyse_and_clean_features via vectorized isin+mask vs per-cell .replace().
Run: python bench_rareval_merge_iter101.py    Drives the full cleaning pass on a 10M discrete column with a rare tail.
Result: NEW 13.5x faster on the merge step, full-frame bit-identical. See _loop_iter_log.md iter101."""
import sys; sys.modules['cupy']=None
import numpy as np, pandas as pd, time
import scipy.stats, numba
from mlframe.preprocessing.cleaning import analyse_and_clean_features

def make(n=10_000_000, seed=0):
    rng=np.random.RandomState(seed)
    a=rng.randint(0,50,n)
    rp=rng.choice(n,300_000,replace=False)
    a[rp]=rng.randint(1000,1080,300_000)
    return pd.DataFrame({"x":a.astype('int64')})

if __name__=="__main__":
    out=make()
    t=time.perf_counter()
    analyse_and_clean_features(out, update_data=True, verbose=0)
    dt=time.perf_counter()-t
    print(f"clean wall: {dt:.3f}s ; nunique_after={out['x'].nunique(dropna=False)}")
    vc=out['x'].value_counts(dropna=False).sort_index()
    print("VC_HASH", pd.util.hash_pandas_object(vc).sum(), "NNA", int(out['x'].isna().sum()))
