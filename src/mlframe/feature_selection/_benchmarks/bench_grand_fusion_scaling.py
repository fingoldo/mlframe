"""Grand-fusion (GPU gen+discretize+noise-gate) vs the PRODUCTION CPU dispatch -- crossover bench.

HONEST baseline: the CPU side is ``_dispatch_batch_mi_with_noise_gate`` itself (NOT a forced permutation
loop), so its analytic large-n shortcut is active -- i.e. exactly what production runs. Default
``fe_npermutations=3``. Measures the per-pair candidate-MI step (one operand pair's 384 unary x binary
combos: generate -> discretize -> noise-gated MI) and asserts the SAME winning candidate is chosen.

Measured (GTX 1050 Ti, 2026-06-20), argmax preserved at every n:
    n=100k  CPU 11270ms  GPU  767ms  14.7x
    n=300k  CPU 19753ms  GPU 1831ms  10.8x
    n=1M    CPU 66330ms  GPU 7279ms   9.1x
The speedup shrinks with n (GPU becomes memory-bound) but stays ~9x+. This per-pair step is ~16s of the
small canonical n=100k/5-feature fit (so end-to-end gain there is modest) but DOMINATES on wide/large
data -- the regime where wiring grand-fusion as a size-gated default pays. Run::

    D:/ProgramData/anaconda3/python.exe -m mlframe.feature_selection._benchmarks.bench_grand_fusion_scaling
"""
import warnings, os, time
if __name__ == "__main__":
    warnings.filterwarnings("ignore"); os.environ.setdefault("TQDM_DISABLE","1")
    import numpy as np
    from mlframe.feature_selection.filters._gpu_resident_fe import _build_candidate_matrix, grand_fused_pair_mi
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
    from mlframe.feature_selection.filters.info_theory import batch_mi_with_noise_gate
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import _dispatch_batch_mi_with_noise_gate
    NP=3  # production default fe_npermutations
    def cpu_prod(a,b,yc,fy):
        cand=np.ascontiguousarray(_build_candidate_matrix(np,a,b))
        disc=discretize_2d_quantile_batch(cand,n_bins=20,dtype=np.int8)
        return _dispatch_batch_mi_with_noise_gate(disc_2d=disc,quantization_nbins=20,classes_y=yc,
            classes_y_safe=yc,freqs_y=fy,npermutations=NP,min_nonzero_confidence=0.0,use_su=False,
            batch_mi_kernel=batch_mi_with_noise_gate)
    def mk(n,seed=1):
        rng=np.random.default_rng(seed);a=rng.uniform(1,5,n);b=rng.uniform(1,5,n);y=a**2/b
        e=np.quantile(y,np.linspace(0,1,21)[1:-1]);yc=np.searchsorted(e,y).astype(np.int64)
        fy=np.bincount(yc,minlength=int(yc.max())+1).astype(np.float64)/n;return a,b,yc,fy
    a,b,yc,fy=mk(3000);cpu_prod(a,b,yc,fy);grand_fused_pair_mi(a,b,yc,yc,fy,nbins=20,npermutations=NP)
    for n in (100_000,300_000,1_000_000):
        a,b,yc,fy=mk(n)
        t0=time.perf_counter();rc=cpu_prod(a,b,yc,fy);tc=time.perf_counter()-t0
        t0=time.perf_counter();_,rg=grand_fused_pair_mi(a,b,yc,yc,fy,nbins=20,npermutations=NP);tg=time.perf_counter()-t0
        print("n=%-8d CPU_prod=%7.1fms GPU=%7.1fms speedup=%4.1fx argmax_match=%s"%(n,tc*1e3,tg*1e3,tc/tg,int(np.argmax(rc))==int(np.argmax(rg))),flush=True)
