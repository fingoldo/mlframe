"""Per-second CPU% (per core summary) and GPU% timeline during a warm 100k x 100 every-family fit; prints a 10 s-bucket table. Usage: util_timeline.py [n] [p]"""
import os, sys, time, threading, subprocess, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
import psutil
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
p = int(sys.argv[2]) if len(sys.argv) > 2 else 100
kw = F.all_families_kwargs(); kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
data = F._make(n, p, 0)
import collections
main = threading.get_ident()
PRE=("_stage_","hybrid_","screen_predictors","_run_fe_step","check_prospective_fe_pairs","_fe_stage_cascade_","_gate_engineered_accuracy","_second_pass","_emit_","score_","generate_","_fit_body","_assign_support","_selection_compaction","run_selection","retain_","prune_","_final")
def stage():
    import sys as _s
    fr=_s._current_frames().get(main); best=None
    while fr is not None:
        nm=fr.f_code.co_name
        if best is None and nm.startswith(PRE): best=nm
        fr=fr.f_back
    return best
def leaf():
    import sys as _s
    fr=_s._current_frames().get(main); g=fr
    while g is not None and 'filters' not in g.f_code.co_filename: g=g.f_back
    return (g.f_code.co_name+':'+str(g.f_lineno)) if g is not None else '?'
rows = []; stop = threading.Event()
def gpu():
    try:
        return float(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.split()[0])
    except Exception:
        return float("nan")
def sampler():
    t0 = time.time(); psutil.cpu_percent(percpu=True)
    while not stop.is_set():
        time.sleep(1.0)
        c = psutil.cpu_percent(percpu=True)
        rows.append((time.time() - t0, sum(c) / len(c), sum(1 for x in c if x > 50), gpu(), stage(), leaf()))
main = threading.get_ident()
fast=[]
def fsamp():
    t0=time.time()
    while not stop.is_set():
        time.sleep(0.1); fast.append((time.time()-t0, stage(), leaf()))
threading.Thread(target=fsamp,daemon=True).start()
th = threading.Thread(target=sampler, daemon=True); th.start()
t = time.time()
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
wall = time.time() - t; stop.set(); th.join()
print(f"WALL {wall:.1f} cores={psutil.cpu_count()}")
print("t0-t1   cpu%mean  cores>50%  gpu%")
for i in range(0, len(rows), 5):
    b = rows[i:i + 10]
    top=collections.Counter(r[4] for r in b).most_common(2)
    print(f"{int(b[0][0]):4d}-{int(b[-1][0]):4d}  {sum(r[1] for r in b)/len(b):6.1f}  {sum(r[2] for r in b)/len(b):6.1f}  {sum(r[3] for r in b)/len(b):6.1f}  {top}")
tl=collections.Counter(r[5] for r in rows if r[0]>75 or 16<r[0]<27)
print("LEAVES late/dip",tl.most_common(14))
tt=collections.Counter((r[1],r[2]) for r in fast if r[0]>72)
print("TAIL",[(k,round(v/10,1)) for k,v in tt.most_common(25)])
print(f"OVERALL cpu% {sum(r[1] for r in rows)/len(rows):.1f}  cores>50 {sum(r[2] for r in rows)/len(rows):.2f}  gpu% {sum(r[3] for r in rows)/len(rows):.1f}")
