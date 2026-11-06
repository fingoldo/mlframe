"""One 100000 x 100 every-family GPU fit (gpu_walls protocol) with a system sampler: own vs other CPU, free memory, page faults, GPU utilisation, every 2 s."""
import os, sys, time, threading, subprocess, warnings
warnings.simplefilter("ignore")
import psutil
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
kw = {f: True for f in F._flag_names()}; kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
data = F._make(100000, 100, 0)
me = psutil.Process()
samples = []; stop = threading.Event()
def gpu_util():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().split()[0])
    except Exception:
        return -1
def sampler():
    prev_me = me.cpu_times(); prev_t = time.perf_counter(); psutil.cpu_percent(percpu=True)
    while not stop.wait(2.0):
        now = time.perf_counter(); ct = me.cpu_times()
        own = ((ct.user + ct.system) - (prev_me.user + prev_me.system)) / (now - prev_t)  # cores
        per = psutil.cpu_percent(percpu=True); tot = sum(per) / 100.0                       # cores busy system-wide
        vm = psutil.virtual_memory(); sw = psutil.swap_memory()
        samples.append((now, own, tot, vm.available / 2**20, sw.used / 2**20, me.memory_info().rss / 2**20, getattr(me.memory_info(), "num_page_faults", 0), gpu_util()))
        prev_me, prev_t = ct, now
th = threading.Thread(target=sampler, daemon=True); th.start()
MRMR._FIT_CACHE.clear()
t = time.perf_counter(); MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data); w = time.perf_counter() - t
stop.set(); th.join()
import statistics as S
own = [s[1] for s in samples]; tot = [s[2] for s in samples]
print("WALL", round(w, 1), "samples", len(samples))
print("own cores avg %.2f p90 %.2f | system cores busy avg %.2f | others avg %.2f" % (S.mean(own), sorted(own)[int(.9 * len(own))], S.mean(tot), S.mean([max(0, t_ - o) for t_, o in zip(tot, own)])))
print("available MB min %.0f avg %.0f | swap used MB max %.0f | rss MB max %.0f" % (min(s[3] for s in samples), S.mean(s[3] for s in samples), max(s[4] for s in samples), max(s[5] for s in samples)))
pf = [s[6] for s in samples]; print("page faults total %d" % (pf[-1] - pf[0]))
g = [s[7] for s in samples if s[7] >= 0]; print("gpu util avg %.0f max %d" % (S.mean(g), max(g)) if g else "gpu n/a")
