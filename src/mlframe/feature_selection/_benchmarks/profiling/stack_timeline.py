"""Per 6-second bucket of a warm 100000 x 100 GPU fit: the innermost mlframe function of the main thread and the number of live mlframe worker threads."""
import collections, os, sys, threading, time, warnings
warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR

kw = F.all_families_kwargs()
kw["nbins_strategy"] = os.environ.get("BENCH_NBINS", "mdlp")
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))
data = F._make(100000, 100, 0)

main_id = threading.get_ident()
buckets = collections.defaultdict(collections.Counter)
workers = collections.defaultdict(list)
stop = threading.Event()
t0 = time.perf_counter()


def innermost(frame):
    """The innermost frame inside mlframe (module:function)."""
    while frame is not None:
        fn = frame.f_code.co_filename.replace("\\", "/")
        if "/mlframe/" in fn and "stack_timeline" not in fn:
            return f"{fn.split('/mlframe/')[-1]}:{frame.f_code.co_name}:{frame.f_lineno}"
        frame = frame.f_back
    return "(outside mlframe)"


def sampler():
    """Sample the main thread's stack and count the other threads that are inside mlframe."""
    while not stop.wait(0.25):
        b = int((time.perf_counter() - t0) // 6)
        frames = sys._current_frames()
        buckets[b][innermost(frames.get(main_id))] += 1
        n = sum(1 for tid, fr in frames.items() if tid not in (main_id, threading.get_ident()) and innermost(fr) != "(outside mlframe)")
        workers[b].append(n)


th = threading.Thread(target=sampler, daemon=True)
th.start()
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
stop.set()
print("WALL", round(time.perf_counter() - t0, 1))
for b in sorted(buckets):
    top = buckets[b].most_common(4)
    w = workers[b]
    print(f"{b*6:4d}-{b*6+6:4d}s workers_in_mlframe avg {sum(w)/max(1,len(w)):4.1f} | " + " | ".join(f"{k} x{v}" for k, v in top))
