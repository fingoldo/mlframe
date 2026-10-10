"""Write ``nvprof_auto.py``: the profiled fit with NVTX ranges around the ~70 most expensive mlframe functions of the last cProfile run.

Then run it under nvprof (CUPTI on PATH) and summarise with ``nvprof_range_summary.py``::

    nvprof --profile-from-start off --print-gpu-summary python nvprof_auto.py <n_rows> > out.txt
    python nvprof_range_summary.py out.txt

The summary gives kernel launches and GPU milliseconds per function: a range with thousands of launches and a few ms of GPU time is launch-bound (fuse it), one with few launches and many ms
is kernel-bound (optimise the kernel).
"""

import pstats
from pathlib import Path

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # this checkout, not an installed copy
from mlframe.feature_selection._benchmarks.profiling._paths import OUT_DIR, PROF_FILE  # noqa: E402

NVPROF_BASE = Path(__file__).with_name("nvprof_fit.py")
p = pstats.Stats(str(PROF_FILE)); st = p.stats
cands = []
for k, v in st.items():
    fn = k[0].replace("\\", "/")
    if "mlframe" not in fn or not fn.endswith(".py") or "/_benchmarks/" in fn: continue  # the profiling scripts themselves run a whole fit when imported
    if v[3] < 0.12: continue
    mod = fn.split("/src/")[-1][:-3].replace("/", ".")
    if mod.endswith(".__init__"): mod = mod[: -len(".__init__")]
    name = k[2]
    if name.startswith("<") or name.startswith("_run_fe_step") : continue
    cands.append((v[3], mod, name))
cands.sort(reverse=True)
seen = set(); out = []
for ct, mod, name in cands[:70]:
    if (mod, name) in seen: continue
    seen.add((mod, name)); out.append((mod, name))
src = NVPROF_BASE.read_text(encoding="utf-8")
inject = '''
import importlib
from cupy.cuda import nvtx
def _tag(modname, name, label):
    try:
        mod = importlib.import_module(modname)
        f = getattr(mod, name)
    except Exception:
        return
    def g(*a, **k):
        nvtx.RangePush(label)
        try: return f(*a, **k)
        finally: nvtx.RangePop()
    g.__name__ = name
    setattr(mod, name, g)
'''
for i, (mod, name) in enumerate(out):
    inject += f'_tag({mod!r}, {name!r}, "A_{name}")\n'
src = src.replace("\nfit(1)\n", inject + "\nfit(1)\n", 1)
(OUT_DIR / "nvprof_auto.py").write_text(src, encoding="utf-8")
print(len(out), "functions tagged")
