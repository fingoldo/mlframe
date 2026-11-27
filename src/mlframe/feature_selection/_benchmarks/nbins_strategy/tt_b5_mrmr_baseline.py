"""Wall and CPU time of ONE MRMR fit at n=20000, p=20 (the same data kind as tt_b5_bench: standard normal columns, y = exp(0.8 x0 + 0.5 x1 + 0.3 x2 + 0.3 e)), for fe_max_steps in {1 (default), 0}.
Waits until no other tt_b5 MRMR job is alive.   python tt_b5_mrmr_baseline.py OUT.jsonl"""
import logging
import json, os, sys, time, warnings
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
import psutil


def others():
    me = os.getpid()
    for pr in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            cl = " ".join(pr.info["cmdline"] or [])
        except Exception as exc:
            logging.getLogger(__name__).debug("tt_b5_mrmr_baseline: ignored %s: %s", type(exc).__name__, exc)
            continue
        if "python" not in (pr.info["name"] or "").lower():
            continue  # skip nohup / sh wrappers, which carry the same command line
        if ("tt_b5_invariance.py" in cl or "tt_b5_queue.py" in cl) and pr.info["pid"] != me:
            return True
    return False


while others():
    time.sleep(10)
import numpy as np, pandas as pd
from mlframe.feature_selection.filters import MRMR
n, p = 20000, 20
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p)).astype(np.float32)
y = np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.3 * rng.standard_normal(n))
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
proc = psutil.Process()
for steps in (0, 1):
    c0 = proc.cpu_times(); t = time.perf_counter()
    m = MRMR(fe_max_steps=steps, verbose=0, n_jobs=1, random_state=0).fit(df, y)
    wall = time.perf_counter() - t; c1 = proc.cpu_times()
    rec = dict(kind="mrmr", n=n, p=p, fe_max_steps=steps, wall_s=round(wall, 2), cpu_s=round(c1.user + c1.system - c0.user - c0.system, 2), peak_wset_MB=round(proc.memory_info().peak_wset / 2 ** 20, 1),
               n_selected=len(m.get_feature_names_out()), names=[str(s) for s in m.get_feature_names_out()][:12], load=psutil.cpu_percent(interval=None))
    open(sys.argv[1], "a").write(json.dumps(rec) + "\n")
    print(rec, flush=True)
