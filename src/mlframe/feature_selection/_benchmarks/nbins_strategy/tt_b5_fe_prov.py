"""Does MRMR's fe_max_steps=1 even PROPOSE log(x0) / sqrt(x0) for a feature with a log / sqrt link, and what happens to the proposal?  Prints the provenance rows whose name
carries a unary op of x0, with in_output, mrmr_gain and dropped_by.   python tt_b5_fe_prov.py FIXTURE [scheme]    (waits until no tt_b5 queue / invariance job is alive)"""
import os, sys, time, warnings
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
import psutil


def others():
    me = os.getpid()
    for pr in psutil.process_iter(["pid", "cmdline", "name"]):
        cl = " ".join(pr.info["cmdline"] or [])
        if "python" in (pr.info["name"] or "").lower() and pr.info["pid"] != me and ("tt_b5_invariance.py" in cl or "tt_b5_queue.py" in cl):
            return True
    return False


while others():
    time.sleep(10)
import pandas as pd
import tt_b5_invariance as I
from mlframe.feature_selection.filters import MRMR
fx = sys.argv[1]
scheme = sys.argv[2] if len(sys.argv) > 2 else "quantile"
X, y, inf, kind = I.make(fx, 3000, 0)
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
t = time.perf_counter()
m = MRMR(binning_scheme=scheme, fe_max_steps=1, verbose=0, n_jobs=1, random_state=0).fit(df, y)
print(fx, scheme, "wall", round(time.perf_counter() - t), "s; output:", list(m.get_feature_names_out()))
pv = m.fe_provenance_
print("provenance rows:", len(pv), "; columns:", list(pv.columns))
sel = pv[pv["feature_name"].astype(str).str.contains(r"log|sqrt|cbrt|inv|recip|abs|exp|square|pow|He2|T2|LL", regex=True) & pv["feature_name"].astype(str).str.contains("x0")]
print(sel[["feature_name", "origin", "in_output", "mrmr_gain", "dropped_by"]].to_string(max_colwidth=40))
print("origins:", pv["origin"].value_counts().to_dict())
