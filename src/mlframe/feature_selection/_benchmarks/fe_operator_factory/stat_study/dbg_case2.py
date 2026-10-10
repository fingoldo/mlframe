"""Debug harness: fit MRMR on the case-2 data and list the rejection records that mention the ``(c, d)`` pair, to see which gate rejected it.
Run: ``python -m mlframe.feature_selection._benchmarks.fe_operator_factory.stat_study.dbg_case2``."""

import warnings

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR

N = 20000
import re
from pathlib import Path

TEST_FILE = Path(__file__).resolve().parents[6] / "tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_gate_vs_elementary.py"
src = TEST_FILE.read_text(encoding="utf-8") if TEST_FILE.exists() else ""
m = re.search(r"^N\s*=\s*([\d_]+)", src, re.M)
N = int(m.group(1).replace("_", "")) if m else N
print("N =", N)
rng = np.random.default_rng(0)
n = N
a, b, c, d, e, f = (rng.random(n) for _ in range(6))
y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2) * np.sin(d / 3)
df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
fs = MRMR(verbose=0, fe_max_steps=2).fit(df, pd.Series(y, name="y"))
print("selected:", list(fs.get_feature_names_out()))
recs = getattr(fs, "_fe_rejection_records_", None) or []
print("rejection records:", len(recs))
for r in recs:
    t = str(r)
    if ("c" in t and "d" in t) and ("log" in t or "sin" in t or "gate" in t or "argmax" in t):
        print("  ", t[:230])
print("---- non-dispersion records")
for r in recs:
    t = str(r)
    if "_by__" in t:
        continue
    print("  ", t[:260])
print("---- attrs")
for k in ("engineered_features_", "fe_pairs_", "_fe_steps_executed_", "selected_pairs_", "feature_engineering_report_"):
    if hasattr(fs, k):
        print(k, str(getattr(fs, k))[:300])
