"""What does MRMR's current FE (fe_max_steps=1) engineer on feature-transform fixtures? (research agent 2)"""
import sys, warnings
import numpy as np, pandas as pd
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0] + "/nbins_strategy")
from tt_a2_feature_transform import make
from mlframe.feature_selection.filters import MRMR

for kind in sys.argv[1:] or ["log", "recip", "square", "sqrt"]:
    for seed in (0, 1):
        x, z, y = make(kind, 20000, seed)
        r = np.random.default_rng(seed + 5)
        X = pd.DataFrame({"x": x, "z": z, **{f"n{i}": r.standard_normal(20000) for i in range(4)}})
        m = MRMR(fe_max_steps=1, verbose=0, random_state=seed).fit(X, y)
        pv = m.fe_provenance_
        print("##", kind, seed, "support_", getattr(m, "support_", None))
        print("##   engineered/selected rows:", pv[pv["in_output"]][["feature_name", "origin", "mrmr_gain"]].to_string(index=False), flush=True)
        print("##   unary-looking names in provenance:", [f for f in pv["feature_name"] if any(t in str(f) for t in ("log", "sqrt", "inv", "recip", "square", "abs", "__LL", "pow"))][:12], flush=True)
