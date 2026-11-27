"""Which value-based component of MRMR breaks rank invariance at fe_max_steps=0 under binning_scheme='quantile'?  Fits raw vs transformed variant of one case under
parameter overrides and prints the selected names.   python tt_b5_diag.py FIXTURE DS VARIANT [steps] [scheme]"""
import logging
import sys, warnings
warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
import tt_b5_invariance as I

fx, ds, var = sys.argv[1], int(sys.argv[2]), sys.argv[3]
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 0
scheme = sys.argv[5] if len(sys.argv) > 5 else "quantile"
X, y, inf, kind = I.make(fx, 3000, ds)
Xv, yv = I.variant(fx, X, y, var)
import pandas as pd
from mlframe.feature_selection.filters import MRMR
NR = dict(fe_residual_rescue_enable=False)
OVR = {
    "default": {},
    "norescue": NR,
    "norescue+policy_keep": dict(NR, redundancy_policy="keep"),
    "norescue+no_synergy_rescue": dict(NR, fe_synergy_prevalence_rescue_enable=False),
    "norescue+no_drop_raw_operands": dict(NR, fe_drop_redundant_raw_operands=False),
    "norescue+no_cond_resid": dict(NR, fe_conditional_residual_enable=False),
    "norescue+all_four_off": dict(NR, redundancy_policy="keep", fe_synergy_prevalence_rescue_enable=False, fe_drop_redundant_raw_operands=False, fe_conditional_residual_enable=False),
}
for name, o in OVR.items():
    out = []
    for _lab, (A, b) in (("raw", (X, y)), (var, (Xv, yv))):
        df = pd.DataFrame(A, columns=[f"x{i}" for i in range(A.shape[1])])
        try:
            m = MRMR(binning_scheme=scheme, fe_max_steps=steps, verbose=0, n_jobs=1, random_state=0, **o).fit(df, b)
            out.append(sorted(map(str, m.get_feature_names_out())))
        except Exception as e:
            logging.getLogger(__name__).debug("tt_b5_diag: ignored %s: %s", type(e).__name__, e)
            out.append("ERR " + repr(e)[:80])
    print(f"{name:22s} same={out[0] == out[1]}  raw={out[0]}  {var}={out[1]}", flush=True)
