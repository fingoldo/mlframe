"""Compare the per-column relevance MRMR records for raw vs a monotone variant (norescue, fe_max_steps=0): is the MI itself invariant and only the borderline decision flipped?"""
import sys, warnings
warnings.simplefilter("ignore")
import pandas as pd
import tt_b5_invariance as I
from mlframe.feature_selection.filters import MRMR
fx, ds, var, scheme = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
X, y, inf, kind = I.make(fx, 3000, ds)
Xv, yv = I.variant(fx, X, y, var)
res = {}
for lab, (A, b) in (("raw", (X, y)), (var, (Xv, yv))):
    df = pd.DataFrame(A, columns=[f"x{i}" for i in range(A.shape[1])])
    m = MRMR(binning_scheme=scheme, fe_max_steps=0, verbose=0, n_jobs=1, random_state=0, fe_residual_rescue_enable=False).fit(df, b)
    pv = m.fe_provenance_
    res[lab] = (list(m.get_feature_names_out()), pv)
    print(lab, "selected", list(m.get_feature_names_out()))
    print(pv.to_string(max_cols=12, max_colwidth=22)[:2500])
