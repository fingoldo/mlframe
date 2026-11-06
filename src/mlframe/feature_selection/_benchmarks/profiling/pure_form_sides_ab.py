"""Same-process A/B of drop_forms_not_beating_their_sides: the edge pool against one thread, 60 synthetic unary_binary candidates, 100000 rows."""
import os, time, warnings
from types import SimpleNamespace
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from mlframe.feature_selection.filters import _fe_pure_form_sides as S, _adaptive_nbins

rng = np.random.default_rng(0)
n, k = 100_000, 12
X = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(k)})
y = (X["x0"].to_numpy() * X["x1"].to_numpy() + 0.3 * rng.standard_normal(n) > 0).astype(int)
cands = []
for i in range(60):
    a, b = f"x{i % k}", f"x{(i * 5 + 1) % k}"
    rec = SimpleNamespace(kind="unary_binary", src_names=(a, b), unary_names=("identity", "identity"), unary_preset="all")
    cands.append(SimpleNamespace(name=f"c{i}", recipe=rec, values=(X[a].to_numpy() * X[b].to_numpy())))

orig_side = S._side_values
S._side_values = lambda recipe, X_: [np.asarray(X_[s].to_numpy(), dtype=np.float64) for s in recipe.src_names]


def run(jobs):
    """One timed call with the edge-jobs pool pinned to ``jobs`` threads."""
    os.environ["MLFRAME_EDGE_JOBS"] = str(jobs)
    t = time.perf_counter()
    out = S.drop_forms_not_beating_their_sides(cands, X, y, classification=True, seed=0)
    return time.perf_counter() - t, [c.name for c in out]


run(1)
res = []
for rep in range(1):
    t1, a = run(1)
    t8, b = run(8)
    assert a == b, "the verdicts differ between one thread and the pool"
    res.append((t1, t8))
    print(f"pair {rep}: 1 thread {t1:.2f} s, pool {t8:.2f} s, kept {len(a)} of {len(cands)}")
print("mean 1 thread %.2f s, pool %.2f s" % (np.mean([r[0] for r in res]), np.mean([r[1] for r in res])))
