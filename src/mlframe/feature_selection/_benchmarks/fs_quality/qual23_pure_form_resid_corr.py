"""qual-23 bench: min_resid_corr floor in retain_usable_pure_forms (FE pure-form retention).

LEVER: the relevance floor `min_resid_corr` in `_adds_nonlinear_value` gates whether a non-separable JOINT engineered
pair form (residual after the additive single-operand basis) is admitted to the linearly-usable retention. The legacy
floor 0.08 rejects a WEAK-but-genuine joint term whose residual-vs-y correlation lands in [0.05, 0.08); lowering it to
0.05 recovers that pure form so a linear downstream can use it.

Method (fit-cap honest): fit MRMR ONCE per (scenario, seed); then call retain_usable_pure_forms on the SAME fitted state
with min_resid_corr=0.08 (OLD) and =0.05 (NEW). Metric = downstream HONEST OOF R^2 of a linear model on
[selected raws + engineered recipes + retained pure forms], measured by 5-fold OOF. NEW wins if it recovers a relevant
pure form that lifts OOF R^2 without regressing the control scenario (where no weak joint term exists -> no-op).

Run: PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.fs_quality.qual23_pure_form_resid_corr
"""
import os, sys, time
os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
import scipy.stats, numba  # noqa
sys.modules.setdefault("cupy", None)
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters._fe_pure_form_retention import retain_usable_pure_forms


def make_data(scenario, seed, n=1100):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n); b = rng.uniform(0.8, 2.5, n); c = rng.uniform(0.5, 3, n); d = rng.normal(size=n)
    noise = {f"z{i}": rng.normal(size=n) for i in range(5)}
    if scenario == "weak_joint":
        # dominant additive linear part + a WEAK non-separable joint ratio term (low resid-corr ~0.05-0.08).
        strong = 3.0 * a + 2.0 * d + 1.5 * c
        weak = 0.45 * (a**2 / b)  # joint, non-separable, weak contribution
        y = strong + weak + 0.4 * rng.normal(size=n)
    else:  # control: purely additive-separable, NO joint term -> NEW must be a no-op
        y = 3.0 * a + 2.0 * d + 1.5 * c + 0.8 * np.sin(b) + 0.4 * rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, **noise})
    return X, pd.Series(y)


def oof_r2(X, y, recipes, raw_cols, seed):
    yv = y.to_numpy(float)
    cols = []
    for nm in raw_cols:
        if nm in X.columns:
            cols.append(np.nan_to_num(X[nm].to_numpy(float)))
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe
    for r in recipes:
        try:
            v = np.nan_to_num(np.asarray(apply_recipe(r, X), float).ravel())
            if v.shape[0] == len(X):
                cols.append(v)
        except Exception:
            pass
    if not cols:
        return float("nan")
    M = np.column_stack(cols)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.zeros_like(yv)
    for tr, te in kf.split(M):
        mdl = make_pipeline(StandardScaler(), LinearRegression()).fit(M[tr], yv[tr])
        pred[te] = mdl.predict(M[te])
    ss_res = float(((yv - pred) ** 2).sum()); ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def run():
    scenarios = ["weak_joint", "control"]
    seeds = [0, 1, 2, 3]
    rows = []
    t0 = time.time(); nfit = 0
    for scen in scenarios:
        for sd in seeds:
            X, y = make_data(scen, sd)
            m = MRMR(max_runtime_mins=1.5, random_seed=sd, verbose=0)
            m.fit(X, y); nfit += 1
            yc = getattr(m, "_fe_prewarp_y_continuous_", None)
            feat_in = list(getattr(m, "feature_names_in_", []) or [])
            sup = np.asarray(getattr(m, "support_", []), dtype=int).ravel()
            raw_cols = [feat_in[i] for i in sup if 0 <= i < len(feat_in)]
            base_recipes = list(getattr(m, "_engineered_recipes_", []) or [])
            res = {}
            for tag, corr in (("OLD", 0.08), ("NEW", 0.05)):
                extra = retain_usable_pure_forms(m, X, yc, seed=sd, min_resid_corr=corr)
                recipes = base_recipes + [r for r, _ in extra]
                r2 = oof_r2(X, y, recipes, raw_cols, sd)
                res[tag] = (r2, len(extra))
            rows.append((scen, sd, res["OLD"][0], res["NEW"][0], res["OLD"][1], res["NEW"][1]))
            print(f"{scen:11s} seed={sd}  OLD_r2={res['OLD'][0]:.4f}(+{res['OLD'][1]})  NEW_r2={res['NEW'][0]:.4f}(+{res['NEW'][1]})  d={res['NEW'][0]-res['OLD'][0]:+.4f}")
    print(f"\nTOTAL MRMR fits: {nfit}  wall={time.time()-t0:.0f}s")
    print("\n=== SUMMARY ===")
    for scen in scenarios:
        sub = [r for r in rows if r[0] == scen]
        wins = sum(1 for r in sub if r[3] > r[2] + 1e-4)
        reg = sum(1 for r in sub if r[3] < r[2] - 1e-4)
        md = float(np.median([r[3] - r[2] for r in sub]))
        print(f"{scen:11s}: NEW>OLD in {wins}/{len(sub)} seeds, regress {reg}/{len(sub)}, median dR2={md:+.4f}")
    return rows


if __name__ == "__main__":
    run()
