"""B5 / question 1: does a monotone map of the TARGET or of a FEATURE change what MRMR selects, per binning_scheme?

Runs MRMR (one fit at a time, n_jobs=1, verbose=0) on fixtures with a known informative set, once on the raw data and once per monotone variant, for
binning_scheme in {quantile, uniform, mdlp} and fe_max_steps in {0, 1}. Every fit appends one JSON line (names from ``get_feature_names_out()``) to the
output file; a finished (fixture, variant, scheme, steps, seed) key is skipped on re-run, so the script can be stopped and resumed.

    python tt_b5_invariance.py OUT.jsonl --steps 0 --group reg
    python tt_b5_invariance.py OUT.jsonl --steps 1 --group reg_small

Analysis: ``python tt_b5_invariance.py OUT.jsonl --analyze``.

Regression fixtures (n rows, 6 columns x0..x5, standard normal unless noted; informative set in brackets):
  lognorm   y = exp(0.8 x0 + 0.5 x1 + 0.3 x2 + 0.3 e)               [0 1 2]  log y is linear
  cube      y = (3 + 0.6 x0 + 0.4 x1 + 0.25 x2 + 0.15 e)^3           [0 1 2]  cbrt y is linear
  weak      y = exp(0.9 x0 + 0.5 x1 + 0.2 x2 + 0.1 x3 + 0.5 e)       [0 1 2 3]  x3 is weak
  linear    y = 10 + 2 x0 + x1 + 0.5 x2 + e                          [0 1 2]
  inter     y = exp(0.7 x0 + 0.5 x1 x2 + 0.3 e)                      [0 1 2]
Classification-by-threshold (label = 1[latent > quantile]); the label is unchanged by ANY monotone map of the latent, so only feature maps are tested:
  c_lin     latent = 0.8 x0 + 0.5 x1 + 0.3 x2 + 0.5 e, cut at the median
  c_skew    x0, x1 lognormal; latent = log x0 + 0.6 log x1 + 0.3 x2 + 0.5 e, cut at the 70 % quantile
  c_tail    latent = 0.6 x0 + 0.5 x1 + 0.5 e, cut at the 90 % quantile
Unary-link fixtures (does the FE search find the transformed column; x0 is the special column, x1 a linear partner, x2..x5 noise):
  u_log     x0 lognormal(1), y = 2 log x0 + 0.5 x1 + 0.5 e
  u_sqrt    x0 uniform(0.1, 10), y = 2 sqrt x0 + 0.5 x1 + 0.3 e
  u_lin     x0 lognormal(1), y = x0 + 0.5 x1 + 0.5 e    (control: the raw column is the right one)
Target variants: raw, log, cbrt.  Feature variants (applied to every column): exp, cube, tanh (tanh(1.5 x)).  Unary-link feature variant: ``lin`` (x0 replaced by the
transformed column that makes the link linear: log x0, sqrt x0, x0) to compare against the raw skewed column.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings

import numpy as np

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)

SCHEMES = ("quantile", "uniform", "mdlp")
NCOL = 6


def _normals(rng, n, p=NCOL):
    return rng.standard_normal((n, p))


GRADED = np.array([1.0, 0.6, 0.35, 0.2, 0.12, 0.08])  # six informative columns of decreasing strength, then six pure-noise columns


def make(fixture: str, n: int, seed: int):
    """Return (X array n x NCOL, y array, informative set of column indices, kind)."""
    rng = np.random.default_rng(1000 + seed)
    X = _normals(rng, n, 12 if fixture.startswith("g_") else NCOL)
    e = rng.standard_normal(n)
    if fixture == "g_lognorm":
        y = np.exp(X[:, :6] @ GRADED * 0.9 + 0.4 * e); inf, kind = set(range(6)), "reg"
    elif fixture == "g_linear":
        y = 10 + 2 * (X[:, :6] @ GRADED) + e; inf, kind = set(range(6)), "reg"
    elif fixture == "g_cube":
        y = (3 + 0.6 * (X[:, :6] @ GRADED) + 0.15 * e) ** 3; inf, kind = set(range(6)), "reg"
    elif fixture == "g_clf":
        lat = X[:, :6] @ GRADED + 0.5 * e
        y = (lat > np.quantile(lat, 0.5)).astype(int); inf, kind = set(range(6)), "clf"
    elif fixture == "lognorm":
        y = np.exp(0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.3 * e); inf, kind = {0, 1, 2}, "reg"
    elif fixture == "cube":
        y = (3 + 0.6 * X[:, 0] + 0.4 * X[:, 1] + 0.25 * X[:, 2] + 0.15 * e) ** 3; inf, kind = {0, 1, 2}, "reg"
    elif fixture == "weak":
        y = np.exp(0.9 * X[:, 0] + 0.5 * X[:, 1] + 0.2 * X[:, 2] + 0.1 * X[:, 3] + 0.5 * e); inf, kind = {0, 1, 2, 3}, "reg"
    elif fixture == "linear":
        y = 10 + 2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + e; inf, kind = {0, 1, 2}, "reg"
    elif fixture == "inter":
        y = np.exp(0.7 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + 0.3 * e); inf, kind = {0, 1, 2}, "reg"
    elif fixture == "c_lin":
        lat = 0.8 * X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2] + 0.5 * e
        y = (lat > np.quantile(lat, 0.5)).astype(int); inf, kind = {0, 1, 2}, "clf"
    elif fixture == "c_skew":
        X[:, 0] = np.exp(rng.standard_normal(n)); X[:, 1] = np.exp(rng.standard_normal(n))
        lat = np.log(X[:, 0]) + 0.6 * np.log(X[:, 1]) + 0.3 * X[:, 2] + 0.5 * e
        y = (lat > np.quantile(lat, 0.7)).astype(int); inf, kind = {0, 1, 2}, "clf"
    elif fixture == "c_tail":
        lat = 0.6 * X[:, 0] + 0.5 * X[:, 1] + 0.5 * e
        y = (lat > np.quantile(lat, 0.9)).astype(int); inf, kind = {0, 1}, "clf"
    elif fixture == "u_log":
        X[:, 0] = np.exp(rng.standard_normal(n)); y = 2 * np.log(X[:, 0]) + 0.5 * X[:, 1] + 0.5 * e; inf, kind = {0, 1}, "reg"
    elif fixture == "u_sqrt":
        X[:, 0] = rng.uniform(0.1, 10, n); y = 2 * np.sqrt(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * e; inf, kind = {0, 1}, "reg"
    elif fixture == "u_lin":
        X[:, 0] = np.exp(rng.standard_normal(n)); y = X[:, 0] + 0.5 * X[:, 1] + 0.5 * e; inf, kind = {0, 1}, "reg"
    else:
        raise ValueError(fixture)
    return X, y, inf, kind


def variant(fixture, X, y, name):
    """Apply a named monotone variant. Returns (X', y')."""
    if name == "raw":
        return X, y
    if name == "log":
        assert y.min() > 0, "log target needs y > 0"
        return X, np.log(y)
    if name == "cbrt":
        return X, np.cbrt(y)
    if name == "exp":
        return np.exp(X), y
    if name == "cube":
        return X ** 3, y
    if name == "tanh":
        return np.tanh(1.5 * X), y
    if name == "lin":
        X2 = X.copy()
        if fixture in ("u_log", "c_skew"):
            X2[:, 0] = np.log(X[:, 0])
            if fixture == "c_skew":
                X2[:, 1] = np.log(X[:, 1])
        elif fixture == "u_sqrt":
            X2[:, 0] = np.sqrt(X[:, 0])
        return X2, y
    raise ValueError(name)


VARIANTS = {
    "g_lognorm": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "g_linear": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "g_cube": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "g_clf": ["raw", "exp", "cube", "tanh"],
    "lognorm": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "cube": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "weak": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "linear": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "inter": ["raw", "log", "cbrt", "exp", "cube", "tanh"],
    "c_lin": ["raw", "exp", "cube", "tanh"],
    "c_skew": ["raw", "lin", "exp", "cube", "tanh"],
    "c_tail": ["raw", "exp", "cube", "tanh"],
    "u_log": ["raw", "lin"],
    "u_sqrt": ["raw", "lin"],
    "u_lin": ["raw"],
}

#: group -> (fixtures, variants override or None)
GROUPS = {
    "reg": (["lognorm", "cube", "weak", "linear", "inter"], None),
    "clf": (["c_lin", "c_skew", "c_tail"], None),
    "graded": (["g_lognorm", "g_linear", "g_cube", "g_clf"], None),
    "graded_small": (["g_lognorm", "g_clf"], ["raw", "log", "cbrt", "exp"]),
    "unary": (["u_log", "u_sqrt", "u_lin"], None),
    # the fe_max_steps=1 fits are tens of seconds each; a reduced grid
    "reg_tiny": (["lognorm", "weak"], ["raw", "log", "exp"]),
    "reg_one": (["lognorm"], ["raw", "log", "exp"]),
    "clf_one": (["c_skew"], ["raw", "lin"]),
    "unary2": (["u_log", "u_sqrt"], None),
    "cube_tiny": (["cube"], ["raw", "cbrt"]),
    "clf_tiny": (["c_lin", "c_skew"], ["raw", "exp", "lin"]),
    "reg_small": (["lognorm", "cube", "weak", "linear"], ["raw", "log", "cbrt", "exp"]),
    "clf_small": (["c_lin", "c_skew"], ["raw", "exp", "lin"]),
}


def fit_names(Xv, yv, scheme, steps, seed, kind, ovr=None):
    """Fit once. Returns (names, wall, core): ``core`` is [(name, mrmr_gain rounded to 6 digits)] of the provenance rows in the output with a finite MI gain, i.e. what the MI greedy
    selected before the later value-based / heuristic stages."""
    import pandas as pd
    from mlframe.feature_selection.filters import MRMR
    df = pd.DataFrame(Xv, columns=[f"x{i}" for i in range(Xv.shape[1])])
    t = time.perf_counter()
    m = MRMR(binning_scheme=scheme, fe_max_steps=steps, verbose=0, n_jobs=1, random_state=seed, **(ovr or {})).fit(df, yv)
    wall = time.perf_counter() - t
    pv = m.fe_provenance_
    core = []
    try:
        for nm, g, ino in zip(pv["feature_name"], pv["mrmr_gain"], pv["in_output"]):
            if bool(ino) and g == g:
                core.append((str(nm), round(float(g), 6)))
    except Exception:
        core = []
    return [str(s) for s in m.get_feature_names_out()], wall, core


#: parameter overrides selectable with --tag; "norescue" switches off the value-based (held-out ridge) residual rescue that breaks rank invariance
TAGS = {"": {}, "norescue": dict(fe_residual_rescue_enable=False)}


def run(out, steps, group, n, seeds, extra_seed_jitter=True, tag="", schemes=SCHEMES):
    fixtures, vs = GROUPS[group]
    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r["fixture"], r["variant"], r["scheme"], r["steps"], r["seed"], r["fitseed"], r.get("tag", "")))
            except Exception:
                pass
    for fx in fixtures:
        for ds in seeds:
            X, y, inf, kind = make(fx, n, ds)
            vlist = vs if vs is not None else VARIANTS[fx]
            vlist = [v for v in vlist if v in VARIANTS[fx] or v in ("raw",)]
            jobs = [(v, 0) for v in vlist]
            if extra_seed_jitter:
                jobs.append(("raw", 1))  # baseline refit with another random_state: the selection jitter of MRMR itself
            for scheme in schemes:
                for v, fitseed in jobs:
                    key = (fx, v, scheme, steps, ds, fitseed, tag)
                    if key in done:
                        continue
                    try:
                        Xv, yv = variant(fx, X, y, v)
                    except AssertionError:
                        continue
                    try:
                        names, wall, core = fit_names(Xv, yv, scheme, steps, fitseed, kind, TAGS[tag])
                        err = None
                    except Exception as ex:  # keep going, record the failure
                        names, wall, err, core = [], float("nan"), repr(ex)[:300], []
                    rec = dict(fixture=fx, variant=v, scheme=scheme, tag=tag, steps=steps, seed=ds, fitseed=fitseed, n=n, informative=sorted(inf), names=names, core=core, wall=wall, error=err)
                    with open(out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{fx:9s} ds{ds} {v:5s} {scheme:8s} steps={steps} fs={fitseed} {wall:6.1f}s {len(names)} cols {names[:6]}", flush=True)


_SRC = re.compile(r"x(\d+)")


def sources(names):
    return sorted({int(m) for nm in names for m in _SRC.findall(nm)})


def direct_raw(names):
    return sorted(int(nm[1:]) for nm in names if re.fullmatch(r"x\d+", nm))


def analyze(path):
    rows = [json.loads(l) for l in open(path)]
    base = {}
    for r in rows:
        if r["variant"] == "raw" and r["fitseed"] == 0:
            base[(r["fixture"], r["scheme"], r["steps"], r["seed"], r.get("tag", ""))] = r
    out = []
    print(f"{'fixture':9s} {'ds':3s} {'steps':5s} {'scheme':8s} {'variant':6s} | same names | same direct-raw | same sources | recall(inf) prec(inf)  | extras vs raw")
    for r in sorted(rows, key=lambda r: (r["steps"], r["fixture"], r["scheme"], r["variant"], r["fitseed"], r["seed"])):
        b = base.get((r["fixture"], r["scheme"], r["steps"], r["seed"], r.get("tag", "")))
        if b is None or r is b:
            continue
        lab = r["variant"] + ("(seed1)" if r["fitseed"] == 1 else "")
        inf = set(r["informative"])
        src = set(sources(r["names"]))
        same_names = set(r["names"]) == set(b["names"])
        same_direct = direct_raw(r["names"]) == direct_raw(b["names"])
        same_src = sources(r["names"]) == sources(b["names"])
        rc, bc = dict(r.get("core", [])), dict(b.get("core", []))
        same_core = set(rc) == set(bc)
        core_gain_diff = max((abs(rc[k] - bc[k]) for k in set(rc) & set(bc)), default=0.0)
        dr = set(direct_raw(r["names"]))
        rec = len(inf & src) / len(inf)
        prec = (len(inf & src) / len(src)) if src else float("nan")
        print(f"{r['fixture']:9s} ds{r['seed']} {r['steps']:5d} {r['scheme'] + ('+' + r['tag'] if r.get('tag') else ''):17s} {lab:12s} | {same_names!s:5s} | {same_direct!s:5s} {sorted(dr)} | {same_src!s:5s} | {rec:.2f} {prec:.2f} | {[nm for nm in r['names'] if nm not in b['names']][:4]}")
        out.append(dict(fixture=r["fixture"], ds=r["seed"], steps=r["steps"], scheme=r["scheme"] + ("+" + r["tag"] if r.get("tag") else ""), variant=lab, seed=r["seed"], same_names=same_names, same_direct=same_direct, same_src=same_src, same_core=same_core, core_gain_diff=core_gain_diff))
    # aggregate: per (steps, scheme, family of variant): share of identical
    import collections
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0.0])
    for o in out:
        fam = "jitter(seed1)" if "(seed1)" in o["variant"] else ("target:" + o["variant"] if o["variant"] in ("log", "cbrt") else "feature:" + o["variant"])
        k = (o["steps"], o["scheme"], fam)
        a = agg[k]
        a[0] += 1; a[1] += o["same_names"]; a[2] += o["same_direct"]; a[3] += o["same_src"]; a[4] += o["same_core"]; a[5] = max(a[5], o["core_gain_diff"])
    print("\nAGGREGATE: (steps, scheme, family) -> cases, identical name set, identical direct-raw set, identical source set")
    for k in sorted(agg):
        a = agg[k]
        print(f"  {k}: {a[0]} cases, final names {a[1]}, direct-raw {a[2]}, sources {a[3]}, MI-core names {a[4]} (max |gain diff| {a[5]:.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--group", default="reg")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    ap.add_argument("--tag", default="", choices=sorted(TAGS))
    a = ap.parse_args()
    if a.analyze:
        analyze(a.out)
    else:
        run(a.out, a.steps, a.group, a.n, [int(s) for s in a.seeds.split(",")], tag=a.tag, extra_seed_jitter=not a.no_jitter, schemes=tuple(a.schemes.split(",")))
