"""Four-pillar bench for default MRMR: F2 across distributions, pure noise, 3-way XOR among noise, conjunction with proxies.

One representative fit per pillar case, timed, with the pillar's own pass/fail contract (mirrors the biz-value tests:
``tests/feature_selection/mrmr/core/test_f2_single_compound_across_distributions.py``,
``tests/feature_selection/biz_val/test_pure_noise_selects_nothing.py``,
``tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_xor3_noise.py``,
``tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_conjunction_decoys.py``). The pytest files remain the contract;
this bench is the fast loop for profiling and optimisation.

Usage::

    python -m mlframe.feature_selection._benchmarks.bench_four_pillars [--only f2,noise,xor3,conj] [--profile out.prof]
        [--f2-default]   # also run F2 with default MRMR() instead of the test's tuned parameters
        [--extra all|name,name] [--panel R] [--strict-extra]   # the hard cases of ``four_pillars_cases`` over a seed panel, judged by Clopper-Pearson bounds

Run once with ``MLFRAME_FE_GPU_STRICT=1`` for the GPU path and once without for the CPU path.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import re
import sys
import time
import warnings

import numpy as np


def _tokens(name) -> set[str]:
    """Tokens."""
    return set(re.split(r"[^A-Za-z0-9]+", str(name)))


def _fit(X, y, **kw):
    """Fit."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        sel = MRMR(verbose=0, **kw).fit(X, y)
        return sel, time.perf_counter() - t


# ----------------------------------------------------------------------------------------------- F2 across distributions
_F2_ID = re.compile(r"[a-zA-Z_]\w*")


def _f2_cols(nm):
    """F2 cols."""
    return set(_F2_ID.findall(nm)) & {"a", "b", "c", "d", "e"}


def _f2_case(profile: str, default: bool):
    """F2 case."""
    sys.path.insert(0, os.getcwd())
    from tests.feature_selection.mrmr.core.test_f2_single_compound_across_distributions import _make, _classify

    df, y = _make(profile, n=10_000, seed=42)
    kw = {} if default else dict(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, n_jobs=1)
    sel, dt = _fit(df, y, **kw)
    names = [str(s) for s in sel.get_feature_names_out()]
    full, frag_ab, frag_cd = _classify(names)
    ok = all("e" not in _f2_cols(nm) for nm in names) and len(full) == 1 and not frag_ab and not frag_cd and "c" in _f2_cols(full[0])
    return ok, dt, names


# ----------------------------------------------------------------------------------------------- pure noise
def _noise_case(kind: str, target: str, strategy):
    """Noise case."""
    sys.path.insert(0, os.getcwd())
    from tests.feature_selection.biz_val.test_pure_noise_selects_nothing import _frame, DISTRIBUTIONS, N

    seed = 11 + DISTRIBUTIONS.index(kind)
    X = _frame(kind, seed)
    rng = np.random.default_rng(seed + 1000)
    y = (rng.random(N) > 0.5).astype(np.int64) if target == "binary" else rng.normal(size=N)
    sel, dt = _fit(X, y, random_state=0, n_jobs=1, nbins_strategy=strategy)
    ok = getattr(sel, "no_significant_features_", False) is True
    return ok, dt, list(sel.get_feature_names_out())


# ----------------------------------------------------------------------------------------------- xor3 among noise
def _xor3_case(y_kind: str, seed: int):
    """Xor3 case."""
    sys.path.insert(0, os.getcwd())
    from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_xor3_noise import _make_frame, _support_problems

    out = _make_frame(y_kind, seed=seed)
    X, y = out[0], out[1]
    sel, dt = _fit(X, y)
    names = list(sel.get_feature_names_out())
    return not _support_problems(names), dt, names


# ----------------------------------------------------------------------------------------------- conjunction with proxies
def _conj_case(perm_seed):
    """Conj case."""
    sys.path.insert(0, os.getcwd())
    from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_conjunction_decoys import _make_frame, _support_problems

    X, y = _make_frame(seed=0, perm_seed=perm_seed)
    sel, dt = _fit(X, y)
    names = list(sel.get_feature_names_out())
    ok = not _support_problems(names)
    return ok, dt, names


# ----------------------------------------------------------------------------------------------- extended hard cases over a seed panel
def _extra_row(case, seed: int):
    """Fit one seed of an extended case and return its contract verdict plus the fit wall."""
    from mlframe.feature_selection._benchmarks import four_pillars_contracts as contracts
    from mlframe.feature_selection.filters.mrmr import MRMR

    kwargs = dict(case.gen_kwargs)
    out = case.gen(seed, **kwargs)
    X, y = out[0], out[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        sel = MRMR(verbose=0, random_state=seed, n_workers=1).fit(X, y)
        dt = time.perf_counter() - t
    if case.kind == "f2":
        verdict = contracts.f2_verdict(sel, lambda sd, n: case.gen(sd, n, **kwargs), seed, **case.contract)
    elif case.kind == "null":
        verdict = contracts.null_verdict(sel)
    else:
        verdict = contracts.cover_verdict(sel, **case.contract)
    verdict["seconds"] = dt
    verdict["selected"] = contracts.selected_names(sel)[:6]
    return verdict


def _run_extra(names: set[str], panel_override: int | None, mode: str) -> list[dict]:
    """Run the selected extended cases (``all`` for every one), print one panel verdict line each and return the rows."""
    from mlframe.feature_selection._benchmarks import four_pillars_contracts as contracts
    from mlframe.feature_selection._benchmarks.four_pillars_cases import CASES

    rows = []
    for case in CASES:
        if "all" not in names and case.name not in names:
            continue
        seeds = panel_override or case.seeds
        per_seed = []
        for seed in range(seeds):
            try:
                per_seed.append(_extra_row(case, seed))
            except Exception as exc:
                warnings.warn(f"extra case {case.name} seed {seed} raised {type(exc).__name__}: {exc}; counted as a failure", stacklevel=1)
                per_seed.append({"ok": False, "n_forbidden": 0, "seconds": float("nan"), "selected": [], "error": f"{type(exc).__name__}: {exc}"})
        pv = contracts.panel(per_seed, power_gate=case.power_gate, fp_gate=case.fp_gate)
        print(f"[{mode}] {contracts.format_panel(case.name, pv)}{'  (report only)' if case.report_only else ''}", flush=True)
        for r in per_seed:
            if not r["ok"]:
                print(f"    miss: selected={r['selected']} {r.get('forbidden', r.get('noise', ''))} {r.get('error', '')}", flush=True)
        rows.append({"case": case.name, "report_only": case.report_only, **pv, "per_seed": per_seed})
    return rows


def _cases(only: set[str], f2_default: bool):
    """Cases."""
    if "f2" in only:
        for p in ("uniform", "scaled_1_5", "heavy_tailed", "mixed", "with_outliers"):
            yield f"f2[{p}]", (lambda p=p: _f2_case(p, False))
            if f2_default:
                yield f"f2_default[{p}]", (lambda p=p: _f2_case(p, True))
    if "noise" in only:
        for strat in (None, "mdlp", "mdlp_validated"):
            for kind, target in (("normal", "binary"), ("id_like", "continuous"), ("categorical", "binary")):
                yield f"noise[{kind}-{target}-{strat}]", (lambda k=kind, t=target, s=strat: _noise_case(k, t, s))
    if "xor3" in only:
        for yk in ("binary", "continuous"):
            yield f"xor3[{yk}-0]", (lambda yk=yk: _xor3_case(yk, 0))
    if "conj" in only:
        yield "conj[native]", (lambda: _conj_case(None))


def main(argv=None):
    """Main."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="f2,noise,xor3,conj")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--f2-default", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--families", action="store_true", help="print the per-family FE wall of every case")
    ap.add_argument("--extra", default=None, help="run the hard cases of four_pillars_cases: 'all' or comma-separated case names")
    ap.add_argument("--panel", type=int, default=None, help="seeds per extended case, overriding each case's own panel size")
    ap.add_argument("--strict-extra", action="store_true", help="exit non-zero when a gating (not report-only) extended case does not PASS")
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if not (args.extra and args.only == "f2,noise,xor3,conj") else set()
    mode = "gpu_strict" if os.environ.get("MLFRAME_FE_GPU_STRICT", "") == "1" else "cpu"
    pr = cProfile.Profile() if args.profile else None
    rows = []
    from mlframe.feature_selection.filters._fe_family_timing import get_fe_family_wall, reset_fe_family_wall

    for name, fn in _cases(only, args.f2_default):
        reset_fe_family_wall()
        if pr:
            pr.enable()
        try:
            ok, dt, names = fn()
            err = None
        except Exception as exc:
            warnings.warn(f"case {name} raised {type(exc).__name__}: {exc}; recorded in the result row", stacklevel=1)
            ok, dt, names, err = False, float("nan"), [], f"{type(exc).__name__}: {exc}"
        if pr:
            pr.disable()
        rows.append({"case": name, "ok": bool(ok), "seconds": round(float(dt), 2), "selected": [str(n) for n in names][:8], "error": err})
        print(f"[{mode}] {'PASS' if ok else 'FAIL'} {name:40s} {dt:8.2f}s  {names[:5]}{'  ' + err if err else ''}", flush=True)
        if args.families:
            top = sorted(get_fe_family_wall().items(), key=lambda kv: -kv[1][0])[:8]
            print("    families: " + "  ".join(f"{k}={w:.1f}s(x{c})" for k, (w, c) in top), flush=True)
    total = sum(r["seconds"] for r in rows if np.isfinite(r["seconds"]))
    n_ok = sum(r["ok"] for r in rows)
    print(f"[{mode}] SUMMARY {n_ok}/{len(rows)} pass, total fit time {total:.1f}s", flush=True)
    extra_rows = _run_extra(set(args.extra.split(",")), args.panel, mode) if args.extra else []
    if pr:
        pr.dump_stats(args.profile)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"mode": mode, "rows": rows, "extra": extra_rows, "total_seconds": total}, fh, indent=1, default=str)
    if args.strict_extra and any(r["verdict"] != "PASS" for r in extra_rows if not r["report_only"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
