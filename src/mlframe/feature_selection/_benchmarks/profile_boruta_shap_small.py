"""cProfile hotspot harness for ``BorutaShap`` at the SMALL representative shape (n~800, p~20).

Run:
    CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 PYTHONPATH=src \
        python -m mlframe.feature_selection._benchmarks.profile_boruta_shap_small

Two scenes are profiled (representative + tiny):
    A) n=800,  p=20  (6 informative + 14 noise), LightGBM 30-tree binary, importance_measure='shap', n_trials=25
    B) n=300,  p=12  (4 informative + 8 noise),  LightGBM 20-tree binary, importance_measure='shap', n_trials=15

Goal: isolate the mlframe-SIDE per-trial work from the (intrinsic, un-optimisable here) booster .fit + TreeSHAP
cost. The dominant wall is third-party (LightGBM .fit + shap TreeExplainer), per the explain() perf note. This
harness ranks ONLY the mlframe-side hotspots: shadow construction, hit counting, history accumulation, the
binomial test, and any per-trial DataFrame rebuild / concat.

VERDICT -- NO ACTIONABLE SPEEDUP (measured 2026-06-11, store-python 3.14, shap 0.52, lgbm 4.6):

Scene A (n=800/p=20/30-tree/25-trial) profiled wall 3.06s. Summing ONLY mlframe-owned functions
(boruta_shap.py / _boruta_shap_*.py / utils.misc) by cProfile tottime gives **0.0288s total = 0.94%** of the fit.
The other 99.06% is intrinsic third-party per-trial cost, none of it optimisable in this file:
  - LightGBM booster ``basic.py:update`` 1.253s (the ``model.fit`` boosting loop)        ~41%
  - LightGBM ``__inner_predict_np2d`` 0.727s (TreeSHAP leaf prediction)                  ~24%
  - shap ``dump_model`` 0.238s + json ``raw_decode`` 0.113s (TreeExplainer reads the     ~11%
    just-refit model each trial; the model is REFIT per trial so the explainer CANNOT be cached)
  - shap ``_tree.py`` Tree.__init__ traversal + copy.deepcopy of the model                ~remainder

Top mlframe-side hotspots (by own tottime, scene A):
  1. create_shadow_features  0.0119s / 25 calls  -> 0.48ms attributed, 0.75ms WARM isolated microbench
  2. feature_importance      0.0036s / 25 calls  (own time only; its 1.335s cumtime is the shap explain() below)
  3. explain                 0.0027s / 25 calls  (own time; dominated by the third-party TreeExplainer it wraps)
     (everything else -- update_importance_history, binomial_H0_test, calculate_hits, test_features -- is <1.2ms
      TOTAL across all 25 trials, i.e. <50us/call: pure cProfile attribution noise per the CLAUDE.md <1ms rule.)

Why the hottest mlframe kernel (create_shadow_features) is NOT optimisable further here:
  - It already runs the homogeneous-numeric FAST PATH (verified: all-float64 frame -> ``to_numpy()`` 2-D view +
    per-column ``_rng.permutation`` into a same-dtype buffer). Warm isolated microbench = 0.75 ms/call, BELOW the
    1ms attribution-noise floor.
  - The per-column permutation loop MUST stay column-by-column to keep the RNG stream bit-identical: a single
    whole-array vectorised shuffle would change which permutation each column receives -> different shadow values
    -> different hits -> SELECTION-ALTERING. Explicitly forbidden.
  - The ``pd.concat([self.X, self.X_shadow])`` builds ``X_boruta``, which the surrogate ``model.fit`` and
    TreeExplainer both consume in full -- no discarded output to prune (per the "audit hot kernels for wasted
    per-call work" rule, the caller uses the entire output).

Bit-identity sanity: full fit is run-to-run bit-identical (selected set + hit vector) on seeds 0 and 7 (n_sel 6
and 9 respectively), confirming the profiled path is deterministic and any future optimisation has a stable golden.

CONCLUSION: BorutaShap is genuinely shap/booster-bound at this (and every) shape -- the model is refit every trial,
so neither the booster .fit nor the rebuilt TreeExplainer can be cached or moved to GPU without changing the
caller-supplied estimator (and GPU TreeSHAP perturbs shap ~1e-6 -> selection-altering at the nanpercentile gate;
see explain() perf note). The mlframe-side per-trial work is already at the floor. No change shipped.
"""
from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
import warnings
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

if TYPE_CHECKING:
    from mlframe.feature_selection.boruta_shap import BorutaShap

warnings.filterwarnings("ignore")


def build_scene(n: int, p: int, informative: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Deterministic make_classification scene -- a few informative columns + noise."""
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=informative,
        n_redundant=0, n_classes=2, n_clusters_per_class=1,
        random_state=seed, shuffle=False,
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(p)]), y


def make_selector(n_estimators: int, n_trials: int) -> BorutaShap:
    """SHAP-driven BorutaShap (the path that actually calls TreeExplainer) at a modest booster size."""
    import lightgbm as lgb
    from mlframe.feature_selection.boruta_shap import BorutaShap

    model = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=31, random_state=0, verbose=-1, n_jobs=1)
    return BorutaShap(
        model=model, importance_measure="shap", classification=True,
        n_trials=n_trials, random_state=0, verbose=False, normalize=True,
    )


def extract_golden(bs: BorutaShap) -> dict:
    """Bit-identity golden: accept/reject/tentative sets + per-feature hit vector + trials run + selection."""
    return {
        "accepted": sorted(bs.accepted),
        "rejected": sorted(bs.rejected),
        "tentative": sorted([str(t) for t in bs.tentative]),
        "hits": [float(h) for h in bs.hits],
        "n_trials_run": int(bs.n_trials_run_),
        "selected_features": sorted(bs.selected_features_),
    }


SCENES = {
    "A_n800_p20": dict(n=800, p=20, informative=6, n_estimators=30, n_trials=25),
    "B_n300_p12": dict(n=300, p=12, informative=4, n_estimators=20, n_trials=15),
}


def run_scene(name: str, cfg: dict, out_dir: Path) -> dict:
    """Plain wall + cProfile pass for one scene; writes the .prof + top-tottime table; returns golden + wall."""
    X, y = build_scene(cfg["n"], cfg["p"], cfg["informative"])

    bs = make_selector(cfg["n_estimators"], cfg["n_trials"])
    t0 = time.perf_counter()
    bs.fit(X, y)
    wall = time.perf_counter() - t0
    golden = extract_golden(bs)

    bs2 = make_selector(cfg["n_estimators"], cfg["n_trials"])
    pr = cProfile.Profile()
    pr.enable()
    bs2.fit(X, y)
    pr.disable()
    golden2 = extract_golden(bs2)
    assert golden2 == golden, f"{name}: profiled run diverged from plain run (non-determinism!)"

    out_dir.mkdir(parents=True, exist_ok=True)
    pr.dump_stats(str(out_dir / f"boruta_shap_small_{name}.prof"))

    s = io.StringIO()
    st = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime")
    st.print_stats(30)
    tottime_txt = s.getvalue()

    s2 = io.StringIO()
    st2 = pstats.Stats(pr, stream=s2).strip_dirs().sort_stats("cumulative")
    st2.print_stats(30)
    cumtime_txt = s2.getvalue()

    (out_dir / f"boruta_shap_small_{name}_tottime.txt").write_text(tottime_txt, encoding="utf-8")
    (out_dir / f"boruta_shap_small_{name}_cumtime.txt").write_text(cumtime_txt, encoding="utf-8")

    print(f"\n========== scene {name}  (n={cfg['n']} p={cfg['p']} trials={cfg['n_trials']} est={cfg['n_estimators']}) ==========")
    print(f"[plain wall] full fit = {wall:.2f}s  | accepted={len(golden['accepted'])} "
          f"rejected={len(golden['rejected'])} tentative={len(golden['tentative'])} "
          f"trials_run={golden['n_trials_run']} hits_sum={sum(golden['hits']):.0f}")
    print(f"\n[tottime top, {name}]\n{tottime_txt}")
    return {"wall": wall, "golden": golden}


def main() -> None:
    """Profile both scenes, write goldens + hotspot tables under ``_results``."""
    out_dir = Path(__file__).parent / "_results"
    print("# BorutaShap SMALL-shape hotspot scan (mlframe-side focus)")
    goldens = {}
    for name, cfg in SCENES.items():
        res = run_scene(name, cfg, out_dir)
        goldens[name] = res["golden"]
    (out_dir / "boruta_shap_small_golden.json").write_text(
        json.dumps(goldens, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\n  goldens -> {out_dir / 'boruta_shap_small_golden.json'}")


if __name__ == "__main__":
    main()
