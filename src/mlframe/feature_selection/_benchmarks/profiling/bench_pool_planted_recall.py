"""Baseline for the FE column-pool plan (``audits/2026-11-09/fe_candidate_pool_plan.md``): per planted-signal generator, does an all-families fit select a column built from every planted source?

Usage: ``python bench_pool_planted_recall.py [ROWS] [COLS] [SEEDS] [GENERATOR,...]`` (the last 30 % of the rows are held out for the linear AUC columns) (defaults 20000 100 1), run from a repository checkout (the generators are loaded from ``tests/feature_selection/_synth/pool_planted.py``). For each generator it prints the wall time, the planted
columns, the selected columns that mention a planted source, and ``recall``: 1 when the selected set holds a column whose name mentions every planted source (a raw column counts only when one source is planted),
else 0, and the noise-only selected columns (names that mention no planted source; the user's criterion is signal in, noise out). Run it once per pool arm (``BENCH_FE_SHAPE_POOL=all|relevant``) and compare the lines; one process, so the timings share the same warm kernels.
"""

import os
import sys
import time
import warnings

warnings.simplefilter("ignore")

from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("pool_planted", Path(__file__).resolve().parents[5] / "tests" / "feature_selection" / "_synth" / "pool_planted.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
GENERATORS = _module.GENERATORS


def _linear_auc(model, X_tr, y_tr, X_te, y_te) -> tuple:
    """Held-out ROC-AUC of a standardised logistic regression on the model's selected (transformed) columns, and on the raw columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def auc(A_tr, A_te) -> float:
        """AUC of the pipeline fitted on ``A_tr``."""
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)).fit(A_tr, y_tr)
        return float(roc_auc_score(y_te, clf.predict_proba(A_te)[:, 1]))

    Z_tr, Z_te = model.transform(X_tr), model.transform(X_te)
    return auc(Z_tr.to_numpy(dtype=float), Z_te.to_numpy(dtype=float)), auc(X_tr.to_numpy(dtype=float), X_te.to_numpy(dtype=float))


def run(rows: int, cols: int, seeds: int, only: str = "") -> None:
    """Fit every generator at ``seeds`` seeds and print one line per fit."""
    kw = F.all_families_kwargs()
    kw["nbins_strategy"] = "mdlp"
    kw["fe_shape_pool"] = os.environ.get("BENCH_FE_SHAPE_POOL", "all")
    chosen = {k: v for k, v in GENERATORS.items() if not only or k in only.split(",")}
    hits: dict = {name: 0 for name in chosen}
    for name, make in chosen.items():
        for seed in range(seeds):
            X, y, planted = make(n=rows, p=cols, seed=seed)
            cut = int(0.7 * len(X))
            X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]
            MRMR._FIT_CACHE.clear()
            t = time.perf_counter()
            m = MRMR(verbose=0, random_seed=seed, n_jobs=1, **kw).fit(X_tr, y_tr)
            wall = time.perf_counter() - t
            selected = list(m.get_feature_names_out())
            gain_auc, raw_auc = _linear_auc(m, X_tr, y_tr, X_te, y_te)
            mention = [c for c in selected if any(s in c for s in planted)]
            recall = int(any(all(s in c for s in planted) for c in selected))
            # a selected column that mentions no planted source is built from noise only: a false positive
            false_pos = [c for c in selected if not any(s in c for s in planted)]
            hits[name] += recall
            print(f"{name:30s} seed {seed} wall {wall:6.1f}s recall {recall} noise-only selected {len(false_pos)} {false_pos[:3]} linear AUC on the selected set {gain_auc:.3f} vs raw columns {raw_auc:.3f} planted {planted} selected {len(selected)} mention {mention[:6]}", flush=True)
    print("recall per generator:", {k: f"{v}/{seeds}" for k, v in hits.items()})


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 20000, int(sys.argv[2]) if len(sys.argv) > 2 else 100, int(sys.argv[3]) if len(sys.argv) > 3 else 1, sys.argv[4] if len(sys.argv) > 4 else "")
