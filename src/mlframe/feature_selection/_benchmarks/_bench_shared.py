"""Shared helpers for the ``bench_iter*``/``bench_*`` shap-proxy-regime benchmark family: small
utilities independently duplicated across those scripts, consolidated here so a fix can't
silently drift out of sync across copies. Each benchmark script stays independently runnable
from this same ``_benchmarks/`` directory -- only the literal duplicated bodies move here.
"""
from __future__ import annotations

from typing import Any, Callable, TextIO

import numpy as np

from mlframe.feature_selection.filters._cluster_aggregate import uf_find  # noqa: F401 -- re-exported for the bench-family call sites


def noise_frame_no_structure(p: int, n: int, seed: int = 0):
    """Pure-noise + ordinary-smooth integer columns with NO planted structure (no modular/lattice pattern): a
    smooth linear threshold of the first two columns is the only signal, so a structure-detector under test
    must stay silent on this frame. Shared negative-control fixture for the wideframe structure-scan benches."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    cols = {f"c{i}": rng.integers(0, 100, n) for i in range(p)}
    X = pd.DataFrame(cols)
    y = ((X["c0"] + 0.7 * X["c1"]) > 85).astype(int).to_numpy()
    return X, y


def logreg_holdout_auc(Xtr, ytr, Xte, yte, cols) -> float:
    """Fit a max_iter=1000 ``LogisticRegression`` on the ``cols`` subset of ``Xtr`` and return its held-out AUC
    on ``Xte`` -- the cross-selector-family cluster-reduction benchmarks' shared evaluation step. Falls back to
    the full column set when ``cols`` is empty."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if len(cols) == 0:
        cols = np.arange(Xtr.shape[1])
    clf = LogisticRegression(max_iter=1000).fit(Xtr.iloc[:, cols], ytr)
    return float(roc_auc_score(yte, clf.predict_proba(Xte.iloc[:, cols])[:, 1]))


def searchsorted_bin_codes(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Right-side ``searchsorted`` bin-code assignment: the shared quantile-edge binning step
    duplicated across the cell-encoding benchmark family."""
    return np.searchsorted(edges, x, side="right")


def build_selector(seed: int = 0) -> Any:
    """Wide-data ShapProxiedFS config: prefilter on, clustering on, exhaustive-approx search, honest re-validation."""
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS

    return ShapProxiedFS(
        classification=True,
        metric="brier",
        optimizer="auto",
        prefilter_top=500,
        cluster_features=True,
        cluster_corr_threshold=0.7,
        top_n=20,
        n_splits=4,
        n_revalidation_models=3,
        trust_guard=True,
        n_anchors=24,
        run_importance_ablation=True,
        within_cluster_refine=True,
        random_state=seed,
        verbose=False,
    )


def random_baseline_brier(y) -> float:
    """Predict the prior on every row (constant probability == positive rate). Reference floor for
    the proxy: anything we ship MUST beat this. y is binary 0/1."""
    p = float(np.asarray(y).mean())
    return float(np.mean((np.asarray(y, dtype=np.float64) - p) ** 2))


def make_dataset(cfg: dict):
    """Build a synthetic shap-proxy regime dataset (X, y, roles) from a benchmark config dict."""
    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    n_noise = max(0, cfg["width"] - cfg["n_informative"] - cfg["n_redundant"])
    X, y, roles = make_regime_dataset(
        n_samples=cfg["n_rows"],
        n_informative=cfg["n_informative"],
        n_redundant=cfg["n_redundant"],
        redundancy_rho=cfg["redundancy_rho"],
        n_noise=n_noise,
        snr=cfg["snr"],
        task="binary",
        seed=cfg["seed"],
    )
    return X, y, roles


def recovered(sel, roles) -> tuple:
    """Count how many of the dataset's informative features the selector recovered: (n_recovered, n_informative)."""
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_)), len(inf)


def recovered_count(sel, roles) -> int:
    """Count how many of the dataset's informative features the selector recovered (count only, no denominator)."""
    inf = {n for n, r in roles.items() if r == "informative"}
    return len(inf & set(sel.selected_features_))


def make_tee_print(orig: Callable, fp: TextIO) -> Callable:
    """Build a ``print`` replacement that writes to both the original stream and ``fp``, tolerating a closed/broken ``fp``."""

    def _tee_print(*a, **kw):
        kw["flush"] = True
        orig(*a, **kw)
        try:
            kw2 = dict(kw)
            kw2["file"] = fp
            kw2["flush"] = True
            orig(*a, **kw2)
        except (OSError, ValueError):
            pass

    return _tee_print


def print_stage_table(timings: dict, total: float) -> None:
    """Print a per-stage wall-time breakdown of the shap-proxy pipeline's fixed stage order, skipping any stage absent from ``timings``."""
    order = ("prefilter", "clustering", "oof_shap", "prescreen", "search", "trust_guard", "revalidation", "importance_ablation", "within_cluster_refine")
    print(f"  total={total:.2f}s  stages:")
    for k in order:
        v = timings.get(k)
        if v is not None:
            print(f"    {k:24s} {v:8.3f}s ({100 * v / total:5.1f}%)")
