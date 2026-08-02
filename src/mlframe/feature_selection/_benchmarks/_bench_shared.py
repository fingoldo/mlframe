"""Shared helpers for the ``bench_iter*``/``bench_*`` shap-proxy-regime benchmark family: small
utilities independently duplicated across those scripts, consolidated here so a fix can't
silently drift out of sync across copies. Each benchmark script stays independently runnable
from this same ``_benchmarks/`` directory -- only the literal duplicated bodies move here.
"""
from __future__ import annotations

from typing import Callable, TextIO


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
