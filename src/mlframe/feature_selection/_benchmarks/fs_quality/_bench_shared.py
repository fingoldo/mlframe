"""Shared helpers for the ``mrmr_*`` fs_quality benchmark family: small utilities independently
duplicated across those scripts, consolidated here so a fix can't silently drift out of sync
across copies. Each benchmark script stays independently runnable from this same
``_benchmarks/fs_quality/`` directory -- only the literal duplicated bodies move here.
"""
from __future__ import annotations

import numpy as np


def build_mrmr(variant, seed: int, variants):
    """Build an MRMR instance for ``variant`` (raw-index-only support_, exact precision/recall against a known
    driver index set): fixed base kwargs overridden by ``variants[variant]``."""
    from mlframe.feature_selection.filters import MRMR

    base = dict(
        fe_max_steps=0,
        interactions_max_order=1,
        full_npermutations=3,
        baseline_npermutations=2,
        random_seed=seed,
        use_gpu=False,
        n_jobs=1,
        verbose=0,
        cv=2,
    )
    base.update(variants[variant])
    return MRMR(**base)


def cell_key(row: dict) -> tuple:
    """Group-by key for one campaign result row: (variant, scenario, n, p, seed)."""
    return (row["variant"], row["scenario"], int(row["n"]), int(row["p"]), int(row["seed"]))


def mean_or_nan(vals) -> float:
    """Mean of the non-None values in ``vals``, or NaN if all are None."""
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")
