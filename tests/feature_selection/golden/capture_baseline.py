"""Capture pre-refactor or intermediate baseline.

Run with::

    # etap 0a (pre-refactor)
    python -m mlframe.tests.feature_selection.golden.capture_baseline --tier pre_refactor

    # etap 9 (intermediate, post-cleanup)
    python -m mlframe.tests.feature_selection.golden.capture_baseline --tier intermediate

The fixed scenario list below mirrors the bench scenarios but uses smaller n
(faster capture, still covers the algorithmic surface).
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from tests.feature_selection.golden import _capture

logger = logging.getLogger("golden_capture")

# Smaller variants of bench scenarios for tractable golden capture.
GOLDEN_SCENARIOS = [
    {"name": "g_n2k_p50_clf", "n": 2000, "p": 50, "informative": 5, "task": "clf", "n_classes": 5, "fe_max_steps": 0},
    {"name": "g_n5k_p100_clf", "n": 5000, "p": 100, "informative": 10, "task": "clf", "n_classes": 5, "fe_max_steps": 0},
    {"name": "g_n2k_p200_clf", "n": 2000, "p": 200, "informative": 15, "task": "clf", "n_classes": 5, "fe_max_steps": 0},
    {"name": "g_n2k_p50_reg", "n": 2000, "p": 50, "informative": 5, "task": "reg", "fe_max_steps": 0},
    {"name": "g_n5k_p100_fe", "n": 5000, "p": 100, "informative": 10, "task": "clf", "n_classes": 5, "fe_max_steps": 1},
]


def _build_data(spec: dict, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a classification or regression fixture from a scenario spec (n, p, informative, task)."""
    if spec["task"] == "clf":
        X, y = make_classification(
            n_samples=spec["n"],
            n_features=spec["p"],
            n_informative=spec["informative"],
            n_redundant=0,
            n_classes=spec["n_classes"],
            n_clusters_per_class=1,
            random_state=seed,
        )
    else:
        X, y = make_regression(
            n_samples=spec["n"],
            n_features=spec["p"],
            n_informative=spec["informative"],
            noise=0.1,
            random_state=seed,
        )
    cols = [f"f_{i}" for i in range(spec["p"])]
    return pd.DataFrame(X, columns=cols), y


def capture_one(spec: dict, tier: str, seed: int = 42) -> dict:
    """Fit MRMR on one scenario and capture its golden snapshot at the given refactor tier."""
    from mlframe.feature_selection.filters import MRMR  # type: ignore

    X, y = _build_data(spec, seed)
    mrmr = MRMR(
        quantization_nbins=10,
        interactions_max_order=1,
        full_npermutations=3,
        baseline_npermutations=2,
        random_seed=seed,
        n_jobs=1,
        verbose=0,
        fe_max_steps=spec.get("fe_max_steps", 0),
        cv=2,
    )
    logger.info("capturing %s (tier=%s)...", spec["name"], tier)
    mrmr.fit(X, y)
    if tier == "pre_refactor":
        return _capture.capture_pre_refactor(mrmr, spec["name"], seed)
    return _capture.capture_intermediate(mrmr, spec["name"], seed)


def main() -> int:
    """CLI entry point: capture golden snapshots for the requested tier/scenarios and write them to disk."""
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=("pre_refactor", "intermediate"), required=True)
    p.add_argument("--scenarios", default="all", help="comma-separated names or 'all'")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.scenarios == "all":
        specs = GOLDEN_SCENARIOS
    else:
        wanted = {s.strip() for s in args.scenarios.split(",")}
        specs = [s for s in GOLDEN_SCENARIOS if s["name"] in wanted]

    target_dir = _capture.PRE_REFACTOR_DIR if args.tier == "pre_refactor" else _capture.INTERMEDIATE_DIR
    for spec in specs:
        snap = capture_one(spec, args.tier, seed=args.seed)
        path = _capture.save_snapshot(snap, target_dir)
        logger.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
