"""Why does RFECV solve parity when nothing else does -- the search, or the estimator inside it?

On the SCM three-way parity bed, `rfecv` recovers the answer key perfectly and identically on all twenty
seeds while every other arm sits at chance with a stability index near zero. That is the sharpest separation
in the benchmark and, as it stands, an unexplained one. Two mechanisms could produce it and they carry
opposite lessons:

* **the search**: evaluating SUBSETS against held-out score, rather than ranking columns one at a time, is
  what finds a structure no single column shows. If so, any wrapper should do it and the recommendation is
  "wrap something" regardless of what is inside.
* **the estimator**: the gradient-boosted tree inside this RFECV can represent a three-way interaction, so
  the subsets containing the operands score better and the search follows. If so, the recommendation is
  "wrap a model that can represent the structure you are looking for", and wrapping a linear model would
  find nothing.

The ablation swaps only the inner estimator, holding the search, the budget, the data and the seeds fixed.
A linear inner model cannot represent parity at all, so it separates the two explanations cleanly: if
recovery survives the swap the search is doing the work, and if it collapses the estimator is.

    python -m mlframe.feature_selection._benchmarks.fs_hybrid.ablation_rfecv_parity --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_results")


def _run_one(estimator_kind: str, seed: int, bed: str, rows: int) -> Dict[str, Any]:
    """Fit RFECV with one inner estimator on one seed of the bed and score what it recovered."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection import extract_selected
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

    from ._scm_beds import build_scm_bed

    frame, labels, truth = build_scm_bed(bed, seed=seed, n_samples=rows)
    if estimator_kind == "lightgbm":
        estimator: Any = lgb.LGBMClassifier(n_estimators=80, verbose=-1, n_jobs=-1, random_state=seed)
    elif estimator_kind == "logistic":
        estimator = LogisticRegression(max_iter=2000, random_state=seed)
    else:
        raise ValueError(f"unknown estimator kind {estimator_kind!r}")

    model = RFECV(
        estimator=estimator,
        cv=3,
        scoring=None,
        verbose=0,
        fi_config=FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min"),
        search_config=SearchConfig(max_refits=12, max_runtime_mins=2.0),
        random_state=seed,
    )
    import pandas as pd

    model.fit(frame, pd.Series(np.asarray(labels)))

    names = [str(column) for column in frame.columns]
    selected = {str(column) for column in extract_selected(model, names) if str(column) in set(names)}
    answer = {str(column) for column in truth["base"]}
    recall = len(selected & answer) / len(answer) if answer else float("nan")
    precision = len(selected & answer) / len(selected) if selected else 0.0
    return {
        "estimator": estimator_kind,
        "bed": bed,
        "seed": seed,
        "n_selected": len(selected),
        "recall": recall,
        "precision": precision,
        "selected": sorted(selected),
        "answer": sorted(answer),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation and print the per-estimator recovery."""
    parser = argparse.ArgumentParser(description="Does RFECV's parity result come from the search or the estimator?")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bed", default="xor3")
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    rows: List[Dict[str, Any]] = []
    for kind in ("lightgbm", "logistic"):
        for seed in range(int(args.seeds)):
            record = _run_one(kind, seed, args.bed, int(args.rows))
            rows.append(record)
            print(f"{kind:<10} seed={seed:<3} |S|={record['n_selected']:<3} recall={record['recall']:.3f} precision={record['precision']:.3f}")

    out = args.out or os.path.join(RESULTS_DIR, f"ablation_rfecv_parity_{args.bed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    print()
    for kind in ("lightgbm", "logistic"):
        subset = [row for row in rows if row["estimator"] == kind]
        recalls = np.asarray([row["recall"] for row in subset], dtype=np.float64)
        sizes = np.asarray([row["n_selected"] for row in subset], dtype=np.float64)
        print(f"{kind:<10} mean recall={recalls.mean():.3f} (min {recalls.min():.3f}, max {recalls.max():.3f})  mean |S|={sizes.mean():.1f}  n={len(subset)}")
    print(f"\nwritten: {out}")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if main() >= 0 else 1)
