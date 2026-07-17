"""Subprocess worker for ``test_suite_fe_linear_recovery.py``.

Runs EXACTLY ONE ``train_mlframe_models_suite`` fit (one generator x distribution x
model x use_mrmr) in a fresh process and prints ``{"R2":..,"span":..}`` after a JSON
marker. Process isolation is mandatory here: two heavy-tailed n=100k suite fits in the
SAME process intermittently leave global FE/model state that makes a later heavy-tail
fit's downstream sklearn pipeline raise "feature names should match" (each fit is green
standalone). The MRMR-endtoend-invariants layer fits each case in a subprocess for the
same reason; this worker reuses that proven pattern via the shared ``_suite_fe_helpers``
scaffold so the recovery thresholds are measured on the real entrypoint, one fit per
process.

Invoked as: ``python _suite_fe_worker.py '<json payload>'`` with payload
``{"gen","dist","model","use_mrmr","seed"}``.
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np


def main(payload: dict) -> None:
    """Subprocess worker entry point: run one FE suite generator/scenario described by payload and report its results."""
    from tests.feature_selection._suite_fe_helpers import (
        GENERATORS,
        run_suite,
        best_test_metric,
        prediction_span_fraction,
    )

    gen = payload["gen"]
    dist = payload["dist"]
    model = payload["model"]
    use_mrmr = bool(payload["use_mrmr"])
    seed = int(payload["seed"])

    case = GENERATORS[gen](seed=seed, distribution=dist)
    # MRMR.fit consumes the GLOBAL np.random stream (independent of random_seed); seed it
    # for a deterministic single fit per process (mirrors the endtoend-invariants worker).
    np.random.seed(seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        entries, _meta = run_suite(
            case.df,
            case.target,
            model=model,
            use_mrmr=use_mrmr,
            random_seed=seed,
        )

    result = dict(
        R2=best_test_metric(entries, "R2"),
        span=prediction_span_fraction(entries),
        n=len(case.df),
        structure=case.structure,
    )
    print("===RESULT_JSON===")
    print(json.dumps(result))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
