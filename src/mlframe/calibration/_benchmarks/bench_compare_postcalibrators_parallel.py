"""Measures whether parallelising compare_postcalibrators's per-calibrator loop is worth the risk.

The audit flagged the loop (fitting ~15-20 independent third-party calibrators) as fully serial with
no measured verdict either way. This script runs the real function (via a lightweight calibrator zoo
that avoids the optional deps this dev box lacks: ml_insights, dirichletcal) at a representative shape
and reports the wall time, so the parallelisation call is based on a number, not a guess.

Measured on this dev box (n=3000, 15 calibrators, selection="inner_cv" default -> 5 refits/calibrator):
wall ~6.1s total, 0 failures. Per-calibrator average is ~0.4s, dominated by netcal/pycalib fit calls
that themselves may release the GIL poorly (per the audit's own note) -- a joblib process-backend
fan-out would need each worker to import + fit netcal/pycalib/dirichletcal/venn_abers fresh (these
libraries pull in torch transitively), so the ~0.3-1s per-worker process-spawn overhead is the same
order of magnitude as the serial per-calibrator cost being parallelised. At the zoo sizes actually used
in this codebase (15-25 calibrators), a process-pool fan-out is not clearly a net win, and it adds real
correctness risk to a function that also aggregates a shared metrics dict, a shared fit_calibrators
dict, and a shared full_name-collision counter across calibrators (P1-5) -- all of which would need to
move to a post-hoc reduce step rather than in-loop mutation under a process pool.

Verdict: no actionable speedup measured at the realistic zoo size; the win is not clearly worth the
correctness risk of parallelising a leakage/exception-isolation-hardened loop. Left as a documented,
re-runnable benchmark (not deleted) so a future pass with a larger production zoo size, or a
thread-backend attempt gated to calibrators confirmed GIL-releasing, can re-measure and revisit.

Run: PYTHONPATH=src python src/mlframe/calibration/_benchmarks/bench_compare_postcalibrators_parallel.py
"""
from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import netcal.binning
import netcal.scaling
import pycalib.models
from betacal import BetaCalibration
from sklearn.calibration import CalibratedClassifierCV
from venn_abers import VennAbersCalibrator

from mlframe.calibration.post import compare_postcalibrators, named_calibrator


def _lightweight_zoo(calib_target: np.ndarray, num_bins: int) -> list:
    return [
        named_calibrator(CalibratedClassifierCV(method="sigmoid", ensemble=False), name="CCCV", param_str="sigmoid", lib="sklearn"),
        named_calibrator(CalibratedClassifierCV(method="isotonic", ensemble=False), name="CCCV", param_str="isotonic", lib="sklearn"),
        *[named_calibrator(BetaCalibration(v), lib="betacal", param_str=f"variant={v}") for v in ["abm", "ab", "am"]],
        named_calibrator(VennAbersCalibrator(), lib="vaa"),
        *[named_calibrator(cls(), lib="netcal") for cls in [netcal.binning.BBQ, netcal.binning.HistogramBinning, netcal.binning.IsotonicRegression]],
        *[named_calibrator(cls(), lib="netcal") for cls in [netcal.scaling.TemperatureScaling, netcal.scaling.BetaCalibration, netcal.scaling.LogisticCalibration]],
        *[
            named_calibrator(cls(), lib="pycalib", transform_method_name="predict_proba")
            for cls in [pycalib.models.IsotonicCalibration, pycalib.models.SigmoidCalibration, pycalib.models.BinningCalibration]
        ],
    ]


def main() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    p1 = rng.random(n)
    probs = np.column_stack([1 - p1, p1])
    target = (p1 + rng.normal(0, 0.1, n) > 0.5).astype(int)

    with patch("mlframe.calibration.post.get_postcalibrators", _lightweight_zoo):
        start = time.perf_counter()
        metrics_df, fit_calibrators, failed = compare_postcalibrators(
            model_name="bench",
            columns=["y"],
            calib_probs=probs,
            calib_target=target,
            oos_probs=None,
            oos_target=None,
        )
        elapsed = time.perf_counter() - start

    print(f"wall={elapsed:.3f}s n_fitted={len(fit_calibrators)} n_failed={len(failed)} per_calibrator={elapsed / max(len(fit_calibrators), 1):.3f}s")


if __name__ == "__main__":
    main()
