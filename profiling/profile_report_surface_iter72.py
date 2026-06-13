"""cProfile the mlframe-own classification report surface at n=200k WITHOUT importing training.core.

The full ``train_mlframe_models_suite`` import segfaults on this py3.14 store build (the known numba+scipy ABI trap),
so per the perf-loop fallback clause this driver exercises the report surface directly: ``fast_calibration_report``
(numeric path, show_plots=False) repeated to mimic the per-class x per-split calls, on a realistic 200k binary scenario.
"""
import cProfile, io, os, pstats, sys
from pathlib import Path
from time import perf_counter as timer

import numpy as np

N = int(os.environ.get("PROFILE_N_ROWS", "200000"))
REPS = int(os.environ.get("PROFILE_REPS", "8"))

from mlframe.metrics.classification._classification_report import fast_calibration_report

rng = np.random.default_rng(42)
# Realistic-ish calibrated-ish binary scores: latent logit -> sigmoid, label ~ Bernoulli(p).
logit = rng.normal(0.0, 1.3, size=N)
p = 1.0 / (1.0 + np.exp(-logit))
y_true = (rng.random(N) < p).astype(np.int64)
y_pred = np.clip(p + rng.normal(0.0, 0.05, size=N), 1e-6, 1 - 1e-6)


def run():
    for _ in range(REPS):
        fast_calibration_report(
            y_true=y_true, y_pred=y_pred, nbins=10, show_plots=False, use_weights=True, verbose=False,
        )


# Warm numba caches once (not profiled).
run()

prof = cProfile.Profile()
t0 = timer()
prof.enable()
run()
prof.disable()
print(f"\n{REPS} fast_calibration_report calls @n={N} in {timer()-t0:.2f}s")

for key, label in [("tottime", "TOTAL TIME"), ("cumulative", "CUMULATIVE")]:
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(key).print_stats(40)
    print(f"\n{'='*80}\nTOP 40 by {label}\n{'='*80}\n{s.getvalue()}")

print(f"\n{'='*80}\nTOP 40 mlframe-own (by tottime)\n{'='*80}")
s = io.StringIO()
pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats("mlframe", 40)
print(s.getvalue())
