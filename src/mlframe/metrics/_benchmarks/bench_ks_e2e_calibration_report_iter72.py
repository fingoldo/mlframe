"""iter72 @200k e2e: full fast_calibration_report (show_plots=False) with the KS inline-ordered gate ON vs OFF.

Paired/interleaved A/B on the whole report function -- the ONLY difference between the two arms is the KS large-n
path (inline-ordered kernel vs pre-gathered kernel), toggled by setting ``_KS_INLINE_ORDERED_MIN_N`` to infinity for
the OLD arm. Both arms call the identical report code, so any wall delta is the KS gather elimination surviving e2e.
Also asserts the returned ks token (and full metrics tuple) is bit-identical across arms.

Run: python -m mlframe.metrics._benchmarks.bench_ks_e2e_calibration_report_iter72
"""
import numpy as np
from time import perf_counter as timer

import mlframe.metrics.classification._classification_extras as extras
from mlframe.metrics.classification._classification_report import fast_calibration_report

N = 200_000
rng = np.random.default_rng(42)
logit = rng.normal(0.0, 1.3, size=N)
p = 1.0 / (1.0 + np.exp(-logit))
y_true = (rng.random(N) < p).astype(np.int64)
y_pred = np.clip(p + rng.normal(0.0, 0.05, size=N), 1e-6, 1 - 1e-6)

_ORIG = extras._KS_INLINE_ORDERED_MIN_N


def call():
    return fast_calibration_report(y_true=y_true, y_pred=y_pred, nbins=10, show_plots=False, use_weights=True)


def set_gate(on):
    extras._KS_INLINE_ORDERED_MIN_N = 150_000 if on else 10**18


# Identity: the 17-scalar metrics tuple (incl. ks at index 10) must match exactly.
set_gate(False); res_old = call()
set_gate(True);  res_new = call()
scal_old = tuple(res_old[:15]); scal_new = tuple(res_new[:15])
assert scal_old == scal_new, f"metrics differ:\n old={scal_old}\n new={scal_new}"  # nosec B101 - internal invariant check in src/mlframe/metrics/_benchmarks, not reachable with untrusted input
print(f"identity OK: ks(old)={res_old[10]!r} == ks(new)={res_new[10]!r}; full 15-scalar tuple ==")

# Warm both arms.
for _ in range(3):
    set_gate(False); call(); set_gate(True); call()

TRIALS = 15
old_t, new_t = [], []
for _ in range(TRIALS):
    set_gate(False); t0 = timer(); call(); old_t.append(timer() - t0)
    set_gate(True);  t0 = timer(); call(); new_t.append(timer() - t0)

extras._KS_INLINE_ORDERED_MIN_N = _ORIG

old_t = np.array(old_t); new_t = np.array(new_t)
faster = int((new_t < old_t).sum())
print(f"e2e fast_calibration_report @n={N}, paired x{TRIALS}:")
print(f"  OLD (gate off): min {old_t.min()*1e3:.2f}ms  med {np.median(old_t)*1e3:.2f}ms")
print(f"  NEW (gate on) : min {new_t.min()*1e3:.2f}ms  med {np.median(new_t)*1e3:.2f}ms")
print(f"  NEW faster in {faster}/{TRIALS} trials | min-speedup {old_t.min()/new_t.min():.3f}x  med {np.median(old_t)/np.median(new_t):.3f}x")
