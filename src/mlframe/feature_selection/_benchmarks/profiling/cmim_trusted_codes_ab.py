"""Same-process A/B of the CMIM device scorer with and without the per-call code guard of batched_cmi_gpu.

A: batched_cmi_gpu is wrapped to force codes_trusted=False (the guard reduces the whole (n, block) candidate matrix twice per reference); B: the scorer as it is, which checks
the codes once on the host and passes codes_trusted=True. 100000 rows x 150 candidates x 40 references, alternating, results equal.
"""
import functools
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from mlframe.feature_selection.filters import _fe_batched_mi as B
from mlframe.feature_selection.filters import _orthogonal_cmim_fe as C

rng = np.random.default_rng(0)
n, K, S, nb = 100_000, 150, 40, 10
eng = rng.integers(0, nb, (K, n))
y = rng.integers(0, 2, n)
sel = [rng.integers(0, nb, n) for _ in range(S)]
names = [f"s{j}" for j in range(S)]
src = [f"e{i % 7}" for i in range(K)]

trusted = B.batched_cmi_gpu


@functools.wraps(trusted)
def untrusted(*args, **kwargs):
    """The call without the caller's trust, so the guard runs."""
    kwargs["codes_trusted"] = False
    return trusted(*args, **kwargs)


def run(guarded: bool):
    """One timed scorer call with the guard on or off."""
    B.batched_cmi_gpu = untrusted if guarded else trusted
    t = time.perf_counter()
    out = C._cmim_scores_device(eng, y, sel, names, src, nb)
    return time.perf_counter() - t, out


run(True)
run(False)
a_t, b_t = [], []
for _ in range(5):
    ta, a = run(True)
    tb, b = run(False)
    a_t.append(ta)
    b_t.append(tb)
    if not (np.allclose(a, b, equal_nan=True)):
        raise AssertionError('np.allclose(a, b, equal_nan=True)')
B.batched_cmi_gpu = trusted
print(f"guard per call: median {np.median(a_t):.3f} s   checked once: median {np.median(b_t):.3f} s   speed-up {np.median(a_t) / np.median(b_t):.2f}x")
