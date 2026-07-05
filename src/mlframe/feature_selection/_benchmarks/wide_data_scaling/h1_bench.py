"""H1: exhaustive C(p,2) joint-MI sweep wall-time vs p, CPU vs GPU (GTX 1050 Ti)."""
import sys, time, math
import numpy as np

sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository/mlframe/src")
sys.path.insert(0, r"D:/Upd/Programming/PythonCodeRepository")

PROG = r"D:/Temp/synergy_scale_bench/progress.txt"
def ck(msg):
    with open(PROG, "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " | " + msg + "\n")
    print(msg, flush=True)

from mlframe.feature_selection.filters.batch_pair_mi_gpu import (
    batch_pair_mi_njit_prange, batch_pair_mi_cuda, batch_pair_mi_cupy,
    _CUDA_AVAIL, _CUPY_AVAIL,
)

ck("H1 start cuda=%s cupy=%s" % (_CUDA_AVAIL, _CUPY_AVAIL))

N = 8000
NBINS = 8
NCLASSES = 2
SEEDS = [0, 1, 2]
PS = [250, 500, 1000, 2000, 5000, 10000]

def make_data(p, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p))
    # quantile bin to 8 int8 codes
    codes = np.empty((N, p), dtype=np.int32)
    for j in range(p):
        q = np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1])
        codes[:, j] = np.searchsorted(q, X[:, j]).astype(np.int32)
    # planted pair interaction on cols 0,1
    inter = (codes[:, 0] >= NBINS // 2) ^ (codes[:, 1] >= NBINS // 2)
    y = inter.astype(np.int32)
    nbins = np.full(p, NBINS, dtype=np.int32)
    freqs_y = np.bincount(y, minlength=NCLASSES).astype(np.float64) / N
    return codes, nbins, y, freqs_y

def all_pairs(p):
    a, b = np.triu_indices(p, k=1)
    return a.astype(np.int64), b.astype(np.int64)

# warmup JIT on tiny data
_c, _nb, _y, _fy = make_data(20, 0)
_pa, _pb = all_pairs(20)
batch_pair_mi_njit_prange(_c, _pa, _pb, _nb, _y, _fy)
if _CUDA_AVAIL:
    try:
        batch_pair_mi_cuda(_c, _pa, _pb, _nb, _y, _fy)
    except Exception as e:
        ck("cuda warmup fail: %r" % e)
ck("warmup done")

results = []  # (p, npairs, cpu_s, cuda_s, cupy_s, mem_mb)

for p in PS:
    npairs = p * (p - 1) // 2
    mem_codes_mb = N * p * 4 / 1e6  # int32
    cpu_times, cuda_times, cupy_times = [], [], []
    cuda_err = cupy_err = None
    for seed in SEEDS:
        codes, nbins, y, fy = make_data(p, seed)
        pa, pb = all_pairs(p)
        # CPU
        t0 = time.perf_counter()
        mi_cpu = batch_pair_mi_njit_prange(codes, pa, pb, nbins, y, fy)
        cpu_times.append(time.perf_counter() - t0)
        # CUDA (only attempt up to a p where it's sane; npairs can be huge)
        if _CUDA_AVAIL and cuda_err is None:
            try:
                t0 = time.perf_counter()
                mi_cuda = batch_pair_mi_cuda(codes, pa, pb, nbins, y, fy)
                cuda_times.append(time.perf_counter() - t0)
                # sanity check vs cpu on first seed
                if seed == 0 and not np.allclose(mi_cpu, mi_cuda, atol=1e-6):
                    ck("WARN p=%d cuda mismatch max=%g" % (p, np.abs(mi_cpu - mi_cuda).max()))
            except Exception as e:
                cuda_err = repr(e)[:120]
                ck("p=%d CUDA fail: %s" % (p, cuda_err))
        # CuPy only for small p (per-pair python loop, very slow) -> skip above 500
        if _CUPY_AVAIL and p <= 500 and cupy_err is None:
            try:
                t0 = time.perf_counter()
                batch_pair_mi_cupy(codes, pa, pb, nbins, y, fy)
                cupy_times.append(time.perf_counter() - t0)
            except Exception as e:
                cupy_err = repr(e)[:120]
        del codes, pa, pb
    cpu_s = np.median(cpu_times)
    cuda_s = np.median(cuda_times) if cuda_times else None
    cupy_s = np.median(cupy_times) if cupy_times else None
    results.append((p, npairs, cpu_s, cuda_s, cupy_s, mem_codes_mb, cuda_err, cupy_err))
    ck(
        "p=%d pairs=%d cpu=%.3fs cuda=%s cupy=%s codes=%.1fMB"
        % (p, npairs, cpu_s, ("%.3fs" % cuda_s) if cuda_s else cuda_err or "skip", ("%.3fs" % cupy_s) if cupy_s else (cupy_err or "skip"), mem_codes_mb)
    )

# Fit O(p^2) curve on CPU pairs/sec to extrapolate
ck("=== H1 RESULTS ===")
print("%-7s %-12s %-10s %-12s %-12s %-10s" % ("p", "pairs", "cpu_s", "cuda_s", "cpu_pps", "cuda_pps"))
for p, npairs, cpu_s, cuda_s, cupy_s, mem, ce, pe in results:
    cpu_pps = npairs / cpu_s if cpu_s else 0
    cuda_pps = npairs / cuda_s if cuda_s else 0
    print(
        "%-7d %-12d %-10.3f %-12s %-12.3e %-10s"
        % (p, npairs, cpu_s, ("%.3f" % cuda_s) if cuda_s else "n/a", cpu_pps, ("%.3e" % cuda_pps) if cuda_pps else "n/a")
    )

# Extrapolation from largest completed CPU point (cost = k * pairs * n; constant pairs/sec)
done = [(p, npairs, cpu_s, cuda_s) for (p, npairs, cpu_s, cuda_s, cs, m, ce, pe) in results]
# Use the largest p that completed for both
cpu_pps_all = [npairs / cpu_s for (p, npairs, cpu_s, cuda_s) in done if cpu_s]
cuda_pps_all = [npairs / cuda_s for (p, npairs, cpu_s, cuda_s) in done if cuda_s]
cpu_pps_ref = np.median(cpu_pps_all)
ck("CPU median throughput: %.3e pairs/sec" % cpu_pps_ref)
if cuda_pps_all:
    cuda_pps_ref = np.median(cuda_pps_all)
    ck("CUDA median throughput: %.3e pairs/sec" % cuda_pps_ref)
else:
    cuda_pps_ref = None

def cross(pps, secs):
    # p where C(p,2)/pps == secs -> p ~ sqrt(2*pps*secs)
    return math.sqrt(2 * pps * secs)
for s in (30, 120, 600):
    line = "wall=%ds: CPU p<=%.0f" % (s, cross(cpu_pps_ref, s))
    if cuda_pps_ref:
        line += ", CUDA p<=%.0f" % cross(cuda_pps_ref, s)
    ck(line)

ck("H1 done")
