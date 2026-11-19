"""THE MRMR wall-time benchmark: 100000 rows x 100 columns, EVERY feature-engineering family on (except the LightGBM surrogate seeder), strict-resident GPU path.

What is measured. ``family_scale.all_families_kwargs()``: every ``fe_*_enable`` family switched on except ``fe_gbm_seeder_enable`` (the LightGBM surrogate seeder costs about 10 s of a GPU fit and adds
no selected feature the other seeders do not find), ``nbins_strategy="mdlp"``, ``random_seed=0``, ``n_jobs=1``, the GPU flags ``MLFRAME_FE_GPU_STRICT / MLFRAME_CMI_GPU / MLFRAME_FE_VRAM_F32 /
MLFRAME_FE_GPU_DISCRETIZE / MLFRAME_FE_GPU_BINNING = 1``. One warm-up fit on a 20000 x 30 frame comes first (numba / cupy / kernel-tuning warm-up), then the 100000 x 100 fit is timed: the line
``WALL <seconds>`` is the number. The scheduler's stage walls are printed on the way.

History of the number on this development box (GTX 1050 Ti 4 GB, Pascal sm_61, 6 SMs; 8 logical cores), warm 100000 x 100, all families, ``mdlp``:

- CPU only, 2026-10-31: 458 s -> about 310 s.
- GPU, 2026-11-01: 237-272 s. 2026-11-03: 91.5 / 101.1 s (the ~100 s target was met). 2026-11-04: 85-86 s. 2026-11-05: about 82 s (``optimal_joint`` binning 96-100 s).
- 2026-11-08: 76-79 s on a quiet machine (repeatable to +-1-3 s). 2026-11-10: 62-69 s, mean CPU load 30 %, mean GPU load 23 %.
- 2026-11-15: 60.7 s (one run, machine not fully quiet: background CPU about 27 %, GPU 83 C, desktop holding 0.5 GB of the card; two concurrent stage sections of 11.7-12.8 s and 10.9-15.1 s, no OOM).
  Later changes (see audits/2026-11-09/_TRACKER.md, the PF11 / CF11 rows) were verified by equality of results and by counters; the strict 100000 x 100 fit of the 2026-11-15 A/Bs spread 267-380 s
  while a foreign CPU-heavy process ran, so only quiet-machine numbers count.
- 2026-11-19, after the kernel-tuning refresh of every stale entry (``retune_all`` on ``mlframe.feature_selection.filters``, 83 GPU and 6 CPU sweeps, 21 minutes; before it dcor_columns_backend, hsic_biased_backend, family_null_backend, fe_gpu_pairs_mi and
  batch_mi_noise_gate ran on their row-count fallbacks) and the stability-vote / KSG changes of the same day (MRMR-VOTE-DEVICE, MRMR-KSG-F32): 100000 x 100 WALL 41.9 and 41.8 s (two runs, GPU 66 -> 71 C, SM 1708-1721 MHz, no throttle, no OOM); the sampler run of the same fit
  42.3 s with mean CPU 53.9 %, mean GPU busy 24.2 % (median 18 %), pool peak 2.83 GB. The split between the refresh and the code changes was not measured (no run on the stale cache with the new code).
- 2026-11-19, 100000 x 200 (``all_families_stage_sampler 100000 200 gpu``): WALL 65.3 s, mean CPU 53.4 %, mean GPU busy 28.3 % (median 20 %, 36 % of the time below 10 %), pool peak 3.39 GB, no OOM. Twice the columns cost 1.54 times the wall; the largest stage
  is still the FE cascade early_a part 1 (14.2 s at 200 columns against 7.8 s at 100, GPU busy 40 %), then score_prospective_pairs 2.4 s; the host side (CPU 53 %, GPU idle below 10 % for 36 % of the fit) is the bound, as at 100 columns.

The target. 50 s was set on 2026-11-08 as a rough guide, not as a hard bar. The real goal (owner, 2026-11-10) is near-100 % CPU and GPU occupancy during the fit with every GPU kernel conforming to
CUDA.md; once that is reached at 100000 x 100, the same work is done at 100000 x 1000. nsys on this fit: kernels busy about 15 % of the window, 133k launches, copies about 1 %: the fit is bound by the
host and by the order of dependent stages, not by the kernels or the bus.

How to run it so the number means something.

- ``PYTHONPATH=<worktree>/src python -m mlframe.feature_selection._benchmarks.profiling.all_families_wall [n] [p]`` (defaults 100000 100). A plain ``python`` imports the main checkout, not the worktree.
- Quiet machine only: check ``nvidia-smi`` memory.used (the desktop holds 0.5-1 GB of the 4 GB card and moves the cupy pool limit), free RAM, and that nothing CPU-heavy runs (Defender scans, scrapers,
  a second pytest): the same code swings by 100+ s with load. Never run two GPU processes at once (a 4 GB card turns the second one into OOM fallbacks).
- No cProfile on the GPU path (its memory flips the host-buffer hoist decision and the pair search falls onto the 50-100x slower recompute route). Use the samplers of this package.
- Thermal state: the harness prints the GPU temperature, SM clock, throttle reasons and CPU clock before and after the fit; ``MLFRAME_BENCH_WAIT_COOL_C=60`` waits for the card to cool first. A wall quoted without them is not comparable (CUDA.md section 6).
- ``MRMR._FIT_CACHE`` memoises by content hash: clear it between same-process fits or a refit returns in 0.4 s.
- After a run grep the log for ``OutOfMemoryError``, ``Out of memory allocating`` and ``falling back``: a fallback changes the op counts and the wall.
- Compare family-wall sums or same-process laps, not single walls, when the machine was not quiet.

Companions (all in this package, same protocol: a warm-up fit first, then the measured fit; run one at a time, nothing else heavy beside it):

- ``all_families_wall``            this file: the wall of the benchmark fit and the per-stage walls of the stage-concurrency scheduler.
- ``all_families_stage_sampler``   main-thread sampler: seconds per innermost FE / screen stage and per innermost frame, GPU busy and CPU load per stage (the occupancy table).
- ``all_families_idle_frames``     the same sampler aimed at the stretches where CPU < 25 % and GPU < 10 % (who waits), with the call chains and the pool memory by stage.
- ``all_families_family_walls``    per-family wall ledger (``fe_family_wall``) of the same fit, longest first, plus the selected features.
- ``all_families_fit_ledger``      the ledger the fitted estimator carries (``fe_family_wall_``, 30000 x 40), a quick sanity probe.
- ``all_families_selection_quality``  held-out AUC (LightGBM, logistic regression, all raw columns) of the selected features: judge a GPU / CPU selection difference by this, not by names.
- ``numba_signature_counts``       numba dispatchers with more than one compiled signature after two fits (recompiles inside a fit).
- ``env_sampled_fit``, ``stack_timeline``, ``utilization``, ``tick_utilization``, ``family_profile``, ``family_scale``: older siblings of the same fit (system sampler, 6 s buckets, idle share, one family at a time).
"""
import logging
import os
import sys
import time
import warnings

warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection.filters.mrmr import MRMR

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
N_COLS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
kw = F.all_families_kwargs()
kw["nbins_strategy"] = "mdlp"
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20000, 30, 1))  # warm-up fit
data = F._make(N_ROWS, N_COLS, 0)


class H(logging.Handler):
    """Prints the scheduler's stage-walls lines with the time since the start of the measured fit."""

    t0 = time.time()

    def emit(self, r):
        m = r.getMessage()
        if "stage walls" in m or "concurrency" in m:
            print("%6.1fs %s" % (time.time() - H.t0, m), flush=True)


lg = logging.getLogger("mlframe.feature_selection.filters._mrmr_fit_impl._fe_stage_concurrency")
lg.setLevel(logging.DEBUG)
lg.addHandler(H())
lg.propagate = False
from mlframe.feature_selection._benchmarks.profiling import hardware_state

if os.environ.get("MLFRAME_BENCH_WAIT_COOL_C"):
    hardware_state.wait_until_cool(float(os.environ["MLFRAME_BENCH_WAIT_COOL_C"]))  # the card cooled to this temperature (C) before the measured fit
print("hardware before: " + hardware_state.line(), flush=True)
H.t0 = time.time()
MRMR._FIT_CACHE.clear()
MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
print("WALL %.1f" % (time.time() - H.t0))
print("hardware after:  " + hardware_state.line(), flush=True)
