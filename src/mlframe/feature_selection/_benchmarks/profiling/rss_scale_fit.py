"""Peak resident memory of one MRMR fit at a given row count, with a timeline of the main thread's innermost mlframe function at each new peak.

Usage: rss_scale_fit.py ROWS [COLS] [all|default]; RSS_TICK_S=20 adds a timeline line of the main thread every 20 s. The frame is a float32 DataFrame (the project dtype), the target a noisy function of five columns. The sampler records the
process RSS every 0.2 s; a new peak is attributed to the main thread's innermost mlframe frame, which names the step that allocated.

LATEST RESULTS (2026-11-21, tree b786940de, `rss_scale_fit.py ROWS COLS all`, one fit per fresh process, runs one after another with nothing else running; GTX 1050 Ti 4 GB, 16 GB host)
    100000 x 100 : 45.4 / 44.0 / 44.3 s, selected [2, 3, 4] every time, peak RSS growth 1.5-1.6 GB. GPU at the start 71-75 C, SM 1708 MHz (1695 at the end of the third run): not throttled.
                   Earlier the same day, before the later changes: 38.8-44.1 s fresh process, about 33 s steady state.
    100000 x 1000: 221-232 s in the fast path (selected [2, 4]; stage:score_one_pair 11-17 s, the device ext-val path declined 0-4 times) and 234-352 s in the slow path (selected [2];
                   score_one_pair 25-38 s, the device path declined 12-30 times by the VRAM cushion); both paths of the SAME fit, one of them per run, about one run in three slow. A/B series of 5 fits each:
                   median 290 s before the day's changes against 296 s after; two of the five older fits took 580 and 702 s (CPU fallbacks of the pair MI / usability scan), none of the five new ones. See the audit tracker
                   (MRMR-TWO-MODES-ROOT-CAUSE-2026-11-21, MRMR-AB-SERIES-2026-11-21).
    500000 x 1000: 896.7 s, selected [2, 3], peak RSS growth 7.6 GB (3.8 x the 2 GB frame). GPU 75 C / 1695 MHz at the start, 69 C / 1708 MHz at the end. The fit ran with the card nearly full: 6 CPU fallbacks
                   ('GPU upload REJECTED ... requested 2.11GB (n_samples=500000, n_cols=1126, dtype=int32) exceeds the safe relative cap 0.94GB', 'GPU path declined for lack of free VRAM'), and the timeline shows
                   about 120 s at 100 % CPU in the CPU pair MI (batch_pair_mi_prange_cols) at t = 480-600 s. The same size took about 604 s in the middle of the day; it was not re-run in between.
    500000 x 1000 (2026-11-22 evening, tree 72a3a5c7e, one fresh process, nothing else running, GPU 70 C / 1721 MHz at the end): 510.0 s, peak RSS growth 4695 MB (2.35 x the frame; 11-21: 896.7 s, 7.6 GB).
                   One 'GPU upload REJECTED' (pair MI, 2.11 GB against a 1.17 GB cap; the feature-major kernel then streamed it) and two 'GPU path declined' (1015-1017 MiB free against the 1024 MiB cushion); one device OOM
                   in the additive-fusion proposal (36 MB, host proposal ran). Main-thread ticks every 60 s: cpu 15-58 %, gpu 0-100 %, i.e. long stretches with one side idle.
    1000000 x 1000: NOT RUN: MRMR.fit refuses before the first stage ('refusing to allocate for n*p=1_000_000_000 (~4.00 GB int32 working set) on a host with 6.73 GB available RAM') after the 4 GB frame is built;
                   status ValueError after 39.8 s. The host would need roughly 4 GB more free RAM than this machine has after the frame.
TARGET KINDS (2026-11-22, `bench_target_strategy_matrix.py 20000 20`, mdlp, one fresh process per cell; BENCH_TARGET picks the kind): binary 32.2 s, multiclass 50.2, regression 50.2, count 54.3, multilabel 49.6,
    multitarget 61.5, heavy_tail 51.6, zero_inflated 48.8. The same grid at the start of the day: regression 298 s, multitarget 495 s, zero_inflated 194 s (the auto-scorer batch declined every continuous target,
    and the default-on target-mix second pass fitted two child MRMRs: now `fe_target_mix_candidates=False`). `support_` lists raw columns only, so a fit that selects engineered columns can show 0 selected.
Wall times on this machine spread by up to 50 % between runs of one code state; read differences below about 15 % as noise unless several fits per side were taken.
"""
from mlframe.feature_selection._benchmarks.binning_label import binning_params
from typing import Any
import os
import sys
import threading
import types
import time
import warnings

import numpy as np
import pandas as pd
import psutil

warnings.simplefilter("ignore", category=FutureWarning); warnings.simplefilter("ignore", category=DeprecationWarning)
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection.filters.mrmr import MRMR

rows = int(sys.argv[1])
cols = int(sys.argv[2]) if len(sys.argv) > 2 else 80
mode = sys.argv[3] if len(sys.argv) > 3 else "default"
if mode == "all":
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F

    kwargs = F.all_families_kwargs()
    kwargs.update(binning_params(os.environ.get("BENCH_NBINS", "mdlp")))
    if os.environ.get("BENCH_QDTYPE"):  # e.g. int16 / int8: A/B of the codes-matrix width
        kwargs["quantization_dtype"] = getattr(np, os.environ["BENCH_QDTYPE"])
else:
    kwargs = {}

rng = np.random.default_rng(0)
# drawn in row chunks of the same generator stream (the same values as one (rows, cols) draw) straight into a column-major buffer: one frame in memory, not the draw plus pandas' copy of it
_buf = np.empty((rows, cols), dtype=np.float32, order="F")
for _r0 in range(0, rows, 50000):
    _buf[_r0 : _r0 + 50000] = rng.standard_normal((min(50000, rows - _r0), cols), dtype=np.float32)
X = pd.DataFrame(_buf, columns=[f"x{i}" for i in range(cols)], copy=False)
del _buf
signal = X["x0"].to_numpy() * X["x1"].to_numpy() + 0.7 * X["x2"].to_numpy() - 0.5 * X["x3"].to_numpy() ** 2 + 0.3 * X["x4"].to_numpy()
_noisy = signal + 0.5 * rng.standard_normal(rows, dtype=np.float32)
TARGET_KIND = os.environ.get("BENCH_TARGET", "binary")  # binary | multiclass | regression | count | multilabel | multitarget | heavy_tail | zero_inflated
if TARGET_KIND == "binary":
    y = pd.Series((_noisy > np.median(signal)).astype(np.int8), name="y")
elif TARGET_KIND == "multiclass":  # four classes cut at the quartiles of the noisy signal
    y = pd.Series(np.searchsorted(np.quantile(_noisy, [0.25, 0.5, 0.75]), _noisy).astype(np.int8), name="y")
elif TARGET_KIND == "regression":
    y = pd.Series(_noisy.astype(np.float32), name="y")
elif TARGET_KIND == "count":  # Poisson counts with a log-linear rate in the signal
    y = pd.Series(rng.poisson(np.exp(0.3 * np.clip(_noisy, -4.0, 4.0))).astype(np.int32), name="y")
elif TARGET_KIND == "multilabel":  # three binary labels, each driven by other columns
    _l1 = X["x5"].to_numpy() + 0.5 * X["x6"].to_numpy() + 0.5 * rng.standard_normal(rows, dtype=np.float32)
    _l2 = X["x7"].to_numpy() * X["x8"].to_numpy() + 0.5 * rng.standard_normal(rows, dtype=np.float32)
    y = pd.DataFrame({"l0": (_noisy > np.median(_noisy)).astype(np.int8), "l1": (_l1 > np.median(_l1)).astype(np.int8), "l2": (_l2 > np.median(_l2)).astype(np.int8)})
elif TARGET_KIND == "multitarget":  # two continuous outputs
    y = pd.DataFrame({"o0": _noisy.astype(np.float32), "o1": (X["x5"].to_numpy() - 0.5 * X["x6"].to_numpy() + 0.3 * rng.standard_normal(rows, dtype=np.float32)).astype(np.float32)})
elif TARGET_KIND == "heavy_tail":  # a continuous target with Cauchy-like tails
    y = pd.Series((signal + 0.5 * rng.standard_cauchy(rows)).astype(np.float32), name="y")
elif TARGET_KIND == "zero_inflated":  # most rows zero, the rest a positive function of the signal
    y = pd.Series(np.where(rng.random(rows) < 0.7, 0.0, np.exp(0.5 * np.clip(_noisy, -4.0, 4.0))).astype(np.float32), name="y")
else:
    raise ValueError(f"unknown BENCH_TARGET {TARGET_KIND!r}")
del _noisy
frame_mb = X.memory_usage(deep=False).sum() / 1e6

import tracemalloc

TRACE = os.environ.get("RSS_TRACEMALLOC") == "1"
CENSUS_AT = float(os.environ.get("RSS_CENSUS_AT_S", "0") or 0)  # print the heaviest live containers at this second and exit
STOP_AFTER = float(os.environ.get("RSS_STOP_AFTER_S", "0") or 0)  # with RSS_TRACEMALLOC=1: print the live allocations at this second and exit
if TRACE:
    tracemalloc.start(6)
trace = types.SimpleNamespace(snapshots=[])  # the sampler thread appends its tracemalloc snapshots here
proc = psutil.Process()
TICK = float(os.environ.get("RSS_TICK_S", "0"))  # seconds between timeline lines of the main thread's step (0: off)
ticks: list[Any] = []
MIN_AVAILABLE = float(os.environ.get("RSS_MIN_AVAILABLE_GB", "1.0")) * 1e9
base = proc.memory_info().rss
peak = [base]
events: list[Any] = []
stop = threading.Event()
main_id: int = threading.main_thread().ident or 0


def proc_cpu() -> Any:
    """System CPU percent since the previous call."""
    return psutil.cpu_percent(interval=None)


def gpu_util() -> str:
    """GPU utilisation percent from nvidia-smi, or ``?``."""
    import subprocess

    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip() or "?"
    except Exception:  # no nvidia-smi: the column stays unknown
        return "?"


def innermost(frame):
    """The innermost mlframe frame of a stack as file:function:line."""
    while frame is not None:
        fn = frame.f_code.co_filename.replace("\\", "/")
        if "/mlframe/" in fn and "rss_scale_fit" not in fn:
            return f"{fn.split('/mlframe/')[-1]}:{frame.f_code.co_name}:{frame.f_lineno}"
        frame = frame.f_back
    return "(outside mlframe)"


def _stack_snapshot() -> "tuple[str, list[str]]":
    """Where the main thread is and where the other threads are, without keeping any stack frame alive afterwards.

    ``sys._current_frames()`` includes the frame of the calling thread itself; a dict of frames held in a local of that very frame is a reference cycle, and every cycle of this kind kept the
    whole stack of every thread, with all its locals (the legs frames of the running families, 300 MB each at 100000 x 1000), alive until the next garbage collection. The sampler's own
    frame is dropped from the dict and the dict lives only inside this function, so the memory the benchmark reports is the fit's, not the sampler's."""
    frames = sys._current_frames()
    frames.pop(threading.get_ident(), None)
    main_where = innermost(frames.get(main_id))
    others: list = []
    for tid, fr in frames.items():
        if tid == main_id:
            continue
        where = innermost(fr)
        if where != "(outside mlframe)" and "threading" not in where:
            others.append(where.split("/")[-1])
    return main_where, others


_avail0 = float(psutil.virtual_memory().available)
_min_avail = [_avail0]  # [lowest system-wide available RAM seen during the fit, bytes]


def sampler():
    """Track the RSS peak and name the step at each new peak above the previous one by 50 MB."""
    t0 = time.perf_counter()
    last_logged = base
    while not stop.wait(0.2):
        rss = proc.memory_info().rss
        _avail = psutil.virtual_memory().available
        if _avail < _min_avail[0]:
            _min_avail[0] = _avail  # the system-wide free RAM: what the paging risk depends on (the RSS also counts working-set pages the system can take back without a write)
        if _avail < MIN_AVAILABLE:
            # the machine is about to page: report what is known and stop the process instead of freezing the laptop
            print(f"ABORTED at t={time.perf_counter() - t0:.1f}s: available memory under {MIN_AVAILABLE / 1e9:.1f} GB, peak RSS growth {(max(peak[0], rss) - base) / 1e6:.0f} MB = {(max(peak[0], rss) - base) / 1e6 / frame_mb:.1f} x the frame", flush=True)
            for when, mb, where in events:
                print(f"  t={when:7.1f}s +{mb:7.0f} MB  {where}", flush=True)
            print(f"  now: {innermost(sys._current_frames().get(main_id))}", flush=True)
            if ticks:  # the timeline of the aborted run is what shows where the CPU and the GPU sat idle
                print("--- timeline of the main thread")
                for when_t, mb_t, where_t in ticks:
                    print(f"  tick t={when_t:7.1f}s +{mb_t:7.0f} MB  {where_t}", flush=True)
            os._exit(3)  # exit-ok: a script run as a process; only os._exit ends the main thread stuck inside the fit, SystemExit here would end this thread alone
        if CENSUS_AT and time.perf_counter() - t0 >= CENSUS_AT:
            from mlframe.feature_selection._benchmarks.profiling import mem_census

            print(f"--- container census at t={time.perf_counter() - t0:.0f}s, RSS growth {(rss - base) / 1e6:.0f} MB (X itself is {frame_mb:.0f} MB)", flush=True)
            print(mem_census.render(mem_census.name_owners(mem_census.census(30))), flush=True)
            os._exit(0)  # exit-ok: a script run as a process; the main thread is inside the fit
        if STOP_AFTER and time.perf_counter() - t0 >= STOP_AFTER and TRACE:  # a snapshot of the plateau: who holds the live memory
            snap = tracemalloc.take_snapshot()
            print(f"--- tracemalloc at t={time.perf_counter() - t0:.0f}s, RSS growth {(rss - base) / 1e6:.0f} MB: the largest live allocations", flush=True)
            for st in snap.statistics("traceback")[:18]:
                print(f"  {st.size / 1e6:8.0f} MB in {st.count} blocks", flush=True)
                for line in st.traceback.format()[-5:]:
                    print("     ", line.strip()[:160], flush=True)
            os._exit(0)  # exit-ok: a script run as a process; the main thread is inside the fit
        if TICK and (not ticks or time.perf_counter() - t0 - ticks[-1][0] >= TICK):
            ticks.append((time.perf_counter() - t0, (rss - base) / 1e6, f"cpu {proc_cpu():3.0f}% gpu {gpu_util():>3}% " + innermost(sys._current_frames().get(main_id))))
        if rss > peak[0]:
            peak[0] = rss
            if TRACE and rss - base > float(os.environ.get("RSS_SNAP_FIRST_MB", "600")) * 1e6 and (not trace.snapshots or rss - trace.snapshots[-1][0] > float(os.environ.get("RSS_SNAP_STEP_MB", "400")) * 1e6):
                snap = tracemalloc.take_snapshot()
                trace.snapshots.append((rss, (rss - base) / 1e6, snap.statistics("traceback")[:14]))
            if rss - last_logged > 50e6:
                last_logged = rss
                _main_where, _others = _stack_snapshot()
                _cpm = sys.modules.get("cupy")
                _pool = ""
                if _cpm is not None:  # the driver of a WDDM card maps the device pool into the process: RSS includes it (measured 2026-11-23, rss_vs_device.py)
                    try:
                        _dp = _cpm.get_default_memory_pool()
                        _pool = f"   [device pool total {_dp.total_bytes() / 1e6:.0f} MB used {_dp.used_bytes() / 1e6:.0f} MB]"
                    except Exception:
                        _pool = ""
                events.append((time.perf_counter() - t0, (rss - base) / 1e6, _main_where + _pool + ("   | threads: " + "; ".join(sorted(set(_others))[:4]) if _others else "")))


_CUSHION_CALLS: list = []
if os.environ.get("RSS_TRACE_CUSHION", "").strip():  # record every VRAM-cushion question (bytes asked, answer) to see which device paths the cushion refuses
    from mlframe.feature_selection.filters import _fe_gpu_vram as _vram

    _orig_cushion = _vram.fe_gpu_has_vram_cushion

    def _traced_cushion(nbytes, *args, **kwargs):
        """The cushion check, remembered with its caller."""
        ok = _orig_cushion(nbytes, *args, **kwargs)
        _CUSHION_CALLS.append((int(nbytes), bool(ok), sys._getframe(1).f_code.co_filename.split("filters")[-1] + ":" + sys._getframe(1).f_code.co_name))
        return ok

    _vram.fe_gpu_has_vram_cushion = _traced_cushion  # type: ignore[assignment]  # the spy keeps the wrapped function's call shape

if os.environ.get("RSS_LOG_INFO", "").strip():  # comma-separated logger names switched to INFO on stdout (a throttled INFO line then shows why a path was declined)
    import logging as _lg

    _h = _lg.StreamHandler(sys.stdout)
    _h.setFormatter(_lg.Formatter("LOG %(name)s: %(message)s"))
    for _ln in os.environ["RSS_LOG_INFO"].split(","):
        _lgr = _lg.getLogger(_ln.strip())
        _lgr.setLevel(_lg.INFO)
        _lgr.addHandler(_h)

_PAIR_CAPTURE: list = []
if os.environ.get("RSS_CAPTURE_PAIRS", "").strip():  # what the per-pair FE search returns for each pair (pair, pair MI, best MI, best configuration): the first pair whose answer differs between two runs
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_core as _pc

    _orig_score_one = _pc._score_one_pair

    def _capturing_score_one(**kwargs):
        """Run the per-pair search and record its answer."""
        out = _orig_score_one(**kwargs)
        try:
            _PAIR_CAPTURE.append({"pair": [int(v) for v in kwargs["raw_vars_pair"]], "pair_mi": float(kwargs["pair_mi"]), "best_mi": None if out[2] is None else float(out[2]), "best_config": repr(out[1])[:200], "entry": repr(out[0])[:1500]})
        except Exception as exc:  # the capture must not break the fit
            _PAIR_CAPTURE.append({"error": repr(exc)})
        return out

    _pc._score_one_pair = _capturing_score_one

_FE_STEP_CAPTURE: list = []
if os.environ.get("RSS_CAPTURE_FE_STEP", "").strip():  # the column pool and the selected variables each time the FE step starts: where two runs of one fit first differ
    from mlframe.feature_selection.filters.mrmr import MRMR as _MRMR_CLS

    _orig_fe_step = _MRMR_CLS._run_fe_step

    def _capturing_fe_step(self, **kwargs):
        """Record what the FE step starts from, then run it."""
        try:
            _FE_STEP_CAPTURE.append({"cols": [str(c) for c in kwargs.get("cols", [])], "selected_vars": sorted(int(v) for v in (kwargs.get("selected_vars") or []))})
        except Exception as exc:  # the capture must not break the fit
            _FE_STEP_CAPTURE.append({"error": repr(exc)})
        out = _orig_fe_step(self, **kwargs)
        try:
            if out is not None:
                _FE_STEP_CAPTURE[-1].update({"out_cols": [str(c) for c in out[1]], "out_selected": sorted(int(v) for v in out[4]), "n_recommended": int(out[5])})
        except Exception as exc:  # the capture must not break the fit
            _FE_STEP_CAPTURE.append({"error_out": repr(exc)})
        return out

    _MRMR_CLS._run_fe_step = _capturing_fe_step

def _sync_gpu() -> None:
    """Block until queued device work finishes, so a wall stopped after a call includes its GPU compute (a launch-only timer under-counts by orders of magnitude)."""
    try:
        import cupy as _cupy
    except ImportError:
        return
    try:
        _cupy.cuda.Device().synchronize()
    except Exception:  # cupy without a usable device: nothing to wait for
        return

_CALL_STATS: dict = {}
_CALL_NONE: dict = {}
for _spec in [x for x in os.environ.get("RSS_TRACE_CALLS", "").split(",") if x.strip()]:  # "package.module:function,...": count and sum the wall of those calls (wrapped on the module attribute)
    import importlib as _il

    _mod_name, _fn_name = _spec.strip().split(":")
    _mod = _il.import_module(_mod_name)
    _fn = getattr(_mod, _fn_name)

    def _make(fn, key):
        """A timing wrapper of ``fn`` recording into ``_CALL_STATS[key]``."""

        def wrapper(*args, **kwargs):
            """Call ``fn`` and add its wall to the stats."""
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                if result is None:
                    _CALL_NONE[key] = _CALL_NONE.get(key, 0) + 1
                return result
            finally:
                _sync_gpu()
                row = _CALL_STATS.setdefault(key, [0, 0.0, []])
                row[0] += 1
                row[1] += time.perf_counter() - t0
                if len(row[2]) < 8:  # the first calls' arguments: type and shape / value
                    row[2].append(",".join(("None" if x is None else type(x).__name__ + str(getattr(x, "shape", x if isinstance(x, (int, float, bool)) else ""))) for x in args[:8]))

        return wrapper

    setattr(_mod, _fn_name, _make(_fn, _spec.strip()))

th = threading.Thread(target=sampler, daemon=True)
th.start()
t = time.perf_counter()
status = "ok"
try:
    _sel = MRMR(verbose=0, random_seed=0, n_jobs=1, **kwargs).fit(X, y)
    _sup = sorted(int(v) for v in np.asarray(getattr(_sel, 'support_', [])).ravel())
    print(f'selected {len(_sup)} columns, sum {sum(_sup)}: {_sup[:20]}', flush=True)
    print(f'engineered {len(getattr(_sel, "_engineered_features_", None) or [])} features', flush=True)  # support_ lists raw columns only: a fit whose raws are all covered by engineered columns has support_ == []
    try:  # which of the five generating columns x0..x4 the output covers, as a raw output column or as a source inside an engineered name
        import re as _re

        _names = [str(c) for c in _sel.get_feature_names_out()]
        _covered = sorted({int(m) for nm in _names for m in _re.findall(r"(?<![A-Za-z0-9])x(\d+)(?![0-9])", nm) if int(m) <= 4})
        print(f"SOURCES covered {_covered} of [0, 1, 2, 3, 4]; {len(_names)} output columns: {_names[:12]}", flush=True)
    except Exception as _exc:
        print("SOURCES unavailable:", type(_exc).__name__, _exc, flush=True)
    if os.environ.get("RSS_FAMILY_WALLS", "").strip():  # the fit's own per-family wall table (summed over threads, so it exceeds the fit wall under concurrency)
        _fw = getattr(_sel, "fe_family_wall_", None) or {}
        print("FAMILY_WALLS " + "; ".join(f"{k}={v['wall_s']:.1f}s/{v['calls']}" for k, v in list(_fw.items())[:45]), flush=True)
    if os.environ.get("MLFRAME_FE_FAMILY_RSS", "").strip() == "1":  # per family: highest RSS above the entry of a call / RSS still held at its exit, max over calls (MB); concurrent families overlap
        from mlframe.feature_selection.filters import _fe_family_timing as _fft

        _rows = sorted(_fft._FE_FAMILY_RSS.items(), key=lambda kv: kv[1][0], reverse=True)
        print("FAMILY_RSS " + "; ".join(f"{k}=+{v[0] / 2**20:.0f}/+{v[1] / 2**20:.0f}" for k, v in _rows[:45]), flush=True)
    if _CALL_STATS:
        print("CALLS " + "; ".join(f"{k}: n={v[0]} wall={v[1]:.1f}s none={_CALL_NONE.get(k, 0)}" for k, v in _CALL_STATS.items()), flush=True)
        for _k, _v in _CALL_STATS.items():
            for _a in _v[2]:
                print("CALLARGS " + _k.split(":")[-1] + " " + _a, flush=True)
    if _CUSHION_CALLS:
        import collections as _co

        _agg: dict = _co.defaultdict(lambda: [0, 0, 0])
        for _nb, _ok, _who in _CUSHION_CALLS:
            _row = _agg[_who]
            _row[0 if _ok else 1] += 1
            _row[2] = max(_row[2], _nb)
        print("CUSHION " + "; ".join(f"{w}: ok={r[0]} refused={r[1]} max={r[2] / 1e6:.0f}MB" for w, r in sorted(_agg.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:14]), flush=True)
    if _PAIR_CAPTURE and os.environ.get("RSS_CAPTURE_PAIRS", "").strip():
        import json as _json2

        with open(os.environ["RSS_CAPTURE_PAIRS"], "w", encoding="utf-8") as _fh2:
            _json2.dump(_PAIR_CAPTURE, _fh2)
    if _FE_STEP_CAPTURE and os.environ.get("RSS_CAPTURE_FE_STEP", "").strip():
        import json as _json

        with open(os.environ["RSS_CAPTURE_FE_STEP"], "w", encoding="utf-8") as _fh:
            _json.dump(_FE_STEP_CAPTURE, _fh)
    _dump = os.environ.get("RSS_DUMP_JSON", "").strip()
    if _dump:  # the fitted state that differs between runs (which stage produced which survivor), for run-to-run diffs
        import json

        _state = {k: getattr(_sel, k, None) for k in ("support_", "feature_names_out_", "selected_features_")}
        _prov: Any = getattr(_sel, "fe_provenance_", None)
        _led: Any = getattr(_sel, "fe_rejection_ledger_", None)
        if hasattr(_led, "to_csv"):  # the per-candidate verdicts, for the first-divergence diff of two runs
            _led.to_csv(_dump + ".ledger.csv", index=False)
        _state["fe_provenance_"] = _prov.to_dict("records") if hasattr(_prov, "to_dict") else repr(_prov)  # the full survivor table, not its truncated repr
        with open(_dump, "w", encoding="utf-8") as _fh:
            json.dump(_state, _fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o), indent=1, sort_keys=True)
except BaseException as exc:  # reported with the peak: a MemoryError at scale is the finding
    status = f"{type(exc).__name__}: {str(exc)[:120]}"
wall = time.perf_counter() - t
stop.set()
def _rss_growth_mb():
    """The process RSS growth over the start of the script, in MB."""
    return (proc.memory_info().rss - base) / 1e6


print(f"RSS growth right after the fit: {_rss_growth_mb():.0f} MB")
import gc


def _census_after_fit(label: str) -> None:
    """The heaviest live containers that hold numpy arrays at this point (``RSS_CENSUS_AFTER_FIT=1``): what keeps the memory of the finished fit."""
    from mlframe.feature_selection._benchmarks.profiling import mem_census

    rows = mem_census.name_owners(mem_census.census(top=14))
    print(f"CENSUS {label} (RSS growth {_rss_growth_mb():.0f} MB):")
    print(mem_census.render(rows), flush=True)


if os.environ.get("RSS_CENSUS_AFTER_FIT", "").strip():
    _census_after_fit("right after the fit, before gc")
if os.environ.get("RSS_STEPS_AFTER_FIT", "").strip():  # which step gives the memory of the finished fit back: waiting, the collector, or a heap trim
    print(f"STEPS threads={threading.active_count()} gc objects={len(gc.get_objects())} rss growth={_rss_growth_mb():.0f} MB", flush=True)
    time.sleep(5)
    print(f"STEPS after 5 s of waiting: {_rss_growth_mb():.0f} MB", flush=True)
    _n = gc.collect()
    print(f"STEPS after gc.collect() ({_n} unreachable): {_rss_growth_mb():.0f} MB; gc objects={len(gc.get_objects())}", flush=True)
    try:
        import ctypes

        ctypes.CDLL("msvcrt")._heapmin()
        print(f"STEPS after _heapmin: {_rss_growth_mb():.0f} MB", flush=True)
    except Exception as _exc:  # not Windows or no msvcrt
        print("STEPS _heapmin unavailable:", _exc, flush=True)
if os.environ.get("RSS_GARBAGE_AFTER_FIT", "").strip():  # what sits in reference cycles at the end of the fit: types of the unreachable objects, and what holds the biggest frames
    import collections as _co

    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.collect()
    gc.set_debug(0)
    _garbage = list(gc.garbage)
    _types = _co.Counter(type(o).__name__ for o in _garbage)
    _arr_bytes = 0
    _seen = set()
    for _o in _garbage:
        if isinstance(_o, np.ndarray):
            _root = _o
            while isinstance(_root.base, np.ndarray):
                _root = _root.base
            if id(_root) not in _seen:
                _seen.add(id(_root))
                _arr_bytes += int(_root.nbytes)
    print(f"GARBAGE {len(_garbage)} unreachable objects holding {_arr_bytes / 1e6:.0f} MB of arrays; types: " + ", ".join(f"{k}={v}" for k, v in _types.most_common(14)), flush=True)
    _frames = [o for o in _garbage if isinstance(o, pd.DataFrame)]
    print(f"GARBAGE DataFrames: {len(_frames)}", flush=True)
    _ids = {id(o) for o in _garbage}
    for _fr in sorted(_frames, key=lambda d: -d.shape[1])[:3]:
        _refs = [r for r in gc.get_referrers(_fr) if id(r) in _ids and r is not _garbage]
        print("  frame", _fr.shape, "held by", [(type(r).__name__, (list(r.keys())[:4] if isinstance(r, dict) else getattr(r, "__name__", ""))) for r in _refs][:6], flush=True)
    _frame_objs = [o for o in _garbage if type(o).__name__ == "frame"]
    _ref_types = _co.Counter(type(r).__name__ for fo in _frame_objs for r in gc.get_referrers(fo) if id(r) in _ids and r is not _garbage)
    print("GARBAGE what refers to the stack frames:", dict(_ref_types.most_common(8)), flush=True)
    _exc_types = _co.Counter(type(o).__name__ for o in _garbage if isinstance(o, BaseException))
    print("GARBAGE exceptions:", dict(_exc_types.most_common(8)), "| tracebacks:", sum(1 for o in _garbage if type(o).__name__ == "traceback"), flush=True)
    _holders: Any = _co.Counter()
    for fo in _frame_objs:
        _big = [k for k, v in fo.f_locals.items() if isinstance(v, (pd.DataFrame, np.ndarray)) and getattr(v, "nbytes", 0) + (v.memory_usage(deep=False).sum() if isinstance(v, pd.DataFrame) else 0) > 50e6]
        if _big:
            _holders[(fo.f_code.co_name, tuple(_big[:3]))] += 1
    print("GARBAGE frames holding a local over 50 MB:", dict(_holders.most_common(8)), flush=True)
    for _t in ("frame", "traceback", "function", "cell", "method"):
        _items = [o for o in _garbage if type(o).__name__ == _t]
        if _items and _t in ("frame", "traceback"):
            print(f"  {_t}: {len(_items)}; first: " + str([(getattr(o, "f_code", None) and o.f_code.co_name) or getattr(getattr(o, "tb_frame", None), "f_code", None) for o in _items[:5]]), flush=True)
    del _garbage, _types, _seen, _frames, _ids
    gc.garbage.clear()
gc.collect()
print(f"after gc.collect(): {_rss_growth_mb():.0f} MB")
if os.environ.get("RSS_CENSUS_AFTER_FIT", "").strip():
    _census_after_fit("after gc.collect()")
try:
    import cupy as _cp

    _cp.get_default_pinned_memory_pool().free_all_blocks()
    print(f"after the pinned pool is freed: {_rss_growth_mb():.0f} MB")
    _cp.get_default_memory_pool().free_all_blocks()
    print(f"after the device pool is freed: {_rss_growth_mb():.0f} MB")
except Exception as exc:  # no cupy
    print("pool release unavailable:", exc)
try:
    import cupy as _cp

    _pin = _cp.get_default_pinned_memory_pool()
    _dev = _cp.get_default_memory_pool()
    print(f"after the fit: pinned host pool {_pin.n_free_blocks()} free blocks; device pool used {_dev.used_bytes() / 1e6:.0f} MB, total {_dev.total_bytes() / 1e6:.0f} MB")
except Exception as exc:  # no cupy
    print("pool stats unavailable:", exc)
print(f"rows={rows} cols={cols} mode={mode} frame={frame_mb:.0f} MB status={status} wall={wall:.1f}s")
print(f"system RAM: available at the start {_avail0 / 1e6:.0f} MB, lowest during the fit {_min_avail[0] / 1e6:.0f} MB, i.e. {(_avail0 - _min_avail[0]) / 1e6:.0f} MB taken ({(_avail0 - _min_avail[0]) / 1e6 / frame_mb:.2f} x the frame)")
print(f"peak RSS growth {(peak[0] - base) / 1e6:.0f} MB = {(peak[0] - base) / 1e6 / frame_mb:.2f} x the frame; RSS before the fit {base / 1e6:.0f} MB")
for when, mb, where in events:
    print(f"  t={when:7.1f}s +{mb:7.0f} MB  {where}")
if ticks:
    print("--- timeline of the main thread")
    for when, mb, where in ticks:
        print(f"  tick t={when:7.1f}s +{mb:7.0f} MB  {where}")

for _rss, mb, stats in trace.snapshots[-2:]:
    print(f"--- tracemalloc at RSS growth {mb:.0f} MB: the largest live allocations")
    for st in stats:
        print(f"  {st.size / 1e6:8.0f} MB in {st.count} blocks")
        for line in st.traceback.format()[-4:]:
            print("     ", line.strip()[:150])
