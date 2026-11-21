"""Peak resident memory of one MRMR fit at a given row count, with a timeline of the main thread's innermost mlframe function at each new peak.

Usage: rss_scale_fit.py ROWS [COLS] [all|default]; RSS_TICK_S=20 adds a timeline line of the main thread every 20 s. The frame is a float32 DataFrame (the project dtype), the target a noisy function of five columns. The sampler records the
process RSS every 0.2 s; a new peak is attributed to the main thread's innermost mlframe frame, which names the step that allocated.
"""
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

warnings.simplefilter("ignore")
for k, v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(k, v)
from mlframe.feature_selection.filters.mrmr import MRMR

rows = int(sys.argv[1])
cols = int(sys.argv[2]) if len(sys.argv) > 2 else 80
mode = sys.argv[3] if len(sys.argv) > 3 else "default"
if mode == "all":
    from mlframe.feature_selection._benchmarks.profiling import family_scale as F

    kwargs = F.all_families_kwargs()
    kwargs["nbins_strategy"] = os.environ.get("BENCH_NBINS", "mdlp")
    if os.environ.get("BENCH_QDTYPE"):  # e.g. int16 / int8: A/B of the codes-matrix width
        kwargs["quantization_dtype"] = getattr(np, os.environ["BENCH_QDTYPE"])
else:
    kwargs = {}

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.standard_normal((rows, cols), dtype=np.float32), columns=[f"x{i}" for i in range(cols)])
signal = X["x0"].to_numpy() * X["x1"].to_numpy() + 0.7 * X["x2"].to_numpy() - 0.5 * X["x3"].to_numpy() ** 2 + 0.3 * X["x4"].to_numpy()
y = pd.Series((signal + 0.5 * rng.standard_normal(rows, dtype=np.float32) > np.median(signal)).astype(np.int8), name="y")
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


def sampler():
    """Track the RSS peak and name the step at each new peak above the previous one by 50 MB."""
    t0 = time.perf_counter()
    last_logged = base
    while not stop.wait(0.2):
        rss = proc.memory_info().rss
        if psutil.virtual_memory().available < MIN_AVAILABLE:
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
                _frames = sys._current_frames()
                _others = []
                for _tid, _fr in _frames.items():
                    if _tid in (main_id, threading.get_ident()):
                        continue
                    _w = innermost(_fr)
                    if _w != "(outside mlframe)" and "threading" not in _w:
                        _others.append(_w.split("/")[-1])
                events.append((time.perf_counter() - t0, (rss - base) / 1e6, innermost(_frames.get(main_id)) + ("   | threads: " + "; ".join(sorted(set(_others))[:4]) if _others else "")))


_CUSHION_CALLS: list = []
if os.environ.get("RSS_TRACE_CUSHION", "").strip():  # record every VRAM-cushion question (bytes asked, answer) to see which device paths the cushion refuses
    from mlframe.feature_selection.filters import _fe_gpu_vram as _vram

    _orig_cushion = _vram.fe_gpu_has_vram_cushion

    def _traced_cushion(nbytes, *args, **kwargs):
        """The cushion check, remembered with its caller."""
        ok = _orig_cushion(nbytes, *args, **kwargs)
        _CUSHION_CALLS.append((int(nbytes), bool(ok), sys._getframe(1).f_code.co_filename.split("filters")[-1] + ":" + sys._getframe(1).f_code.co_name))
        return ok

    _vram.fe_gpu_has_vram_cushion = _traced_cushion

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
    if os.environ.get("RSS_FAMILY_WALLS", "").strip():  # the fit's own per-family wall table (summed over threads, so it exceeds the fit wall under concurrency)
        _fw = getattr(_sel, "fe_family_wall_", None) or {}
        print("FAMILY_WALLS " + "; ".join(f"{k}={v['wall_s']:.1f}s/{v['calls']}" for k, v in list(_fw.items())[:45]), flush=True)
    if _CALL_STATS:
        print("CALLS " + "; ".join(f"{k}: n={v[0]} wall={v[1]:.1f}s none={_CALL_NONE.get(k, 0)}" for k, v in _CALL_STATS.items()), flush=True)
        for _k, _v in _CALL_STATS.items():
            for _a in _v[2]:
                print("CALLARGS " + _k.split(":")[-1] + " " + _a, flush=True)
    if _CUSHION_CALLS:
        import collections as _co

        _agg = _co.defaultdict(lambda: [0, 0, 0])
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
        _prov = getattr(_sel, "fe_provenance_", None)
        _led = getattr(_sel, "fe_rejection_ledger_", None)
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

gc.collect()
print(f"after gc.collect(): {_rss_growth_mb():.0f} MB")
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
