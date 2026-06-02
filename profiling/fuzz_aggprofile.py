"""Aggregating random-combo profiler + bug-hunt over the fuzz combo space.

Why this exists: ad-hoc profiling that varies only the MODEL (and uses default
configs) never exercises the encoding / imputer / scaler / outlier / MRMR-FE
branches that the fuzz axes route through. This reuses the fuzz harness's exact
per-combo config application (preprocessing / outlier / feature-selection /
mrmr) over many RANDOM non-torch combos, accumulating mlframe-side non-njit
self-time per function across the diverse branches, and reporting any combo
that throws or hangs (a real bug / pathological-cost signal with all axes on).

Robustness: each combo runs in its OWN subprocess with a wall-clock timeout, so
a hanging branch (e.g. O(n^2) outlier detection, or a torch path that slips the
filter) is killed and the sweep continues instead of stalling the whole run.

Usage (from repo root):
    python profiling/fuzz_aggprofile.py [N] [--n-rows R] [--timeout T]
    python profiling/fuzz_aggprofile.py --one <idx> [--n-rows R]   # internal
"""
from __future__ import annotations
import argparse, io, os, random, subprocess, sys, time, traceback, dataclasses, collections, pstats, cProfile

# Make the package importable whether invoked as a path script or via -m.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_TORCH = {"mlp", "lstm", "gru", "transformer", "tabnet", "ft_transformer", "node", "saint"}


def _is_torch(combo) -> bool:
    if any(m in _TORCH for m in combo.models):
        return True
    rec = getattr(combo, "recurrent_model", None) or getattr(combo, "recurrent_model_cfg", None)
    return rec not in (None, "none", "None")


def _build_pool():
    """Deterministic shuffled pool of non-torch combos (same in runner + worker)."""
    from tests.training._fuzz_combo import enumerate_combos
    combos = enumerate_combos(target=150, master_seed=20260601)
    pool = [c for c in combos if not _is_torch(c)]
    random.Random(20260602).shuffle(pool)
    return pool


def _run_one(idx: int, n_rows: int) -> None:
    """Profile a single combo by pool index; emit PROF:/ELAPSED:/ERR: lines."""
    from tests.training._fuzz_combo import build_frame_for_combo
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor
    from tests.training.test_fuzz_suite import (
        _preprocessing_for_combo, _config_for_models, _configs_for_combo,
        _custom_pre_pipelines_for_combo, _outlier_detector_for_combo,
    )
    from mlframe.training import FeatureSelectionConfig, OutlierDetectionConfig, OutputConfig
    from mlframe.training.core import train_mlframe_models_suite

    pool = _build_pool()
    if idx >= len(pool):
        print("ERR:index out of range", flush=True)
        return
    combo = dataclasses.replace(pool[idx], n_rows=n_rows)
    print(f"COMBO:{combo.short_id()}:{combo.models}:{combo.target_type}:fe={combo.use_mrmr_fs}:ens={combo.use_ensembles}", flush=True)
    t0 = time.time()
    pr = cProfile.Profile()
    try:
        df, tcol, _ = build_frame_for_combo(combo)
        fte = SimpleFeaturesAndTargetsExtractor(target_column=tcol, regression=(combo.target_type == "regression"))
        pr.enable()
        train_mlframe_models_suite(
            df=df, target_name=combo.short_id(), model_name=combo.short_id(),
            features_and_targets_extractor=fte, mlframe_models=list(combo.models),
            hyperparams_config=_config_for_models(combo.models, combo.n_rows, iterations=3,
                                                  early_stopping_rounds=getattr(combo, "early_stopping_rounds_cfg", 0)),
            preprocessing_config=_preprocessing_for_combo(combo),
            outlier_detection_config=OutlierDetectionConfig(detector=_outlier_detector_for_combo(combo)),
            feature_selection_config=FeatureSelectionConfig(
                use_mrmr_fs=combo.use_mrmr_fs,
                custom_pre_pipelines=_custom_pre_pipelines_for_combo(combo) or {},
                mrmr_kwargs=({"verbose": 0, "max_runtime_mins": 1, "n_workers": 1, "quantization_nbins": 5,
                              "use_simple_mode": True, "min_nonzero_confidence": 0.9, "max_consec_unconfirmed": 3,
                              "full_npermutations": 3} if combo.use_mrmr_fs else None),
            ),
            output_config=OutputConfig(data_dir=os.path.join(_REPO, "profiling", "_results", "aggprof"), models_dir="models"),
            use_ordinary_models=True, use_mlframe_ensembles=combo.use_ensembles, verbose=0,
            **_configs_for_combo(combo),
        )
        pr.disable()
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats("mlframe", 50)
        for l in s.getvalue().splitlines():
            if ".py:" in l and "mlframe" in l:
                p = l.split(None, 5)
                try:
                    tt = float(p[1])
                    if tt > 0.02 and not any(x in l for x in ("njit", "warmup", "prewarm", "_kernel")):
                        print(f"PROF:{tt:.4f}:{p[5].split(os.sep)[-1].strip()}", flush=True)
                except (ValueError, IndexError):
                    pass
        print(f"ELAPSED:{time.time()-t0:.1f}", flush=True)
    except Exception as e:
        try: pr.disable()
        except Exception: pass
        print(f"ERR:{type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def _run_sweep(n: int, n_rows: int, timeout: float) -> None:
    pool = _build_pool()
    n = min(n, len(pool))
    print(f"aggregating profile over {n} random NON-torch combos @ n={n_rows}, per-combo timeout={timeout:.0f}s", flush=True)
    agg = collections.defaultdict(lambda: [0.0, 0])
    fails = []
    for idx in range(n):
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--one", str(idx), "--n-rows", str(n_rows)]
        combo_id = "?"
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=_REPO)
            out = res.stdout
        except subprocess.TimeoutExpired as te:
            out = (te.stdout or b"").decode("utf-8", "replace") if isinstance(te.stdout, bytes) else (te.stdout or "")
            for l in out.splitlines():
                if l.startswith("COMBO:"):
                    combo_id = l.split(":", 2)[1]
            fails.append((combo_id, f"TIMEOUT>{timeout:.0f}s"))
            print(f"  [{idx:2}] TIMEOUT {combo_id} (>{timeout:.0f}s)", flush=True)
            continue
        seen_funcs = []
        elapsed = "?"; err = None
        for l in out.splitlines():
            if l.startswith("COMBO:"): combo_id = l.split(":", 2)[1]
            elif l.startswith("PROF:"):
                _, tt, fn = l.split(":", 2); agg[fn][0] += float(tt); agg[fn][1] += 1
            elif l.startswith("ELAPSED:"): elapsed = l.split(":", 1)[1]
            elif l.startswith("ERR:"): err = l.split(":", 1)[1]
        if err is not None:
            fails.append((combo_id, err))
            print(f"  [{idx:2}] FAIL {combo_id}: {err}", flush=True)
        else:
            print(f"  [{idx:2}] OK   {combo_id} ({elapsed}s)", flush=True)
    print(f"\n=== AGG DONE: {len(fails)} failures / {n} combos ===", flush=True)
    print("=== top mlframe non-njit hotspots (aggregate self-time across combos) ===", flush=True)
    for fn, (tot, nc) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {tot:7.3f}s  seen_in={nc:2}  {fn[:62]}", flush=True)
    print("=== FAILURES / TIMEOUTS ===", flush=True)
    for cid, msg in fails:
        print(f"  {cid}: {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=25)
    ap.add_argument("--one", type=int, default=None)
    ap.add_argument("--n-rows", type=int, default=10000)
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()
    if a.one is not None:
        _run_one(a.one, a.n_rows)
    else:
        _run_sweep(a.n, a.n_rows, a.timeout)


if __name__ == "__main__":
    main()
