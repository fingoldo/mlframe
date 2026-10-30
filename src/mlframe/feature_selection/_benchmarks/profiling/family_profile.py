"""Profile ONE MRMR fit with ONE FE family on (the rest off), CPU or GPU path: cProfile top tottime / cumtime, the family wall, and optionally the same fit's BASE wall for the cost.

python -m mlframe.feature_selection._benchmarks.profiling.family_profile <family> [--n 100000] [--p 100] [--mode cpu|gpu] [--top 30] [--no-base] [--binning mdlp] [--dump-every SECONDS]

``--dump-every`` prints every thread's stack at that interval (where a fit that does not finish is spending its time). The full profile is written to ``<profile dir>/prof_<family>_<mode>_<binning>.pstats``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import time
import warnings

from mlframe.feature_selection._benchmarks.profiling import family_scale as F
from mlframe.feature_selection._benchmarks.profiling._paths import OUT_DIR


def binning_kwargs(binning: str) -> dict:
    """MRMR arguments of one binning setup: ``none_<method>`` is no adaptive strategy with that quantization method, anything else is the ``nbins_strategy`` name."""
    if binning.startswith("none_"):
        return {"nbins_strategy": None, "quantization_method": binning[5:]}
    return {"nbins_strategy": binning}


def _parse() -> argparse.Namespace:
    """Command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("family")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--p", type=int, default=100)
    ap.add_argument("--mode", default="cpu")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--dump-every", type=int, default=0)
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--binning", default="mdlp", help="mdlp | mdlp_validated | optimal_joint | fd | sturges | none_quantile | none_uniform")
    return ap.parse_args()


def _set_mode(mode: str) -> None:
    """Hide the GPU for ``cpu``; for ``gpu`` set the strict-resident switches the GPU benchmarks use (an explicit environment value wins)."""
    if mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return
    for key, value in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
        os.environ.setdefault(key, value)


def _fit(a: argparse.Namespace, data: tuple, on: "str | None", profile: bool) -> tuple:
    """One timed fit of ``data`` with family ``on`` (``None``/``BASE``: every family off; ``DEFAULTS``: the library defaults) after a warm-up fit with the same flags; returns ``(wall, profiler, model)``."""
    from mlframe.feature_selection.filters import _fe_family_timing as FT
    from mlframe.feature_selection.filters.mrmr import MRMR

    kw = {} if on == "DEFAULTS" else {f: False for f in F._flag_names()}
    kw.update(binning_kwargs(a.binning))
    if on and on not in ("BASE", "DEFAULTS"):
        kw[f"fe_{on}_enable"] = True
    MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*F._make(20_000, 30, 1))  # warm JIT / kernels with the same flags
    FT.reset_fe_family_wall()
    pr = cProfile.Profile() if profile else None
    t0 = time.perf_counter()
    if pr:
        pr.enable()
    model = MRMR(verbose=0, random_seed=0, n_jobs=1, **kw).fit(*data)
    if pr:
        pr.disable()
    return time.perf_counter() - t0, pr, model


def main() -> None:
    """Fit once with the family on under cProfile and print the wall, the selected features, the family walls and the top functions."""
    a = _parse()
    if a.dump_every:
        import faulthandler

        faulthandler.dump_traceback_later(a.dump_every, repeat=True)
    warnings.simplefilter("ignore")
    _set_mode(a.mode)
    from mlframe.feature_selection.filters import _fe_family_timing as FT

    data = F._make(a.n, a.p, 0)
    base = None
    if not a.no_base:
        base, _, _ = _fit(a, data, None, False)
        print(f"BASE wall {base:.1f}s", flush=True)
    wall, pr, model = _fit(a, data, a.family, True)
    print(f"{a.family} [{a.mode}/{a.binning}] wall {wall:.1f}s" + (f"  cost over BASE {wall - base:+.1f}s" if base else ""), flush=True)
    print("selected:", [str(s) for s in model.get_feature_names_out()][:12])
    print("family walls:", {k: round(v[0], 2) for k, v in sorted(FT.get_fe_family_wall().items(), key=lambda kv: -kv[1][0])[:8]})
    assert pr is not None
    pr.dump_stats(str(OUT_DIR / f"prof_{a.family}_{a.mode}_{a.binning}.pstats"))
    for key in ("tottime", "cumulative"):
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(a.top)
        print(f"--- top {a.top} by {key}")
        print("\n".join(line[:190] for line in buf.getvalue().splitlines() if line.strip())[:9000])


if __name__ == "__main__":
    main()
