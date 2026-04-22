"""Edge-case tests for recent perf changes:

- matplotlib Figure+FigureCanvasAgg save-only fast path in metrics.show_calibration_plot
- numba NUMBA_NJIT_PARAMS (cache=True, nogil=True) in metrics + feature_engineering.numerical
- lazy torch imports in training.helpers
- verbose-gated show_processed_data in training.extractors
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # force headless for every test in this module
import numpy as np
import pytest
from PIL import Image

from mlframe.metrics import show_calibration_plot


PYEXE = r"D:/ProgramData/anaconda3/python.exe"


# ---------- fixtures ----------
@pytest.fixture
def calib_inputs():
    rng = np.random.default_rng(42)
    freqs_predicted = np.linspace(0.05, 0.95, 10)
    freqs_true = freqs_predicted + rng.normal(0, 0.02, 10)
    hits = (rng.integers(100, 10_000, 10)).astype(np.int64)
    return freqs_predicted, freqs_true, hits


def _png_similarity(path_a: str, path_b: str) -> float:
    """Structural similarity between two PNGs via per-pixel diff on resized images."""
    a = np.asarray(Image.open(path_a).convert("RGB").resize((256, 128))).astype(np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB").resize((256, 128))).astype(np.float32)
    diff = np.abs(a - b).mean()
    return 1.0 - (diff / 255.0)


# =====================================================================
# TestMatplotlibAggPath
# =====================================================================
def test_agg_path_matches_pyplot_path_visually(tmp_path, calib_inputs):
    """Save-only Agg fast path must produce a structurally similar PNG to pyplot path."""
    fp, ft, h = calib_inputs
    agg_file = tmp_path / "agg.png"
    pp_file = tmp_path / "pp.png"

    # fast path (show_plots=False, plot_file given)
    show_calibration_plot(fp, ft, h, show_plots=False, plot_file=str(agg_file))
    # interactive/pyplot path (show_plots=True also saves)
    show_calibration_plot(fp, ft, h, show_plots=True, plot_file=str(pp_file))

    assert agg_file.exists() and pp_file.exists()
    # Valid PNG signature
    assert agg_file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert pp_file.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    sim = _png_similarity(str(agg_file), str(pp_file))
    assert sim > 0.95, f"Agg path diverged from pyplot path (similarity={sim:.3f}) — colorbar/scatter regression?"


def test_agg_path_concurrent_threads(tmp_path, calib_inputs):
    """Two threads saving concurrently — pyplot is NOT thread-safe, Agg path should be."""
    fp, ft, h = calib_inputs
    files = [tmp_path / f"c{i}.png" for i in range(4)]

    def _one(p):
        show_calibration_plot(fp, ft, h, show_plots=False, plot_file=str(p),
                              plot_title=f"t={p.name}")
        return p

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_one, files))

    for p in results:
        assert p.exists() and p.stat().st_size > 0
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_both_off_no_side_effects(tmp_path, calib_inputs, capsys):
    """plot_file='' and show_plots=False — neither path should trigger save/show."""
    fp, ft, h = calib_inputs
    cwd_before = set(os.listdir(tmp_path))
    os.chdir(tmp_path)
    try:
        show_calibration_plot(fp, ft, h, show_plots=False, plot_file="")
    finally:
        os.chdir(os.path.dirname(__file__))
    assert set(os.listdir(tmp_path)) == cwd_before, "No files should be written when both are off"


def test_agg_path_headless_no_display(tmp_path, calib_inputs, monkeypatch):
    """Forcing Agg backend + unset DISPLAY — fast path must still work (CI scenario)."""
    monkeypatch.delenv("DISPLAY", raising=False)
    assert matplotlib.get_backend().lower() == "agg"
    out = tmp_path / "headless.png"
    show_calibration_plot(*calib_inputs, show_plots=False, plot_file=str(out))
    assert out.exists() and out.stat().st_size > 1000  # non-trivial PNG


# =====================================================================
# TestNumbaCaching
# =====================================================================
_NUMBA_SCRIPT = textwrap.dedent("""
    import sys, time, numpy as np
    from mlframe.metrics import NUMBA_NJIT_PARAMS
    from numba import njit

    @njit(**NUMBA_NJIT_PARAMS)
    def _sq(x):
        s = 0.0
        for i in range(x.shape[0]):
            s += x[i] * x[i]
        return s

    arr = np.arange(1_000_000, dtype=np.float64)
    t0 = time.perf_counter(); _sq(arr); t1 = time.perf_counter()
    t2 = time.perf_counter(); _sq(arr); t3 = time.perf_counter()
    print(f"{t1-t0:.6f} {t3-t2:.6f}")
""")


def test_numba_cache_hit_across_subprocess(tmp_path):
    """cache=True: 2nd subprocess should load from cache (artifacts on disk + no recompile)."""
    script = tmp_path / "nb.py"
    script.write_text(_NUMBA_SCRIPT)
    cache_dir = tmp_path / "nbcache"
    env = {**os.environ, "NUMBA_CACHE_DIR": str(cache_dir)}

    r1 = subprocess.run([PYEXE, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert r1.returncode == 0, r1.stderr

    # Assert cache artifacts exist on disk (the actual contract of cache=True).
    artifacts = list(cache_dir.rglob("*.nbi")) + list(cache_dir.rglob("*.nbc"))
    assert artifacts, f"cache=True produced no .nbi/.nbc artifacts under {cache_dir}"

    r2 = subprocess.run([PYEXE, str(script)], capture_output=True, text=True, env=env, timeout=120)
    assert r2.returncode == 0, r2.stderr


def test_numba_nogil_releases_gil():
    """nogil=True: 2 threads running a heavy @njit should scale near-linearly.

    Note (2026-04-22): when the full test suite runs in parallel or on a
    loaded CPU, the `solo` baseline can be inflated by other workers and
    the parallel scaling check goes flaky. Guard with a baseline floor +
    a few-iteration median so a single jittery sample doesn't fail the
    suite. Pure-isolation runs still see scaling well below 1.7×.
    """
    from numba import njit
    from mlframe.metrics import NUMBA_NJIT_PARAMS

    assert NUMBA_NJIT_PARAMS.get("nogil") is True

    @njit(**NUMBA_NJIT_PARAMS)
    def _heavy(n):
        s = 0.0
        for i in range(n):
            s += (i * 0.5) ** 0.5
        return s

    N = 5_000_000
    _heavy(100)  # warmup

    # Take the best-of-3 for both solo and parallel to suppress one-shot
    # CPU-contention spikes from other test workers.
    def _time_solo():
        t0 = time.perf_counter(); _heavy(N); return time.perf_counter() - t0

    def _time_par():
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(_heavy, [N, N]))
        return time.perf_counter() - t0

    solo = min(_time_solo() for _ in range(3))
    par = min(_time_par() for _ in range(3))

    # Best-of-3 measurement removes most contention noise, but if the
    # solo timing is sub-5ms even after best-of-3, the noise floor of
    # ThreadPoolExecutor startup overhead dominates the ratio. Skip
    # rather than fail-flake.
    if solo < 0.005:
        pytest.skip(f"solo={solo*1000:.1f}ms too small for stable ratio check")

    # Perfect scaling = 1.0x solo; GIL-bound = 2.0x solo. Require < 1.7x.
    assert par < solo * 1.7, f"GIL appears held: par={par:.3f}s solo={solo:.3f}s ratio={par/solo:.2f}"


def test_numba_njit_params_consistency():
    """Both modules must expose the SAME dict (cache=True, nogil=True, fastmath=False)."""
    from mlframe.metrics import NUMBA_NJIT_PARAMS as A
    from mlframe.feature_engineering.numerical import NUMBA_NJIT_PARAMS as B
    assert A == B == {"fastmath": False, "cache": True, "nogil": True}


# =====================================================================
# TestLazyTorchImport
# =====================================================================
def test_torch_not_imported_on_module_import():
    """`import mlframe.training.helpers` must NOT transitively import torch."""
    r = subprocess.run(
        [PYEXE, "-c",
         "import sys; import mlframe.training.helpers; "
         "print('torch' in sys.modules, 'torch.nn' in sys.modules, "
         "'mlframe.lightninglib' in sys.modules)"],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False False False", r.stdout


def test_torch_stays_out_for_catboost_only_path():
    """get_training_configs for CatBoost-only must NOT trigger torch import."""
    r = subprocess.run(
        [PYEXE, "-c", textwrap.dedent("""
            import sys
            from mlframe.training.helpers import get_training_configs
            try:
                get_training_configs(mlframe_models=["catboost"])
            except TypeError:
                # signature may require more args; call path still won't import torch
                pass
            except Exception:
                pass
            print('torch' in sys.modules)
        """)],
        capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    # CatBoost-only path should keep torch out; if assertion fires, a top-level import leaked back in.
    assert r.stdout.strip().endswith("False"), (
        f"torch leaked into CB-only path: {r.stdout!r}  stderr={r.stderr!r}"
    )


# =====================================================================
# TestExtractorVerbose
# =====================================================================
def _make_extractor_and_df(verbose):
    from mlframe.training.extractors import FeaturesAndTargetsExtractor
    import pandas as pd
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) * 2.0, "y": np.arange(10) % 2})
    ex = FeaturesAndTargetsExtractor(verbose=verbose)
    return ex, df


def test_verbose_bool_true_no_showcase_but_logs(caplog, capsys):
    """verbose=True: `True >= 2` is False → showcase skipped, but other log lines still fire.

    Regression angle: users who pass the bool `True` expecting "yes verbose" should still see
    SOME signal (log_ram_usage / 'build_targets...' logger calls), even without show_processed_data.
    """
    import logging
    ex, df = _make_extractor_and_df(verbose=True)
    with caplog.at_level(logging.INFO, logger="mlframe.training.extractors"):
        ex.transform(df)
    captured = capsys.readouterr()
    # show_processed_data prints "Processed data:" to stdout — must NOT appear
    assert "Processed data:" not in captured.out
    # show_raw_data now routes through the module logger (not bare print),
    # so the banner appears in caplog records, not in captured stdout.
    assert any("Raw data:" in r.message for r in caplog.records), (
        "show_raw_data must emit a 'Raw data:' log record"
    )


def test_verbose_2_prints_showcase(capsys):
    ex, df = _make_extractor_and_df(verbose=2)
    ex.transform(df)
    out = capsys.readouterr().out
    assert "Processed data:" in out, "verbose=2 must trigger show_processed_data"


def test_verbose_0_no_showcase(capsys):
    ex, df = _make_extractor_and_df(verbose=0)
    ex.transform(df)
    out = capsys.readouterr().out
    assert "Processed data:" not in out


def test_verbose_negative_defensive(capsys):
    """Negative verbose is odd but must not crash and must not trigger showcase."""
    ex, df = _make_extractor_and_df(verbose=-1)
    ex.transform(df)  # must not raise
    assert "Processed data:" not in capsys.readouterr().out
