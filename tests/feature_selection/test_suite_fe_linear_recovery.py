"""End-to-end FS -> FE -> model -> metric guards for LINEAR / RIDGE recovery through
``train_mlframe_models_suite`` (mlframe's MAIN entry point), 2026-06-12.

Bug class these guard
---------------------
MRMR synthesizes the dominant engineered feature (a ratio / product / polynomial /
trig product / cluster aggregate) on a target whose signal rides in that feature's
MAGNITUDE, but ``transform()`` USED to hand the downstream model the feature's 10-level
MI QUANTILE CODE instead of its continuous value. Rank survives, magnitude is destroyed;
a LINEAR model then caps its predictions far below the (heavy) tail and test-R2 collapses
(measured ~0.002 on the user's CASE2, with prediction-span fraction ~0.001). A TREE never
notices -- it splits on a rank code just as well -- so only a LINEAR-downstream, end-to-end
suite test surfaces it. See ``MRMR_FE_TEST_GAP_ANALYSIS.md`` and
``test_biz_value_mrmr_continuous_engineered_linear_r2.py`` (the companion deterministic +
n=100k CASE2 pin -- this file broadens the guard across generators/distributions/models,
it does NOT duplicate that pin).

Two assertions per recovery case, both calibrated by MEASUREMENT (margin below the value):
* ``test-R2 >= floor`` -- the magnitude reached the model well enough to fit.
* ``prediction_span_fraction >= floor`` -- the SHARP signature guard: the bug capped this
  near 0.001 on heavy-tail / large-magnitude targets; a healthy linear fit spans most of
  the target range. Asserted everywhere, the binding guard on the heavy-tail families.

Process isolation
-----------------
Each suite fit runs in a FRESH subprocess (``_suite_fe_worker.py``). Two heavy-tailed
n=100k suite fits in the SAME process intermittently corrupt downstream-pipeline global
state ("feature names should match" at sklearn replay); every fit is green standalone.
The MRMR-endtoend-invariants layer isolates per-case for the identical reason. This is a
process-state artifact, NOT the magnitude-stripping bug under test.

Measured values (seeds 0/1, 2026-06-12) baked into every threshold below as comments.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys

import pytest


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# repo root (parent of ``tests/``) so the worker can ``import tests.feature_selection.*``.
_REPO_ROOT = os.path.dirname(os.path.dirname(_TEST_DIR))


def _run_fit(*, gen: str, dist: str, model: str, use_mrmr: bool, seed: int = 0, timeout: int = 600) -> dict:
    """Run ONE suite fit in a fresh subprocess; return ``{"R2","span","n","structure"}``.

    Retries once on a Windows OOM / paging-file transient (the repo's concurrent-load
    retry policy) before treating the failure as real.
    """
    src_dir = os.environ.get("MRMR_SRC_DIR")
    pyutilz = os.environ.get("MRMR_PYUTILZ_DIR")
    env = dict(os.environ)
    env["NUMBA_DISABLE_CUDA"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MLFRAME_DISABLE_HNSW"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    pp = env.get("PYTHONPATH", "")
    # _REPO_ROOT enables ``import tests.feature_selection.*`` in the worker; src/pyutilz
    # are usually already on the inherited PYTHONPATH but are added defensively.
    extra = [p for p in (_REPO_ROOT, src_dir, pyutilz, _TEST_DIR) if p]
    if extra:
        env["PYTHONPATH"] = os.pathsep.join(extra + ([pp] if pp else []))

    payload = json.dumps(dict(gen=gen, dist=dist, model=model, use_mrmr=use_mrmr, seed=seed))
    worker = os.path.join(_TEST_DIR, "_suite_fe_worker.py")

    last_err = ""
    out = ""
    for attempt in range(2):
        proc = subprocess.run(  # nosec B603 -- fixed local argv (sys.executable/git + literal args), no shell, no untrusted input
            [sys.executable, worker, payload],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        out = proc.stdout or ""
        marker = "===RESULT_JSON==="
        if marker in out:
            for line in reversed(out.split(marker, 1)[1].strip().splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    return json.loads(line)
        last_err = (proc.stderr or "")[-3000:]
        if any(t in last_err.lower() for t in ("paging file", "memoryerror", "cannot allocate", "oom")) and attempt == 0:
            import time as _t

            _t.sleep(90)
            continue
        break
    raise RuntimeError(
        f"worker failed for gen={gen} dist={dist} model={model} mrmr={use_mrmr} seed={seed}\n"
        f"--- stderr tail ---\n{last_err}\n--- stdout tail ---\n{out[-1500:]}"
    )


# ---------------------------------------------------------------------------
# Family 1 -- LINEAR magnitude recovery: model="linear", use_mrmr=True.
#
# (gen, dist, r2_floor, span_floor, slow, measured-comment). Each floor sits clearly
# BELOW the measured value AND clearly ABOVE the bug's signature (R2~0.002, span~0.001).
# bilinear-normal is intentionally EXCLUDED: a product of two normals is not linearly
# recoverable after FE (measured R2~0.14), so it is no magnitude-recovery target.
# ---------------------------------------------------------------------------
_LINEAR_RECOVERY = [
    # ratio_heavytail uniform n=100000 (user's CASE2 shape): measured R2 0.707/0.724,
    # span 0.410/0.452. Floors WEAKENED vs the dedicated n=100k CASE2 pin (R2>=0.99):
    # this generator adds 5% noise + _pos() guards + a hidden f term, so the recoverable
    # ceiling is ~0.71, not ~1.0. Floors still ~100x above the magnitude-stripping
    # signature (R2 0.002 / span 0.001). SLOW (n=100k, ~30s/fit).
    pytest.param("ratio_heavytail", "uniform", 0.60, 0.30, True, id="linear-ratio_heavytail-uniform"),
    # bilinear uniform n=40000: measured R2 0.979/0.983, span 0.970/0.982.
    pytest.param("bilinear", "uniform", 0.90, 0.80, False, id="linear-bilinear-uniform"),
    # poly normal n=40000: measured R2 0.950/0.995, span 1.003/1.014.
    pytest.param("poly", "normal", 0.85, 0.80, False, id="linear-poly-normal"),
    # trig_product uniform n=40000: measured R2 0.972, span 0.676/0.761.
    pytest.param("trig_product", "uniform", 0.85, 0.50, False, id="linear-trig_product-uniform"),
    # cluster_linear normal n=20000: measured R2 0.998, span 0.999/1.006.
    pytest.param("cluster_linear", "normal", 0.95, 0.85, False, id="linear-cluster_linear-normal"),
]


@pytest.mark.parametrize("gen,dist,r2_floor,span_floor,slow", _LINEAR_RECOVERY)
def test_linear_magnitude_recovery(gen, dist, r2_floor, span_floor, slow, request):
    """``train_mlframe_models_suite(model="linear", use_mrmr=True)`` must reach the
    calibrated test-R2 AND prediction-span floors on each linearly-recoverable
    FE-synthesizable target. The span floor is the sharp guard against the capped-
    prediction signature of the quantized-engineered-feature bug."""
    if slow:
        request.node.add_marker(pytest.mark.slow)
    res = _run_fit(gen=gen, dist=dist, model="linear", use_mrmr=True, seed=0)
    assert res["R2"] >= r2_floor, (
        f"linear+MRMR test-R2={res['R2']:.4f} < floor {r2_floor} on {gen}/{dist} "
        f"({res['structure']}); a collapse here means the synthesized feature reached the "
        f"linear model magnitude-stripped (quantized) again, or FE failed to synthesize it."
    )
    assert res["span"] >= span_floor, (
        f"linear+MRMR prediction_span_fraction={res['span']:.4f} < floor {span_floor} on "
        f"{gen}/{dist}; the quantized-output bug capped this near 0.001 -- predictions are "
        f"not spanning the target range, the magnitude-stripping signature."
    )


# ---------------------------------------------------------------------------
# Family 2 -- FE delivers a USABLE signal: linear+MRMR(FE) R2 materially exceeds
# raw-only linear (use_mrmr=False) on a NONLINEAR target a linear model cannot fit raw.
# poly: raw-only linear R2 measured -0.003 (genuinely helpless on a**3-3a+0.5b**2);
# FE-on R2 0.950/0.995 -> gap ~0.95. Margin set far BELOW the gap.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "gen,dist,min_uplift",
    [
        # poly normal: raw R2 ~ -0.003, FE R2 ~0.95 -> gap ~0.95; require >= 0.50.
        pytest.param("poly", "normal", 0.50, id="uplift-poly-normal"),
    ],
)
def test_fe_delivers_usable_signal_uplift(gen, dist, min_uplift):
    """FE must turn an un-linear-fittable target into a linearly-fittable one: linear+MRMR
    test-R2 must exceed raw-only linear test-R2 by a calibrated margin. Pre-fix the
    engineered feature was a rank code and this uplift collapsed."""
    raw = _run_fit(gen=gen, dist=dist, model="linear", use_mrmr=False, seed=0)
    fe = _run_fit(gen=gen, dist=dist, model="linear", use_mrmr=True, seed=0)
    uplift = fe["R2"] - raw["R2"]
    assert uplift >= min_uplift, (
        f"FE uplift (linear+MRMR R2 {fe['R2']:.4f} - raw-only R2 {raw['R2']:.4f} = "
        f"{uplift:.4f}) < required {min_uplift} on {gen}/{dist}. FE is no longer delivering "
        f"a usable continuous signal a linear model can fit -- the magnitude-stripping bug."
    )


# ---------------------------------------------------------------------------
# Family 3 -- RIDGE parity: the magnitude recovery is not linear-only. Same heavy-tail
# target, model="ridge". Measured R2 0.707/0.724, span 0.410/0.452 (matches the linear
# numbers -- ridge with a tiny penalty fits the same continuous magnitude). SLOW.
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize(
    "gen,dist,r2_floor,span_floor",
    [
        pytest.param("ratio_heavytail", "uniform", 0.60, 0.30, id="ridge-ratio_heavytail-uniform"),
    ],
)
def test_ridge_magnitude_recovery(gen, dist, r2_floor, span_floor):
    """The fix is not linear-only: ``model="ridge"`` reaches the same magnitude-recovery
    floors on the heavy-tail target."""
    res = _run_fit(gen=gen, dist=dist, model="ridge", use_mrmr=True, seed=0)
    assert res["R2"] >= r2_floor, (
        f"ridge+MRMR test-R2={res['R2']:.4f} < floor {r2_floor} on {gen}/{dist} "
        f"({res['structure']}); magnitude-stripping would collapse this as it does linear."
    )
    assert res["span"] >= span_floor, (
        f"ridge+MRMR prediction_span_fraction={res['span']:.4f} < floor {span_floor} on "
        f"{gen}/{dist}; the capped-prediction signature of the quantized-output bug."
    )
