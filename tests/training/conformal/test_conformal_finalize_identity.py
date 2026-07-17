"""CPX30 identity gate: hoisting the alpha-invariant APS argsort+cumsum out of the
alpha loop in conformal_classification_report must be BIT-IDENTICAL to the pre-hoist
code across all alphas (aps + lac scores).

OLD baseline is loaded from `git show HEAD:.../_conformal_finalize.py` so we compare two
real artifacts, not a from-memory rewrite (per CLAUDE.md A/B methodology rule 4).
"""

import importlib.util
import subprocess  # nosec B404 -- test-only local trusted subprocess invocation (fixed argv, no shell, no untrusted input)
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REL = "src/mlframe/training/_conformal_finalize.py"
REPO = Path(__file__).resolve().parents[3]


def _load_old_module() -> types.ModuleType:
    try:
        src = subprocess.check_output(["git", "show", f"HEAD:{REL}"], cwd=REPO, text=True, stderr=subprocess.DEVNULL)  # nosec B603 B607 -- fixed local argv (sys.executable/git + literal args), not a partial/searched path from untrusted input, no shell
    except Exception as e:  # pragma: no cover - env without HEAD
        pytest.skip(f"cannot load OLD baseline via git show: {e}")
    # Mirror package context so its relative imports (.composite.*) resolve.

    spec = importlib.util.spec_from_loader("mlframe.training._conformal_finalize_OLD", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "mlframe.training"
    mod.__file__ = str(REPO / REL)
    sys.modules["mlframe.training._conformal_finalize_OLD"] = mod
    exec(compile(src, str(REPO / REL), "exec"), mod.__dict__)  # nosec B102 -- exec of locally-authored trusted source (repo module text or a literal test string), never untrusted input
    return mod


def _make(n_test=4000, n_cal=1500, k=9, seed=7):
    rng = np.random.default_rng(seed)
    tl = rng.standard_normal((n_test, k))
    cl = rng.standard_normal((n_cal, k))
    tp = np.exp(tl)
    tp /= tp.sum(1, keepdims=True)
    cp = np.exp(cl)
    cp /= cp.sum(1, keepdims=True)
    return dict(
        test_probs=tp,
        test_target=rng.integers(0, k, n_test),
        calib_probs=cp,
        calib_target=rng.integers(0, k, n_cal),
        classes=np.arange(k),
        alphas=[round(0.05 * i, 4) for i in range(1, 12)],
    )


@pytest.mark.parametrize("score", ["aps", "lac"])
def test_cpx30_identity_old_vs_new(score):
    from mlframe.training._conformal_finalize import conformal_classification_report as new_fn

    old_fn = _load_old_module().conformal_classification_report
    data = _make()
    old = old_fn(score=score, **data)
    new = new_fn(score=score, **data)

    assert old["alphas"] == new["alphas"]
    for a in old["per_alpha"]:
        for key in ("nominal_coverage", "achieved_coverage", "mean_set_size"):
            ov, nv = old["per_alpha"][a][key], new["per_alpha"][a][key]
            assert ov == nv, f"score={score} alpha={a} {key}: OLD {ov!r} != NEW {nv!r}"
