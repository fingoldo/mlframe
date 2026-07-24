"""Unit coverage for ``_fe_pure_form_retention_gpu_resident.py``'s
``adds_nonlinear_value_batch_gpu_resident``.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this module (the resident batched twin of the
per-candidate ``_adds_nonlinear_value`` pure-form retention gate) had zero test references anywhere in
the suite. Pins the no-cupy / degenerate-input fallback contract fully offline, plus selection-equivalence
against a from-scratch sklearn CPU reference on a real CUDA device when available.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters._fe_pure_form_retention_gpu_resident import adds_nonlinear_value_batch_gpu_resident


def _need_cuda() -> bool:
    """Whether a usable CUDA device is present this process."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


def _single_operand_basis(x):
    """The 6-function additive single-operand basis this module's docstring pins: standardized x, x^2,
    x^3, sign*sqrt|x|, sign*log1p|x|, 1/(|x|+1)."""
    xs = (x - x.mean()) / (x.std() + 1e-12)
    return [xs, xs**2, xs**3, np.sign(xs) * np.sqrt(np.abs(xs)), np.sign(xs) * np.log1p(np.abs(xs)), 1.0 / (np.abs(xs) + 1.0)]


def _cpu_reference_gate(form_vals, xa, xb, rel_y, min_resid_frac, min_resid_corr):
    """From-scratch CPU reference mirroring ``_fe_pure_form_retention._adds_nonlinear_value`` exactly."""
    fv = np.asarray(form_vals, dtype=np.float64)
    f_std = float(np.std(fv))
    if f_std <= 1e-12:
        return False
    Xr = np.column_stack(_single_operand_basis(xa) + _single_operand_basis(xb))
    Xr_scaled = StandardScaler().fit_transform(Xr)
    lr = LinearRegression().fit(Xr_scaled, fv)
    resid = fv - lr.predict(Xr_scaled)
    if float(np.std(resid)) < min_resid_frac * f_std:
        return False
    u = resid - resid.mean()
    v = np.asarray(rel_y, dtype=np.float64) - np.asarray(rel_y, dtype=np.float64).mean()
    su, sv = float(np.std(resid)), float(np.std(rel_y))
    if su < 1e-12 or sv < 1e-12:
        return False
    ssu, ssv = float((u * u).sum()), float((v * v).sum())
    if ssu <= 0.0 or ssv <= 0.0:
        return False
    corr = abs(float((u * v).sum()) / float(np.sqrt(ssu * ssv)))
    return corr >= min_resid_corr


def test_returns_none_for_empty_pool():
    """A zero-candidate pool is a well-defined degenerate case: ``None`` (caller falls through to the
    exact per-candidate CPU path, which is a no-op loop anyway)."""
    rng = np.random.default_rng(0)
    assert adds_nonlinear_value_batch_gpu_resident([], [], [], [], rng.normal(size=100), min_resid_frac=0.10, min_resid_corr=0.08) is None


def test_returns_none_when_cupy_unavailable(monkeypatch):
    """When ``import cupy`` fails, the whole batch must fall back to ``None`` (exact per-candidate CPU
    path), not raise."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        """Raise on ``import cupy``, delegate every other import to the real one."""
        if name == "cupy":
            raise ImportError("no cupy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    rng = np.random.default_rng(1)
    n = 200
    form_values = [rng.normal(size=n)]
    result = adds_nonlinear_value_batch_gpu_resident(
        form_values, [("a", "b")], ["a", "b"], [rng.normal(size=n), rng.normal(size=n)], rng.normal(size=n),
        min_resid_frac=0.10, min_resid_corr=0.08,
    )
    assert result is None


def test_returns_none_when_operand_missing_from_base_set():
    """A candidate whose source-pair name isn't in ``base_names`` is a shape/logic mismatch -- the whole
    batch must fall back to ``None`` (exact CPU path), rather than silently skip just that candidate."""
    pytest.importorskip("cupy")
    if not _need_cuda():
        pytest.skip("no CUDA")
    rng = np.random.default_rng(2)
    n = 200
    form_values = [rng.normal(size=n)]
    result = adds_nonlinear_value_batch_gpu_resident(
        form_values, [("a", "missing_operand")], ["a", "b"], [rng.normal(size=n), rng.normal(size=n)], rng.normal(size=n),
        min_resid_frac=0.10, min_resid_corr=0.08,
    )
    assert result is None


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_batch_matches_cpu_reference_gate_on_mixed_candidates():
    """The resident batched gate must return the SAME boolean verdict as the from-scratch sklearn CPU
    reference for every candidate in a pool mixing a genuinely non-separable+relevant joint form, a
    purely-additive (separable) form, and a non-separable-but-target-irrelevant form."""
    pytest.importorskip("cupy")
    rng = np.random.default_rng(3)
    n = 4000
    xa = rng.normal(size=n)
    xb = rng.normal(size=n)
    rel_y = 0.6 * xa + 0.4 * xb + 0.05 * rng.normal(size=n)

    # Candidate 0: genuinely non-separable (a*b product) AND relevant to y (since y correlates with a,b).
    joint_relevant = xa * xb
    # Candidate 1: purely additive/separable in its own basis -- the additive-basis OLS should nearly
    # fully explain it, leaving a tiny residual (non-separable gate rejects).
    separable = 3.0 * xa + 2.0 * xb**2
    # Candidate 2: non-separable (product) but built from operands uncorrelated with rel_y.
    xc = rng.normal(size=n)
    xd = rng.normal(size=n)
    joint_irrelevant = xc * xd

    form_values = [joint_relevant, separable, joint_irrelevant]
    src_pairs = [("a", "b"), ("a", "b"), ("c", "d")]
    base_names = ["a", "b", "c", "d"]
    base_columns = [xa, xb, xc, xd]
    min_resid_frac, min_resid_corr = 0.10, 0.08

    got = adds_nonlinear_value_batch_gpu_resident(
        form_values, src_pairs, base_names, base_columns, rel_y,
        min_resid_frac=min_resid_frac, min_resid_corr=min_resid_corr,
    )
    assert got is not None
    assert len(got) == 3

    want = [
        _cpu_reference_gate(joint_relevant, xa, xb, rel_y, min_resid_frac, min_resid_corr),
        _cpu_reference_gate(separable, xa, xb, rel_y, min_resid_frac, min_resid_corr),
        _cpu_reference_gate(joint_irrelevant, xc, xd, rel_y, min_resid_frac, min_resid_corr),
    ]
    assert got == want
