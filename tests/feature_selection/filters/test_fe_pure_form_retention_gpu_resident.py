"""GPU-vs-host parity tests for ``_fe_pure_form_retention_gpu_resident.adds_nonlinear_value_batch_gpu_resident``
(mrmr_audit_2026-07-20 test_coverage.md #12). This resident twin of the per-candidate non-separability
gate inside ``retain_usable_pure_forms`` shipped with zero direct tests -- only exercised transitively
through full MRMR fits with ``MLFRAME_FE_GPU_STRICT_RESIDENT=1`` set. Builds a standalone host reference
implementation of the SAME 12-column additive-basis mean-centered-OLS math the module's own docstring
claims equivalence to, and checks selection-equivalent (boolean-identical) verdicts against it."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_pure_form_retention_gpu_resident import (
    adds_nonlinear_value_batch_gpu_resident,
)


def _host_basis(xcol: np.ndarray) -> np.ndarray:
    """The exact 6-function additive single-operand basis (standardized x, x^2, x^3, signed-sqrt,
    signed-log1p, reciprocal-magnitude), matching both the CPU sklearn path and the GPU kernel."""
    xs = (xcol - xcol.mean()) / (xcol.std() + 1e-12)
    return np.column_stack([xs, xs * xs, xs * xs * xs, np.sign(xs) * np.sqrt(np.abs(xs)), np.sign(xs) * np.log1p(np.abs(xs)), 1.0 / (np.abs(xs) + 1.0)])


def _host_verdict(fv: np.ndarray, xa: np.ndarray, xb: np.ndarray, y: np.ndarray, min_resid_frac: float, min_resid_corr: float) -> bool:
    """Standalone host reference: mean-centered OLS of fv on the 12-column additive basis of (xa, xb),
    then the same two-legged gate (non-separability + relevance) the GPU kernel implements."""
    f_std = float(np.std(fv))
    if f_std <= 1e-12:
        return False
    Xr = np.column_stack([_host_basis(xa), _host_basis(xb)])
    Xc = Xr - Xr.mean(axis=0, keepdims=True)
    fc = fv - fv.mean()
    beta, *_ = np.linalg.lstsq(Xc, fc, rcond=None)
    resid = fc - Xc @ beta
    resid_std = float(np.std(resid))
    if resid_std < min_resid_frac * f_std:
        return False
    yc = y - y.mean()
    num = float(np.dot(resid, yc))
    denom = float(np.sqrt(np.dot(resid, resid) * np.dot(yc, yc)))
    corr = abs(num / denom) if denom > 1e-12 else 0.0
    return corr >= min_resid_corr


def _make_pool(n=800, seed=0):
    """Two raw operands + three candidate forms: a genuine non-separable+relevant ratio, a separable
    (additive) form that should fail the non-separability leg, and a non-separable-but-irrelevant form."""
    rng = np.random.default_rng(seed)
    xa = rng.standard_normal(n)
    xb = rng.uniform(0.5, 3.0, n)
    y = xa**2 / xb + 0.05 * rng.standard_normal(n)

    form_genuine = xa**2 / xb  # non-separable in (xa, xb) AND relevant to y
    form_separable = xa + xb  # a plain additive sum -- fully explained by the additive basis
    noise_target = rng.standard_normal(n)
    form_irrelevant = np.sin(xa) * np.cos(xb)  # non-separable but uncorrelated with y

    return dict(
        form_values=[form_genuine, form_separable, form_irrelevant],
        src_pairs=[("a", "b"), ("a", "b"), ("a", "b")],
        base_names=["a", "b"],
        base_columns=[xa, xb],
        rel_y=y,
        xa=xa,
        xb=xb,
        y=y,
        _unused=noise_target,
    )


class TestGpuHostParity:
    """The GPU-resident batched verdicts must exactly match the standalone host reference per candidate."""

    def test_verdicts_match_host_reference_across_seeds(self):
        """Across 5 seeds, the GPU-resident batched verdicts must be boolean-identical to the host reference."""
        for seed in range(5):
            pool = _make_pool(seed=seed)
            gpu_verdicts = adds_nonlinear_value_batch_gpu_resident(
                pool["form_values"], pool["src_pairs"], pool["base_names"], pool["base_columns"],
                pool["rel_y"], min_resid_frac=0.10, min_resid_corr=0.08,
            )
            assert gpu_verdicts is not None, f"seed={seed}: expected a real (non-fallback) GPU verdict list"
            host_verdicts = [_host_verdict(fv, pool["xa"], pool["xb"], pool["y"], min_resid_frac=0.10, min_resid_corr=0.08) for fv in pool["form_values"]]
            assert list(gpu_verdicts) == host_verdicts, f"seed={seed}: GPU {gpu_verdicts} != host {host_verdicts}"

    def test_genuine_nonseparable_relevant_form_is_kept(self):
        """Sanity check on the parity fixture itself: the ratio form must actually be retained."""
        pool = _make_pool(seed=1)
        gpu_verdicts = adds_nonlinear_value_batch_gpu_resident(
            pool["form_values"], pool["src_pairs"], pool["base_names"], pool["base_columns"],
            pool["rel_y"], min_resid_frac=0.10, min_resid_corr=0.08,
        )
        assert gpu_verdicts[0] is True

    def test_separable_form_is_rejected(self):
        """The plain additive sum must fail the non-separability leg."""
        pool = _make_pool(seed=1)
        gpu_verdicts = adds_nonlinear_value_batch_gpu_resident(
            pool["form_values"], pool["src_pairs"], pool["base_names"], pool["base_columns"],
            pool["rel_y"], min_resid_frac=0.10, min_resid_corr=0.08,
        )
        assert gpu_verdicts[1] is False


class TestDegenerateInputsReturnNone:
    """Empty pool / missing operand / too-few rows must degrade to ``None`` (caller falls back to CPU)."""

    def test_empty_pool_returns_none(self):
        """An empty candidate pool hits the explicit `not form_values` early guard."""
        assert adds_nonlinear_value_batch_gpu_resident([], [], [], [], np.array([1.0, 2.0]), min_resid_frac=0.1, min_resid_corr=0.08) is None

    def test_missing_operand_name_returns_none(self):
        """A src_pairs operand absent from base_names falls back to the exact CPU path."""
        pool = _make_pool(seed=2)
        result = adds_nonlinear_value_batch_gpu_resident(
            pool["form_values"], [("a", "nonexistent")], pool["base_names"], pool["base_columns"],
            pool["rel_y"], min_resid_frac=0.10, min_resid_corr=0.08,
        )
        assert result is None

    def test_too_few_rows_returns_none(self):
        """n < 2 rows returns None rather than proceeding into a degenerate OLS solve."""
        result = adds_nonlinear_value_batch_gpu_resident(
            [np.array([1.0])], [("a", "b")], ["a", "b"], [np.array([1.0]), np.array([2.0])],
            np.array([1.0]), min_resid_frac=0.1, min_resid_corr=0.08,
        )
        assert result is None
