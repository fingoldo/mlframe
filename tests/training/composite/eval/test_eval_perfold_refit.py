"""M2: per-fold transform refit closes the in-fold transform-leakage gap.

The leak (finding M2, ``discovery/_eval.py`` global fit at line ~71 +
``discovery/_screening_tiny._tiny_cv_rmse_y_scale``): a candidate transform's
``fitted_params`` (e.g. ``linear_residual``'s alpha/beta, or a flexible
spline/poly residual's coefficients) are fit ONCE on every valid TRAIN row,
then those GLOBAL params are reused to compute ``T = forward(y, base, params)``
for every inner tiny-CV fold -- including each fold's held-out validation rows.
Because the global fit already absorbed the structure of the rows that later
become a fold's validation set, the held-out RMSE the tiny-CV reports is
OPTIMISTIC: the params peeked at the very rows they are scored against. This
optimism is largest for FLEXIBLE transforms (high-degree poly / spline
residuals) on smaller samples, where the global fit can absorb noise.

The cure -- and the ``_eval``-side contract this file pins -- is
:func:`refit_transform_on_fold`: re-fit the transform params on each fold's
TRAIN rows ONLY, then ``forward``/``inverse`` the held-out fold with those
fold-local params. A held-out fold scored that way is honest by construction
(its params never saw it).

These tests:

* ``test_biz_val_perfold_refit_removes_global_fit_optimism`` -- the biz_value
  test. On a synthetic where a flexible residual transform overfits the global
  sample, the honest per-fold-refit held-out RMSE is materially WORSE (larger)
  than the optimistic global-fit number, i.e. the leak inflated the score and
  the refit removes that inflation. A regression that silently drops the
  per-fold refit (reverting to global params) collapses the gap and fails.
* unit tests pinning the ``refit_transform_on_fold`` contract: no-leakage by
  construction (only fold rows enter the fit), degenerate-fold ``None``
  fallback, fitted-domain refinement, and groups threading.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mlframe.training.composite.discovery._eval import refit_transform_on_fold
from mlframe.training.composite.transforms import Transform


# ----------------------------------------------------------------------
# A flexible polynomial-residual transform whose global fit overfits a
# small noisy sample. ``T = y - poly_d(base)`` where ``poly_d`` is the
# degree-``d`` least-squares fit of y on base. With ``d`` high relative to
# n, the global fit chases noise -> exactly the regime where reusing the
# global params on held-out fold rows is optimistic.
# ----------------------------------------------------------------------
_POLY_DEGREE = 13


def _poly_fit(y: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    n = y.shape[0]
    deg = min(_POLY_DEGREE, max(1, n - 1))
    # Fit on the train rows handed in. polyfit is least-squares.
    coef = np.polyfit(base, y, deg)
    return {"coef": coef.tolist()}


def _poly_forward(y: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    coef = np.asarray(params["coef"], dtype=np.float64)
    g = np.polyval(coef, np.asarray(base, dtype=np.float64))
    return np.asarray(y, dtype=np.float64) - g


def _poly_inverse(t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    coef = np.asarray(params["coef"], dtype=np.float64)
    g = np.polyval(coef, np.asarray(base, dtype=np.float64))
    return np.asarray(t_hat, dtype=np.float64) + g


def _poly_domain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64)
    ok = np.isfinite(base)
    if y is None:
        return ok
    return ok & np.isfinite(np.asarray(y, dtype=np.float64))


def _make_poly_transform() -> Transform:
    return Transform(
        name="poly_residual_testonly",
        forward=_poly_forward,
        inverse=_poly_inverse,
        fit=_poly_fit,
        domain_check=_poly_domain,
        description="test-only flexible polynomial residual",
    )


def _make_data(n: int = 220, seed: int = 7):
    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(-2.5, 2.5, size=n))
    # True mean is a gentle cubic; the residual T = y - true(base) is pure
    # noise. A flexible degree-9 GLOBAL poly fit will absorb part of that noise
    # (overfit), which is what the held-out fold then unfairly benefits from.
    true_g = 0.4 * base**3 - 1.1 * base
    noise = rng.normal(0.0, 1.0, size=n)
    y = true_g + noise
    return y.astype(np.float64), base.astype(np.float64)


def _kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return [perm[i::k] for i in range(k)]


def _cv_heldout_rmse(
    y: np.ndarray,
    base: np.ndarray,
    transform: Transform,
    *,
    k: int,
    seed: int,
    per_fold_refit: bool,
) -> float:
    """Held-out CV RMSE mirroring the tiny-CV structure.

    Downstream "model" is the simplest honest baseline: predict the TRAIN-fold
    mean of T, then invert with the params actually used for that fold. This
    isolates the transform-leakage effect (no tree noise). With
    ``per_fold_refit=False`` the params are the GLOBAL fit (leaky); with
    ``True`` the params are refit on each fold's TRAIN rows via
    ``refit_transform_on_fold`` (honest).
    """
    n = y.shape[0]
    folds = _kfold_indices(n, k, seed)
    global_params = transform.fit(y, base)
    sq_err: list[float] = []
    for vi in range(k):
        val_idx = folds[vi]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != vi])
        if per_fold_refit:
            refit = refit_transform_on_fold(
                transform,
                y[train_idx],
                base[train_idx],
            )
            params = global_params if refit is None else refit[0]
        else:
            params = global_params
        # T on the train fold uses the same params we score with.
        t_train = transform.forward(y[train_idx], base[train_idx], params)
        t_pred_const = float(np.mean(t_train))
        t_hat_val = np.full(val_idx.shape[0], t_pred_const)
        y_hat_val = transform.inverse(t_hat_val, base[val_idx], params)
        diff = y_hat_val - y[val_idx]
        finite = np.isfinite(diff)
        if finite.any():
            sq_err.extend((diff[finite] ** 2).tolist())
    if not sq_err:
        return float("nan")
    return float(np.sqrt(np.mean(sq_err)))


# ----------------------------------------------------------------------
# biz_value
# ----------------------------------------------------------------------
def test_biz_val_perfold_refit_removes_global_fit_optimism():
    """Global-fit reuse reports an OPTIMISTIC held-out RMSE; per-fold refit is
    honest and strictly worse.

    Measured (8-seed sweep): mean honest RMSE 1.140 vs optimistic 0.973 =
    ratio 1.171 on the degree-13 poly residual + cubic+noise synthetic (the
    flexible global fit absorbs noise -> the leaky held-out score is flattered).
    Floor 1.10 (~6% below the measured 1.171) so seed noise does not trip it; a
    regression that drops the per-fold refit collapses the gap (ratio -> 1.0)
    and fails.
    """
    transform = _make_poly_transform()
    k = 5
    opt_rmses: list[float] = []
    hon_rmses: list[float] = []
    for seed in range(8):
        y, base = _make_data(n=160, seed=seed)
        opt = _cv_heldout_rmse(
            y,
            base,
            transform,
            k=k,
            seed=100 + seed,
            per_fold_refit=False,
        )
        hon = _cv_heldout_rmse(
            y,
            base,
            transform,
            k=k,
            seed=100 + seed,
            per_fold_refit=True,
        )
        assert np.isfinite(opt) and np.isfinite(hon)
        opt_rmses.append(opt)
        hon_rmses.append(hon)
    mean_opt = float(np.mean(opt_rmses))
    mean_hon = float(np.mean(hon_rmses))
    # The honest (per-fold-refit) RMSE must be MATERIALLY larger than the
    # leaky global-fit RMSE: the global fit's optimism is real and the refit
    # removes it. Floor 10% gap; measured gap is ~17%.
    assert mean_hon >= mean_opt * 1.10, (
        f"per-fold refit should expose global-fit optimism: "
        f"honest mean RMSE {mean_hon:.4f} vs optimistic {mean_opt:.4f} "
        f"(ratio {mean_hon / mean_opt:.3f}); a ratio ~1.0 means the per-fold "
        f"refit was silently dropped (leak reinstated)."
    )
    # And per seed the honest number is never BETTER than the leaky one by
    # more than tiny noise -- the leak can only inflate (lower) the score.
    for o, h in zip(opt_rmses, hon_rmses):
        assert h >= o * 0.98, (
            f"honest RMSE {h:.4f} cannot be meaningfully better than the leaky global-fit RMSE {o:.4f}; the leak only ever flatters the score."
        )


# ----------------------------------------------------------------------
# unit: refit_transform_on_fold contract
# ----------------------------------------------------------------------
def test_refit_only_sees_fold_rows():
    """No leakage by construction: the params returned for a fold are computed
    ONLY from the rows handed in (identical to fitting the transform directly on
    those rows), never from the global sample."""
    transform = _make_poly_transform()
    y, base = _make_data(n=180, seed=3)
    folds = _kfold_indices(180, 5, seed=11)
    train_idx = np.concatenate([folds[j] for j in range(1, 5)])
    out = refit_transform_on_fold(transform, y[train_idx], base[train_idx])
    assert out is not None
    fold_params, valid = out
    direct = transform.fit(y[train_idx], base[train_idx])
    np.testing.assert_allclose(
        np.asarray(fold_params["coef"]),
        np.asarray(direct["coef"]),
    )
    # Differs from the global fit (proof the fold did NOT use all rows).
    global_params = transform.fit(y, base)
    assert not np.allclose(
        np.asarray(fold_params["coef"]),
        np.asarray(global_params["coef"]),
    )
    assert valid.shape == (train_idx.shape[0],)
    assert bool(valid.all())


def test_refit_returns_none_on_too_few_rows():
    """A fold with fewer than ``min_valid_rows`` valid rows returns None so the
    caller keeps the global params (bit-stable fallback, never a crash)."""
    transform = _make_poly_transform()
    y = np.array([1.0])
    base = np.array([0.5])
    assert refit_transform_on_fold(transform, y, base, min_valid_rows=2) is None


def test_refit_drops_invalid_domain_rows():
    """Pre-fit domain filter drops non-finite rows; the returned mask marks
    exactly the surviving rows and the fit uses only those."""
    transform = _make_poly_transform()
    y = np.array([0.0, 1.0, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    base = np.array([0.1, 0.2, 0.3, np.inf, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    out = refit_transform_on_fold(transform, y, base)
    assert out is not None
    _params, valid = out
    expected = np.isfinite(y) & np.isfinite(base)
    np.testing.assert_array_equal(valid, expected)
    assert int(valid.sum()) == 8


def test_refit_returns_none_on_degenerate_params():
    """A fit that flags ``is_degenerate`` yields None so the caller does not
    score the held-out fold on a near-identity refit."""

    def _degen_fit(y, base):
        return {"alpha": 0.0, "beta": 0.0, "is_degenerate": True}

    transform = Transform(
        name="degen_testonly",
        forward=lambda y, b, p: np.asarray(y, dtype=np.float64),
        inverse=lambda t, b, p: np.asarray(t, dtype=np.float64),
        fit=_degen_fit,
        domain_check=lambda y, b: np.isfinite(np.asarray(b, dtype=np.float64)) & (np.isfinite(np.asarray(y, dtype=np.float64)) if y is not None else True),
        description="test-only degenerate-flag transform",
    )
    y, base = _make_data(n=60, seed=1)
    assert refit_transform_on_fold(transform, y, base) is None


def test_refit_fitted_domain_refinement_narrows_mask():
    """A ``domain_check_fitted`` hook that drops rows once params exist narrows
    the returned mask (T15 parity)."""

    def _fit(y, base):
        # Offset chosen so that base + offset <= 0 for the most-negative
        # bases -> the fitted-domain hook (base + offset > 0) drops them,
        # exercising the T15 narrowing path. (A params-free domain_check
        # cannot see this offset, so it lets those rows through.)
        return {"offset": 0.8}

    def _fwd(y, base, p):
        return np.asarray(y, dtype=np.float64) - np.log(
            np.asarray(base, dtype=np.float64) + p["offset"],
        )

    def _inv(t, base, p):
        return np.asarray(t, dtype=np.float64) + np.log(
            np.asarray(base, dtype=np.float64) + p["offset"],
        )

    def _domain(y, base):
        b = np.asarray(base, dtype=np.float64)
        ok = np.isfinite(b)
        if y is None:
            return ok
        return ok & np.isfinite(np.asarray(y, dtype=np.float64))

    def _domain_fitted(y, base, p):
        # Only rows with base + offset > 0 are in the true (log) domain.
        return np.asarray(base, dtype=np.float64) + p["offset"] > 0.0

    transform = Transform(
        name="logshift_testonly",
        forward=_fwd,
        inverse=_inv,
        fit=_fit,
        domain_check=_domain,
        domain_check_fitted=_domain_fitted,
        description="test-only log-shift transform with fitted domain",
    )
    base = np.array([-1.0, -0.6, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    y = np.linspace(0.0, 1.0, base.shape[0])
    out = refit_transform_on_fold(transform, y, base)
    assert out is not None
    params, valid = out
    # The fitted-domain hook keeps only rows with base + offset > 0; with the
    # fitted offset=0.8 that drops base=-1.0 only. Compute the expected mask
    # from the actual fitted offset.
    offset = float(params["offset"])
    expected = np.isfinite(base) & np.isfinite(y) & ((base + offset) > 0.0)
    np.testing.assert_array_equal(valid, expected)
    # The fitted-domain refinement must have actually narrowed something
    # relative to the params-free pre-fit mask, exercising the T15 path.
    assert int(valid.sum()) < int((np.isfinite(base) & np.isfinite(y)).sum())


def test_refit_threads_groups_when_fit_accepts():
    """``groups_fold`` is passed to a grouped fit only when the fit signature
    accepts it, and sliced to the surviving rows."""
    seen: dict[str, Any] = {}

    def _grouped_fit(y, base, groups=None):
        seen["groups"] = None if groups is None else np.asarray(groups).copy()
        return {"alpha": 0.0, "beta": float(np.mean(y))}

    transform = Transform(
        name="grouped_testonly",
        forward=lambda y, b, p, groups=None: np.asarray(y, dtype=np.float64),
        inverse=lambda t, b, p, groups=None: np.asarray(t, dtype=np.float64),
        fit=_grouped_fit,
        domain_check=lambda y, b: np.isfinite(np.asarray(b, dtype=np.float64)) & (np.isfinite(np.asarray(y, dtype=np.float64)) if y is not None else True),
        description="test-only grouped transform",
        requires_groups=True,
    )
    y = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    base = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    groups = np.array([0, 0, 1, 1, 1])
    out = refit_transform_on_fold(transform, y, base, groups_fold=groups)
    assert out is not None
    # Row 2 (y NaN) is dropped pre-fit; groups must be sliced to match.
    np.testing.assert_array_equal(seen["groups"], np.array([0, 0, 1, 1]))


def test_refit_groups_ignored_when_fit_rejects():
    """A 2-arg fit (no ``groups``) never receives the kwarg even if groups are
    supplied -- no TypeError, no leakage."""
    transform = _make_poly_transform()  # fit(y, base) -- no groups param
    y, base = _make_data(n=60, seed=2)
    groups = np.zeros(60, dtype=int)
    out = refit_transform_on_fold(transform, y, base, groups_fold=groups)
    assert out is not None  # did not raise on the unexpected kwarg
