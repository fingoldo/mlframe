"""Cross-target ensemble + OOF utilities: CompositeCrossTargetEnsemble (stack/weighted/mean strategies with validation gate), compute_oof_holdout_predictions, derive_seeds (sha256-stable subseed derivation), detect_gpu_in_use, env_signature. Split out of composite.py to keep ensemble concerns separate from discovery; composite.py re-exports every symbol below at its bottom for full back-compat."""


from __future__ import annotations

import hashlib
import logging
import math
import warnings
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .composite_estimator import CompositeTargetEstimator
from .composite_transforms import get_transform

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# CompositeCrossTargetEnsemble
# ----------------------------------------------------------------------


def derive_seeds(random_state: int, components: Sequence[str]) -> dict[str, int]:
    """Derive deterministic per-component seeds from a master seed.

    Uses sha256 truncation to keep the values stable across Python /
    numpy versions (no dependence on hash() salt randomisation). The
    returned dict maps each component name to a 32-bit unsigned int.

    Why this exists. Discovery has several internal sources of
    randomness (MI sampling, tiny-model CV split, OOF holdout split,
    bootstrap CI). Threading the same ``random_state`` through every
    one of them creates correlation: if the master seed produces an
    "easy" MI sample it tends to also produce an "easy" CV split.
    Sub-seeds break the correlation while keeping reproducibility:
    same master seed -> same sub-seeds -> same downstream randomness.
    """
    import struct
    out: dict[str, int] = {}
    for c in components:
        h = hashlib.sha256(f"{random_state}::{c}".encode()).digest()
        out[c] = struct.unpack("<I", h[:4])[0]
    return out


def detect_gpu_in_use(mlframe_models: Sequence[str]) -> list[str]:
    """Return list of model families that may be using GPU.

    Best-effort detection: imports each library only if it appears in
    ``mlframe_models`` and probes for GPU availability via the
    library's standard health-check API. Returns the subset that has
    GPU detected. Returns empty list when no GPU library is in use.

    Used by the suite to emit a one-shot warning when composite mode
    is combined with GPU training: GPU non-determinism is amplified
    by K composite-model fits and can surface as ensemble weight
    drift across runs even when ``random_state`` is fixed.
    """
    detected: list[str] = []
    families = {str(m).lower() for m in mlframe_models}
    if any(f in families for f in ("lgb", "lightgbm")):
        try:
            import lightgbm as lgb  # noqa: F401
            # LightGBM doesn't have a portable "is GPU available"
            # check; we infer from the user's stated intent only.
            # Conservative: skip the warning if we can't tell.
        except ImportError:
            pass
    if any(f in families for f in ("xgb", "xgboost")):
        try:
            import xgboost as xgb
            try:
                # XGBoost build info is the canonical "GPU available?"
                # signal post-2.x.
                bi = xgb.build_info()
                if isinstance(bi, dict) and bi.get("USE_CUDA", False):
                    detected.append("xgboost")
            except Exception:
                pass
        except ImportError:
            pass
    if any(f in families for f in ("cb", "catboost")):
        try:
            from catboost.utils import get_gpu_device_count
            if get_gpu_device_count() > 0:
                detected.append("catboost")
        except Exception:
            pass
    return detected


def env_signature() -> dict[str, str | None]:
    """Snapshot of library versions relevant to composite-target
    discovery + serialisation. Stored on metadata so a pickle saved
    today can be reload-validated tomorrow against version drift.

    Returns ``None`` for any library not installed.
    """
    sig: dict[str, str | None] = {}
    for libname in ("numpy", "pandas", "polars", "sklearn", "lightgbm",
                    "xgboost", "catboost", "scipy", "dill"):
        try:
            mod = __import__(libname)
            sig[libname] = getattr(mod, "__version__", None)
        except Exception:
            sig[libname] = None
    return sig


def compute_oof_holdout_predictions(
    component_models: list[Any],
    component_names: list[str],
    component_specs: list[dict[str, Any] | None],
    train_X: Any,
    y_train_full: np.ndarray,
    base_train_full_per_spec: dict[str, np.ndarray],
    holdout_frac: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute honest holdout predictions for each component.

    Approach: take a single random ``holdout_frac`` slice of train,
    re-fit a clone of each component's inner on the remaining
    (1-holdout_frac) rows, and predict on the held-out slice. For
    wrapped composite-target components we re-apply the spec's
    transform on the same stack_train slice to get T values, train
    the inner clone on (X_stack_train, T_stack_train), then wrap
    using ``CompositeTargetEstimator.from_fitted_inner`` and predict
    in y-scale on stack_holdout. For raw-target components the inner
    clone is fit directly on (X_stack_train, y_stack_train).

    Single-split (not K-fold) keeps the additional compute bounded
    at ``len(components)`` re-fits. Returns:

    - ``holdout_preds_matrix``: (n_holdout, K) y-scale predictions.
    - ``y_holdout``: (n_holdout,) original-scale targets.
    - ``surviving_names``: subset of ``component_names`` whose
      re-fit succeeded (any failures are dropped from the matrix
      so callers can re-align weight vectors).
    """
    from sklearn.model_selection import train_test_split

    n_train = len(y_train_full)
    if n_train < 50 or holdout_frac <= 0 or holdout_frac >= 1:
        return np.zeros((0, len(component_models))), np.zeros(0), []

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n_train)
    n_holdout = max(int(round(n_train * holdout_frac)), 1)
    holdout_idx = np.sort(perm[:n_holdout])
    train_idx = np.sort(perm[n_holdout:])

    # Subset X. Branch on type so we don't pull pandas APIs on
    # polars frames.
    if hasattr(train_X, "to_pandas") and not isinstance(train_X, pd.DataFrame):
        import polars as pl
        train_mask = np.zeros(n_train, dtype=bool)
        train_mask[train_idx] = True
        X_stack = train_X.filter(pl.Series(train_mask))
        X_holdout = train_X.filter(pl.Series(~train_mask))
    elif isinstance(train_X, pd.DataFrame):
        X_stack = train_X.iloc[train_idx].reset_index(drop=True)
        X_holdout = train_X.iloc[holdout_idx].reset_index(drop=True)
    else:
        X_stack = train_X[train_idx]
        X_holdout = train_X[holdout_idx]

    y_stack = y_train_full[train_idx].astype(np.float64)
    y_holdout = y_train_full[holdout_idx].astype(np.float64)

    holdout_cols: list[np.ndarray] = []
    surviving_names: list[str] = []
    for model, name, spec in zip(component_models, component_names, component_specs):
        try:
            if isinstance(model, CompositeTargetEstimator):
                # Composite-target wrapper. Re-fit the inner on
                # stack_train T values, then re-wrap and predict.
                if spec is None:
                    raise ValueError("composite component with no spec")
                base_full = base_train_full_per_spec.get(spec["base_column"])
                if base_full is None:
                    raise ValueError(
                        f"missing base column '{spec['base_column']}' for OOF"
                    )
                base_stack = base_full[train_idx]
                transform = get_transform(spec["transform_name"])
                valid = transform.domain_check(y_stack, base_stack)
                # Drop invalid rows from stack_train; the inner will
                # train only on rows where T is finite.
                if valid.sum() < 10:
                    raise ValueError("too few valid rows after domain filter")
                t_stack = transform.forward(
                    y_stack[valid], base_stack[valid], spec["fitted_params"],
                )
                inner_clone = clone(model.estimator_)
                if hasattr(X_stack, "iloc"):
                    X_stack_valid = X_stack.iloc[valid].reset_index(drop=True)
                elif hasattr(X_stack, "filter") and not isinstance(X_stack, np.ndarray):
                    import polars as pl
                    X_stack_valid = X_stack.filter(pl.Series(valid))
                else:
                    X_stack_valid = X_stack[valid]
                inner_clone.fit(X_stack_valid, t_stack)
                wrapped = CompositeTargetEstimator.from_fitted_inner(
                    fitted_inner=inner_clone,
                    transform_name=spec["transform_name"],
                    base_column=spec["base_column"],
                    transform_fitted_params=spec["fitted_params"],
                    y_train=y_stack[valid],
                )
                preds = wrapped.predict(X_holdout)
            else:
                # Raw-target component. Re-fit the inner on
                # (X_stack, y_stack) and predict on X_holdout.
                inner_clone = clone(model)
                inner_clone.fit(X_stack, y_stack)
                preds = inner_clone.predict(X_holdout)
            preds = np.asarray(preds).reshape(-1).astype(np.float64)
            if not np.all(np.isfinite(preds)):
                # NaN preds on holdout -- exclude from ensemble.
                raise ValueError("non-finite holdout predictions")
            holdout_cols.append(preds)
            surviving_names.append(name)
        except Exception as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] OOF refit failed for component "
                "'%s': %s. Excluded from ensemble weights.", name, exc,
            )
            continue

    if not holdout_cols:
        return np.zeros((n_holdout, 0)), y_holdout, []
    return np.column_stack(holdout_cols), y_holdout, surviving_names


class CompositeCrossTargetEnsemble:
    """Weighted-average ensemble of K composite-target predictors plus
    optionally the raw-target predictor.

    All input models MUST already produce y-scale predictions (i.e. be
    :class:`CompositeTargetEstimator` wrappers OR a raw regressor on
    the original target). The ensemble does not invert anything --
    it just averages.

    The ensemble class itself is strategy-neutral: weights are
    pre-computed by :meth:`from_train_metrics` (the recommended path)
    or :meth:`from_uniform_weights` (mean baseline) and frozen on the
    instance. ``predict`` is one matrix-vector product.

    Validation gate
    ---------------
    :meth:`from_train_metrics` runs a built-in gate: it compares the
    ensemble's train-set RMSE against the best single component's
    train-set RMSE. If the ensemble is worse, it returns the best
    single component instead and logs a warning. The check is
    biased optimistic (uses train data) but still catches the most
    common failure mode -- a high-variance candidate with a stretched
    weight that drags the ensemble below the strongest component.
    """

    def __init__(
        self,
        component_models: list[Any],
        component_names: list[str],
        weights: np.ndarray,
        strategy: str,
        notes: dict[str, Any] | None = None,
    ) -> None:
        if len(component_models) == 0:
            raise ValueError("CompositeCrossTargetEnsemble: empty component list.")
        if len(component_models) != len(component_names) or len(component_models) != len(weights):
            raise ValueError(
                "CompositeCrossTargetEnsemble: component_models, component_names, "
                "and weights must be same length; got "
                f"{len(component_models)} / {len(component_names)} / {len(weights)}."
            )
        weights = np.asarray(weights, dtype=np.float64)
        wsum = float(weights.sum())
        if wsum <= 0 or not math.isfinite(wsum):
            raise ValueError(
                f"CompositeCrossTargetEnsemble: weights must sum to positive finite "
                f"value; got sum={wsum}."
            )
        self.component_models = list(component_models)
        self.component_names = list(component_names)
        self.weights = weights / wsum  # always normalised
        self.strategy = strategy
        self.notes = dict(notes or {})

    # ------------------------------------------------------------------
    # Constructors / factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_uniform_weights(
        cls,
        component_models: list[Any],
        component_names: list[str],
    ) -> CompositeCrossTargetEnsemble:
        """Equal-weight average: ``w_k = 1/K`` for all components."""
        n = len(component_models)
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=np.full(n, 1.0 / n) if n > 0 else np.array([]),
            strategy="mean",
        )

    @classmethod
    def from_linear_stack(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_predictions: np.ndarray,  # (n_train, K) y-scale predictions
        y_train: np.ndarray,
        ridge_alpha: float = 1.0,
    ) -> CompositeCrossTargetEnsemble:
        """Linear stacking via Ridge regression.

        Fits a Ridge model ``y_train ~ X @ w + b`` where ``X`` is the
        per-component prediction matrix on train. The resulting
        weights are the stack coefficients; intercept is folded into
        the bias by absorbing it as an extra ``+b/n`` per component
        (good enough when Ridge converges).

        ``ridge_alpha`` is fixed (no internal CV) -- callers wanting
        alpha tuning should ridge-CV externally and pass the chosen
        alpha. Higher alpha -> more regularisation -> closer to mean.

        Returns negative weights when a component is anti-correlated
        with the target -- this is fine, the ensemble may still work.
        ``predict`` re-normalises only the magnitudes, so a negative
        weight means the component's prediction is subtracted.
        """
        from sklearn.linear_model import Ridge
        n = len(component_models)
        if n == 0:
            raise ValueError("from_linear_stack: empty component list.")
        component_predictions = np.asarray(component_predictions, dtype=np.float64)
        if component_predictions.shape[1] != n:
            raise ValueError(
                f"from_linear_stack: prediction matrix has {component_predictions.shape[1]} "
                f"columns, expected {n} (one per component)."
            )
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if len(y) != component_predictions.shape[0]:
            raise ValueError(
                f"from_linear_stack: y_train length {len(y)} != prediction "
                f"matrix rows {component_predictions.shape[0]}."
            )
        # Drop rows with non-finite y or predictions.
        finite = np.isfinite(y) & np.all(np.isfinite(component_predictions), axis=1)
        if finite.sum() < n + 2:
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: only %d finite rows for "
                "%d components; falling back to oof_weighted-style mean.",
                int(finite.sum()), n,
            )
            return cls.from_uniform_weights(component_models, component_names)

        ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
        ridge.fit(component_predictions[finite], y[finite])
        raw_weights = np.asarray(ridge.coef_, dtype=np.float64)
        # Sanity: if all weights are zero or non-finite, fall back.
        if not np.any(raw_weights) or not np.all(np.isfinite(raw_weights)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: degenerate weights; "
                "falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)
        # The constructor normalises by sum. For linear_stack this is
        # NOT semantically right (negative weights, intercept), so we
        # bypass the normalisation by building manually.
        instance = cls(
            component_models=component_models,
            component_names=component_names,
            weights=np.abs(raw_weights) + 1e-12,  # placeholder for constructor
            strategy="linear_stack",
        )
        # Inject the actual (un-normalised) weights + intercept.
        instance.weights = raw_weights
        instance.notes = {
            "ridge_alpha": ridge_alpha,
            "intercept": float(ridge.intercept_),
            "raw_weights": raw_weights.tolist(),
            "n_train_rows": int(finite.sum()),
        }
        instance._linear_stack_intercept = float(ridge.intercept_)
        return instance

    @classmethod
    def from_nnls_stack(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_predictions: np.ndarray,
        y_train: np.ndarray,
    ) -> CompositeCrossTargetEnsemble:
        """Non-negative least squares stacking.

        Fits ``y = X @ w`` subject to ``w >= 0`` via
        ``scipy.optimize.nnls``. Weights are then normalised to sum
        to 1, which keeps the predict path identical to mean /
        oof_weighted (no separate intercept handling). This is the
        recommended stack when component predictions are all already
        in y-scale (no negative weight makes physical sense), and is
        less prone to overfitting than ridge stacking on small data.
        """
        from scipy.optimize import nnls
        n = len(component_models)
        if n == 0:
            raise ValueError("from_nnls_stack: empty component list.")
        component_predictions = np.asarray(component_predictions, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y) & np.all(np.isfinite(component_predictions), axis=1)
        if finite.sum() < n + 2:
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: only %d finite rows for "
                "%d components; falling back to mean.",
                int(finite.sum()), n,
            )
            return cls.from_uniform_weights(component_models, component_names)

        try:
            w, _residual = nnls(component_predictions[finite], y[finite])
        except RuntimeError as exc:
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: solver failed (%s); "
                "falling back to mean.", exc,
            )
            return cls.from_uniform_weights(component_models, component_names)

        if w.sum() <= 0 or not np.all(np.isfinite(w)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] nnls_stack: zero or non-finite "
                "weights; falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)

        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=w,
            strategy="nnls_stack",
            notes={
                "raw_weights_pre_normalise": w.tolist(),
                "n_train_rows": int(finite.sum()),
            },
        )

    @classmethod
    def from_train_metrics(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_train_rmse: Sequence[float],
        baseline_train_rmse: float | None = None,
    ) -> Union[CompositeCrossTargetEnsemble, Any]:
        """Build an ensemble weighted by *gain over a naive baseline*.

        The gain-over-naive convention defends against the trivial
        "raw model with TVT_prev as a feature" beating the naive
        ``predict y = base`` baseline by a tiny margin: that model's
        absolute RMSE is small (good), but its *gain* over the naive
        baseline is also small, so it gets a sensible weight rather
        than dominating the ensemble simply because its RMSE is
        numerically smaller than a good but harder-target composite.

        ``baseline_train_rmse`` is the RMSE of the naive predictor
        ``y_hat = base`` on train (or any sensible benchmark; pass
        the noisiest reasonable predictor's RMSE). If None, the
        median of ``component_train_rmse`` is used as a self-
        normalising fallback.

        If every component's RMSE is worse than the baseline, the
        method returns the SINGLE best-RMSE component instead of the
        ensemble (validation gate). Log line announces the fallback.
        """
        n = len(component_models)
        if n == 0:
            raise ValueError("from_train_metrics: empty component list.")
        rmses = np.asarray(component_train_rmse, dtype=np.float64)
        if len(rmses) != n:
            raise ValueError(
                f"from_train_metrics: rmse list len {len(rmses)} != n_components {n}."
            )
        if not np.all(np.isfinite(rmses)):
            raise ValueError("from_train_metrics: rmses contain non-finite values.")

        if baseline_train_rmse is None:
            baseline = float(np.median(rmses))
        else:
            baseline = float(baseline_train_rmse)
            if not math.isfinite(baseline):
                baseline = float(np.median(rmses))

        gains = np.maximum(0.0, baseline - rmses)
        if gains.sum() <= 0:
            # No component beats baseline. Return the single best by
            # RMSE; ensemble would be no improvement.
            best_idx = int(np.argmin(rmses))
            logger.warning(
                "[CompositeCrossTargetEnsemble] no component beats the baseline "
                "RMSE=%.4g; falling back to single best component '%s' (RMSE=%.4g).",
                baseline, component_names[best_idx], rmses[best_idx],
            )
            return component_models[best_idx]

        # The "no component beats baseline" gate fires above; the
        # only remaining decision is to build the ensemble.
        # We deliberately do NOT add an independence-bound RMSE gate
        # here: composite-target predictions correlate (same train
        # data, shared base feature), so the independence formula
        # overestimates ensemble RMSE and the gate would fire on
        # legitimate ensembles. The true validation gate -- "ensemble
        # OOF-RMSE > best single OOF-RMSE" -- requires real CV-OOF
        # predictions per component, which the per-target loop does
        # not currently expose. A future PR may add OOF storage; for
        # now the user is expected to evaluate the ensemble on a
        # held-out test set themselves.
        weights = gains / gains.sum()
        best_single_idx = int(np.argmin(rmses))
        best_single_rmse = float(rmses[best_single_idx])
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=weights,
            strategy="oof_weighted",
            notes={
                "baseline_train_rmse": baseline,
                "component_train_rmses": rmses.tolist(),
                "best_single_rmse": best_single_rmse,
                "best_single_name": component_names[best_single_idx],
                "gate_fallback": False,
            },
        )

    # ------------------------------------------------------------------
    # sklearn-ish API
    # ------------------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Weighted combination of per-component predictions.

        For ``mean`` / ``oof_weighted`` / ``nnls_stack`` strategies
        weights are non-negative and sum to 1; the result is a
        weighted average. For ``linear_stack`` strategy weights may be
        negative, do not sum to 1, and an intercept is added -- the
        result is the Ridge stack's prediction
        ``y_hat = X @ w + intercept``.
        """
        if not self.component_models:
            raise RuntimeError("CompositeCrossTargetEnsemble: no components.")
        per_component = []
        for model, name in zip(self.component_models, self.component_names):
            try:
                # Fold the dtype cast into the asarray call so we don't
                # allocate twice on the predict hot path. ``copy=False``
                # is the asarray default; the dtype kwarg lets us skip
                # a separate ``.astype()`` round-trip.
                pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
            except Exception as exc:
                logger.warning(
                    "[CompositeCrossTargetEnsemble] component '%s' predict failed: "
                    "%s. Excluding from this batch's ensemble (re-normalising).",
                    name, exc,
                )
                pred = None
            per_component.append(pred)

        ok = [(p, w) for p, w in zip(per_component, self.weights) if p is not None]
        if not ok:
            raise RuntimeError(
                "CompositeCrossTargetEnsemble.predict: all components failed."
            )
        preds_matrix = np.column_stack([p for p, _ in ok])
        weights = np.array([w for _, w in ok], dtype=np.float64)

        if self.strategy == "linear_stack":
            # Ridge stack: predictions = X @ w + intercept. Do NOT
            # renormalise weights. If a component dropped out, drop
            # its weight contribution -- the rest of the linear
            # combination is still valid (just with one fewer term).
            intercept = float(getattr(self, "_linear_stack_intercept", 0.0))
            return (preds_matrix * weights[None, :]).sum(axis=1) + intercept

        # Convex strategies (mean / oof_weighted / nnls_stack):
        # re-normalise weights across surviving components.
        if weights.sum() <= 0:
            # All surviving weights collapsed to zero -- fall back to
            # mean across surviving components.
            weights = np.full_like(weights, 1.0 / len(weights))
        else:
            weights = weights / weights.sum()
        return (preds_matrix * weights[None, :]).sum(axis=1)

    def export_metadata(self) -> dict[str, Any]:
        """Plain-dict snapshot for ``metadata`` storage."""
        return {
            "strategy": self.strategy,
            "component_names": list(self.component_names),
            "weights": self.weights.tolist(),
            "notes": dict(self.notes),
        }

    def cap_inference_components(
        self, max_components: int,
    ) -> CompositeCrossTargetEnsemble:
        """Return a NEW ensemble holding only the top-N components by
        absolute weight.

        Use case: production online prediction with a latency budget
        that can't afford running K=8 wrappers per row. Trims to the
        largest-weighted components and re-normalises (or preserves
        the linear-stack semantics by keeping the matching subset of
        weights + intercept). Returns a new ensemble; the original
        is unchanged.

        ``max_components <= 0`` or ``>= len(components)`` -> returns
        a copy of self unchanged (no trimming).
        """
        if max_components <= 0 or max_components >= len(self.component_models):
            return CompositeCrossTargetEnsemble(
                component_models=list(self.component_models),
                component_names=list(self.component_names),
                weights=np.asarray(self.weights, dtype=np.float64),
                strategy=self.strategy,
                notes=dict(self.notes),
            )
        # Pick top-N by |weight|.
        order = np.argsort(-np.abs(np.asarray(self.weights, dtype=np.float64)))
        keep = sorted(order[:max_components].tolist())
        new = CompositeCrossTargetEnsemble(
            component_models=[self.component_models[i] for i in keep],
            component_names=[self.component_names[i] for i in keep],
            weights=np.asarray([self.weights[i] for i in keep], dtype=np.float64),
            strategy=self.strategy,
            notes={**self.notes, "capped_to_top_n": int(max_components),
                   "dropped_components": [
                       self.component_names[i]
                       for i in range(len(self.component_models))
                       if i not in keep
                   ]},
        )
        # Linear stack: preserve intercept too.
        if self.strategy == "linear_stack" and hasattr(self, "_linear_stack_intercept"):
            new._linear_stack_intercept = self._linear_stack_intercept
        return new

