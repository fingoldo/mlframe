"""``CompositeCrossTargetEnsemble`` -- the multi-component composite ensemble class.

Wave 101 (2026-05-21): split out from ``composite_ensemble.py`` to keep that
file below the 1k-line monolith threshold. Behaviour preserved bit-for-bit;
the class is re-exported from ``composite_ensemble`` so existing imports
continue to work.
"""
from __future__ import annotations

import logging
import math
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge, RidgeCV

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False

logger = logging.getLogger(__name__)


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
        is_convex: bool = True,
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
        # ``is_convex=True``: weights are a convex combination (non-negative, sum to 1) -- normalise
        # so the predict path can rely on the invariant. ``is_convex=False``: weights are the raw
        # output of a constrained / unconstrained solver (Ridge linear_stack, NNLS without renorm)
        # and downstream code must NOT assume sum=1; predict for these strategies does an
        # additive linear combination (no renormalisation on the surviving subset).
        if is_convex:
            wsum = float(weights.sum())
            if wsum <= 0 or not math.isfinite(wsum):
                raise ValueError(
                    f"CompositeCrossTargetEnsemble: convex weights must sum to positive finite "
                    f"value; got sum={wsum}."
                )
            weights = weights / wsum
        else:
            if not np.all(np.isfinite(weights)):
                raise ValueError(
                    "CompositeCrossTargetEnsemble: weights contain non-finite values."
                )
        self.component_models = list(component_models)
        self.component_names = list(component_names)
        self.weights = weights
        self.strategy = strategy
        self.is_convex = bool(is_convex)
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
        # C-P2-3: fail fast with a self-attributed message rather than falling through to
        # ``__init__``'s generic "empty component list" check. Callers reach this point because
        # they had no fit candidates, not because of an internal bug -- surfacing the true cause
        # at the call site lets ops investigate the upstream candidate-discovery step.
        if n == 0:
            raise ValueError(
                "from_uniform_weights: empty component list. The caller must supply at least "
                "one component model + name. Check upstream component-discovery (no surviving "
                "components after fit / gate filtering)."
            )
        return cls(
            component_models=component_models,
            component_names=component_names,
            weights=np.full(n, 1.0 / n),
            strategy="mean",
        )

    @classmethod
    def from_linear_stack(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_predictions: np.ndarray,  # (n_train, K) y-scale predictions
        y_train: np.ndarray,
        ridge_alpha: float | None = None,
        ridge_alpha_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        sample_weight: np.ndarray | None = None,
    ) -> CompositeCrossTargetEnsemble:
        """Linear stacking via Ridge regression with internal CV alpha selection.

        Fits a Ridge model ``y_train ~ X @ w + b`` where ``X`` is the
        per-component prediction matrix on train. The resulting weights
        are the stack coefficients; the intercept is stored separately
        and added at predict time (``y_hat = X @ w + intercept``).

        When ``ridge_alpha`` is None (default), the alpha is chosen via
        ``RidgeCV`` over ``ridge_alpha_grid`` using built-in efficient
        leave-one-out CV. Pass a concrete ``ridge_alpha`` to bypass CV.

        Returns negative weights when a component is anti-correlated
        with the target -- this is fine, the ensemble may still work.
        ``linear_stack`` is a non-convex NO-REFIT strategy: predict does
        no renormalisation, so a negative weight means the component's
        prediction is subtracted.
        """
        # ENS-Low-7: Ridge / RidgeCV hoisted to module-top imports.
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

        # Optional per-row weighting for the Ridge fit. Ridge / RidgeCV both accept sample_weight natively;
        # we slice it on the same ``finite`` mask used for X / y so weight rows stay aligned with kept rows.
        _ridge_sw = None
        if sample_weight is not None:
            _ridge_sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if _ridge_sw.shape[0] != component_predictions.shape[0]:
                raise ValueError(
                    f"from_linear_stack: sample_weight length {_ridge_sw.shape[0]} != "
                    f"prediction matrix rows {component_predictions.shape[0]}."
                )
            if not np.all(np.isfinite(_ridge_sw)) or (_ridge_sw < 0).any():
                raise ValueError("from_linear_stack: sample_weight must be finite and non-negative.")
            _ridge_sw = _ridge_sw[finite]

        if ridge_alpha is None:
            ridge = RidgeCV(alphas=tuple(ridge_alpha_grid), fit_intercept=True)
            ridge.fit(component_predictions[finite], y[finite], sample_weight=_ridge_sw)
            chosen_alpha = float(getattr(ridge, "alpha_", ridge_alpha_grid[0]))
        else:
            ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
            ridge.fit(component_predictions[finite], y[finite], sample_weight=_ridge_sw)
            chosen_alpha = float(ridge_alpha)
        raw_weights = np.asarray(ridge.coef_, dtype=np.float64)
        # Sanity: if all weights are zero or non-finite, fall back.
        if not np.any(raw_weights) or not np.all(np.isfinite(raw_weights)):
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: degenerate weights; "
                "falling back to mean."
            )
            return cls.from_uniform_weights(component_models, component_names)
        # WEIGHT-NEGATIVE-WARN: Ridge can produce negative coefficients when a component is
        # anti-correlated with the target. That's mathematically fine for a linear stack but
        # operators routinely interpret negative-weight ensembles as a bug; surface them at WARN
        # so the suite log makes the situation explicit. The ensemble is still built (the predict
        # path supports negative weights via is_convex=False).
        _neg_mask = raw_weights < 0
        if _neg_mask.any():
            # C-P2-1: surface a quantitative impact estimate so operators see HOW MUCH of the ensemble's
            # combined |weight| comes from negative-coefficient members. A scalar like 0.62 means
            # 62% of the absolute-weight mass is anti-correlated, which is a much clearer signal than
            # the per-member list alone.
            _neg_share = float(np.sum(np.abs(raw_weights[_neg_mask])) / max(np.sum(np.abs(raw_weights)), 1e-300))
            logger.warning(
                "[CompositeCrossTargetEnsemble] linear_stack: %d/%d components have negative Ridge "
                "coefficients (anti-correlated with target on train; %.1f%% of |weight| mass). Ensemble "
                "is still built; the predict path subtracts the weighted contribution of those members. "
                "Affected: %s.",
                int(_neg_mask.sum()), len(raw_weights), 100.0 * _neg_share,
                ", ".join(f"{component_names[i]} (w={raw_weights[i]:.4g})" for i in np.flatnonzero(_neg_mask).tolist()),
            )
        # Ridge stack: weights may be negative and do not sum to 1; signal this with
        # is_convex=False so the constructor skips the convex-sum normalisation that
        # would otherwise destroy the Ridge fit. Eliminates the historical
        # build-with-placeholder-then-mutate dance.
        instance = cls(
            component_models=component_models,
            component_names=component_names,
            weights=raw_weights,
            strategy="linear_stack",
            notes={
                "ridge_alpha": chosen_alpha,
                "ridge_alpha_was_cv_selected": ridge_alpha is None,
                "ridge_alpha_grid": list(ridge_alpha_grid) if ridge_alpha is None else None,
                "intercept": float(ridge.intercept_),
                "raw_weights": raw_weights.tolist(),
                "n_train_rows": int(finite.sum()),
            },
            is_convex=False,
        )
        instance._linear_stack_intercept = float(ridge.intercept_)
        # SOLVER-COPY: ``component_predictions[finite]`` already returns a copy (boolean indexing on
        # ndarray always allocates), so the prior explicit ``.copy()`` was a 256-MB-per-pickle
        # duplicate. Same for ``y[finite]``. Keep a reference instead of a copy of a copy.
        instance._linear_stack_train_preds = component_predictions[finite]
        instance._linear_stack_train_y = y[finite]
        instance._linear_stack_ridge_alpha = chosen_alpha
        return instance

    @classmethod
    def from_nnls_stack(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_predictions: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> CompositeCrossTargetEnsemble:
        """Non-negative least squares stacking.

        Fits ``y = X @ w`` subject to ``w >= 0`` via
        ``scipy.optimize.nnls``. Weights are kept AS NNLS computed them
        (no post-hoc renormalisation to sum=1) -- renormalising would
        return a predictor that differs from the one the gate (and
        any downstream RMSE-on-y-scale evaluation) measured. The
        is_convex=False flag signals this to downstream consumers so
        they don't assume sum=1.

        sample_weight: optional per-row weights. scipy.nnls has no native sample_weight kwarg, so we
        emulate weighted least squares by row-scaling: replace ``A x = b`` with ``diag(sqrt(w)) A x =
        diag(sqrt(w)) b``. The NNLS minimiser of the scaled system is identical to the weighted-LS
        minimiser of the original system because ||sqrt(w) (Ax - b)||_2^2 == sum_i w_i (a_i x - b_i)^2.
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

        _A_for_nnls = component_predictions[finite]
        _b_for_nnls = y[finite]
        _nnls_sw = None
        if sample_weight is not None:
            _nnls_sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if _nnls_sw.shape[0] != component_predictions.shape[0]:
                raise ValueError(
                    f"from_nnls_stack: sample_weight length {_nnls_sw.shape[0]} != "
                    f"prediction matrix rows {component_predictions.shape[0]}."
                )
            if not np.all(np.isfinite(_nnls_sw)) or (_nnls_sw < 0).any():
                raise ValueError("from_nnls_stack: sample_weight must be finite and non-negative.")
            _nnls_sw = _nnls_sw[finite]
            _sqrt_w = np.sqrt(_nnls_sw).reshape(-1, 1)
            _A_for_nnls = _A_for_nnls * _sqrt_w
            _b_for_nnls = _b_for_nnls * _sqrt_w.reshape(-1)

        try:
            w, _residual = nnls(_A_for_nnls, _b_for_nnls)
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

        instance = cls(
            component_models=component_models,
            component_names=component_names,
            weights=w,
            strategy="nnls_stack",
            notes={
                "raw_weights": w.tolist(),
                "n_train_rows": int(finite.sum()),
            },
            # Don't renormalise: the predictor returned must match the one NNLS solved
            # for so the gate's measured RMSE corresponds to the deployed model.
            is_convex=False,
        )
        # SOLVER-COPY: boolean-indexing already produces a copy; the prior explicit ``.copy()`` was
        # a redundant 256-MB-per-pickle duplicate. Keep the bool-indexed view directly.
        instance._nnls_stack_train_preds = component_predictions[finite]
        instance._nnls_stack_train_y = y[finite]
        return instance

    @classmethod
    def from_train_metrics(
        cls,
        component_models: list[Any],
        component_names: list[str],
        component_train_rmse: Sequence[float] | None = None,
        baseline_train_rmse: float | None = None,
        component_oof_rmse: Sequence[float] | None = None,
        baseline_oof_rmse: float | None = None,
    ) -> Union[CompositeCrossTargetEnsemble, Any]:
        """Build an ensemble weighted by *gain over a naive baseline*.

        The gain-over-naive convention defends against the trivial
        "raw model with a lag feature" beating the naive
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
        # VAL-LEAK (from_train_metrics): prefer OOF RMSE when supplied. Train-set RMSE is biased
        # because train rows were seen at fit -- using it for weight derivation is the
        # same selection problem as using val (already burned for ES). When ``component_oof_rmse``
        # is given we rank on OOF; otherwise we fall back to train_rmse with a WARN so the operator
        # knows the gate is biased optimistic.
        if component_oof_rmse is not None:
            rmses = np.asarray(component_oof_rmse, dtype=np.float64)
            if len(rmses) != n:
                raise ValueError(
                    f"from_train_metrics: component_oof_rmse list len {len(rmses)} != n_components {n}."
                )
            if baseline_oof_rmse is not None:
                baseline_train_rmse = baseline_oof_rmse
        else:
            if component_train_rmse is None:
                raise ValueError(
                    "from_train_metrics: must supply either component_oof_rmse "
                    "(preferred, honest holdout) or component_train_rmse "
                    "(biased optimistic fallback); got neither."
                )
            rmses = np.asarray(component_train_rmse, dtype=np.float64)
            if len(rmses) != n:
                raise ValueError(
                    f"from_train_metrics: rmse list len {len(rmses)} != n_components {n}."
                )
            logger.warning(
                "[CompositeCrossTargetEnsemble] from_train_metrics: ranking on TRAIN RMSE which is "
                "biased optimistic (rows seen at fit). Pass component_oof_rmse=... for an honest "
                "cross-validated weighting."
            )
        if not np.all(np.isfinite(rmses)):
            raise ValueError("from_train_metrics: rmses contain non-finite values.")

        if baseline_train_rmse is None:
            # MEDIAN-BASELINE: previously fell back to ``np.median(rmses)`` which by construction
            # discards the worse-than-median half of the candidate pool entirely. That's a hidden
            # contract surprise: the caller passed K components expecting a K-component ensemble
            # and got a (K/2)-component one with no log line. We now use the WORST component RMSE
            # (numerically the largest) as the baseline so every component that beats the worst
            # contributes a non-zero weight, AND we WARN the caller that no explicit baseline was
            # passed -- the operator should plug in a real benchmark (naive predictor / median
            # baseline_train_rmse / dataset variance) for production runs.
            baseline = float(np.max(rmses))
            logger.warning(
                "[CompositeCrossTargetEnsemble] from_train_metrics: no baseline_train_rmse passed; "
                "defaulting to max(component_train_rmse)=%.4g so every component beats baseline. "
                "Pass an explicit baseline (e.g. naive predictor RMSE) to get a real gain-over-naive weighting.",
                baseline,
            )
        else:
            baseline = float(baseline_train_rmse)
            if not math.isfinite(baseline):
                baseline = float(np.max(rmses))

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

        # Track surviving indices so linear_stack can refit Ridge on exactly the columns
        # whose components produced predictions for this batch.
        surviving_idx = [i for i, p in enumerate(per_component) if p is not None]
        ok = [(per_component[i], self.weights[i]) for i in surviving_idx]
        if not ok:
            raise RuntimeError(
                "CompositeCrossTargetEnsemble.predict: all components failed."
            )
        preds_matrix = np.column_stack([p for p, _ in ok])
        weights = np.array([w for _, w in ok], dtype=np.float64)

        if not getattr(self, "is_convex", True):
            # Non-convex strategies (linear_stack, nnls_stack): weights are the raw solver output,
            # possibly negative (Ridge) or with arbitrary sum (NNLS). When every component is present
            # predict is the exact deployed stack. When a component drops out at predict time we keep
            # only the surviving columns' raw weights (plus the linear_stack intercept) -- a NO-REFIT
            # policy so predict stays a pure deterministic function of the inputs: the same input yields
            # the same output across repeated / batched calls. The earlier behaviour refit a fresh
            # solver on a stashed (n_train, K) train matrix, which (a) made predict depend on which
            # columns dropped, (b) was non-deterministic across batches when different batches lost
            # different components, and (c) forced a multi-GB stash to survive every pickle. Dropping a
            # column from a linear stack is a sensible, deterministic fallback; the alternative refit was
            # not worth the leakage/RAM/non-determinism cost.
            full_weights = np.asarray(self.weights, dtype=np.float64)
            full_intercept = float(getattr(self, "_linear_stack_intercept", 0.0))
            if len(surviving_idx) == len(self.component_models):
                return (preds_matrix * full_weights[None, :]).sum(axis=1) + full_intercept

            surviving_weights = full_weights[surviving_idx]
            _intercept = full_intercept if self.strategy == "linear_stack" else 0.0
            logger.warning(
                "[CompositeCrossTargetEnsemble] %s: %d of %d components dropped out at predict time; "
                "combining surviving columns with their original weights (no refit, deterministic).",
                self.strategy, len(self.component_models) - len(surviving_idx),
                len(self.component_models),
            )
            return (preds_matrix * surviving_weights[None, :]).sum(axis=1) + _intercept

        # Convex strategies (mean / oof_weighted): re-normalise across surviving components.
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
            "is_convex": bool(getattr(self, "is_convex", True)),
            "intercept": float(getattr(self, "_linear_stack_intercept", 0.0)),
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
            copy_inst = CompositeCrossTargetEnsemble(
                component_models=list(self.component_models),
                component_names=list(self.component_names),
                weights=np.asarray(self.weights, dtype=np.float64),
                strategy=self.strategy,
                notes=dict(self.notes),
                is_convex=getattr(self, "is_convex", True),
            )
            for _attr in (
                "_linear_stack_intercept", "_linear_stack_train_preds",
                "_linear_stack_train_y", "_linear_stack_ridge_alpha",
                "_nnls_stack_train_preds", "_nnls_stack_train_y",
            ):
                if hasattr(self, _attr):
                    setattr(copy_inst, _attr, getattr(self, _attr))
            return copy_inst
        # Pick top-N by |weight|.
        # Wave 57 (2026-05-20): lexsort with component-index tiebreaker so tied
        # weights (NNLS shrinkage saturation, convex weights pinned at 0) don't
        # silently flip which components survive across stack-row orderings.
        _abs_w = np.abs(np.asarray(self.weights, dtype=np.float64))
        order = np.lexsort((np.arange(len(_abs_w)), -_abs_w))
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
            is_convex=getattr(self, "is_convex", True),
        )
        # Carry over the linear/NNLS stash to the trimmed ensemble. predict (NO-REFIT) never reads
        # these; the column-slice below only keeps the stash aligned for the votenrank diagnostic
        # assertion, which expects the train-matrix columns to match the component ordering.
        if self.strategy == "linear_stack" and hasattr(self, "_linear_stack_intercept"):
            new._linear_stack_intercept = self._linear_stack_intercept
        for _attr in (
            "_linear_stack_train_preds", "_linear_stack_train_y", "_linear_stack_ridge_alpha",
            "_nnls_stack_train_preds", "_nnls_stack_train_y",
        ):
            if hasattr(self, _attr):
                # cap_inference_components(N) stores only N components, so slice the train-matrix
                # columns to the kept set to keep the stash aligned with the new component ordering.
                _val = getattr(self, _attr)
                if _attr.endswith("_train_preds") and _val is not None:
                    setattr(new, _attr, np.asarray(_val)[:, keep])
                else:
                    setattr(new, _attr, _val)
        return new

    def discard_train_matrix(self) -> CompositeCrossTargetEnsemble:
        """Drop the stashed (n_train, K) training-prediction matrix and target vector.

        The stash is NO-REFIT-era dead weight: predict no longer refits a solver on component
        dropout, so no method consumes these arrays. They are ~``8 * n_train * (K + 1)`` bytes that
        would otherwise survive every ``save_mlframe_model`` round-trip (``__getstate__`` already
        strips them before pickling). Call this method to strip them eagerly off a live instance;
        the returned ensemble keeps predicting normally. Operation is in-place and returns ``self``
        so call sites can chain.
        """
        for _attr in (
            "_linear_stack_train_preds",
            "_linear_stack_train_y",
            "_nnls_stack_train_preds",
            "_nnls_stack_train_y",
        ):
            if hasattr(self, _attr):
                delattr(self, _attr)
        self.notes["train_matrix_discarded"] = True
        return self

    # Names of the (n_train, K) stash attributes that must never reach a pickle: at TB scale they
    # dominate the serialized blob (~8 * n_train * (K+1) bytes) and survive every save round-trip.
    _TRAIN_MATRIX_ATTRS = (
        "_linear_stack_train_preds",
        "_linear_stack_train_y",
        "_nnls_stack_train_preds",
        "_nnls_stack_train_y",
    )

    def __getstate__(self) -> dict[str, Any]:
        """Strip the (n_train, K) training-prediction stash from the pickle.

        The dropout-refit path is the only consumer of these arrays and the deployed default is no-refit
        renormalisation (predict falls back when they are absent), so persisting them just bloats every save by
        GBs on large frames. The unpickled instance keeps predicting; only the deprecated refit-on-dropout path
        becomes unavailable, which is the intended trade.
        """
        state = dict(self.__dict__)
        for _attr in self._TRAIN_MATRIX_ATTRS:
            state.pop(_attr, None)
        notes = dict(state.get("notes") or {})
        notes["train_matrix_discarded"] = True
        state["notes"] = notes
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

