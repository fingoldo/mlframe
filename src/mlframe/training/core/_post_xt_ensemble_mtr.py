"""MTR per-column ensemble carved out of ``_phase_composite_post_xt_ensemble``.

Holds ``MTRPerColumnEqualMeanEnsemble`` (the per-target equal-mean / NNLS wrapper for MULTI_TARGET_REGRESSION cross-target ensembling) and ``_build_mtr_per_column_ensemble`` (registers it into the suite ``models`` / ``metadata`` dicts). The parent module re-exports both names for backward compatibility.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..composite_post_shim import PrePipelinePredictShim

logger = logging.getLogger("mlframe.training.core._phase_composite_post")


class MTRPerColumnEqualMeanEnsemble:
    """E2 (F-34, 2026-05-31) + E3 (F-34, 2026-05-31): per-column ensemble
    for MULTI_TARGET_REGRESSION cross-target ensembling.

    Wraps a list of K trained component models (each producing (N, K)
    predictions on input X). Two strategies:

      * ``strategy="equal_mean"`` (default): equal weight 1 / n_components
        per component per column. No fit() needed; works immediately.
      * ``strategy="nnls"`` (E3): non-negative least-squares per-column
        weights learned from a held-out (X, y) set via .fit(X, y).
        Weights are normalised to sum to 1 (or fall back to equal-mean
        when NNLS returns the all-zero degenerate solution). Per
        target column k, solves: y[:, k] = A_k @ w_k, w_k >= 0 where
        A_k is the (N, n_components) component-prediction matrix on
        the held-out X. Independent solve per column so the K targets
        can have different optimal component mixtures.

    The class keeps its original name (``MTRPerColumnEqualMeanEnsemble``)
    for backward compatibility with the existing ``isinstance`` checks
    and ``models`` dict entries; the ``strategy`` kwarg controls the
    behaviour. Future PR can wire honest-OOF (held-out preds from
    cross-validated folds) by calling .fit() with the OOF stack
    instead of an in-sample held-out set; the predict() contract is
    unchanged.

    The wrapper exposes the standard sklearn-shape predict(X) ->
    np.ndarray so the suite's downstream save / report layers treat
    it as any other regressor.
    """

    def __init__(
        self,
        components,
        component_names,
        n_targets: int,
        *,
        strategy: str = "equal_mean",
        weights: np.ndarray = None,
    ):
        if not components:
            raise ValueError("MTRPerColumnEqualMeanEnsemble requires at least 1 component")
        if strategy not in ("equal_mean", "nnls"):
            raise ValueError(
                f"strategy must be 'equal_mean' or 'nnls'; got {strategy!r}"
            )
        self._components = list(components)
        self._component_names = list(component_names) if component_names else [
            f"comp{i}" for i in range(len(self._components))
        ]
        self._n_targets = int(n_targets)
        self._strategy = strategy
        # Pre-supplied weights take precedence (e.g. a future PR that
        # computes honest-OOF weights externally and injects them).
        # Shape contract: (n_components, n_targets).
        if weights is not None:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (len(self._components), self._n_targets):
                raise ValueError(
                    f"weights shape {weights.shape} != "
                    f"({len(self._components)}, {self._n_targets})"
                )
            self._weights = weights
            self._strategy = "nnls"  # caller provided weights -> use them
        else:
            # Equal-mean default: 1 / n_components per (component, target).
            self._weights = np.full(
                (len(self._components), self._n_targets),
                1.0 / len(self._components),
                dtype=np.float64,
            )

    @property
    def components(self):
        return tuple(self._components)

    @property
    def component_names(self):
        return tuple(self._component_names)

    @property
    def n_targets(self) -> int:
        return self._n_targets

    @property
    def strategy(self) -> str:
        return self._strategy

    @property
    def weights(self) -> np.ndarray:
        """(n_components, n_targets) array. Columns are per-target
        weight vectors; rows are component contributions. Each column
        sums to 1.0 by construction (equal_mean) or post-normalisation
        (nnls). Defensive copy."""
        return self._weights.copy()

    def fit(self, X, y) -> "MTRPerColumnEqualMeanEnsemble":
        """E3: fit per-column NNLS weights from a held-out (X, y) set.

        For each target column k, solves:
            y[:, k] = A_k @ w_k,  subject to w_k >= 0
        where A_k[:, j] = self._components[j].predict(X)[:, k].
        Then normalises w_k to sum to 1 (so the prediction is a convex
        combination). When NNLS returns the all-zero degenerate
        solution (no component fits the column), falls back to
        equal-mean for that column.

        No-op when strategy == "equal_mean" (the equal-weight ensemble
        doesn't depend on training data).

        Args:
            X: held-out features (n_holdout, n_features).
            y: held-out targets (n_holdout, n_targets) or (n_holdout,)
               for K=1.

        Returns: self (sklearn convention).
        """
        if self._strategy == "equal_mean":
            return self  # no-op
        from scipy.optimize import nnls as _nnls

        # Gather (n_components, N, K) prediction stack on the held-out X.
        comp_preds = []
        for c in self._components:
            p = np.asarray(c.predict(X), dtype=np.float64)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            comp_preds.append(p)
        stacked = np.stack(comp_preds, axis=0)  # (n_comp, N, K)

        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_arr.shape[1] != self._n_targets:
            raise ValueError(
                f"y.shape[1] = {y_arr.shape[1]} != n_targets = {self._n_targets}"
            )

        n_comp = len(self._components)
        weights = np.zeros((n_comp, self._n_targets), dtype=np.float64)
        for k in range(self._n_targets):
            A_k = stacked[:, :, k].T  # (N, n_comp)
            b_k = y_arr[:, k]
            w_k, _residual = _nnls(A_k, b_k, maxiter=200)
            w_sum = float(w_k.sum())
            if w_sum > 0:
                # Keep RAW NNLS weights (not normalised to sum to 1).
                # Normalising distorts the optimal fit when component
                # predictions don't bracket the target -- e.g. if all
                # components emit values smaller than y, optimal NNLS
                # may produce weights that sum to >1 (boosting the
                # prediction); normalising would force a convex-
                # combination interpretation that loses the optimum.
                # Trade: the per-column weights are no longer
                # interpretable as "probabilities" / a "convex
                # combination", but the predictions stay optimal.
                weights[:, k] = w_k
            else:
                # Degenerate: NNLS returned all-zero (e.g. all
                # component preds are exactly zero for this column).
                # Fall back to equal-mean for THIS column only.
                weights[:, k] = 1.0 / n_comp
        self._weights = weights
        return self

    def predict(self, X) -> np.ndarray:
        preds_stack = []
        for c in self._components:
            p = np.asarray(c.predict(X))
            if p.ndim == 1:
                # Single-target component? Promote (N,) to (N, 1) so the
                # stack shape is consistent; downstream caller must
                # ensure all components have the same n_targets dimension.
                p = p.reshape(-1, 1)
            preds_stack.append(p)
        stacked = np.stack(preds_stack, axis=0)  # (n_components, N, K)
        # Per-column weighted sum: (n_comp, N, K) @ (n_comp, K) -> (N, K).
        # Uses einsum so the per-column weight is applied to the
        # matching column of each component's preds; equal-mean
        # collapses to stacked.mean(axis=0) by construction.
        return np.einsum("cnk,ck->nk", stacked, self._weights)

    def __repr__(self) -> str:
        return (
            f"MTRPerColumnEqualMeanEnsemble("
            f"n_components={len(self._components)}, "
            f"n_targets={self._n_targets}, "
            f"strategy={self._strategy!r}, "
            f"components={self._component_names!r})"
        )


def _build_mtr_per_column_ensemble(
    *, _tt_e, _orig_tname, models, metadata, target_by_type,
    oof_weights=None,
) -> None:
    """Build a per-column ensemble for an MTR target.

    Strategy auto-selected:
      * ``oof_weights`` (n_components, n_targets) supplied -> inject honest train-K-fold OOF NNLS weights.
      * Otherwise -> equal_mean default.

    Mutates ``models`` and ``metadata`` in place to mirror the single-target CT_ENSEMBLE registration shape.
    """
    _orig_entries = (models or {}).get(_tt_e, {}).get(_orig_tname, []) or []
    _components: list[Any] = []
    _component_names: list[str] = []
    for _i, _entry in enumerate(_orig_entries):
        _inner = getattr(_entry, "model", None) or _entry
        if not hasattr(_inner, "predict"):
            continue
        _pp = getattr(_entry, "pre_pipeline", None)
        _name = f"raw#{_i}"
        _components.append(PrePipelinePredictShim(_inner, _pp, _name))
        _component_names.append(_name)

    if len(_components) < 2:
        logger.info(
            "[MTR CT_ENSEMBLE] target='%s': only %d component(s) "
            "available; need >=2 for an ensemble. Skipping.",
            _orig_tname, len(_components),
        )
        return

    # Probe K (n_targets) from the routed target_by_type entry.
    try:
        _y_full = (target_by_type or {}).get(_tt_e, {}).get(_orig_tname)
        _y_arr = np.asarray(_y_full) if _y_full is not None else None
        _K = int(_y_arr.shape[1]) if _y_arr is not None and _y_arr.ndim == 2 else 1
    except Exception:
        _K = 1

    # Honest-OOF NNLS when valid precomputed weights are supplied; equal_mean otherwise.
    _use_nnls = (
        oof_weights is not None
        and getattr(oof_weights, "shape", None) == (len(_components), _K)
    )
    if oof_weights is not None and not _use_nnls:
        logger.warning(
            "[MTR CT_ENSEMBLE] target='%s': supplied OOF weights shape %s != (%d, %d); using equal-mean.",
            _orig_tname, getattr(oof_weights, "shape", None), len(_components), _K,
        )
    _strategy_label = "per_column_nnls_oof" if _use_nnls else "per_column_equal_mean"
    _ensemble_model = MTRPerColumnEqualMeanEnsemble(
        components=_components,
        component_names=_component_names,
        n_targets=_K,
        weights=(oof_weights if _use_nnls else None),
    )
    _ens_entry = SimpleNamespace(
        model=_ensemble_model,
        pre_pipeline=None,
        # mirror the structural keys the standard CT_ENSEMBLE entry
        # carries so downstream consumers that look for these attrs
        # don't KeyError
        model_name=f"CT_ENSEMBLE_MTR[{','.join(_component_names)}]",
        target_name=_orig_tname,
        ct_ensemble=True,
        mtr_ensemble=True,
        ensemble_strategy=_strategy_label,
        n_components=len(_components),
        component_names=tuple(_component_names),
    )
    models.setdefault(_tt_e, {}).setdefault(_orig_tname, []).append(_ens_entry)
    metadata.setdefault("mtr_ct_ensemble", {}).setdefault(
        str(_tt_e), {})[_orig_tname] = {
        "strategy": _strategy_label,
        "n_components": len(_components),
        "component_names": list(_component_names),
        "n_targets": _K,
    }
    logger.info(
        "[MTR CT_ENSEMBLE] target='%s' (K=%d): %s ensemble built "
        "over %d components: %s.",
        _orig_tname, _K, _strategy_label, len(_components), _component_names,
    )
