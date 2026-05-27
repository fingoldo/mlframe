"""ShapProxiedFS -- feature selection by ranking subsets via summed SHAP values.

Idea (Mazzanti / TDS, extended): train one model on all features, compute SHAP values once, then
approximate the OOS prediction of a model trained on any feature subset ``S`` by the coalition value
``base + sum_{j in S} phi_j``. Rank subsets by a proper ML metric of ``(y, proxy_pred)`` instead of
honestly retraining 2^n models, then honestly re-validate the cheap top-N to pick the final subset.

Pipeline: fit big model -> OOF SHAP (per-fold base) -> proxy-rank subsets (exact numba/GPU brute
force for n<=~22, else beam/greedy/GA/annealing/gradient) -> proxy-trust guard -> honest re-validate
top-N on a disjoint holdout -> expose the sklearn selector contract.

Honest about limits: the proxy attributes the *full* model restricted to S, not a model retrained on
S, so it under-credits subsets that drop features whose signal correlated survivors could recover
(the empirical "<50% coverage breaks down" wall). Hence the trust guard, the disjoint-holdout
re-validation, the ``min_selected_ratio`` knob, and the importance-top-k ablation in the report.

sklearn contract mirrors BorutaShap: ``support_`` (bool mask in input-column order),
``selected_features_`` (names in input order), ``feature_names_in_``, ``n_features_in_``;
``transform`` uses name-based ``X.loc[:, selected]``; ``NotFittedError`` before fit.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

_EXACT_OPTIMIZERS = {"bruteforce", "bruteforce_gpu"}
_HEURISTIC_OPTIMIZERS = {"beam", "greedy_forward", "greedy_backward", "multistart", "genetic", "annealing", "gradient"}


class ShapProxiedFS(BaseEstimator, TransformerMixin):
    """SHAP-coalition-proxy feature selector (sklearn transformer)."""

    def __init__(
        self,
        model=None,
        classification: bool = True,
        metric: Optional[str] = None,
        optimizer: str = "auto",
        *,
        out_of_fold: bool = True,
        n_splits: int = 5,
        n_models: int = 1,
        min_features: int = 1,
        max_features: Optional[int] = None,
        top_n: int = 30,
        holdout_size: float = 0.25,
        revalidate: bool = True,
        n_revalidation_models: int = 3,
        lambda_stab: float = 0.5,
        parsimony_tol: float = 0.02,
        min_selected_ratio: float = 0.0,
        trust_guard: bool = True,
        n_anchors: int = 30,
        spearman_floor: float = 0.6,
        run_importance_ablation: bool = True,
        beam_width: int = 8,
        brute_force_max_features: int = 22,
        use_gpu: bool = False,
        n_jobs: int = -1,
        random_state: int = 0,
        verbose: bool = True,
        tqdm: bool = False,
    ):
        self.model = model
        self.classification = classification
        self.metric = metric
        self.optimizer = optimizer
        self.out_of_fold = out_of_fold
        self.n_splits = n_splits
        self.n_models = n_models
        self.min_features = min_features
        self.max_features = max_features
        self.top_n = top_n
        self.holdout_size = holdout_size
        self.revalidate = revalidate
        self.n_revalidation_models = n_revalidation_models
        self.lambda_stab = lambda_stab
        self.parsimony_tol = parsimony_tol
        self.min_selected_ratio = min_selected_ratio
        self.trust_guard = trust_guard
        self.n_anchors = n_anchors
        self.spearman_floor = spearman_floor
        self.run_importance_ablation = run_importance_ablation
        self.beam_width = beam_width
        self.brute_force_max_features = brute_force_max_features
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.tqdm = tqdm
        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _to_pandas(X):
        try:
            import polars as pl
        except ImportError:
            pl = None
        if pl is not None and isinstance(X, pl.DataFrame):
            from mlframe.training.utils import get_pandas_view_of_polars_df

            return get_pandas_view_of_polars_df(X)
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

    def _coerce_target(self, y):
        try:
            import polars as pl
            if isinstance(y, pl.Series):
                y = y.to_numpy()
        except ImportError:
            pass
        y = np.asarray(y)
        if self.classification:
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError(
                    f"ShapProxiedFS(classification=True) supports binary targets only; got {len(classes)} classes."
                )
            self.classes_ = classes
            y = (y == classes[1]).astype(np.float64)
        else:
            y = y.astype(np.float64)
        return y

    def _resolve_optimizer(self, n_features: int) -> str:
        opt = self.optimizer
        if opt != "auto":
            return opt
        from mlframe.feature_selection._shap_proxy_search import total_subsets

        if n_features <= self.brute_force_max_features:
            n_sub = total_subsets(n_features, self.min_features, self.max_features)
            if n_sub <= 2_000_000:
                return "bruteforce_gpu" if self.use_gpu else "bruteforce"
        return "beam"

    def _run_search(self, optimizer, phi, base, y):
        """Dispatch to the chosen optimizer; returns list of (proxy_loss, feature_idx tuple)."""
        kw = dict(classification=self.classification, metric=self.metric,
                  max_card=self.max_features, top_n=self.top_n)
        if optimizer == "bruteforce":
            from mlframe.feature_selection._shap_proxy_search import brute_force_top_n

            return brute_force_top_n(phi, base, y, min_card=self.min_features,
                                     parallel=(phi.shape[1] >= 14), **kw)
        if optimizer == "bruteforce_gpu":
            from mlframe.feature_selection._shap_proxy_gpu import brute_force_top_n_gpu, gpu_available

            if gpu_available():
                return brute_force_top_n_gpu(phi, base, y, min_card=self.min_features, **kw)
            logger.warning("ShapProxiedFS: use_gpu requested but no CUDA device; falling back to numba brute force.")
            from mlframe.feature_selection._shap_proxy_search import brute_force_top_n

            return brute_force_top_n(phi, base, y, min_card=self.min_features, parallel=True, **kw)
        from mlframe.feature_selection import _shap_proxy_heuristics as H

        if optimizer == "beam":
            return H.beam_search(phi, base, y, beam_width=self.beam_width, min_card=self.min_features, **kw)
        if optimizer == "greedy_forward":
            return H.greedy_forward(phi, base, y, classification=self.classification, metric=self.metric,
                                    max_card=self.max_features, top_n=self.top_n)
        if optimizer == "greedy_backward":
            return H.greedy_backward(phi, base, y, classification=self.classification, metric=self.metric,
                                     min_card=self.min_features, top_n=self.top_n)
        if optimizer == "multistart":
            return H.multistart_local(phi, base, y, rng=self._rng, **kw)
        if optimizer == "genetic":
            return H.genetic(phi, base, y, rng=self._rng, **kw)
        if optimizer == "annealing":
            return H.simulated_annealing(phi, base, y, rng=self._rng, **kw)
        if optimizer == "gradient":
            from mlframe.feature_selection._shap_proxy_gradient import gradient_top_n

            return gradient_top_n(phi, base, y, classification=self.classification, metric=self.metric,
                                  random_state=int(self.random_state), top_n=self.top_n)
        raise ValueError(f"Unknown optimizer={optimizer!r}")

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        from sklearn.base import clone

        from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

        X = self._to_pandas(X).reset_index(drop=True)
        X.columns = [str(c) for c in X.columns]
        self.feature_names_in_ = np.asarray(list(X.columns))
        self.n_features_in_ = int(X.shape[1])
        n_features = self.n_features_in_
        y = self._coerce_target(y)

        model_template = self.model if self.model is not None else make_default_estimator(
            self.classification, random_state=int(self.random_state))

        # Disjoint holdout for honest re-validation + trust guard (avoids winner's curse).
        stratify = y if self.classification else None
        idx_all = np.arange(len(X))
        idx_search, idx_hold = train_test_split(
            idx_all, test_size=self.holdout_size, random_state=int(self.random_state),
            shuffle=True, stratify=stratify)
        X_search, y_search = X.iloc[idx_search].reset_index(drop=True), y[idx_search]
        X_hold, y_hold = X.iloc[idx_hold].reset_index(drop=True), y[idx_hold]

        # OOF / in-sample SHAP attribution on the search set.
        phi, base, y_phi = compute_shap_matrix(
            model_template, X_search, y_search, classification=self.classification,
            out_of_fold=self.out_of_fold, n_splits=self.n_splits, n_models=self.n_models,
            rng=self._rng, tqdm_desc=("shap-oof" if self.tqdm else None))

        optimizer = self._resolve_optimizer(n_features)
        candidates = self._run_search(optimizer, phi, base, y_phi)

        # min_selected_ratio guard: the proxy degrades for small subsets (the <50% wall).
        if self.min_selected_ratio > 0:
            filtered = [(l, c) for l, c in candidates if len(c) / n_features >= self.min_selected_ratio]
            candidates = filtered or candidates  # never return empty
        if not candidates:
            raise RuntimeError("ShapProxiedFS: search produced no candidate subsets.")

        report: dict = dict(optimizer=optimizer, n_candidates=len(candidates),
                            proxy_best=dict(features=tuple(candidates[0][1]), proxy_loss=candidates[0][0]))

        # Proxy-trust diagnostic.
        if self.trust_guard:
            from mlframe.feature_selection._shap_proxy_revalidate import proxy_trust_guard

            report["trust"] = proxy_trust_guard(
                phi, base, y_phi, model_template, X_search, X_hold, y_hold,
                classification=self.classification, metric=self.metric, n_anchors=self.n_anchors,
                rng=self._rng, min_card=self.min_features, max_card=self.max_features,
                spearman_floor=self.spearman_floor, n_jobs=self.n_jobs)

        # Honest re-validation of the top-N on the disjoint holdout.
        if self.revalidate:
            from mlframe.feature_selection._shap_proxy_revalidate import revalidate_top_n

            best_idx, ranked, baseline = revalidate_top_n(
                candidates, model_template, X_search, y_search, X_hold, y_hold,
                classification=self.classification, metric=self.metric,
                n_models=self.n_revalidation_models, lambda_stab=self.lambda_stab,
                parsimony_tol=self.parsimony_tol, rng=self._rng, n_jobs=self.n_jobs)
            report["revalidation"] = dict(ranked=ranked[: self.top_n], random_baseline=baseline)
        else:
            best_idx = tuple(candidates[0][1])

        # Importance-top-k ablation (unique-value gate vs plain SHAP importance).
        if self.run_importance_ablation and best_idx:
            from mlframe.feature_selection._shap_proxy_revalidate import importance_topk_ablation

            report["importance_ablation"] = importance_topk_ablation(
                phi, best_idx, model_template, X_search, y_search, X_hold, y_hold,
                classification=self.classification, metric=self.metric)

        # Expose sklearn contract: names in INPUT column order.
        best_set = set(int(i) for i in best_idx)
        self.selected_features_ = [c for i, c in enumerate(self.feature_names_in_) if i in best_set]
        self.support_ = np.array([i in best_set for i in range(n_features)], dtype=bool)
        self.shap_proxy_report_ = report
        if self.verbose:
            logger.info("ShapProxiedFS: optimizer=%s selected %d/%d features: %s",
                        optimizer, len(self.selected_features_), n_features, self.selected_features_)
        return self

    # ------------------------------------------------------------------ transform
    def transform(self, X):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError("ShapProxiedFS.transform called before fit.")
        X = self._to_pandas(X)
        X.columns = [str(c) for c in X.columns]
        selected = list(self.selected_features_)
        if hasattr(X, "loc"):
            return X.loc[:, selected]
        return X[selected]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError("ShapProxiedFS.get_support called before fit.")
        return np.where(self.support_)[0] if indices else self.support_
