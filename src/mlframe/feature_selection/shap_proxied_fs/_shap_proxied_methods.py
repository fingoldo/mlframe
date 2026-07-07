"""``ShapProxiedMethodsMixin`` -- resolver / coercion / preflight helpers for :class:`ShapProxiedFS`.

Carved out of ``shap_proxied_fs/__init__`` to keep the package facade under the 1k-LOC ceiling.
All methods operate on ``self`` (constructor state lives on the concrete class); heavy deps are
lazy-imported in-body as in the original, so this module needs only a few module-scope names.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_resolvers import _resolve_brute_force_n_sub_gate


class ShapProxiedMethodsMixin:
    """Resolver + coercion + preflight helpers for :class:`ShapProxiedFS` (see module docstring)."""

    # Constructor state lives on the concrete ``ShapProxiedFS`` class (see its ``__init__``); these
    # annotations declare the contract so mypy can type-check this mixin's methods on ``self``.
    booster_kind: Optional[str]
    brute_force_max_features: int
    classification: bool
    max_features: Optional[int]
    min_features: int
    optimizer: str
    use_gpu: bool
    revalidation_mmr_jaccard_threshold: Optional[float]
    revalidation_ucb_stdev_multiplier: Optional[float]
    classes_: Any
    # Provided by ``ShapProxiedFitMixin`` (the concrete class inherits both).
    fit: Callable[..., Any]

    def _resolve_booster_kind(self) -> str:
        """Pick the default booster family. ``None`` (default) auto-detects by availability so
        environments with only catboost installed still work without the user pinning the kwarg;
        an explicit user value always wins. Raises ``ValueError`` on an unknown kind so the typo
        surfaces at fit time instead of silently falling back."""
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _VALID_BOOSTER_KINDS

        if self.booster_kind is not None:
            kind = str(self.booster_kind).lower()
            if kind not in _VALID_BOOSTER_KINDS:
                raise ValueError(f"ShapProxiedFS: unknown booster_kind={self.booster_kind!r}; " f"expected one of {_VALID_BOOSTER_KINDS}.")
            return kind
        # Auto-detect: prefer xgboost (the historical default; SHAP fast paths are most mature
        # there), fall back to catboost when only catboost is installed.
        try:
            import xgboost  # noqa: F401
            return "xgboost"
        except ImportError:
            from mlframe.feature_selection.shap_proxied_fs._shap_proxy_catboost import catboost_available

            if catboost_available():
                return "catboost"
            raise ImportError("ShapProxiedFS: neither xgboost nor catboost is installed; install one or pass an " "explicit ``model=`` template.")

    def _resolve_revalidation_ucb_stdev_multiplier(self, n_features: int) -> float:
        """Width-dependent default for the revalidation UCB stdev multiplier.

        Explicit user value (set on the instance) always wins. ``None`` (the auto sentinel) routes
        to ``0.6`` at ``n_features >= 10000`` and ``1.0`` below. Tightening at wide regimes cuts
        revalidation wall-clock by firing the early-stop sooner; smaller-regime calibration
        (iter34 default 1.0) stays untouched because the proxy residual spread is narrower there.
        """
        # bench-attempt-rejected (iter48, 2026-05-29): tried a third step at width>=20000 -> k=0.4
        # on top of the iter41 0.6 step. C4 (width=20000) measured reval 8.96s -> 8.87s (1% on
        # ~9s, well under noise floor). The iter41 0.6 step already saturates the gate at C4:
        # post-prefilter the top_n=20 candidates' proxy_loss values cluster tightly (near-duplicate
        # SHAP-aware picks from the same 88-feature stage-B survivors), so the residual delta std
        # is already small and dropping k further does not push (proxy + slack) above best_so_far
        # for the un-evaluated tail. The reval wall remaining at ~9s is the floor: ucb_min_eval_size
        # default ``max(n_workers, 3)`` first batch (~8 candidates) + 1-2 post-batch dispatches at
        # the iter41 setting. Any further reval cut needs a different mechanism (smaller min_eval_size,
        # or a corrector-aware sort that breaks the proxy_loss tie), not k tightening.
        if self.revalidation_ucb_stdev_multiplier is not None:
            return float(self.revalidation_ucb_stdev_multiplier)
        return 0.6 if int(n_features) >= 10000 else 1.0

    def _resolve_revalidation_mmr_jaccard_threshold(self, n_features: int) -> float | None:
        """Width-dependent default for the iter50 MMR Jaccard de-duplication threshold.

        Explicit user value (set on the instance) always wins, including ``0.0`` (no dedup) and
        any value in ``(0, 1]``. ``None`` (auto sentinel) routes to ``0.3`` at
        ``n_features >= 20000`` (where iter49 measured >0.7 pairwise overlap among the top_n=20
        corrector-sorted candidates after the SHAP-aware cluster picks) and ``None`` (disabled)
        below, since smaller-width top_n is less redundant and the floor risk (dropping a winner
        in genuinely distinct candidates) outweighs the wall-clock gain.
        """
        if self.revalidation_mmr_jaccard_threshold is not None:
            return float(self.revalidation_mmr_jaccard_threshold)
        return 0.3 if int(n_features) >= 20000 else None

    @staticmethod
    def _mmr_filter_by_jaccard(candidates, tau: float) -> list[int]:
        """Greedy MMR dedup over an already-(corrector-)sorted candidate list.

        Returns the list of kept candidate indices (preserving the input order). A candidate is
        kept if its Jaccard distance to every previously-kept candidate exceeds ``tau``; otherwise
        it is dropped as a near-duplicate. Jaccard distance is ``1 - |A intersect B| / |A union B|``;
        ``tau=0`` keeps everything (only exact duplicates would drop), ``tau=1`` keeps only the
        first candidate. Defensive: if ``tau`` is misconfigured such that no second candidate
        clears the gate, the first candidate (corrector-sorted winner) is always kept; the caller
        will never receive an empty list. Candidates' feature indices (``c[1]``) are interpreted
        as sets of unit/feature ids.
        """
        kept: list[int] = []
        kept_sets: list[set] = []
        for i, cand in enumerate(candidates):
            s = set(int(x) for x in cand[1])
            if not kept_sets:
                kept.append(i)
                kept_sets.append(s)
                continue
            min_dist = 1.0
            for ks in kept_sets:
                union = len(s | ks)
                if union == 0:
                    dist = 0.0
                else:
                    dist = 1.0 - (len(s & ks) / union)
                if dist < min_dist:
                    min_dist = dist
                    if min_dist <= tau:
                        break
            if min_dist > tau:
                kept.append(i)
                kept_sets.append(s)
        # Defensive: keep at least the top-1 (sorted-by-corrector winner) if the gate dropped all.
        if not kept and candidates:
            kept.append(0)
        return kept

    @staticmethod
    def preflight(X, y, *, classification: bool = True, **kwargs):
        """Cheap "will-it-shine?" check BEFORE a full fit: returns a recommendation
        (run / caution / fallback) + dataset diagnostics + reasons. See ``_shap_proxy_preflight``."""
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_preflight import preflight as _preflight

        return _preflight(X, y, classification=classification, **kwargs)

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
        # ShapProxiedFS is single-output: the booster fit + OOF-SHAP + numba proxy-loss kernels all assume a 1D target.
        # A 2D y (multilabel / multi-output) silently survived np.unique/astype and only blew up deep inside numba with an
        # opaque TypingError; surface a clear error at fit entry instead.
        if y.ndim > 1:
            extra = "" if y.ndim != 2 or y.shape[1] != 1 else " (pass y.ravel() for a single-column 2D target)."
            raise ValueError(f"ShapProxiedFS supports a single-output 1D target only; got y with shape {y.shape}." + extra)
        if self.classification:
            classes = np.unique(y)
            if len(classes) != 2:
                raise ValueError(f"ShapProxiedFS(classification=True) supports binary targets only; got {len(classes)} classes.")
            self.classes_ = classes
            y = (y == classes[1]).astype(np.float64)
        else:
            y = y.astype(np.float64)
        return y

    def _resolve_optimizer(self, n_features: int) -> str:
        """Pick the optimizer for the post-prescreen candidate pool.

        At ``optimizer="auto"``, brute force is the preferred path when (a) the post-prescreen
        ``n_features`` is at or below the user's ``brute_force_max_features`` ceiling AND (b) the
        exhaustive subset count fits the dispatcher's ``brute_force_n_sub_gate`` feasibility cap.
        Otherwise the dispatcher falls back to beam over the SAME candidate pool. At default
        ``max_features=None`` the n_sub gate effectively caps brute-force dispatch at n<=26
        (2^26 = 67M < 80M default gate); n in {27, 28} runs beam over the wider prescreen pool.
        """
        opt = self.optimizer
        if opt != "auto":
            return opt
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import total_subsets

        if n_features <= self.brute_force_max_features:
            n_sub = total_subsets(n_features, self.min_features, self.max_features)
            if n_sub <= _resolve_brute_force_n_sub_gate():
                return "bruteforce_gpu" if self.use_gpu else "bruteforce"
        return "beam"

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

    def get_feature_names_out(self, input_features=None):
        """TODO C (2026-05-28): sklearn TransformerMixin convention. Returns the
        selected feature names; pairs with the existing ``get_support`` so both
        sklearn API surfaces work uniformly with the other mlframe selectors
        (MRMR has get_feature_names_out, RFECV has both, ShapProxied previously
        had only get_support). Surfaced by the shared FS contract suite.
        """
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError("ShapProxiedFS.get_feature_names_out called before fit.")
        return np.asarray(self.selected_features_, dtype=object)
