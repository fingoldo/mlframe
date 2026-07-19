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

# gt_08 auto-path-ONLY SNR tightening. The opt-in ``su_seeded_interactions=True`` flag keeps its
# original defaults (snr_z=3.0 / snr_abs_floor=1e-3, unchanged) -- these only apply when
# ``proxy_mode="auto"`` engaged the screen AND the caller left the opt-in knobs at their factory
# default (an explicit override always wins). Pre-flip 6-bed x 3-seed bench
# (_benchmarks/bench_shap_interaction_proxy.py) measured one spurious gate fire on the
# ``additive_redundant`` bed at the factory snr_z=3.0 (best_synergy=0.024489 vs gate=0.022797, a
# near-miss driven by the redundant near-duplicate columns' correlated marginal SU); snr_z=4.5
# silenced it (gate rose to 0.025961) while leaving every genuine-synergy bed (xor2, mixed, mult,
# xor_distract) fully detected across all 3 seeds -- see that bench's committed output.
_AUTO_SNR_Z_DEFAULT = 3.0
_AUTO_SNR_Z_TIGHTENED = 4.5
_AUTO_SNR_ABS_FLOOR_DEFAULT = 1e-3


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
    proxy_mode: str
    su_seeded_interactions: bool
    su_seeded_snr_z: float
    su_seeded_snr_abs_floor: float
    random_state: int
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

    def _su_screen_enabled(self) -> bool:
        """Resolve whether the su_seeded synergy screen must run for this fit.

        True when the caller pinned ``su_seeded_interactions=True`` explicitly, OR when
        ``proxy_mode="auto"`` -- the auto mode ALWAYS runs the screen (its built-in permutation-null
        SNR gate is the safe-condition detector: empty ``kept`` reproduces the additive path
        byte-identically, non-empty ``kept`` enables the same rescue/augmentation ``su_seeded_
        interactions=True`` already ships). Every call site that previously read
        ``self.su_seeded_interactions`` directly to gate the screen/rescue/augmentation must read this
        instead, so "auto" and the opt-in flag share one behaviour path.
        """
        return bool(self.su_seeded_interactions) or str(self.proxy_mode).lower() == "auto"

    def _su_screen_snr_z(self) -> float:
        """Resolve the su_seeded screen's ``snr_z`` gate, tightened for the auto-only path.

        The opt-in ``su_seeded_interactions=True`` flag always gets its literal ``su_seeded_snr_z``
        (explicit opt-in keeps its historical defaults byte-for-byte). ``proxy_mode="auto"`` without
        the explicit flag tightens the default (see ``_AUTO_SNR_Z_TIGHTENED``) to silence the one
        spurious gate fire the pre-flip bench measured on a redundant-column bed; a caller who
        explicitly pins ``su_seeded_snr_z`` still wins even under "auto".
        """
        if not self.su_seeded_interactions and str(self.proxy_mode).lower() == "auto" and float(self.su_seeded_snr_z) == _AUTO_SNR_Z_DEFAULT:
            return _AUTO_SNR_Z_TIGHTENED
        return float(self.su_seeded_snr_z)

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
        """Coerce ``X`` to a pandas DataFrame for the booster/SHAP pipeline: polars frames go through the zero-copy Arrow view, plain ndarrays get synthetic ``f0..fN`` column names, pandas frames pass through unchanged."""
        try:
            import polars as pl
        except ImportError:
            pl = None  # type: ignore[assignment]  # optional-dependency sentinel, guarded by the `pl is not None` check below
        if pl is not None and isinstance(X, pl.DataFrame):
            from mlframe.training.utils import get_pandas_view_of_polars_df

            return get_pandas_view_of_polars_df(X)
        if isinstance(X, pd.DataFrame):
            return X
        arr = np.asarray(X)
        return pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

    def _coerce_target(self, y):
        """Coerce ``y`` to the 1D float64 target the booster + numba proxy-loss kernels require, rejecting multi-output ``y`` with a clear error (rather than an opaque numba TypingError downstream); for ``classification=True`` also validates exactly 2 classes and binarises against ``classes_``."""
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
        """Subset ``X`` to the fitted ``selected_features_`` columns; raises ``NotFittedError`` if called before ``fit``."""
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
        """Fit the selector on ``(X, y)`` then return ``X`` subset to the selected features."""
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False):
        """Boolean support mask over the original input columns (or, if ``indices=True``, the integer positions where it is True); raises ``NotFittedError`` if called before ``fit``."""
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "support_"):
            raise NotFittedError("ShapProxiedFS.get_support called before fit.")
        return np.where(self.support_)[0] if indices else self.support_

    def get_feature_names_out(self, input_features=None):
        """Selected feature names (sklearn ``get_feature_names_out`` convention); raises ``NotFittedError`` if called before ``fit``."""
        from sklearn.exceptions import NotFittedError

        if not hasattr(self, "selected_features_"):
            raise NotFittedError("ShapProxiedFS.get_feature_names_out called before fit.")
        return np.asarray(self.selected_features_, dtype=object)
