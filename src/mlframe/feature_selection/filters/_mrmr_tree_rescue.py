"""MRMRTreeRescued: MRMR with a gated tree-importance rescue for its selection-gate collapse on interaction data.

MRMR's marginal-MI greedy STRUCTURALLY under-selects on interaction-heavy data: a pure-interaction operand
(``y = a*b``, XOR, sign products) has ~zero MARGINAL MI per operand, so the greedy never selects it and the feature
is lost. This is the SELECTION gate, not FE or binning (confirmed by the MDLP-collapse diagnostic + the FE-cap bench:
raising the FE synergy cap or injecting tree products does NOT help - the greedy discards them). On the standard
madelon FS benchmark MRMR collapses to <=3 features (downstream lgbm 0.69 vs 0.87 on all features).

This subclass adds a cheap, gated RESCUE: after a normal MRMR fit, when the selection is small relative to a WIDE
feature pool (the under-selection regime), it fits one shallow gradient-boosted tree - which branches on the
informative operands regardless of their marginal MI - and UNIONS its top-K importance features into ``support_``.
Gated to the collapse regime, so it is a BYTE-IDENTICAL no-op wherever MRMR already selects well.

MEASURED (3-seed): madelon mrmr_fe 0.6885 -> +rescue 0.7999 (+0.111, all seeds +0.10..0.12, std 0.0084);
synth and hard_synth byte-identical no-ops (the gate does not fire). The rescue extends ``support_`` only, so
``transform`` / ``get_feature_names_out`` / ``get_support`` flow it through unchanged. (round4_mrmr_tree_rescue_bench
/ round4_mrmr_rescue_confirm.)
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pandas as pd

from .mrmr import MRMR

logger = logging.getLogger(__name__)


def _rank_by_importance(imp) -> list:
    """Indices of positive importances, highest first; ties keep column order (a reversed ascending argsort preferred the highest index)."""
    imp = np.asarray(imp, dtype=float)
    return [int(i) for i in np.argsort(-imp, kind="stable") if imp[i] > 0]


class MRMRTreeRescued(MRMR):
    """MRMR + a gated shallow-GBM importance rescue for the under-selection (interaction-heavy) regime.

    Extra params (all else as MRMR):
      tree_rescue : "auto" (default) fires the rescue only when MRMR under-selects on a wide frame; True/"always"
        fires it whenever p > tree_rescue_min_p; False/"off"/None disables it (behaves exactly like MRMR).
      tree_rescue_top_k : number of shallow-GBM importance features to union into the selection (default 20).
      tree_rescue_min_p : only consider the rescue when n_features_in_ exceeds this (default 60 - narrow frames
        do not have the marginal-MI blind spot at scale and MRMR selects fine).
      tree_rescue_min_ratio / tree_rescue_min_features : "auto" under-selection threshold - fire when the raw
        selected count < max(tree_rescue_min_features, ceil(tree_rescue_min_ratio * p)) (default 5 / 0.04).
      tree_rescue_n_estimators / tree_rescue_max_depth : the shallow GBM (default 80 / 3 - cheap, ~0.4s).
    """

    # The rescue's own params; the rest are forwarded to MRMR (404 params - enumerating them in the signature would
    # be unmaintainable), so the ctor keeps **kwargs and we report the merged param set for sklearn introspection.
    _TREE_RESCUE_PARAMS = ("tree_rescue", "tree_rescue_top_k", "tree_rescue_min_p", "tree_rescue_min_ratio",
                           "tree_rescue_min_features", "tree_rescue_n_estimators", "tree_rescue_max_depth")

    def __init__(self, *args, tree_rescue="auto", tree_rescue_top_k: int = 20, tree_rescue_min_p: int = 60,
                 tree_rescue_min_ratio: float = 0.04, tree_rescue_min_features: int = 5,
                 tree_rescue_n_estimators: int = 80, tree_rescue_max_depth: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.tree_rescue = tree_rescue
        self.tree_rescue_top_k = int(tree_rescue_top_k)
        self.tree_rescue_min_p = int(tree_rescue_min_p)
        self.tree_rescue_min_ratio = float(tree_rescue_min_ratio)
        self.tree_rescue_min_features = int(tree_rescue_min_features)
        self.tree_rescue_n_estimators = int(tree_rescue_n_estimators)
        self.tree_rescue_max_depth = int(tree_rescue_max_depth)

    @classmethod
    def _get_param_names(cls):
        """Return the union of MRMR's param names and the rescue's own, so sklearn's get_params/set_params/clone round-trip correctly despite the varargs ctor."""
        # The varargs ctor hides params from sklearn's introspection; report MRMR's params + the rescue's own so
        # get_params / set_params / clone round-trip (clone reconstructs via **kwargs, which the ctor accepts).
        return sorted(set(MRMR._get_param_names()) | set(cls._TREE_RESCUE_PARAMS))

    # ------------------------------------------------------------------
    def _tree_rescue_should_fire(self) -> bool:
        """Decide whether the post-fit rescue should run, based on ``tree_rescue`` mode, pool width, and (in "auto" mode) whether MRMR under-selected relative to the collapse-regime threshold."""
        mode = self.tree_rescue
        if not mode or (isinstance(mode, str) and mode.lower() in ("off", "false", "none")):
            return False
        p = int(getattr(self, "n_features_in_", 0) or 0)
        if p <= self.tree_rescue_min_p:
            return False
        if mode is True or (isinstance(mode, str) and mode.lower() in ("always", "true")):
            return True
        # "auto": fire only on under-selection (the collapse regime)
        n_sel = int(np.asarray(getattr(self, "support_", np.array([]))).size)
        floor = max(self.tree_rescue_min_features, math.ceil(self.tree_rescue_min_ratio * p))
        return n_sel < floor

    def _apply_tree_rescue(self, X, y):
        """Gated post-fit rescue: union the shallow-GBM importance top-K into ``support_``. No-op + safe on any error."""
        try:
            if not self._tree_rescue_should_fire():
                return
            import lightgbm as lgb
            from sklearn.utils.multiclass import type_of_target

            cols = list(getattr(self, "feature_names_in_", []))
            # numeric coercion (best-effort): the rescue augments the NUMERIC informative features the greedy missed.
            if hasattr(X, "columns"):
                Xf = X.reindex(columns=cols) if cols else X
                # Fast path for the common all-numeric frame: a single ``np.array(..., float, copy=True)`` gather +
                # in-place NaN->0 replaces the per-column ``apply(pd.to_numeric).fillna().to_numpy()`` (~3
                # passes). Byte-identical to the slow path on numeric frames (``to_numeric`` is a no-op there,
                # ``fillna(0.0)`` only touches NaN, never inf - matched by ``isnan``). A non-numeric column
                # makes ``asarray(float)`` raise; fall back to the lenient coerce-to-NaN-then-zero path so a
                # mixed frame still rescues exactly as before (the bad column becomes all-zeros, not a skip).
                # ``copy=True`` (not ``np.asarray``'s copy-if-needed) is required: pandas>=3's Copy-on-Write
                # makes ``np.asarray(df, dtype=float)`` return a READ-ONLY, non-owning view when the frame is
                # already float64 (no dtype conversion needed, so no copy is forced) -- the very next in-place
                # ``Xnum[np.isnan(Xnum)] = 0.0`` then raised ``ValueError: assignment destination is read-only``
                # on every all-numeric, NaN-containing frame under pandas 3.x (CI: py3.11/3.12 shards resolving
                # pandas==3.0.5). Silently fell through to the slow ``except`` path there instead of crashing
                # loudly (both branches are inside the same ``try``), so the fast path was quietly dead on the
                # entire pandas-3.x matrix leg -- always paying the ~3-pass ``apply``/``to_numeric``/``fillna``
                # cost this fast path exists to avoid.
                try:
                    # NaN is left in place: LightGBM handles missing values natively, whereas a 0.0 fill sits inside most features' range
                    # and the GBM splits on the imputed value as if it were observed.
                    Xnum = np.array(Xf, dtype=float, copy=True)
                except (ValueError, TypeError):
                    _coerced = Xf.apply(pd.to_numeric, errors="coerce")
                    _nan_fill = int(_coerced.isna().to_numpy().sum())
                    if _nan_fill:
                        logger.warning(
                            "tree_rescue: %d cell(s) across the feature frame were unparseable and coerced to 0.0 "
                            "(non-numeric columns become all-zero, biasing the rescue LGBM). Columns: %s",
                            _nan_fill, list(Xf.columns[_coerced.isna().any().to_numpy()]),
                        )
                    Xnum = _coerced.fillna(0.0).to_numpy(dtype=float)
            else:
                Xnum = np.asarray(X, dtype=float)
            if Xnum.shape[1] != int(self.n_features_in_):
                return  # column mismatch (e.g. transformed input) -> skip rescue, keep MRMR's selection
            yv = np.asarray(y).ravel()
            is_clf = type_of_target(yv) in ("binary", "multiclass")
            seed = int(self._effective_random_seed() or 0)
            Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
            m = Est(n_estimators=self.tree_rescue_n_estimators, max_depth=self.tree_rescue_max_depth,
                    num_leaves=2 ** self.tree_rescue_max_depth, learning_rate=0.1,
                    n_jobs=getattr(self, "n_jobs", -1), verbose=-1, random_state=int(seed))
            # ``factors_to_use`` restricts the FIT, not just the ranking. Fitting on the full frame and filtering afterwards let excluded
            # columns consume split budget and shift the importances of the allowed ones, so the exclusion was a display filter for this path.
            allowed = getattr(self, "factors_to_use", None)
            allowed = None if allowed is None else sorted({int(a) for a in allowed if 0 <= int(a) < Xnum.shape[1]})
            if allowed is not None and not allowed:
                return  # every column excluded: nothing to rescue from
            _fit_cols = np.asarray(allowed, dtype=np.int64) if allowed is not None else None
            m.fit(Xnum if _fit_cols is None else Xnum[:, _fit_cols], yv)
            _imp_fit = np.asarray(m.feature_importances_, dtype=float)
            if _fit_cols is None:
                imp = _imp_fit
            else:
                # Scatter back to full width so the ranking, the top-k cut and the logged shares all speak in the caller's column indices.
                imp = np.zeros(Xnum.shape[1], dtype=float)
                imp[_fit_cols] = _imp_fit
            order = _rank_by_importance(imp)
            if allowed is not None:
                _allowed_set = set(allowed)
                order = [i for i in order if i in _allowed_set]
            order = order[: self.tree_rescue_top_k]
            existing = {int(i) for i in np.asarray(self.support_, dtype=np.int64)}
            added = [i for i in order if i not in existing]
            if added:
                self.support_ = np.concatenate([np.asarray(self.support_, dtype=np.int64), np.asarray(added, dtype=np.int64)])
                self.n_features_ = int(self.support_.size)
                # Report what each rescued feature was rescued ON. The importance is IN-SCREEN: the GBM saw the same rows MRMR did, so it
                # carries a winner's-curse bias and is evidence to read sceptically, not a held-out gain. Saying so is the point: a count and
                # a list of names gave the reader no way to tell whether a rescued feature was worth anything.
                _imp_total = float(imp.sum())
                _shares = {int(i): (float(imp[i]) / _imp_total if _imp_total > 0.0 else 0.0) for i in added}
                self.tree_rescue_importances_ = {(str(cols[i]) if cols else int(i)): _shares[int(i)] for i in added}
                logger.info(
                    "[MRMR] tree-rescue: under-selected (%d of %d raw) -> added %d shallow-GBM feature(s) on IN-SCREEN importance "
                    "(same rows as the fit, so biased upward; not a held-out gain): %s",
                    len(existing),
                    int(self.n_features_in_),
                    len(added),
                    ", ".join(f"{str(cols[i]) if cols else i}={_shares[int(i)]:.4f}" for i in added[:8]),
                )
        except Exception as e:  # never let the rescue break a successful MRMR fit
            warnings.warn(f"MRMRTreeRescued: tree-rescue degraded ({type(e).__name__}: {e}); selection unchanged", stacklevel=2)

    def fit(self, X, y, *args, **kwargs):
        """Fit MRMR normally, then apply the gated tree-importance rescue on top of the resulting ``support_``."""
        super().fit(X, y, *args, **kwargs)
        self._apply_tree_rescue(X, y)
        return self
