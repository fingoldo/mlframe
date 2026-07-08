"""MRMRTreeRescued: MRMR with a gated tree-importance rescue for its selection-gate collapse on interaction data.

MRMR's marginal-MI greedy STRUCTURALLY under-selects on interaction-heavy data: a pure-interaction operand
(``y = a*b``, XOR, sign products) has ~zero MARGINAL MI per operand, so the greedy never selects it and the feature
is lost. This is the SELECTION gate, not FE or binning (confirmed by the MDLP-collapse diagnostic + the FE-cap bench:
raising the FE synergy cap or injecting tree products does NOT help -- the greedy discards them). On the standard
madelon FS benchmark MRMR collapses to <=3 features (downstream lgbm 0.69 vs 0.87 on all features).

This subclass adds a cheap, gated RESCUE: after a normal MRMR fit, when the selection is small relative to a WIDE
feature pool (the under-selection regime), it fits one shallow gradient-boosted tree -- which branches on the
informative operands regardless of their marginal MI -- and UNIONS its top-K importance features into ``support_``.
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


class MRMRTreeRescued(MRMR):
    """MRMR + a gated shallow-GBM importance rescue for the under-selection (interaction-heavy) regime.

    Extra params (all else as MRMR):
      tree_rescue : "auto" (default) fires the rescue only when MRMR under-selects on a wide frame; True/"always"
        fires it whenever p > tree_rescue_min_p; False/"off"/None disables it (behaves exactly like MRMR).
      tree_rescue_top_k : number of shallow-GBM importance features to union into the selection (default 20).
      tree_rescue_min_p : only consider the rescue when n_features_in_ exceeds this (default 60 -- narrow frames
        do not have the marginal-MI blind spot at scale and MRMR selects fine).
      tree_rescue_min_ratio / tree_rescue_min_features : "auto" under-selection threshold -- fire when the raw
        selected count < max(tree_rescue_min_features, ceil(tree_rescue_min_ratio * p)) (default 5 / 0.04).
      tree_rescue_n_estimators / tree_rescue_max_depth : the shallow GBM (default 80 / 3 -- cheap, ~0.4s).
    """

    # The rescue's own params; the rest are forwarded to MRMR (404 params -- enumerating them in the signature would
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
                # Fast path for the common all-numeric frame: a single ``np.asarray(..., float)`` gather +
                # in-place NaN->0 replaces the per-column ``apply(pd.to_numeric).fillna().to_numpy()`` (~3
                # passes). Byte-identical to the slow path on numeric frames (``to_numeric`` is a no-op there,
                # ``fillna(0.0)`` only touches NaN, never inf -- matched by ``isnan``). A non-numeric column
                # makes ``asarray(float)`` raise; fall back to the lenient coerce-to-NaN-then-zero path so a
                # mixed frame still rescues exactly as before (the bad column becomes all-zeros, not a skip).
                try:
                    Xnum = np.asarray(Xf, dtype=float)
                    if np.isnan(Xnum).any():
                        Xnum[np.isnan(Xnum)] = 0.0
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
            seed = getattr(self, "random_seed", None) or getattr(self, "random_state", None) or 0
            Est = lgb.LGBMClassifier if is_clf else lgb.LGBMRegressor
            m = Est(n_estimators=self.tree_rescue_n_estimators, max_depth=self.tree_rescue_max_depth,
                    num_leaves=2 ** self.tree_rescue_max_depth, learning_rate=0.1,
                    n_jobs=getattr(self, "n_jobs", -1), verbose=-1, random_state=int(seed))
            m.fit(Xnum, yv)
            imp = np.asarray(m.feature_importances_, dtype=float)
            order = [int(i) for i in np.argsort(imp)[::-1] if imp[i] > 0][: self.tree_rescue_top_k]
            # respect factors_to_use if the user restricted the pool
            allowed = getattr(self, "factors_to_use", None)
            if allowed is not None:
                allowed = set(int(a) for a in allowed)
                order = [i for i in order if i in allowed]
            existing = {int(i) for i in np.asarray(self.support_, dtype=np.int64)}
            added = [i for i in order if i not in existing]
            if added:
                self.support_ = np.concatenate([np.asarray(self.support_, dtype=np.int64), np.asarray(added, dtype=np.int64)])
                self.n_features_ = int(self.support_.size)
                logger.info(
                    "[MRMR] tree-rescue: under-selected (%d of %d raw) -> added %d shallow-GBM " "importance feature(s) [%s]",
                    len(existing),
                    int(self.n_features_in_),
                    len(added),
                    ", ".join(str(cols[i]) for i in added[:8]) if cols else str(added[:8]),
                )
        except Exception as e:  # never let the rescue break a successful MRMR fit
            warnings.warn(f"MRMRTreeRescued: tree-rescue degraded ({type(e).__name__}: {e}); selection unchanged", stacklevel=2)

    def fit(self, X, y, *args, **kwargs):
        """Fit MRMR normally, then apply the gated tree-importance rescue on top of the resulting ``support_``."""
        super().fit(X, y, *args, **kwargs)
        self._apply_tree_rescue(X, y)
        return self
