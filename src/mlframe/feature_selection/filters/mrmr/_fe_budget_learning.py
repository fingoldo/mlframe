"""Per-family FE budget learning around one ``MRMR`` fit (``fe_budget_learning``).

Before the fit, a persisted per-family budget (keyed by dataset fingerprint) scales the triplet / quadruplet / adaptive-arity
seed_k/top_count quotas; after it, each family's credit per wall-second reallocates the budget and persists it. Carved out of
``_mrmr_class.py`` (1k-LOC budget). Both halves are best-effort: a failure logs a warning and leaves the fit unscaled / the
budget unchanged.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def apply_learned_fe_budgets(self, X, loaded_budgets, quota_snapshot: dict) -> None:
    """Scale the per-family quotas on ``self`` by the loaded (or equal-split) budget and record the pre-fit state the post-fit
    update needs. Every original quota value is written into ``quota_snapshot`` so the caller's ``finally`` can restore it."""
    try:
        from mlframe.feature_selection.filters._fe_family_budget import dataset_fingerprint as _fe_budget_fp, load_budgets as _fe_load_budgets

        _fe_budget_cols = list(X.columns) if hasattr(X, "columns") else [str(i) for i in range(np.asarray(X).shape[1])]
        self._fe_budget_fingerprint_ = _fe_budget_fp(len(_fe_budget_cols), _fe_budget_cols)
        if loaded_budgets is None:
            loaded_budgets = _fe_load_budgets(fingerprint=self._fe_budget_fingerprint_)
        # ``_FE_FAMILY_WALL`` (``_fe_family_timing.py``) is PROCESS-GLOBAL by design (nested
        # fits / composite-discovery passes accumulate into it so a whole-run summary via
        # ``log_fe_family_summary()`` reflects the whole suite) - it is never reset here
        # (that would break other, unrelated consumers of the same ledger). Instead, snapshot
        # it now so the post-fit block below can compute THIS FIT's own wall-time delta
        # (post-fit snapshot minus this one), not the process-cumulative total across every
        # fit since process start (which would make credit/wall ROI increasingly wrong the
        # longer a training service has been running - verified: without this delta, ROI
        # after several fits in the same process was dominated by stale history and the
        # learned budget stopped changing at all).
        from mlframe.feature_selection.filters._fe_family_timing import get_fe_family_wall as _fe_get_wall_pre

        self._fe_budget_wall_pre_fit_ = _fe_get_wall_pre()
        _fe_budget_quota_attrs = {
            "triplet": ("fe_hybrid_orth_triplet_seed_k", "fe_hybrid_orth_triplet_top_count"),
            "quadruplet": ("fe_hybrid_orth_quadruplet_seed_k", "fe_hybrid_orth_quadruplet_top_count"),
            "adaptive_arity": ("fe_hybrid_orth_adaptive_arity_seed_k",),
        }
        _fe_equal_share = 1.0 / max(len(_fe_budget_quota_attrs), 1)
        # Stash the base budget used THIS fit (loaded, or equal-split when nothing was
        # persisted yet) so the post-fit reallocation block below compounds on top of it
        # instead of restarting from equal-split every fit (which would make learning never
        # accumulate across successive fits).
        self._fe_budget_prev_ = dict(loaded_budgets) if loaded_budgets else {f: _fe_equal_share for f in _fe_budget_quota_attrs}
        if loaded_budgets:
            for _fam, _attrs in _fe_budget_quota_attrs.items():
                _fam_fraction = loaded_budgets.get(_fam, _fe_equal_share)
                _fam_scale = _fam_fraction / _fe_equal_share  # 1.0 at equal-split; <1 shrinks, >1 grows
                for _attr in _attrs:
                    _orig_val = int(getattr(self, _attr, 0) or 0)
                    quota_snapshot[_attr] = _orig_val
                    setattr(self, _attr, max(1, round(_orig_val * _fam_scale)))
            logger.info("[MRMR] fe_budget_learning: applied loaded budgets %s (fingerprint=%s).", loaded_budgets, self._fe_budget_fingerprint_)
    except Exception as _fe_budget_exc:
        logger.warning("[MRMR] fe_budget_learning: pre-fit budget load/scale failed (%s); proceeding with unscaled quotas.", _fe_budget_exc)


def update_fe_budgets_after_fit(self) -> None:
    """Reallocate and persist the per-family budget from this fit's credit and wall time; publishes ``self.fe_family_budget_``."""
    try:
        from mlframe.feature_selection.filters._fe_family_budget import (
            family_credit as _fe_family_credit,
            family_roi as _fe_family_roi,
            persist_budgets as _fe_persist_budgets,
            reallocate_budgets as _fe_reallocate_budgets,
        )
        from mlframe.feature_selection.filters._fe_family_timing import get_fe_family_wall as _fe_get_wall

        _fe_wall_post_fit = _fe_get_wall()
        _fe_wall_pre_fit = getattr(self, "_fe_budget_wall_pre_fit_", None) or {}
        # This fit's OWN wall delta, not the process-cumulative total (see the pre-fit
        # snapshot comment above for why the ledger itself is never reset).
        _fe_wall_snapshot = {
            _fam: (
                _post[0] - _fe_wall_pre_fit.get(_fam, (0.0, 0))[0],
                _post[1] - _fe_wall_pre_fit.get(_fam, (0.0, 0))[1],
            )
            for _fam, _post in _fe_wall_post_fit.items()
        }
        _fe_credit = _fe_family_credit(getattr(self, "fe_provenance_", None))
        _fe_roi = _fe_family_roi(_fe_credit, _fe_wall_snapshot)
        _fe_budget_kwargs = dict(getattr(self, "fe_budget_kwargs", None) or {})
        _fe_tracked_families = ("triplet", "quadruplet", "adaptive_arity")
        _fe_equal_share = 1.0 / len(_fe_tracked_families)
        # Compound on top of the budget actually used THIS fit (loaded pre-fit, or
        # equal-split when nothing was persisted yet) - restarting from equal-split every
        # fit would make learning never accumulate across successive fits.
        _fe_prev_budgets = dict(getattr(self, "_fe_budget_prev_", None) or {f: _fe_equal_share for f in _fe_tracked_families})
        for _fam in _fe_tracked_families:
            _fe_prev_budgets.setdefault(_fam, _fe_equal_share)
        _fe_budgets_before = dict(_fe_prev_budgets)
        _fe_budgets_after = _fe_reallocate_budgets(_fe_roi, base_budget=_fe_prev_budgets, **_fe_budget_kwargs)
        _fe_persist_budgets(_fe_budgets_after, fingerprint=getattr(self, "_fe_budget_fingerprint_", None))
        self.fe_family_budget_ = dict(
            wall=_fe_wall_snapshot,
            credit=_fe_credit,
            roi=_fe_roi,
            budgets_before=_fe_budgets_before,
            budgets_after=_fe_budgets_after,
        )
    except Exception as _fe_budget_post_exc:
        logger.warning("[MRMR] fe_budget_learning: post-fit credit/reallocate/persist failed (%s); no budget update this fit.", _fe_budget_post_exc)
