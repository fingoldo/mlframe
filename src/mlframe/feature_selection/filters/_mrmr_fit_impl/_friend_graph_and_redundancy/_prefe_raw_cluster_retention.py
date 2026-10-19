"""Stages of ``_prefe_raw_sole_parent_pass``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._group1 import _engineered_operand_map
# --- end imports ---


def _prefe_raw_cluster_retention(self, cols, _substituted, selected_vars, _raw_names_set):
    """Stage of ``_prefe_raw_sole_parent_pass``."""
    _cm_for_raw_retention = getattr(self, "cluster_members_", None)
    if isinstance(_cm_for_raw_retention, dict):
        _dcd_for_swaps = getattr(self, "dcd_", None)
        _swapped_idx = {int(_e["anchor"]) for _e in ((_dcd_for_swaps or {}).get("swap_log") or []) if isinstance(_dcd_for_swaps, dict) and _e.get("anchor") is not None}
        _swapped_names = {cols[_i] for _i in _swapped_idx if 0 <= _i < len(cols)}
        for _anchor, _members in _cm_for_raw_retention.items():
            if _anchor not in _swapped_names:
                continue
            _substituted.add(_anchor)
            if isinstance(_members, (list, tuple, set)):
                _substituted.update(_members)
    # MULTI-PARENT OPERAND SCOPE (guards a past regression): a raw feature that
    # is an OPERAND of a SURVIVING multi-parent engineered feature is NOT covered
    # by the sole-parent ``_substituted`` exclusion above, so the original blanket
    # re-add resurrected EVERY such operand - including ones whose entire signal
    # flowed into the engineered child (e.g. ``y = a**2/b + log(c)*sin(d)``: raw
    # ``a, c, d`` carry NO information about ``y`` beyond ``div(sqr(a),abs(b))`` and
    # ``mul(log(c),sin(d))``, yet were re-added with ``support_rank -1`` and no gain,
    # padding the support with three redundant columns). The post-FE re-selection
    # ALREADY judged them redundant via the Fleuret conditional-MI redundancy term.
    # We restore the OLD CORRECT behaviour by deferring to that verdict for such
    # operands at large n (where the conditional-MI estimate is reliable), while
    # keeping the protective unconditional re-add at small n (the regime the
    # protection was built and validated for) and for raws NOT consumed by any
    # surviving engineered feature (the originally-intended absorbed-by-unrelated case).

    # Map each raw-operand name -> list of surviving ENGINEERED survivor column indices that consume it.
    _eng_operands_of: dict = {}  # raw_name -> list[engineered survivor col idx]
    _engineered_operand_map(selected_vars, cols, _raw_names_set, _eng_operands_of)
    # Sample-size scope: the small-n regime the protection was BUILT and
    # validated for (n=500 / 2000 / 3000 fixtures). At large n the post-FE re-selection's
    # conditional-MI redundancy term is statistically reliable - its drop of a redundant
    # operand IS the OLD CORRECT behaviour - so we do NOT override it there. ``_RR_PROTECT_MAX_N``
    # sits well above the largest validated fixture (3000) and far below the regression case (1e5).
    return _eng_operands_of
