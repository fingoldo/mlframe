"""Cluster-aggregate exclusion stage of ``_assign_support``, split out to keep that function under its length ceiling."""

from __future__ import annotations


def _resolve_cluster_aggregate_exclusions(self, selected_vars, cols, cached_MIs):
    """Stage of ``_assign_support``."""
    # Imported here, not at module top: ``_assign_support`` imports this module, so a top-level import would be circular.
    from ._assign_support import _collect_cluster_representative_keeps, _strip_pseudo_remix_aggregate_members

    _ca_final_excl = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
    _cm_final = getattr(self, "cluster_members_", None)
    _raw_names_cmfinal = set(self.feature_names_in_)
    # Raw cluster representatives a DCD-aggregate-anchor swap would otherwise strip: force-kept / force-ADDED
    # below so every collapsed cluster retains >=1 raw column. Initialised at function-body level (NOT inside
    # the ``isinstance(_cm_final, dict)`` block) because it is referenced unconditionally further down - a
    # fit whose ``cluster_members_`` is not a dict (e.g. dcd_enable=False) must not hit an UnboundLocalError.
    _ca_keep_raw: set = set()
    if isinstance(_cm_final, dict):
        _collect_cluster_representative_keeps(self, _cm_final, _raw_names_cmfinal, selected_vars, cols, cached_MIs, _ca_final_excl, _ca_keep_raw)
    # PSEUDO-REMIX SELF-SOURCE PROTECTION. A conditional-gate / binned-numeric-agg /
    # row-argmax anchor (``gate_mask__a__b`` / ``binagg_mean(d|qbin(a))`` / ``argmax__a__b``) is a
    # LOSSY threshold/binning RE-MIX of its raw source(s): it cannot carry a raw operand's private
    # LINEAR term (a binary gate of ``a`` does not span ``10*a``). When the clustering folds a RAW
    # column into a cluster ANCHORED by such a pseudo-remix BUILT FROM that raw and strips the raw as
    # a "member", a genuine private term is lost (test_private_raw_a_kept: raw ``a`` with a dominant
    # ``10*a`` term clustered under ``gate_mask__a__b`` and dropped). Mirror the redundancy gate's
    # ``_is_pseudo_remix_child`` exclusion here: never strip a RAW column that the cluster pairs with
    # a pseudo-remix BUILT FROM that raw, in EITHER direction -
    #   (A) pseudo-remix ANCHOR + raw-source MEMBER  (``gate_mask__a__b`` anchors raw ``a``); or
    #   (B) raw ANCHOR + pseudo-remix MEMBER of it    (``x2`` anchors ``gate_mask__x2__x1``).
    # The lossy gate/binagg/argmax cannot carry the raw's continuous value a LINEAR downstream needs
    # (measured: a 5-class LogReg macro-F1 0.62 when x2 was stripped as such a cluster anchor vs >0.70
    # protected; and the test_private_raw_a_kept ``10*a`` case for direction A). Engineered members +
    # genuine (non-pseudo) aggregate members are untouched -> byte-identical when no such pairing exists.
    _ca_final_excl = _strip_pseudo_remix_aggregate_members(self, _ca_final_excl, _cm_final)
    return _ca_final_excl, _ca_keep_raw
