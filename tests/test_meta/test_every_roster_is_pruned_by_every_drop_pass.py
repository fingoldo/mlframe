"""Every engineered roster is pruned by every pass that drops engineered columns.

Two hand-maintained lists of roster attribute names did the same job in the same file: the Spearman-dedup pass filtered 18 of them and the
unified-gate pass below it filtered 27. A column the dedup dropped therefore stayed in the other nine rosters until a later reconciliation
happened to catch it, which it did, so nothing leaked to a user - but the divergence was invisible and one reconciliation pass away from
mattering, and every new FE family had to be added to both by hand.

Both passes read ``FE_ROSTER_ATTRS`` now, which is the tuple the package already maintains as the list of engineered rosters.
"""

from __future__ import annotations

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters" / "_mrmr_fit_impl"


def _roster_attrs():
    """The package's own list of engineered roster attribute names."""
    import sys

    sys.path.insert(0, str(_SRC.parents[3]))
    from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS

    return set(FE_ROSTER_ATTRS)


def _literal_roster_tuples(path: pathlib.Path):
    """Every tuple literal in the module that looks like a hand-written list of roster names."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Tuple, ast.List)):
            continue
        names = [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
        if len(names) >= 5 and all(n.endswith("_features_") or n.endswith("_candidates_") for n in names):
            out.append((node.lineno, names))
    return out


def test_no_pass_carries_its_own_hand_written_roster_list():
    """A second copy of the roster list is how the two passes drifted; there must not be one."""
    offenders = []
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "_fe_roster_attrs.py":
            continue  # the one place the list is allowed to live
        for lineno, names in _literal_roster_tuples(path):
            offenders.append(f"{path.name}:{lineno} ({len(names)} roster names)")
    assert not offenders, (
        "hand-written roster-name list(s) outside _fe_roster_attrs.py: " + ", ".join(offenders) + ". Iterate FE_ROSTER_ATTRS instead, so a "
        "new FE family is pruned by every drop pass without being added to each by hand."
    )


def test_the_shared_drop_prunes_every_roster():
    """``drop_from_fe_rosters`` (the one implementation both drop passes call) removes a dropped column from every roster and keeps the rest in order."""
    from types import SimpleNamespace

    from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS, drop_from_fe_rosters

    est = SimpleNamespace(**{name: ["keep_a", "gone", "keep_b"] for name in FE_ROSTER_ATTRS})
    drop_from_fe_rosters(est, {"gone"})
    assert all(getattr(est, name) == ["keep_a", "keep_b"] for name in FE_ROSTER_ATTRS)
    bare = SimpleNamespace()
    drop_from_fe_rosters(bare, {"gone"})
    assert all(getattr(bare, name) == [] for name in FE_ROSTER_ATTRS)


def test_the_shared_tuple_covers_what_the_passes_used_to_list():
    """Every roster either pass named by hand is still in the shared tuple, so nothing stopped being pruned."""
    attrs = _roster_attrs()
    previously_listed = {
        "hybrid_orth_features_", "mi_greedy_features_", "kfold_te_features_", "count_encoding_features_", "frequency_encoding_features_",
        "cat_num_interaction_features_", "missingness_indicator_features_", "missingness_count_features_", "missingness_pattern_features_",
        "pairwise_ratio_features_", "pairwise_log_ratio_features_", "grouped_delta_features_", "lagged_diff_features_",
        "grouped_agg_features_", "composite_group_agg_features_", "grouped_quantile_features_", "cat_pair_features_",
        "cat_triple_features_", "numeric_decompose_features_", "modular_features_", "group_distance_features_", "rare_category_features_",
        "conditional_residual_features_", "conditional_dispersion_features_", "wavelet_features_", "rankgauss_features_",
        "temporal_agg_features_",
    }
    missing = sorted(previously_listed - attrs)
    assert not missing, f"roster(s) the drop passes used to prune are absent from FE_ROSTER_ATTRS: {missing}"
