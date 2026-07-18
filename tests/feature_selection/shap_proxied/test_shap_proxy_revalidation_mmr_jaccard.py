"""Unit tests for the iter50 MMR Jaccard de-duplication of revalidation candidates.

Pure-Python contract checks: no fit, no joblib, no booster work. Validates that:
  - ``None`` (sentinel) routes to 0.3 at ``n_features >= 20000`` and disabled below.
  - Explicit user value overrides the auto for any width (including 0.0 = no dedup).
  - The auto threshold boundary is exactly 20000 (inclusive: >=20000 -> 0.3).
  - The MMR greedy filter keeps the first candidate, drops near-duplicates, and preserves order.
  - The defensive floor never returns an empty list, even when tau drops everything.
  - The filter is deterministic / stable given identical inputs.
"""

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def test_default_is_none_sentinel():
    """Default is none sentinel."""
    sel = ShapProxiedFS()
    assert sel.revalidation_mmr_jaccard_threshold is None


def test_auto_below_threshold_returns_none_disabled():
    """Auto below threshold returns none disabled."""
    sel = ShapProxiedFS()
    assert sel._resolve_revalidation_mmr_jaccard_threshold(1) is None
    assert sel._resolve_revalidation_mmr_jaccard_threshold(100) is None
    assert sel._resolve_revalidation_mmr_jaccard_threshold(19999) is None


def test_auto_at_and_above_threshold_returns_zero_point_three():
    """Auto at and above threshold returns zero point three."""
    sel = ShapProxiedFS()
    assert sel._resolve_revalidation_mmr_jaccard_threshold(20000) == 0.3
    assert sel._resolve_revalidation_mmr_jaccard_threshold(50000) == 0.3
    assert sel._resolve_revalidation_mmr_jaccard_threshold(100000) == 0.3


def test_user_pinned_overrides_auto_below_threshold():
    """User pinned overrides auto below threshold."""
    sel = ShapProxiedFS(revalidation_mmr_jaccard_threshold=0.5)
    assert sel.revalidation_mmr_jaccard_threshold == 0.5
    assert sel._resolve_revalidation_mmr_jaccard_threshold(500) == 0.5


def test_user_pinned_overrides_auto_above_threshold():
    """User pinned overrides auto above threshold."""
    sel = ShapProxiedFS(revalidation_mmr_jaccard_threshold=0.15)
    assert sel.revalidation_mmr_jaccard_threshold == 0.15
    assert sel._resolve_revalidation_mmr_jaccard_threshold(50000) == 0.15


def test_user_pinned_zero_is_no_dedup_baseline():
    # 0.0 = MMR disabled-equivalent: only an exact-duplicate (distance 0) would drop.
    """User pinned zero is no dedup baseline."""
    sel = ShapProxiedFS(revalidation_mmr_jaccard_threshold=0.0)
    assert sel.revalidation_mmr_jaccard_threshold == 0.0
    # 0.0 propagated as float, not coerced back to None
    assert sel._resolve_revalidation_mmr_jaccard_threshold(20000) == 0.0


def test_mmr_keeps_first_candidate_always():
    """Mmr keeps first candidate always."""
    candidates = [(0.10, [1, 2, 3]), (0.11, [1, 2, 3, 4]), (0.12, [9, 10])]
    # tau=0.3: cand_1 has Jaccard distance 1 - 3/4 = 0.25 to cand_0, dropped.
    # cand_2 has Jaccard distance 1.0 to cand_0 (no overlap) and 1.0 to dropped (irrelevant); kept.
    kept = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.3)
    assert kept[0] == 0
    assert 2 in kept
    assert 1 not in kept


def test_mmr_drops_near_duplicates_at_default_tau():
    # Two near-duplicates (overlap > 1-tau = 0.7) should collapse to first.
    """Mmr drops near duplicates at default tau."""
    candidates = [(0.10, [1, 2, 3, 4, 5]), (0.11, [1, 2, 3, 4, 6]), (0.12, [1, 2, 3, 4, 7])]
    kept = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.3)
    # Each pair has Jaccard distance 1 - 4/6 = 0.333... > 0.3, so all kept (boundary check).
    assert kept == [0, 1, 2]
    # At tau=0.5 (distance must EXCEED 0.5): 0.333 < 0.5 -> only first kept.
    kept2 = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.5)
    assert kept2 == [0]


def test_mmr_preserves_order():
    """Mmr preserves order."""
    candidates = [(0.1, [1]), (0.2, [2]), (0.3, [3]), (0.4, [4])]
    # All disjoint -> all kept, in input order.
    kept = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.3)
    assert kept == [0, 1, 2, 3]


def test_mmr_defensive_floor_never_empty():
    # tau just below 1.0 would drop everything except the seed; tau exactly 1.0 keeps only seed.
    """Mmr defensive floor never empty."""
    candidates = [(0.1, [1, 2]), (0.2, [3, 4]), (0.3, [5, 6])]
    # tau=1.0: distance must exceed 1.0 (impossible) -> only seed kept (the defensive baseline).
    kept = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 1.0)
    assert kept == [0]
    # Empty input still returns empty (no defensive seed if nothing to seed with).
    assert ShapProxiedFS._mmr_filter_by_jaccard([], 0.3) == []


def test_mmr_tau_zero_keeps_all_non_identical():
    # tau=0.0 means "drop only exact duplicates" (distance must exceed 0 strictly).
    """Mmr tau zero keeps all non identical."""
    candidates = [(0.1, [1, 2]), (0.2, [1, 2, 3])]
    kept = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.0)
    # distance = 1 - 2/3 = 0.333 > 0 -> both kept.
    assert kept == [0, 1]
    # Exact duplicate input -> second one drops at tau=0.
    candidates_dup = [(0.1, [1, 2, 3]), (0.2, [1, 2, 3])]
    kept_dup = ShapProxiedFS._mmr_filter_by_jaccard(candidates_dup, 0.0)
    assert kept_dup == [0]


def test_mmr_deterministic_repeat_calls():
    # Stable, side-effect free: repeated calls yield identical kept indices.
    """Mmr deterministic repeat calls."""
    candidates = [(0.1, [1, 2, 3]), (0.11, [1, 2, 4]), (0.12, [5, 6, 7]), (0.13, [1, 2, 3, 8])]
    a = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.3)
    b = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.3)
    c = ShapProxiedFS._mmr_filter_by_jaccard(candidates, 0.3)
    assert a == b == c


def test_resolver_returns_float_type_when_enabled():
    """Resolver returns float type when enabled."""
    sel = ShapProxiedFS()
    out = sel._resolve_revalidation_mmr_jaccard_threshold(20000)
    assert isinstance(out, float)
    out2 = sel._resolve_revalidation_mmr_jaccard_threshold(19999)
    assert out2 is None


def test_mmr_does_not_perturb_ucb_or_other_knobs():
    # iter50 lever is independent of iter34/iter41 UCB knobs; setting MMR should not move them.
    """Mmr does not perturb ucb or other knobs."""
    sel = ShapProxiedFS()
    sel2 = ShapProxiedFS(revalidation_mmr_jaccard_threshold=0.2)
    assert sel.revalidation_ucb_enabled == sel2.revalidation_ucb_enabled
    assert sel._resolve_revalidation_ucb_stdev_multiplier(20000) == sel2._resolve_revalidation_ucb_stdev_multiplier(20000)
    assert sel.parsimony_tol == sel2.parsimony_tol
    assert sel.top_n == sel2.top_n
