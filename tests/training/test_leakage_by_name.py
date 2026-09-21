"""A feature named like the targets is post-outcome-suspect even when it barely correlates with y.

A production run fed ``target_end_date`` (and its month / sin / cos decompositions) as FEATURES while training on
``target_total_hours`` / ``target_total_charge``; BaselineDiagnostics ranked ``target_end_date_month`` top-3. A contract's
end date is known only after the work is done, so the correlation-based leakage gate (|corr| >= 0.99) never fires on it.
"""

from __future__ import annotations

from mlframe.training.core._leakage_by_name import target_named_features


def test_shared_target_prefix_is_flagged():
    targets = ["target_total_hours", "target_total_charge", "target_hourly_rate"]
    features = ["target_end_date", "target_end_date_month", "budget_amount", "desc_len", "target_total_hours"]
    assert target_named_features(features, targets) == ["target_end_date", "target_end_date_month"]


def test_a_target_is_never_flagged_as_its_own_leak():
    assert target_named_features(["y"], ["y"]) == []


def test_derived_from_a_single_target_name_is_flagged():
    assert target_named_features(["y_month", "x"], ["y"]) == ["y_month"]


def test_unrelated_names_are_left_alone():
    targets = ["revenue", "revenue_growth"]
    assert target_named_features(["reviews_count", "region"], targets) == []


def test_no_targets_flags_nothing():
    assert target_named_features(["target_end_date"], []) == []
