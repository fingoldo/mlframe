"""Audit 2026-11-20 finding 39-F4: a named target must never receive another target's persisted ensemble parameters."""

from mlframe.training.core.predict import _resolve_chosen_ensemble_params


def _meta():
    """One persisted entry, for target A only."""
    return {"ensembles_chosen_params": {"regression": {"A": {"rrf_k": 10}}}}


def test_named_target_missing_from_bucket_gets_no_params():
    """Target B is absent; the single entry belongs to A, so B gets the defaults ({}), not A's rrf_k."""
    assert _resolve_chosen_ensemble_params(_meta(), target_type="regression", target_name="B") == {}


def test_named_target_missing_without_target_type_gets_no_params():
    """Even with no target type, a named target absent from the map must not borrow the only entry."""
    assert _resolve_chosen_ensemble_params(_meta(), target_name="B") == {}


def test_no_target_asked_for_still_uses_the_only_entry():
    """With no target named, the single persisted entry is still the answer (legacy single-target models)."""
    assert _resolve_chosen_ensemble_params(_meta()) == {"rrf_k": 10}
