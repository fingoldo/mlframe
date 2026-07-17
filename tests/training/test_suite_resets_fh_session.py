"""Regression test for fix #5: train_mlframe_models_suite must reset the FH session token.

Pre-fix ``main.py:175`` only called ``reset_phase_registry()``; the FH ``InMemoryKey`` session
(``feature_handling/fingerprint.py:_CURRENT_SESSION``) was NEVER rotated. Two consecutive suite
calls in the same process kept the same SessionToken, and any ``id(train_df)`` recycling between
suites could collide on a cached entry whose state belonged to the prior suite.

Post-fix the suite imports ``reset_session`` from the FH fingerprint module and calls it alongside
``reset_phase_registry()`` at suite entry. Each suite call gets a distinct SessionToken so cached
entries from a prior suite cannot accidentally hit.
"""

from __future__ import annotations


import pytest

from mlframe.training.core import main as core_main
from mlframe.training.feature_handling import (
    current_session,
    reset_session,
)


def test_main_module_imports_reset_session_from_fingerprint():
    """Structural smoke: the symbol must be wired into core.main as ``reset_fh_session``."""
    assert hasattr(core_main, "reset_fh_session"), (
        "core.main must import reset_session from feature_handling.fingerprint so suite entry can rotate the FH cache namespace"
    )
    # And it must point to the actual fingerprint helper (not some shadowed local).
    from mlframe.training.feature_handling.fingerprint import (
        reset_session as fp_reset,
    )

    assert core_main.reset_fh_session is fp_reset


def test_suite_entry_rotates_fh_session_token():
    """Behavioural: invoking the suite must produce a fresh SessionToken even if validation later
    raises. Pre-fix the token persisted; post-fix every suite entry rotates it.
    """
    # Pin the starting token, then call the suite with an invalid ``df`` so the function returns
    # via the early validation TypeError. The reset_phase_registry + reset_fh_session lines (175-)
    # run BEFORE the validation, so the session must have rotated by the time the call unwinds.
    reset_session()  # ensure a known starting point
    token_before = current_session().session_id

    with pytest.raises(TypeError, match="df must be pandas"):
        core_main.train_mlframe_models_suite(
            df=12345,  # invalid type triggers early TypeError after the resets
            target_name="y",
            model_name="m",
            features_and_targets_extractor=object(),  # not used before the df validation
        )

    token_after = current_session().session_id
    assert token_after != token_before, (
        "suite entry must rotate the FH session token; without it consecutive suites can collide on recycled id(train_df) for in-memory cache entries"
    )


def test_two_consecutive_suite_calls_produce_distinct_session_tokens():
    """End-to-end behavioural: two suite calls back-to-back must each get a fresh token."""
    seen = []

    def _capture(*args, **kwargs):
        # Capture the current token immediately after the validation-failing call returns; with
        # the fix, each call rotates the token before validation rejects the bogus df.
        with pytest.raises(TypeError):
            core_main.train_mlframe_models_suite(
                df=999,
                target_name="y",
                model_name="m",
                features_and_targets_extractor=object(),
            )
        seen.append(current_session().session_id)

    _capture()
    _capture()
    assert seen[0] != seen[1], "consecutive suite calls must produce distinct FH session tokens"
