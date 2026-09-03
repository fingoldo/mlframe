"""MODELS-14 (2026-08-05 audit): create_ctr_params's CrossEntropy check on ``params["loss_function"]`` must be
exact-value membership, not a substring test. A bare string is iterable char-by-char in Python, so
``"CrossEntropy" in "QueryCrossEntropy"`` is True (substring match) even though QueryCrossEntropy is a
distinct loss function that DOES support TargetBorderCount.
"""

from __future__ import annotations

import random as _stdlib_random

from mlframe.models.tuning import create_ctr_params


def test_query_cross_entropy_string_does_not_falsely_match_cross_entropy():
    """loss_function='QueryCrossEntropy' (a bare string) must NOT be treated as CrossEntropy."""
    forced_rng = _stdlib_random.Random(0)
    forced_rng.random = lambda: 0.9  # force the generate_valid_candidates branch to run

    res_query = create_ctr_params(GPU_ENABLED=False, params={"loss_function": "QueryCrossEntropy"}, stdlib_rng=forced_rng, random_state=0)
    forced_rng2 = _stdlib_random.Random(0)
    forced_rng2.random = lambda: 0.9
    res_cross = create_ctr_params(GPU_ENABLED=False, params={"loss_function": "CrossEntropy"}, stdlib_rng=forced_rng2, random_state=0)

    query_has_target_border_count = any("TargetBorderCount" in line for line in (res_query or []))
    cross_has_target_border_count = any("TargetBorderCount" in line for line in (res_cross or []))
    assert query_has_target_border_count, "QueryCrossEntropy must still allow TargetBorderCount (distinct loss function)"
    assert not cross_has_target_border_count, "CrossEntropy must exclude TargetBorderCount"
