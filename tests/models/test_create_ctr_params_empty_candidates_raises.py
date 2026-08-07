"""MODELS-10 (2026-08-05 audit): ``create_ctr_params`` must raise a clear error if
``generate_valid_candidates`` returns no candidates for a main_type, instead of an opaque
``StopIteration`` from ``next(iter(cands))`` on an empty result.
"""

from __future__ import annotations

import pytest

from mlframe.models.tuning import create_ctr_params


def test_empty_candidates_raises_clear_error(monkeypatch):
    """generate_valid_candidates returning [] must raise ValueError, not StopIteration."""
    # create_ctr_params and generate_valid_candidates both actually live in tuning_rules.py --
    # tuning.py is a thin re-export facade (see that module's docstring), so patching the
    # facade's own generate_valid_candidates binding would not affect the real internal call.
    import mlframe.models.tuning_rules as mod

    monkeypatch.setattr(mod, "generate_valid_candidates", lambda *a, **k: [])
    # stdlib_rng.random() > 0.5 gate must be forced True so generate_valid_candidates is actually called.
    import random as _stdlib_random

    forced_rng = _stdlib_random.Random(0)
    monkeypatch.setattr(forced_rng, "random", lambda: 0.9)

    with pytest.raises(ValueError, match="no candidates"):
        create_ctr_params(GPU_ENABLED=False, stdlib_rng=forced_rng)
