"""Sensor: CB Pool cache MUST be cleared between suite calls.

Pre-fix shape (race + state-leak audit finding P0): _phase_config_setup.py
imported _CB_POOL_CACHE from mlframe.training.trainer module. That symbol was
a DEAD STUB at trainer.py:217 -- an empty dict that no code wrote to or read
from. The real CatBoost-train Pool cache lived in mlframe.training._cb_pool.

Result: suite-startup .clear() ran on the dead stub, succeeded silently, and
the LIVE cache kept its entries from the prior suite call. Python recycles
id() across suites; the next suite call could see an id() collision and fetch
a stale Pool with stale binned features + labels. CatBoost.fit short-circuits
on isinstance(X, Pool) so it didn't re-validate -- wrong model with no warning.

Post-fix: trainer.py aliases the real cache from _cb_pool, AND _phase_config_setup
imports directly from _cb_pool. Both routes scrub the live cache.
"""
from __future__ import annotations


def test_trainer_alias_points_to_live_cache():
    """trainer._CB_POOL_CACHE must be the SAME object as _cb_pool._CB_POOL_CACHE."""
    from mlframe.training import trainer
    from mlframe.training import _cb_pool
    assert trainer._CB_POOL_CACHE is _cb_pool._CB_POOL_CACHE, (
        "trainer._CB_POOL_CACHE must alias the live cache in _cb_pool, "
        "otherwise clear() at suite startup silently scrubs an empty dict."
    )


def test_trainer_side_clear_actually_scrubs_live_cache():
    """The exact pattern used by tests + _phase_config_setup pre-fix: trainer-side clear()
    must purge entries from the live cache."""
    from mlframe.training import trainer
    from mlframe.training import _cb_pool

    # Write to the live cache directly.
    _cb_pool._CB_POOL_CACHE.clear()  # start clean
    _cb_pool._CB_POOL_CACHE[("test_key",)] = "live_value"

    # Clear via the trainer-side name (the pre-fix call pattern).
    trainer._CB_POOL_CACHE.clear()

    # Live cache must now be empty -- pre-fix it would still hold "live_value" because
    # trainer.py:217 was a dead stub independent of _cb_pool._CB_POOL_CACHE.
    assert len(_cb_pool._CB_POOL_CACHE) == 0, (
        f"trainer-side clear didn't scrub live cache; pre-fix bug regression. "
        f"Live cache still has: {dict(_cb_pool._CB_POOL_CACHE)}"
    )


def test_phase_config_setup_imports_from_cb_pool():
    """The fix routes the import through _cb_pool directly. Verify the source
    file does NOT contain the stale ``from mlframe.training.trainer import _CB_POOL_CACHE``
    pattern -- if a future refactor reintroduces it, this test fails."""
    import pathlib
    # Derive path from installed package; previous hardcoded D:/ path
    # raised FileNotFoundError on any other machine.
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "_phase_config_setup.py"
    ).read_text(encoding="utf-8")
    assert "from mlframe.training._cb_pool import _CB_POOL_CACHE" in src, (
        "_phase_config_setup.py must import _CB_POOL_CACHE from _cb_pool (the live cache), "
        "not from trainer (which was the dead-stub source pre-fix)."
    )
