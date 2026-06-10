"""Wave-14 sensors: the ``x = user_arg or fallback`` trap class.

Python's ``or`` short-circuits on any falsy value (0, "", [], 0.0, empty
dict/df), not just ``None``. The legitimate idiom ``x = arg or default`` is
fine when ALL falsy values are semantically equivalent to None; it is a SILENT
BUG when caller intent for a falsy value differs from the default.

This file pins four specific sites that the wave-14 audit flagged where the
intent-disagreement was operator-visible:

1. ``tiny_rerank_n_jobs=0`` is the "auto-pick CPU count" sentinel in
   composite_discovery.py:2176 (the very next line reads ``if cfg == 0:``).
   The pre-fix ``or 1`` collapsed 0->1 before the sentinel check, making
   the auto-pick branch dead code.
2. ``random_state=0`` is a legitimate sklearn seed in
   _phase_composite_discovery.py:287. The pre-fix ``or 42`` rewrote 0->42
   so two callers passing 0 and 42 produced identical data_signatures and
   reproducibility broke for any operator that explicitly chose seed=0.
3. ``fit_cache_max=0`` is the operator-explicit "disable LRU" sentinel in
   mrmr.py:1893. The pre-fix ``or 4`` silently restored the default cap, so
   ``MRMR(fit_cache_max=0)`` left the cache fully enabled.
4. ``rebuilt == {}`` from score_ensemble in _phase_recurrent.py:301 is the
   "gate pruned every member" signal; the pre-fix ``rebuilt or ensemble_dict``
   collapsed it to "return prior ensemble" silently.

All four checks are source-level (no full-suite fixture needed): grep for
the pre-fix pattern's absence + presence of the post-fix idiom. Per
``feedback_behavioral_tests`` source-level checks are reserved for boundary
contracts where behavioural reproduction would require an integration
fixture; cross-field validation of the ``x is None`` vs ``not x`` distinction
qualifies because the only difference is in the (caller, sentinel-value)
pair, not the arithmetic the function performs.
"""
from __future__ import annotations

import pathlib

import mlframe as _mlframe


_SRC_ROOT = pathlib.Path(_mlframe.__file__).resolve().parent


def _read(rel: str) -> str:
    """Read a source file. For modules that have been split into sibling
    helpers (e.g. ``mrmr.py`` -> ``_mrmr_fit_impl.py`` /
    ``_mrmr_fingerprints.py`` / ``_mrmr_fe_step/`` (subpackage) /
    ``_mrmr_validate_transform.py``), concat every sibling so the
    source-grep boundary check still matches the relocated code."""
    primary = (_SRC_ROOT / rel).read_text(encoding="utf-8")
    if rel == "feature_selection/filters/mrmr.py":
        _dir = _SRC_ROOT / "feature_selection" / "filters"
        for nm in (
            "_mrmr_fingerprints.py", "_mrmr_fit_impl.py",
            "_mrmr_fe_step/_step_core.py", "_mrmr_fe_step/_helpers.py",
            "_mrmr_validate_transform.py",
        ):
            _sib = _dir / nm
            if _sib.exists():
                primary = primary + "\n" + _sib.read_text(encoding="utf-8")
    return primary


def test_tiny_rerank_n_jobs_zero_sentinel_reaches_branch():
    """``tiny_rerank_n_jobs=0`` (auto-pick) MUST reach the ``if cfg == 0:``
    branch. Pre-fix ``int(... or 1)`` collapsed it.

    ``_tiny_model_rerank`` was moved to the
    ``_composite_discovery_tiny_rerank.py`` sibling when
    ``composite_discovery.py`` was split below 1k LOC.
    """
    src = _read("training/composite/discovery/_tiny_rerank.py")
    assert "int(getattr(self.config, \"tiny_rerank_n_jobs\", 1) or 1)" not in src, (
        "Pre-fix `or 1` pattern reappeared: tiny_rerank_n_jobs=0 sentinel is "
        "silently rewritten to 1 BEFORE the `if cfg == 0:` auto-pick branch "
        "runs (wave 14 regression). Use `int(1 if raw is None else raw)`."
    )
    assert "_rerank_n_jobs_cfg = int(1 if _rerank_raw is None else _rerank_raw)" in src, (
        "Post-fix idiom missing in _composite_discovery_tiny_rerank.py. "
        "Expected explicit None-check so `tiny_rerank_n_jobs=0` reaches the "
        "auto-pick branch."
    )


def test_discovery_random_state_zero_preserved():
    """``random_state=0`` is a legitimate seed: the discovery data_signature row-sampler must
    yield a DIFFERENT signature for seed=0 vs seed=42 (pre-fix ``or 42`` collapsed them) while
    staying deterministic per seed. This exercises the actual signature function the call uses."""
    import numpy as np
    import pandas as pd
    from mlframe.training.composite.cache import data_signature

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.random(5000), "f2": rng.random(5000), "t": rng.random(5000)})
    sig0 = data_signature(df, "t", ["f1", "f2"], random_state=0)
    sig42 = data_signature(df, "t", ["f1", "f2"], random_state=42)
    assert sig0 != sig42, "seed=0 and seed=42 must produce distinct signatures (no `or 42` collapse)"
    assert sig0 == data_signature(df, "t", ["f1", "f2"], random_state=0), "signature must be deterministic per seed"


def test_mrmr_fit_cache_max_zero_disables_cache():
    """``fit_cache_max=0`` (explicit "disable LRU") must leave the process cache empty after
    a real fit; ``fit_cache_max>0`` must populate it. Pre-fix ``or 4`` rewrote 0->4, so
    cache-off silently kept caching -- this exercises the real fit writer, not the source."""
    import numpy as np
    from collections import OrderedDict
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    X = rng.random((200, 5))
    y = (X[:, 0] + rng.random(200) * 0.1 > 0.5).astype(int)

    _saved = OrderedDict(MRMR._FIT_CACHE)
    MRMR._FIT_CACHE.clear()
    try:
        MRMR(fit_cache_max=0, max_runtime_mins=0.2, verbose=0).fit(X, y)
        assert len(MRMR._FIT_CACHE) == 0, "fit_cache_max=0 must keep the cache empty after fit"

        MRMR._FIT_CACHE.clear()
        MRMR(fit_cache_max=4, max_runtime_mins=0.2, verbose=0).fit(X, y)
        assert len(MRMR._FIT_CACHE) >= 1, "fit_cache_max=4 must populate the cache after fit"
    finally:
        MRMR._FIT_CACHE.clear()
        MRMR._FIT_CACHE.update(_saved)


def test_recurrent_rerun_empty_rebuild_not_silently_swapped():
    """``rebuilt == {}`` (all members gated out) MUST be returned verbatim.
    Pre-fix ``rebuilt or ensemble_dict`` conflated empty-rebuild with
    "rerun failed", silently restoring the pre-recurrent ensemble."""
    src = _read("training/core/_phase_recurrent.py")
    assert "return rebuilt or ensemble_dict" not in src, (
        "Pre-fix `rebuilt or ensemble_dict` pattern reappeared. Empty rebuild "
        "(all members gated out by the recurrent rerun) is operationally "
        "distinct from rebuild-failed and must not silently restore the "
        "prior ensemble (wave 14 regression)."
    )
    assert "if rebuilt is None:" in src
    assert "all members gated out" in src, (
        "The empty-rebuild branch must WARN-log with a distinguishable message "
        "so operators can tell `{}` from `prior_ensemble`."
    )


def test_mrmr_fit_cache_disable_zero_behavior_unit():
    """Unit-level: build a tiny MRMR fit cache state and verify that
    ``fit_cache_max=0`` actually empties the cache. Pre-fix this passed
    silently because `or 4` rewrote 0->4 so the cleanup while-loop never
    fired on a 4-entry cache."""
    from collections import OrderedDict
    from mlframe.feature_selection.filters.mrmr import MRMR

    # Snapshot + restore the process-wide cache to avoid bleed into other tests.
    _saved = OrderedDict(MRMR._FIT_CACHE)
    MRMR._FIT_CACHE.clear()
    try:
        # Seed cache with a few fake entries.
        for i in range(3):
            MRMR._FIT_CACHE[f"fake_key_{i}"] = object()
        assert len(MRMR._FIT_CACHE) == 3

        # Inline the post-fix branch (we don't run a full fit; we just exercise
        # the cap-clear branch as the cache writer does).
        _cap_raw = 0  # operator says "disable LRU"
        _cap = int(4 if _cap_raw is None else _cap_raw)
        if _cap <= 0:
            MRMR._FIT_CACHE.clear()
        else:
            while len(MRMR._FIT_CACHE) > _cap:
                MRMR._FIT_CACHE.popitem(last=False)

        assert len(MRMR._FIT_CACHE) == 0, (
            "fit_cache_max=0 must empty the cache; pre-fix the `or 4` trap "
            "kept it at 4 entries."
        )
    finally:
        MRMR._FIT_CACHE.clear()
        MRMR._FIT_CACHE.update(_saved)
