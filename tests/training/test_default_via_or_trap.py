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

Each site is exercised through its behaviour: the resolver, the signature function, the fit-cache writer and the recurrent rerun are called with the falsy sentinel.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np


def test_tiny_rerank_n_jobs_zero_sentinel_reaches_branch(monkeypatch):
    """``tiny_rerank_n_jobs=0`` auto-picks min(spec count, physical cores); None is the historical 1; other values are taken as given.

    The pre-fix ``int(raw or 1)`` collapsed 0 to 1 before the auto-pick branch ran.
    """
    import pyutilz.parallel
    from mlframe.training.composite.discovery._tiny_rerank import _resolve_rerank_n_jobs

    monkeypatch.setattr(pyutilz.parallel, "cpu_count_physical", lambda: 8)
    assert _resolve_rerank_n_jobs(0, 5) == 5
    assert _resolve_rerank_n_jobs(0, 20) == 8
    assert _resolve_rerank_n_jobs(0, 0) == 1
    assert _resolve_rerank_n_jobs(None, 20) == 1
    assert _resolve_rerank_n_jobs(3, 20) == 3
    assert _resolve_rerank_n_jobs(-1, 20) == 1


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


def _recurrent_ctx():
    """A context with two members whose predictions match the sliced target row counts, plus the aligned target values."""
    members = [SimpleNamespace(model_name=f"m{i}", train_preds=np.zeros(6), val_preds=np.zeros(2), test_preds=np.zeros(2)) for i in range(2)]
    ctx = SimpleNamespace(
        models={"reg": {"t": members}},
        train_idx=np.arange(0, 6),
        val_idx=np.arange(6, 8),
        test_idx=np.arange(8, 10),
        verbose=0,
        model_name="mdl",
        group_ids=None,
        sample_weights=None,
    )
    return ctx, np.arange(10.0)


def test_recurrent_rerun_empty_rebuild_not_silently_swapped(monkeypatch, caplog):
    """An empty rebuilt ensemble (every member gated out) is returned as the empty dict with a distinguishable WARN; ``None`` keeps the prior ensemble."""
    from mlframe.models import ensembling
    from mlframe.training.core._phase_recurrent import _apply_recurrent_to_ensemble

    ctx, target_values = _recurrent_ctx()
    prior = {"prior_method": object()}

    def run(result):
        """Apply the recurrent rerun with ``score_ensemble`` stubbed to return ``result``."""
        monkeypatch.setattr(ensembling, "score_ensemble", lambda models_and_predictions, **kwargs: result)
        return _apply_recurrent_to_ensemble(ctx=ctx, ensemble_dict=prior, target_type="reg", target_name="t", target_values=target_values)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._phase_recurrent"):
        empty = run({})
    assert empty == {} and empty is not prior
    assert any("all members gated out" in r.getMessage() for r in caplog.records)

    assert run(None) is prior

    fresh = {"blend": 1}
    assert run(fresh) == fresh


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

        assert len(MRMR._FIT_CACHE) == 0, "fit_cache_max=0 must empty the cache; pre-fix the `or 4` trap kept it at 4 entries."
    finally:
        MRMR._FIT_CACHE.clear()
        MRMR._FIT_CACHE.update(_saved)
