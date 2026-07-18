"""Regression tests for MEDIUM findings of the 2026-05-17 audit
(feature_handling/ scope, wave 3).

Each test pins a specific bug surfaced by the audit. Tests must fail
on pre-fix code and pass on post-fix.
"""

from __future__ import annotations

import tempfile
import threading
from typing import List

import numpy as np
import pytest

# =====================================================================
# M-FH-01 - providers._is_secret_field whole-token matching
# =====================================================================


def test_m_fh_01_secret_field_no_false_positive_on_tokenizer() -> None:
    """Pre-fix substring matching flagged ``tokenizer``, ``monkey``, ``author``,
    ``keychain`` as secret-bearing because the patterns ``token`` / ``key`` /
    ``auth`` appeared as substrings. Post-fix uses whole-token matching after
    splitting on non-alphanumerics so only true secret-bearing names trigger.
    """
    from mlframe.training.feature_handling.providers import _is_secret_field

    # Genuine positives still match.
    assert _is_secret_field("api_key")
    assert _is_secret_field("API_KEY")
    assert _is_secret_field("X-Auth-Token")
    assert _is_secret_field("Authorization")
    assert _is_secret_field("bearer_token")
    assert _is_secret_field("password")
    assert _is_secret_field("client.secret")

    # False positives that the pre-fix substring path caught.
    assert not _is_secret_field("tokenizer")
    assert not _is_secret_field("tokenizer_path")
    assert not _is_secret_field("monkey")
    assert not _is_secret_field("author")
    assert not _is_secret_field("keychain")  # whole-token "key" not present
    assert not _is_secret_field("monkeys")


# =====================================================================
# M-FH-02 - providers._scrub_dict recursive
# =====================================================================


def test_m_fh_02_scrub_recursive_into_nested_dicts() -> None:
    """Pre-fix ``_scrub_dict`` only walked the top-level params dict, so a
    secret stored under ``params['headers']['Authorization']`` leaked through
    ``model_dump`` / ``__repr__``. Post-fix recurses into nested dicts AND
    lists of dicts."""
    from mlframe.training.feature_handling.providers import EmbeddingProvider

    p = EmbeddingProvider(
        kind="custom",
        model="m",
        params={
            "headers": {"Authorization": "Bearer secret123"},
            "extras": [{"api_key": "leaked"}],
            "nested": {"deep": {"password": "hunter2"}},  # nosec B105 -- synthetic fixture value asserting the scrubber redacts it, not a real credential
        },
    )
    dumped = p.model_dump()
    s = repr(dumped)
    assert "secret123" not in s
    assert "leaked" not in s
    assert "hunter2" not in s
    # The scrub marker should be present.
    assert "***" in s

    # __repr__ also recurses.
    r = repr(p)
    assert "secret123" not in r
    assert "leaked" not in r
    assert "hunter2" not in r


def test_m_fh_02_signature_stable_across_nested_secret_change() -> None:
    """The signature property hashes scrubbed params; a swapped nested secret
    must not change the cache signature (matches the documented R2-6 invariant
    for top-level secrets)."""
    from mlframe.training.feature_handling.providers import EmbeddingProvider

    p1 = EmbeddingProvider(
        kind="custom",
        model="m",
        params={"headers": {"Authorization": "Bearer A"}},
    )
    p2 = EmbeddingProvider(
        kind="custom",
        model="m",
        params={"headers": {"Authorization": "Bearer B"}},
    )
    assert p1.signature == p2.signature


# =====================================================================
# M-FH-03 - providers.from_uri URL-decodes model name
# =====================================================================


def test_m_fh_03_from_uri_url_decodes_model_name() -> None:
    """Pre-fix the model name kept literal ``%2F`` / ``%20`` because we never
    invoked ``urllib.parse.unquote`` on it. Post-fix the stored model field is
    the decoded form."""
    from mlframe.training.feature_handling.providers import EmbeddingProvider

    p = EmbeddingProvider.from_uri("hf://BAAI%2Fbge-small-en-v1.5")
    assert p.model == "BAAI/bge-small-en-v1.5"

    p2 = EmbeddingProvider.from_uri("hf://my%20org%2Fmodel?device=cpu")
    assert p2.model == "my org/model"
    assert p2.params.get("device") == "cpu"


# =====================================================================
# M-FH-04 - cache_backend._evict_to_caps fast-path
# =====================================================================


def test_m_fh_04_evict_to_caps_fast_path_under_entry_cap() -> None:
    """Pre-fix ``_evict_to_caps`` did ``os.listdir`` + per-entry ``stat`` on
    every write even when no eviction was needed. Post-fix short-circuits
    when the sidecar entry count is already at-or-below ``max_entries`` and
    ``max_size_mb`` is unset.

    We patch ``os.listdir`` to count calls; with the fast path active on
    writes well under the cap, ``listdir`` should not fire during evict.
    """
    from mlframe.training.feature_handling import cache_backend as cb_mod

    with tempfile.TemporaryDirectory() as tmp:
        backend = cb_mod.LocalDiskBackend(tmp, max_entries=1000)
        # Prime a few entries.
        for i in range(5):
            backend.write(f"k{i}", b"x" * 8)

        # Count listdir calls inside _evict_to_caps. We patch the module's
        # os.listdir reference for the duration of the next write.
        calls = {"n": 0}
        orig_listdir = cb_mod.os.listdir

        def counting_listdir(*args, **kwargs):
            """Counting listdir."""
            calls["n"] += 1
            return orig_listdir(*args, **kwargs)

        cb_mod.os.listdir = counting_listdir
        try:
            backend.write("k_new", b"y" * 8)
        finally:
            cb_mod.os.listdir = orig_listdir

        # Fast-path means _evict_to_caps does not invoke listdir when the
        # entry count is well under the cap. ``list_keys`` would also call
        # listdir, but the write path does not.
        assert calls["n"] == 0, f"_evict_to_caps invoked os.listdir under the cap; expected fast-path. calls={calls['n']}"


def test_m_fh_04_eviction_still_works_when_over_cap(monkeypatch) -> None:
    """Sanity: the fast-path optimisation must not skip eviction when the
    entry count actually exceeds the cap."""
    from mlframe.training.feature_handling.cache_backend import LocalDiskBackend

    # Force strictly-increasing LRU timestamps via a fake clock so eviction order is deterministic without a
    # wall-clock sleep. ``_touch_lru`` reads ``time.time()`` from the cached stdlib module.
    import time as _time_mod

    _counter = {"t": 1_700_000_000.0}

    def _tick() -> float:
        """Tick."""
        _counter["t"] += 1.0
        return _counter["t"]

    monkeypatch.setattr(_time_mod, "time", _tick)

    with tempfile.TemporaryDirectory() as tmp:
        backend = LocalDiskBackend(tmp, max_entries=3)
        for i in range(6):
            backend.write(f"key_{i}", b"v" * 4)
        kept = set(backend.list_keys())
        # The newest 3 keys should survive (LRU = oldest evicted first).
        assert len(kept) == 3
        assert "key_5" in kept


# =====================================================================
# M-FH-05 - fingerprint cache key includes column-name digest
# =====================================================================


def test_m_fh_05_fp_cache_key_includes_column_signature() -> None:
    """Pre-fix key was ``(id(df), n_cols)``; two frames with the same column
    count but different schemas could collide under id-recycling. Post-fix
    the key triples in the column-name tuple hash."""
    from mlframe.training.feature_handling import fingerprint as fp_mod

    # Synthetic stand-in objects so we can control id() exactly.
    class _FakeDF:
        """Groups tests covering fake d f."""
        def __init__(self, cols: List[str]):
            self.columns = cols

        def __len__(self):
            return 0

    fp_mod._fingerprint_cache.clear()
    a = _FakeDF(["col_a", "col_b"])
    b = _FakeDF(["col_x", "col_y"])  # same length, different names

    fp = fp_mod.ContentFingerprint(
        n_rows=0,
        n_cols=2,
        column_dtypes_hash="aa",
        sampled_rows_hash="bb",
    )
    fp_mod._fp_cache_put(a, fp)

    # If b's id collided with a's (which can happen after `del a` in real
    # code), the pre-fix key (id, n_cols) would return a stale hit.
    # Post-fix the column-name digest prevents that.
    # We can't easily force id collisions, but we can verify the key
    # function returns different tuples for same-id-same-len-different-cols.
    key_a = fp_mod._fp_cache_key(a)
    key_b = fp_mod._fp_cache_key(b)
    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b, f"different-schema frames produced the same fp cache key: {key_a}"


# =====================================================================
# M-FH-06 - custom_handler.fit_transform narrowed TypeError catch
# =====================================================================


def test_m_fh_06_fit_transform_does_not_swallow_unrelated_typeerror() -> None:
    """Pre-fix the blanket ``except (AttributeError, TypeError)`` in
    ``CustomHandler.fit_transform`` swallowed any TypeError raised inside the
    underlying transformer, silently demoting to fit+transform and re-running
    the broken call. Post-fix only the y-signature-mismatch TypeError is
    caught; other TypeErrors propagate so the operator sees the real bug.
    """
    from mlframe.training.feature_handling.custom_handler import CustomHandler
    from mlframe.training.feature_handling.handlers import CustomParams

    class BuggyTransformer:
        """Groups tests covering buggy transformer."""
        def fit(self, X, y=None):
            """Fit."""
            return self

        def fit_transform(self, X, y=None):
            # An unrelated TypeError -- not a signature mismatch.
            """Fit transform."""
            raise TypeError("buggy: cannot subtract 'str' from 'int'")

        def transform(self, X):
            """Transform."""
            return X

    params = CustomParams(transformer=BuggyTransformer(), output_kind="dense")
    handler = CustomHandler(column="c", params=params)

    # Use a tiny pandas DF so _extract_column doesn't need polars.
    import pandas as pd

    df = pd.DataFrame({"c": [1, 2, 3]})

    with pytest.raises(TypeError, match="buggy"):
        handler.fit_transform(df, y=[0, 1, 0])


def test_m_fh_06_fit_transform_still_falls_back_on_signature_mismatch() -> None:
    """Sanity: the fit-only fallback still kicks in when the transformer
    rejects ``y`` because its signature is ``fit_transform(X)`` only."""
    from mlframe.training.feature_handling.custom_handler import CustomHandler
    from mlframe.training.feature_handling.handlers import CustomParams

    class UnsupervisedOnly:
        """Groups tests covering unsupervised only."""
        def fit(self, X):
            """Fit."""
            self.fitted = True
            return self

        def fit_transform(self, X):
            """Fit transform."""
            self.fitted = True
            return np.asarray(X) * 2

        def transform(self, X):
            """Transform."""
            return np.asarray(X) * 2

    params = CustomParams(transformer=UnsupervisedOnly(), output_kind="dense")
    handler = CustomHandler(column="c", params=params)
    import pandas as pd

    df = pd.DataFrame({"c": [1, 2, 3]})
    out = handler.fit_transform(df, y=[0, 1, 0])
    assert out is not None


# =====================================================================
# M-FH-07 - text_detection stride covers full column
# =====================================================================


def test_m_fh_07_column_to_string_list_covers_full_range() -> None:
    """Pre-fix the integer-step formula ``step = max(1, n // max_sample)``
    silently truncated to the head of the column when ``n`` was between
    ``max_sample`` and ``2 * max_sample`` (e.g. n=15_000, step=1, took the
    first 10_000). Post-fix ``np.linspace`` spreads the sample over [0, n-1]
    so the last row is always reachable.
    """
    pl = pytest.importorskip("polars")
    from mlframe.training.feature_handling.text_detection import _column_to_string_list

    n = 25_000
    # Use a marker at the very last index that pre-fix sampling would miss.
    values = ["mid"] * (n - 1) + ["TAIL_MARKER"]
    df = pl.DataFrame({"c": values})

    sampled = _column_to_string_list(df, "c", max_sample=10_000)
    assert "TAIL_MARKER" in sampled, "linspace stride must include the final row; pre-fix integer step silently dropped the trailing slice"
    # And the sample size is reasonable (bounded by max_sample).
    assert len(sampled) <= 10_000


# =====================================================================
# M-FH-08 - registry.prewarm atomic submit
# =====================================================================


def test_m_fh_08_prewarm_dedupes_under_concurrent_callers() -> None:
    """Pre-fix the future cache lookup, submit, and store were three separate
    steps with no lock - two threads hitting a fresh signature could each
    submit a load job (which on a GPU doubles the VRAM peak). Post-fix the
    triple is under ``_REGISTRY_LOCK``; only one Future per signature is
    submitted.
    """
    from mlframe.training.feature_handling import registry as reg_mod

    # Snapshot then clear futures so the test is hermetic.
    saved = dict(reg_mod._PREWARM_FUTURES)
    reg_mod._PREWARM_FUTURES.clear()

    load_calls = {"n": 0}
    load_lock = threading.Lock()
    load_started = threading.Event()
    release_load = threading.Event()

    class _SlowProvider:
        """Groups tests covering slow provider."""
        signature = "test-medium-prewarm-08"

        def acquire(self):
            """Acquire."""
            with load_lock:
                load_calls["n"] += 1
            load_started.set()
            release_load.wait(timeout=5.0)

        def release(self):
            """Release."""
            pass

        def transform(self, texts):  # not used here
            """Transform."""
            return np.zeros((len(texts), 1), dtype=np.float32)

    provider = _SlowProvider()
    try:
        N = 8
        futs = []

        def fire():
            """Fire."""
            futs.append(reg_mod.prewarm(provider))

        threads = [threading.Thread(target=fire) for _ in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)
        load_started.wait(timeout=2.0)

        # All N callers must observe the same Future.
        assert len({id(f) for f in futs}) == 1, "prewarm submitted multiple Futures under concurrent callers"

        release_load.set()
        # Drain the future.
        futs[0].result(timeout=5.0)

        # acquire() must have been called exactly once.
        assert load_calls["n"] == 1, f"acquire() ran {load_calls['n']} times under concurrent prewarm"
    finally:
        # Restore state to avoid leaking into sibling tests.
        release_load.set()
        reg_mod._PREWARM_FUTURES.clear()
        reg_mod._PREWARM_FUTURES.update(saved)
        try:
            reg_mod.shutdown_all()
        except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
            pass
