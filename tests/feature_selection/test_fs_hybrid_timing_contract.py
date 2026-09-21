"""The three things that make a cell's timing quotable: a drained memo, an anchor, and a versioned store.

Each of these exists because of a way a number could be wrong while looking right. The memo drain stops an
arm's cost being measured as a dictionary lookup; the anchor stops two machines' seconds being compared as
though they were the same unit; the schema version stops a resume from averaging two different quantities.
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any, Dict, List

import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._anchor import ANCHOR_VERSION, measure_anchor
from mlframe.feature_selection._benchmarks.fs_hybrid._cell_store import SCHEMA_VERSION, JsonlCellStore, SchemaVersionMismatchError
from mlframe.feature_selection._benchmarks.fs_hybrid._memo import _MRMR_MODULE, assert_memo_drained, drain_memo_caches


class _FakeLock:
    """A context manager standing in for the identity cache's real lock."""

    def __init__(self) -> None:
        self.entered = 0

    def __enter__(self) -> "_FakeLock":
        """Record that the drain took the lock."""
        self.entered += 1
        return self

    def __exit__(self, *exc: Any) -> bool:
        """Never suppress an exception raised inside the drain."""
        return False


def _fake_mrmr_module(fit_entries: int, identity_entries: int, *, clear_works: bool = True) -> types.ModuleType:
    """Build a stand-in for the mrmr module carrying both caches.

    A fake rather than a real fit: this test is about whether the DRAIN is verified, and driving a real
    MRMR fit here would make the test slow and would couple it to MRMR's own configuration surface.
    """
    module = types.ModuleType("fake_mrmr")
    fit_cache: Dict[str, int] = {f"fit{i}": i for i in range(fit_entries)}
    identity_cache: Dict[str, int] = {f"id{i}": i for i in range(identity_entries)}

    class _MRMR:
        """Stand-in exposing only the drain surface the memo module uses."""

        _FIT_CACHE = fit_cache

        @classmethod
        def clear_fit_cache(cls) -> int:
            """Clear the fit cache, or refuse to, depending on how the fake was built."""
            if not clear_works:
                raise RuntimeError("cache lock unavailable")
            size = len(cls._FIT_CACHE)
            cls._FIT_CACHE.clear()
            return size

    module.MRMR = _MRMR  # type: ignore[attr-defined]
    module._MRMR_IDENTITY_FP_CACHE = identity_cache  # type: ignore[attr-defined]
    module._MRMR_IDENTITY_FP_LOCK = _FakeLock()  # type: ignore[attr-defined]
    return module


def test_drain_clears_both_caches_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both caches empty afterwards is what `verified` means, and both counts are reported."""
    module = _fake_mrmr_module(fit_entries=3, identity_entries=5)
    monkeypatch.setitem(sys.modules, _MRMR_MODULE, module)

    drain = drain_memo_caches()

    assert drain.verified is True
    assert drain.attempted is True
    assert drain.fit_entries_cleared == 3
    assert drain.identity_entries_cleared == 5
    assert len(module.MRMR._FIT_CACHE) == 0
    assert len(module._MRMR_IDENTITY_FP_CACHE) == 0
    assert module._MRMR_IDENTITY_FP_LOCK.entered == 1


def test_drain_reports_unverified_when_the_clear_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed clear leaves a populated memo, so the cell's timing must be marked unusable, not published.

    This is the whole point of the module. A fit that hits the memo returns the right answer instantly, so
    nothing downstream looks wrong -- only the number is.
    """
    module = _fake_mrmr_module(fit_entries=2, identity_entries=2, clear_works=False)
    monkeypatch.setitem(sys.modules, _MRMR_MODULE, module)

    drain = drain_memo_caches()

    assert drain.verified is False
    assert "clear_fit_cache" in drain.reason
    assert drain.as_dict()["memo_drained"] is False


def test_drain_reports_unverified_when_entries_survive(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear that returns cleanly but leaves entries behind is still a failed drain."""
    module = _fake_mrmr_module(fit_entries=1, identity_entries=0)
    monkeypatch.setitem(sys.modules, _MRMR_MODULE, module)
    # A cache that refills itself is the realistic version of this: another thread mid-fit writes its
    # entry back immediately after the clear.
    original = module.MRMR.clear_fit_cache

    def _clear_then_refill() -> int:
        """Clear the cache and immediately put an entry back, as a concurrent fit would."""
        count = original()
        module.MRMR._FIT_CACHE["late"] = 1
        return count

    monkeypatch.setattr(module.MRMR, "clear_fit_cache", _clear_then_refill)

    drain = drain_memo_caches()

    assert drain.verified is False
    assert "survived the drain" in drain.reason


def test_drain_is_verified_when_mrmr_was_never_imported(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unimported module holds no memoized fit, so there is nothing to drain and nothing unknown."""
    monkeypatch.delitem(sys.modules, _MRMR_MODULE, raising=False)

    drain = drain_memo_caches()

    assert drain.verified is True
    assert drain.attempted is False
    assert drain.as_dict()["memo_drained"] is True


def test_assert_memo_drained_raises_rather_than_returning_a_bad_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The strict entry point refuses to let an unverifiable drain produce a number at all."""
    monkeypatch.setitem(sys.modules, _MRMR_MODULE, _fake_mrmr_module(1, 1, clear_works=False))

    with pytest.raises(RuntimeError, match="memo state is unknown"):
        assert_memo_drained()


def test_anchor_is_positive_and_repeatable_within_an_order_of_magnitude() -> None:
    """The anchor must actually run and must not vary wildly on an idle-ish machine.

    The tolerance is deliberately loose: this host runs many processes, and a tight bound would make the
    test a contention detector. What it pins is that the workload is real (non-zero) and not accidentally
    optimised away to a constant.
    """
    readings = [measure_anchor() for _ in range(3)]

    assert all(reading.anchor_s > 0.0 for reading in readings)
    assert all(reading.anchor_version == ANCHOR_VERSION for reading in readings)
    fastest, slowest = min(r.anchor_s for r in readings), max(r.anchor_s for r in readings)
    assert slowest < fastest * 50, f"anchor readings span {fastest:.4f}..{slowest:.4f}s, which is not one workload"


def test_anchor_record_carries_its_version() -> None:
    """A stored anchor without its version could be silently compared across a workload change."""
    record = measure_anchor().as_dict()

    assert set(record) == {"anchor_s", "anchor_process_s", "anchor_version"}
    assert record["anchor_version"] == ANCHOR_VERSION


def test_store_stamps_the_schema_version_without_mutating_the_caller_dict(tmp_path: Any) -> None:
    """The stamp lands in the file and NOT on the caller's object, so a retry re-stamps rather than inherits."""
    store = JsonlCellStore(tmp_path / "cells.jsonl")
    record: Dict[str, Any] = {"cell_key": "a", "status": "ok"}

    store.append(record)

    assert "schema_version" not in record
    written = json.loads((tmp_path / "cells.jsonl").read_text(encoding="utf-8").strip())
    assert written["schema_version"] == SCHEMA_VERSION


def test_store_treats_an_unversioned_record_as_version_one(tmp_path: Any) -> None:
    """Records written before versioning existed carry version one's shape, so they must read as version one."""
    path = tmp_path / "cells.jsonl"
    path.write_text(json.dumps({"cell_key": "old", "status": "ok"}) + "\n", encoding="utf-8")

    assert JsonlCellStore(path).schema_versions() == {1}
    assert JsonlCellStore(path).assert_single_schema_version() == 1


def test_store_refuses_a_file_that_mixes_schema_versions(tmp_path: Any) -> None:
    """Resume fails loudly on a mixed file: every aggregate over it would be a mixture of two quantities."""
    path = tmp_path / "cells.jsonl"
    lines: List[str] = [json.dumps({"cell_key": "a", "schema_version": 1}), json.dumps({"cell_key": "b", "schema_version": 2})]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SchemaVersionMismatchError, match="mixes record schema versions"):
        JsonlCellStore(path).assert_single_schema_version()


def test_store_refuses_a_file_from_a_future_schema(tmp_path: Any) -> None:
    """A file written by newer code is not readable here, and guessing at it would be worse than failing."""
    path = tmp_path / "cells.jsonl"
    path.write_text(json.dumps({"cell_key": "a", "schema_version": SCHEMA_VERSION + 7}) + "\n", encoding="utf-8")

    with pytest.raises(SchemaVersionMismatchError, match="reads at most"):
        JsonlCellStore(path).assert_single_schema_version()


def test_store_append_survives_many_writes_under_the_lock(tmp_path: Any) -> None:
    """Every appended record must be recoverable: a lock that dropped writes would look like unrun cells."""
    store = JsonlCellStore(tmp_path / "cells.jsonl")
    for index in range(50):
        store.append({"cell_key": f"k{index}", "status": "ok"})

    assert len(store.load()) == 50
    assert store.assert_single_schema_version() == SCHEMA_VERSION
