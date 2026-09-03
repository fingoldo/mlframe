"""Wave 48 (2026-05-20): TOCTOU (Time-of-Check vs Time-of-Use) race audit.

Audit class: code that checks file/directory existence then takes action
assuming the check is still valid, but a concurrent worker / external cleanup
could change the filesystem state in between -- raising FileNotFoundError /
FileExistsError uncaught.

5 P2 + 4 Low = 9 fixes applied:

  P2:
    1. training/composite_cache.py:828 (DiscoveryCache.invalidate)
       exists+remove -> try/remove except FileNotFoundError.
    2. feature_engineering/transformer/_key_bank.py:228-234 (save_key_bank tmp_dir)
       Shared "<fingerprint>.tmp" path -> UUID-stamped per-worker tmp + exist_ok=True.
    3. feature_engineering/transformer/_key_bank.py:255-259 (save_key_bank final rename)
       Loser-tolerant rmtree+rename (content-addressable cache: loser's bytes equiv).
    4. feature_selection/wrappers/_rfecv.py:509 (_load_checkpoint)
       Dropped exists precheck; added FileNotFoundError/OSError to except.
    5. estimators/pipelines.py:43-49 (_verify_sidecar)
       Dropped isfile precheck; try-open with FileNotFoundError -> return True.

  Low (cosmetic redundant-precheck removals):
    6. training/io.py:313-316 (load_save_meta_sidecar) -- drop exists precheck.
    7. training/feature_handling/cache.py:336-340 (_read_from_disk) -- drop exists precheck.
    8. training/feature_handling/cache_backend.py:144 (DiskBackend.exists docstring contract).
    9. feature_selection/_benchmarks/kernel_tuning_cache/cli.py:45,86 -- try-open / try-remove.
"""

from __future__ import annotations

import importlib
import os
import contextlib
import tempfile
from pathlib import Path

MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    """Read."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The race itself.
#
# Behavioural since 2026-09-03. These asserted exact source spellings -- an `if os.path.exists`
# / `os.remove` pair absent, an `except FileNotFoundError` / `return False` pair present, and so
# on. Two problems with that. They break on any reindent or rename while the behaviour is
# untouched; and, more to the point, the behavioural sensors further down do NOT cover what they
# stood in for. Every one of those exercises the file-NEVER-EXISTED case, which a redundant
# `if exists()` precheck satisfies exactly as happily as the try/except does.
#
# The defect is the file vanishing BETWEEN the check and the use: a parallel hyperopt suite
# sharing cache_dir invalidating the same key, an external cleanup cron clearing checkpoints
# mid-fit. So these make the probe lie -- existence answers yes, the file is really gone. A
# precheck implementation then walks into the operation and raises FileNotFoundError; one that
# simply performs the operation and catches does not. That is the difference the source pins were
# encoding, and unlike the pins it is observable.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _existence_probes_lie():
    """Answer yes to every existence probe while the file is really gone.

    Kept narrowly scoped around the single call under test: os.path.exists is consulted by plenty
    of unrelated machinery, so this must not be left installed.
    """
    import os.path

    real_exists, real_isfile = os.path.exists, os.path.isfile
    real_p_exists, real_p_isfile = Path.exists, Path.is_file
    os.path.exists = lambda _p: True
    os.path.isfile = lambda _p: True
    Path.exists = lambda _self, **_kw: True
    Path.is_file = lambda _self: True
    try:
        yield
    finally:
        os.path.exists, os.path.isfile = real_exists, real_isfile
        Path.exists, Path.is_file = real_p_exists, real_p_isfile


def test_invalidate_survives_the_entry_vanishing_mid_call() -> None:
    """Two parallel suites sharing cache_dir invalidate the same key; the loser must not crash."""
    from mlframe.training.composite.cache import DiscoveryCache

    with tempfile.TemporaryDirectory() as td:
        c = DiscoveryCache(cache_dir=td)
        with _existence_probes_lie():
            assert c.invalidate("a_key_that_is_not_there") is False


def test_rfecv_load_checkpoint_survives_the_checkpoint_vanishing_mid_call() -> None:
    """An external cleanup cron clears the checkpoint between the probe and the open, mid-fit."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rf = RFECV.__new__(RFECV)
    with tempfile.TemporaryDirectory() as td:
        rf.checkpoint_path = str(Path(td) / "gone.pkl")
        with _existence_probes_lie():
            assert rf._load_checkpoint() is None


def test_load_save_meta_sidecar_survives_the_sidecar_vanishing_mid_call() -> None:
    """A concurrent bundle rewrite replaces the sidecar; the reader falls through to None."""
    from mlframe.training.io import load_save_meta_sidecar

    with tempfile.TemporaryDirectory() as td:
        with _existence_probes_lie():
            assert load_save_meta_sidecar(str(Path(td) / "bundle.bin")) is None


def test_key_bank_save_uses_uuid_tmp_dir() -> None:
    """Key bank save uses uuid tmp dir."""
    src = _read("feature_engineering/transformer/_key_bank.py")
    # The fix introduces UUID-stamped tmp + ignore_errors / try-except on rename.
    assert 'fingerprint + ".tmp."' in src and "_uuid.uuid4().hex[:8]" in src
    # The rename must be wrapped in try/except OSError.
    assert "tmp_dir.rename(final_dir)" in src and "except OSError as _rn_err" in src


def test_verify_sidecar_survives_the_sidecar_vanishing_mid_call(monkeypatch) -> None:
    """The pickle-integrity sidecar check must not raise when the sidecar goes mid-verification.

    Behavioural since 2026-09-03. This concatenated estimators/pipelines.py, utils/safe_pickle.py
    and the pyutilz implementation, then asserted one old spelling absent and either of two newer
    ones present -- a pin already rewritten twice as the helper moved between those three files,
    and one that says nothing about what happens when the sidecar disappears mid-call. The
    tolerated-miss path is fail-closed by default and opens only under
    MLFRAME_ALLOW_UNVERIFIED_PICKLE, so the test sets it rather than depending on the environment.

    `test_rfecv_load_checkpoint_tolerates_missing_file` and
    `test_io_load_save_meta_sidecar_drops_redundant_precheck` were removed here rather than
    rewritten: their subjects are now covered by a race test above plus the never-existed sensor
    below, which together say strictly more than the spellings did.
    """
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    from mlframe.utils.safe_pickle import verify_sidecar

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle.pkl"
        bundle.write_bytes(b"payload")
        with _existence_probes_lie():
            assert verify_sidecar(str(bundle)) is True


def test_feature_handling_cache_read_drops_redundant_precheck() -> None:
    """Feature handling cache read drops redundant precheck."""
    src = _read("training/feature_handling/cache.py")
    assert "if not os.path.exists(path):\n            return None\n        allow_pickle" not in src


def test_cache_backend_exists_documents_advisory_contract() -> None:
    """Cache backend exists documents advisory contract."""
    src = _read("training/feature_handling/cache_backend.py")
    assert "Advisory existence check" in src
    assert "TOCTOU" in src


def test_kernel_tuning_cli_show_and_clear_tolerate_missing() -> None:
    """Kernel tuning cli show and clear tolerate missing."""
    src = _read("feature_selection/_benchmarks/kernel_tuning_cache/cli.py")
    # _cmd_show now uses try/open; the missing file message branches via FileNotFoundError.
    assert 'except FileNotFoundError:\n        print(f"# no cache at {path}"' in src
    # _cmd_clear similarly tolerates a race.
    assert 'except FileNotFoundError:\n        print(f"# already removed:' in src


# ---------------------------------------------------------------------------
# Behavioural sensors: trigger the race outcome and assert no crash.
# ---------------------------------------------------------------------------


def test_invalidate_missing_key_returns_false_no_crash() -> None:
    """invalidate() on a key that never existed must return False, not raise."""
    from mlframe.training.composite.cache import DiscoveryCache

    with tempfile.TemporaryDirectory() as td:
        c = DiscoveryCache(cache_dir=td)
        # Key never written -> file does not exist -> invalidate must return False.
        assert c.invalidate("nonexistent_key_xyz") is False


def test_rfecv_load_checkpoint_missing_path_returns_none() -> None:
    """Rfecv load checkpoint missing path returns none."""
    from mlframe.feature_selection.wrappers.rfecv import RFECV

    rf = RFECV.__new__(RFECV)
    rf.checkpoint_path = "/nonexistent_dir_xyz/no_such_file.pkl"
    # Must return None silently (FileNotFoundError handled).
    assert rf._load_checkpoint() is None


def test_load_save_meta_sidecar_missing_returns_none() -> None:
    """Load save meta sidecar missing returns none."""
    from mlframe.training.io import load_save_meta_sidecar

    with tempfile.TemporaryDirectory() as td:
        bundle_path = os.path.join(td, "nonexistent.bin")
        # No bundle, no sidecar -> must return None silently.
        result = load_save_meta_sidecar(bundle_path)
        assert result is None


def test_verify_sidecar_missing_returns_true(monkeypatch) -> None:
    """``verify_sidecar`` moved from estimators.pipelines to
    utils.safe_pickle, and the default flipped from silent-true to
    fail-closed (returns False on missing sidecar) for the RCE-bypass
    guard. The ``tolerate-missing`` intent this test pins is now the
    env-opt-in path -- set ``MLFRAME_ALLOW_UNVERIFIED_PICKLE=1`` so the
    function returns True with a WARN (matching the test's original
    contract).
    """
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    try:
        from mlframe.utils.safe_pickle import verify_sidecar as _verify_sidecar
    except ImportError:
        # Back-compat: fall through to the legacy location if the user's
        # checkout predates the safe_pickle carve.
        from mlframe.estimators.pipelines import _verify_sidecar  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        bundle_path = os.path.join(td, "fake.bin")
        Path(bundle_path).write_bytes(b"dummy")
        # No sidecar exists -> must return True with WARN (env-opt-in path).
        assert _verify_sidecar(bundle_path) is True
