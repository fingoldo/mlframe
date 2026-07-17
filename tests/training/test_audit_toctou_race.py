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
import tempfile
from pathlib import Path



MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    """Read."""
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_composite_cache_invalidate_uses_try_remove() -> None:
    """Composite cache invalidate uses try remove."""
    src = _read("training/composite/cache_store.py")
    # The fix replaces exists+remove with try/remove.
    assert "if os.path.exists(path):\n            os.remove(path)" not in src
    assert "try:\n            os.remove(path)\n        except FileNotFoundError:\n            return False" in src


def test_key_bank_save_uses_uuid_tmp_dir() -> None:
    """Key bank save uses uuid tmp dir."""
    src = _read("feature_engineering/transformer/_key_bank.py")
    # The fix introduces UUID-stamped tmp + ignore_errors / try-except on rename.
    assert 'fingerprint + ".tmp."' in src and "_uuid.uuid4().hex[:8]" in src
    # The rename must be wrapped in try/except OSError.
    assert "tmp_dir.rename(final_dir)" in src and "except OSError as _rn_err" in src


def test_rfecv_load_checkpoint_tolerates_missing_file() -> None:
    """Rfecv load checkpoint tolerates missing file."""
    src = _read("feature_selection/wrappers/rfecv/__init__.py")
    # The pre-fix `if not path or not os.path.exists(path)` is replaced
    # with `if not path: return None` plus FileNotFoundError on open.
    assert "if not path or not os.path.exists(path):" not in src
    assert "except FileNotFoundError:\n            return None" in src
    # The broader except now includes OSError.
    assert "OSError" in src


def test_pipelines_verify_sidecar_tolerates_missing() -> None:
    """The verify_sidecar helper moved out of estimators/pipelines.py into
    utils/safe_pickle.py, then further out into pyutilz.core.safe_pickle (mlframe's
    utils/safe_pickle.py now only wraps it), and the implementation switched from
    ``try/except FileNotFoundError`` to a direct ``if not isfile(sidecar)``
    branch. Functionally equivalent: missing sidecar is tolerated (returns
    True under MLFRAME_ALLOW_UNVERIFIED_PICKLE=1 env var, False default
    fail-closed; the test's "tolerate" intent matches the env-opt-in
    path). Accept either source shape so the sensor stays valid post-move.
    """
    import pyutilz.core.safe_pickle as _pyutilz_safe_pickle

    facade = _read("estimators/pipelines.py")
    sibling = _read("utils/safe_pickle.py")
    upstream = Path(_pyutilz_safe_pickle.__file__).read_text(encoding="utf-8")
    src = facade + "\n" + sibling + "\n" + upstream
    # Old leak-through pattern (silent-true on missing) must still be gone.
    assert "if not os.path.isfile(sidecar):\n        return True" not in src
    # Post-fix: either the old try/except FileNotFoundError shape OR the
    # new ``if not isfile(sidecar):`` direct branch with explicit env-var
    # gate.
    assert "except FileNotFoundError:" in src or "if not isfile(sidecar):" in src
    assert "return True" in src


def test_io_load_save_meta_sidecar_drops_redundant_precheck() -> None:
    """Io load save meta sidecar drops redundant precheck."""
    src = _read("training/io.py")
    assert "if not os.path.exists(sidecar):\n        return None\n    try:" not in src
    assert "except FileNotFoundError:\n        return None" in src


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
