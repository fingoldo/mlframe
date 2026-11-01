"""Suite metadata of every kind needs a sha256 sidecar before it is read (DESER-07).

Legacy ``joblib`` metadata used to load before any integrity check. It now goes through the same sidecar gate as the pickle kinds.
"""

import joblib
import pytest

from mlframe.training.core._metadata_loader import load_metadata_file
from mlframe.utils.safe_pickle import write_sidecar


@pytest.fixture(autouse=True)
def _no_unverified_opt_out(monkeypatch):
    """Start from the default: the unverified-load opt-out is unset."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)


def test_joblib_metadata_without_a_sidecar_is_refused(tmp_path):
    """A joblib metadata file with no sidecar raises before any bytes are unpickled."""
    path = tmp_path / "metadata.joblib"
    joblib.dump({"ok": 1}, path)
    with pytest.raises(RuntimeError, match="sha256 sidecar"):
        load_metadata_file(str(path), "joblib", "test")


def test_joblib_metadata_with_a_matching_sidecar_loads(tmp_path):
    """The same joblib file loads once its sidecar matches."""
    path = tmp_path / "metadata.joblib"
    joblib.dump({"ok": 1}, path)
    write_sidecar(str(path))
    assert load_metadata_file(str(path), "joblib", "test") == {"ok": 1}
