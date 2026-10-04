"""Regression test for audit2 F1: read_trained_models must block path traversal by DEFAULT.

Pre-fix the commonpath containment check ran only when trusted_root was explicitly passed
(it defaults to None), so a malicious featureset ("../.." / absolute path) escaped inference_folder
and reached the joblib.load pickle. Now the check always runs, defaulting the root to inference_folder.
"""

import pandas as pd
import pytest

from mlframe.inference.predict import read_trained_models


@pytest.mark.parametrize(
    "malicious_featureset",
    ["../../../../etc/passwd", "../../secret", "/abs/escape", "..\\..\\win"],
)
def test_traversal_featureset_rejected_without_trusted_root(malicious_featureset):
    """Traversal featureset rejected without trusted root."""
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with pytest.raises(ValueError, match="not inside trusted_root"):
        read_trained_models(malicious_featureset, X, inference_folder="infer")


def test_legitimate_featureset_not_rejected_by_containment(tmp_path, monkeypatch):
    """A normal relative featureset inside inference_folder passes the containment check (it then fails
    later for a missing dir, NOT with the traversal ValueError) — confirms the guard isn't over-broad."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    monkeypatch.chdir(tmp_path)
    models, X_out = read_trained_models("my_featureset", X, inference_folder="infer")
    assert models == {}
    assert X_out is X
