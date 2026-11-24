"""Cluster 12 regression: unblinding respects label boundaries and tolerates non-object mapping JSON (audit 2026-11-20, finding 12-F16).

``unblind_text`` replaced labels by bare substring, so ``arm_b`` rewrote inside ``arm_bx``. It now matches whole labels, longest first, in one pass.
``load_mapping`` crashed on a JSON array (no ``.items()``); a non-object mapping now means "not blinded".
"""

import json

from mlframe.feature_selection._benchmarks.fs_hybrid._blinding import MAPPING_SUFFIX, load_mapping, unblind_text


def test_label_is_not_rewritten_inside_a_longer_token():
    """``arm_b`` is replaced as a whole token, not inside ``arm_bx`` (12-F16)."""
    out = unblind_text("arm_b and arm_bx", {"Real": "arm_b"})
    assert out == "Real and arm_bx"


def test_longest_label_wins_in_one_pass():
    """With overlapping labels the longer one is substituted, not a prefix of it (12-F16)."""
    out = unblind_text("arm_aa", {"Short": "arm_a", "Long": "arm_aa"})
    assert out == "Long"


def test_non_object_mapping_json_is_not_blinded(tmp_path):
    """A JSON array mapping file means no mapping, not an exception (12-F16)."""
    results = tmp_path / "results.json"
    results.write_text("{}", encoding="utf-8")
    (tmp_path / ("results.json" + MAPPING_SUFFIX)).write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert load_mapping(str(results)) is None
