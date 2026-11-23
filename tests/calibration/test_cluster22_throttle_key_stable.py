"""Cluster 22 regression: the post-train calibrator's throttle key is a stable name, not a line number (audit 2026-11-20, finding 22-F43)."""

from pathlib import Path

import mlframe.calibration._post_train_calibrators as mod


def test_throttle_key_has_no_line_number():
    """The log_throttle key is a stable name; a line-number key would drift on every edit (22-F43)."""
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert '"post_train_calib_target_coerce_failed"' in src
    assert '"_post_train_calibrators:257"' not in src
