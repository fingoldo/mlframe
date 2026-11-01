"""The InfoNet checkpoint is pinned by sha256: its first use records the digest and a changed file is refused (NET2-03).

Before the fix nothing checked the checkpoint's bytes, so a replaced file was loaded with ``torch.load``.
"""

from pathlib import Path

import pytest

from mlframe.feature_selection.filters import _neural_mi as nm


def test_first_use_records_the_digest_and_a_matching_later_load_passes(tmp_path):
    """The first check writes a sidecar; a second check on the unchanged bytes passes."""
    ckpt = tmp_path / "infonet_pretrained.pt"
    ckpt.write_bytes(b"weights-v1")
    nm._verify_infonet_checkpoint(ckpt)
    assert Path(str(ckpt) + ".sha256").exists()
    nm._verify_infonet_checkpoint(ckpt)


def test_a_replaced_checkpoint_is_refused(tmp_path):
    """Changing the bytes after the digest was recorded makes the check refuse the load."""
    ckpt = tmp_path / "infonet_pretrained.pt"
    ckpt.write_bytes(b"weights-v1")
    nm._verify_infonet_checkpoint(ckpt)
    ckpt.write_bytes(b"tampered-weights")
    with pytest.raises(RuntimeError, match="does not match its recorded sha256"):
        nm._verify_infonet_checkpoint(ckpt)
