"""A symlink inside ``trusted_root`` that points outside it is not accepted as a path inside the root (DSR2-01).

``validate_trusted_path`` compared lexical absolute paths, so ``root/link/payload.pkl`` passed the commonpath check even when
``root/link`` pointed at a directory elsewhere. Both sides are now resolved with ``realpath``.
"""

import os

import pytest

from mlframe.core.helpers import validate_trusted_path


def _make_escaping_link(root, outside):
    """Create ``root/link`` pointing at ``outside``; skip when the platform will not create symlinks for this user."""
    link = root / "link"
    try:  # audit-exempt: except_skip_masks_call_under_test -- only the symlink creation is guarded, the skip names a host without symlink rights
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create symlinks on this host: {exc}")
    return link


def test_symlink_inside_root_that_points_outside_is_rejected(tmp_path):
    """A path that reaches the outside target only through an in-root symlink is refused."""
    root = tmp_path / "trusted"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _make_escaping_link(root, outside)
    with pytest.raises(ValueError, match="not inside trusted_root"):
        validate_trusted_path(str(root / "link" / "payload.pkl"), str(root))


def test_regular_file_inside_root_is_still_accepted(tmp_path):
    """Resolving symlinks does not reject ordinary paths inside the root."""
    root = tmp_path / "trusted"
    root.mkdir()
    (root / "model.pkl").write_bytes(b"x")
    assert validate_trusted_path(str(root / "model.pkl"), str(root)) is None
