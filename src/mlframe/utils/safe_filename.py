"""File-name stems built from names the caller chose (model, target and pipeline names) that cannot leave their directory or name a device.

A name containing a path separator either writes into a subfolder that does not exist or, when absolute, makes ``os.path.join`` drop the directory
altogether; a colon is an NTFS alternate-data-stream separator and a drive marker; ``CON.dump`` names a device on Windows. Ordinary names, spaces and
underscores included, pass through unchanged, so existing file names and the model-name lookups built on them keep matching.
"""

from __future__ import annotations

import re

__all__ = ["safe_file_stem"]

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))})


def safe_file_stem(name: object) -> str:
    """``name`` with path separators, drive colons, wildcards and control characters replaced by ``_`` and a reserved Windows device name prefixed with ``_``."""
    stem = _UNSAFE.sub("_", str(name))
    if stem.split(".", 1)[0].strip().upper() in _RESERVED:
        stem = "_" + stem
    return stem
