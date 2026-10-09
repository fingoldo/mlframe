"""Where the profiling scripts read and write their artefacts.

``MLFRAME_PROFILE_DIR`` overrides the default (the system temp directory)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

OUT_DIR = Path(os.environ.get("MLFRAME_PROFILE_DIR", tempfile.gettempdir()))
PROF_FILE = OUT_DIR / "gpu_fit.prof"
SRC_ROOT = Path(__file__).resolve().parents[4]  # .../src


def add_src_to_path() -> None:
    """Make ``import mlframe`` resolve to this checkout when the script is run directly."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
