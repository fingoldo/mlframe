"""Where the profiling scripts read and write their artefacts.

``MLFRAME_PROFILE_DIR`` overrides the default (the system temp directory)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

OUT_DIR = Path(os.environ.get("MLFRAME_PROFILE_DIR", tempfile.gettempdir()))  # tmp-path-ok: developer benchmark scratch directory, run by one user on their own machine
PROF_FILE = OUT_DIR / "gpu_fit.prof"
