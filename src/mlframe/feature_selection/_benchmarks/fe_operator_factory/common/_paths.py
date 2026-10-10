"""Where the factory scripts read committed results from and write scratch outputs to.

Fresh outputs go to ``scratch_dir`` (``<system temp>/mlframe_fe_operator_factory/<sub-package>``; nothing is written into the repository). Aggregation scripts read ``data_dir``,
which is the committed ``results/`` folder of the sub-package, or the scratch folder when ``--fresh`` is on the command line.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path(tempfile.gettempdir()) / "mlframe_fe_operator_factory"
FRESH_FLAG = "--fresh"


def scratch_dir(sub: str) -> Path:
    """Writable output folder for sub-package ``sub`` (created on demand)."""
    out = SCRATCH_ROOT / sub
    out.mkdir(parents=True, exist_ok=True)
    return out


def results_dir(sub: str) -> Path:
    """Committed raw results of sub-package ``sub``."""
    return PKG_DIR / sub / "results"


def data_dir(sub: str) -> Path:
    """Folder an aggregation script reads: the scratch folder when ``--fresh`` is passed on the command line, else the committed results."""
    return scratch_dir(sub) if FRESH_FLAG in sys.argv[1:] else results_dir(sub)
