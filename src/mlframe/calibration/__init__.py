"""Probability calibration quality and post-hoc calibrators.

The ``post`` submodule pulls heavy optional deps (netcal, pycalib,
ml_insights, betacal, venn-abers); it is NOT eager-imported. Use:

    from mlframe.calibration.post import ...

Submodules:
    quality        - calibration quality assessment (reliability diagrams, ECE, ...).
    post           - post-hoc calibration methods (isotonic, Platt, beta, Venn-Abers, ...).
    probabilities  - probability transformations and diagnostics.
"""

from mlframe.calibration.quality import *  # noqa: F401,F403
from mlframe.calibration.probabilities import *  # noqa: F401,F403
