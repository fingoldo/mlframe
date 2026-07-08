"""Prediction, explainability, and post-training analysis.

Submodules:
    predict         - batch and streaming inference (was inference.py).
    explainability  - SHAP and permutation-based explanation wrappers.
    postanalysis    - post-training analysis utilities.
"""

from __future__ import annotations


from mlframe.inference.predict import *
from mlframe.inference.explainability import *
from mlframe.inference.postanalysis import *
