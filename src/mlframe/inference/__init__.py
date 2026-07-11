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
from mlframe.inference.logical_constraints import apply_logical_constraints, discover_logical_constraints
from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint
from mlframe.inference.time_budget_ensemble import TimeBudgetEnsemble
from mlframe.inference.recursive_forecast import recursive_multi_step_forecast
