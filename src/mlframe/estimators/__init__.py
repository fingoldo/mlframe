"""Custom estimators and sklearn-compatible wrappers.

Submodules:
    base            - generic wrapper classes for early-stopping aware fitting.
    custom          - bespoke sklearn-compatible estimators.
    baselines       - RFE / baseline-model wrappers.
    early_stopping  - overfitting detection for non-native ES models.
    pipelines       - sklearn Pipeline subclass with mlframe extensions.
"""

from __future__ import annotations


from mlframe.estimators.base import *
from mlframe.estimators.custom import *
from mlframe.estimators.baselines import *
from mlframe.estimators.early_stopping import *
from mlframe.estimators.pipelines import *
