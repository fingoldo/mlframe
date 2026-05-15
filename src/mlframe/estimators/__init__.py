"""Custom estimators and sklearn-compatible wrappers.

Submodules:
    base            - generic wrapper classes for early-stopping aware fitting.
    custom          - bespoke sklearn-compatible estimators.
    baselines       - RFE / baseline-model wrappers.
    early_stopping  - overfitting detection for non-native ES models.
    pipelines       - sklearn Pipeline subclass with mlframe extensions.
"""

from mlframe.estimators.base import *  # noqa: F401,F403
from mlframe.estimators.custom import *  # noqa: F401,F403
from mlframe.estimators.baselines import *  # noqa: F401,F403
from mlframe.estimators.early_stopping import *  # noqa: F401,F403
from mlframe.estimators.pipelines import *  # noqa: F401,F403
