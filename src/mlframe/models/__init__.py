"""Model operations: ensembling, hyperparameter tuning, splitting.

Submodules:
    ensembling    - stacking, blending, voting, weighted ensembles.
    optimization  - hyperparameter optimization (Optuna, GridSearch wrappers).
    tuning        - tuning utilities and CV-aware fit-predict loops.
    selection     - train / test / cv-fold splitting (was model_selection.py).
    rf_proximity  - Breiman random-forest proximity matrix + outlier measure.
"""

from __future__ import annotations


from mlframe.models.ensembling import *
from mlframe.models.optimization import *
from mlframe.models.tuning import *
from mlframe.models.selection import *
from mlframe.models.rf_proximity import *
from mlframe.models.lgbm_defaults import default_lgbm_params
from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic
