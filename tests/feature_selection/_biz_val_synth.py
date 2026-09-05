"""Re-export shim for the shared business-value synthetic data generators.

The implementation now lives verbatim in
:mod:`tests.feature_selection._synth.biz_val_synth`; this module exists only so that the
~44 existing importers under ``tests/`` keep working without an edit. Nothing here adds
behaviour -- every name below is the same object as in the ``_synth`` module.

New tests should not import from this shim. The scenario generators are slated to move
into production (``mlframe.data.datasets``); once they land there, import them from
production and let this shim shrink. Until then, import from
``tests.feature_selection._synth.biz_val_synth`` directly.
"""

from tests.feature_selection._synth.biz_val_synth import make_signal_plus_noise
from tests.feature_selection._synth.biz_val_synth import make_correlated_redundant
from tests.feature_selection._synth.biz_val_synth import make_3way_xor
from tests.feature_selection._synth.biz_val_synth import make_polynomial_target
from tests.feature_selection._synth.biz_val_synth import make_imbalanced
from tests.feature_selection._synth.biz_val_synth import make_heavy_tail_skewed
from tests.feature_selection._synth.biz_val_synth import make_latent_reflections
from tests.feature_selection._synth.biz_val_synth import make_two_latent_groups
from tests.feature_selection._synth.biz_val_synth import as_df
from tests.feature_selection._synth.biz_val_synth import support_indices
from tests.feature_selection._synth.biz_val_synth import signal_overlap
from tests.feature_selection._synth.biz_val_synth import signal_recovery_count
from tests.feature_selection._synth.biz_val_synth import downstream_auc
from tests.feature_selection._synth.biz_val_synth import baseline_signal_auc
from tests.feature_selection._synth.biz_val_synth import _XREF as _XREF
from tests.feature_selection._synth.biz_val_synth import _build_linear as _build_linear
from tests.feature_selection._synth.biz_val_synth import _build_quadratic_classif as _build_quadratic_classif
from tests.feature_selection._synth.biz_val_synth import _build_redundant_multi as _build_redundant_multi
from tests.feature_selection._synth.biz_val_synth import _build_redundant_quadratic as _build_redundant_quadratic
from tests.feature_selection._synth.biz_val_synth import _train_holdout_split as _train_holdout_split
from tests.feature_selection._synth.biz_val_synth import _logreg_auc as _logreg_auc
from tests.feature_selection._synth.biz_val_synth import _quantile_bin_local as _quantile_bin_local
from tests.feature_selection._synth.biz_val_synth import _mi_one as _mi_one

# Literal list of string constants on purpose: vulture only honours ``__all__`` when it
# parses as an ``ast.Assign`` to a list/tuple of string constants, and its ``_ignore_import``
# exemption applies only to files literally named ``__init__.py``. Without this, every
# re-export above would fire as an unused import at confidence 90 in the blocking
# ``tests-lint-blocking`` vulture job.
__all__ = [
    "make_signal_plus_noise",
    "make_correlated_redundant",
    "make_3way_xor",
    "make_polynomial_target",
    "make_imbalanced",
    "make_heavy_tail_skewed",
    "make_latent_reflections",
    "make_two_latent_groups",
    "as_df",
    "support_indices",
    "signal_overlap",
    "signal_recovery_count",
    "downstream_auc",
    "baseline_signal_auc",
    "_XREF",
    "_build_linear",
    "_build_quadratic_classif",
    "_build_redundant_multi",
    "_build_redundant_quadratic",
    "_train_holdout_split",
    "_logreg_auc",
    "_quantile_bin_local",
    "_mi_one",
]
