"""The pipeline cache key tells a pre-pipeline with the target mix on from the same one with it off, and leaves the legacy key alone when no digest is given."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.training.core._phase_train_one_target import _compute_pipeline_cache_key
from mlframe.training.core._phase_train_one_target_cache_helpers import pre_pipeline_params_digest


def _pipe(mix: bool) -> Pipeline:
    """A pre-pipeline of an MRMR with the target mix on or off, and a scaler."""
    return Pipeline([("pre", MRMR(verbose=0, n_jobs=1, fe_target_mix_enable=mix)), ("scaler", StandardScaler())])


def test_mix_flag_changes_the_digest_and_the_key():
    """MRMR with the mix on and off give different digests, so the keys differ for the same strategy, name and features."""
    d_on, d_off = pre_pipeline_params_digest(_pipe(True)), pre_pipeline_params_digest(_pipe(False))
    assert d_on and d_off and d_on != d_off
    k_on = _compute_pipeline_cache_key("imp1_scale1_enc0", "mrmr", (False, False), False, [], [], [], pre_pipeline_digest=d_on)
    k_off = _compute_pipeline_cache_key("imp1_scale1_enc0", "mrmr", (False, False), False, [], [], [], pre_pipeline_digest=d_off)
    assert k_on != k_off


def test_empty_digest_keeps_the_legacy_key():
    """No digest gives exactly the key the function built before the parameter existed."""
    legacy = _compute_pipeline_cache_key("imp1_scale1_enc0", "mrmr", (False, False), False, [], [], [])
    assert _compute_pipeline_cache_key("imp1_scale1_enc0", "mrmr", (False, False), False, [], [], [], pre_pipeline_digest="") == legacy


def test_no_pre_pipeline_gives_an_empty_digest():
    """A missing pre-pipeline has no parameters to fold in."""
    assert pre_pipeline_params_digest(None) == ""
