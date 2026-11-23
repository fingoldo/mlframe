"""Cluster 32 regression: the ordered-encoder bench times calls on prebuilt inputs, and both arms agree (audit 2026-11-20, finding 32-F20)."""

import inspect

import numpy as np

from mlframe.training._benchmarks import bench_ordered_target_encoder as bench


def test_timed_helpers_do_not_generate_inputs():
    """The timed helpers take inputs as arguments and never call the generator (32-F20)."""
    for fn in (bench._run_separate_calls, bench._run_batch):
        assert "_make_batch_inputs(" not in inspect.getsource(fn)


def test_batch_and_separate_arms_agree_on_prebuilt_inputs():
    """Both arms run on one prebuilt input set and their per-column encodings match (32-F20)."""
    from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode, ordered_target_encode_batch

    columns, y, order = bench._make_batch_inputs(500, 20, 3)
    batch = ordered_target_encode_batch(columns, y, order=order, smoothing=1.0)
    for name, cats in columns.items():
        single = ordered_target_encode(cats, y, order=order, smoothing=1.0)
        np.testing.assert_allclose(np.asarray(batch[name]), np.asarray(single))
