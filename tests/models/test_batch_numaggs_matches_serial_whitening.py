"""batch_numaggs computes the same means as the serial path (audit 2026-11-20, finding 18-F5)."""
import numpy as np

from mlframe.models.ensembling import base


def test_batch_path_matches_the_serial_reference_row_for_row():
    """Each row of the batch output equals the serial call with whiten_means=False: the batch path must not whiten the means (batch-vs-serial parity)."""
    rng = np.random.default_rng(0)
    predictions = rng.uniform(0.1, 0.9, size=(6, 40))
    names_len = len(base.compute_numerical_aggregates_numba(predictions[0], geomean_log_mode=False, directional_only=False, whiten_means=False))

    batch = base.batch_numaggs(predictions, names_len, {}, means_only=True)

    for i in range(predictions.shape[0]):
        serial = base.compute_numerical_aggregates_numba(predictions[i], geomean_log_mode=False, directional_only=False, whiten_means=False)
        np.testing.assert_allclose(batch[i], np.asarray(serial, dtype=np.float32), rtol=1e-5, atol=1e-6)
