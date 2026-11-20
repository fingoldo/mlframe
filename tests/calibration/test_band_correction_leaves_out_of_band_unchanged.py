"""Out-of-band predictions are unchanged by the band correction, including the default clip (audit 2026-11-20, finding 22-F5)."""
import numpy as np

from mlframe.calibration.prediction_band_correction import apply_prediction_band_correction


def test_out_of_band_values_are_not_clipped():
    """-0.3 is outside (0.4, 0.6] and must stay -0.3; the in-band 0.5 is corrected to 0.75."""
    out = apply_prediction_band_correction(np.array([-0.3, 0.5]), lo=0.4, hi=0.6, factor=1.5)
    assert abs(out[0] - (-0.3)) < 1e-12
    assert abs(out[1] - 0.75) < 1e-12
