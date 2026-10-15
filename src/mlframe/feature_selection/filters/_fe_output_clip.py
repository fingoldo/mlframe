"""Replay-output clip quantiles shared by the FE operators that clip their fit-time output.

A replay clips its output to the fit-time quantiles so heavy tails cannot dominate a linear consumer.
"""

OUTPUT_CLIP_QUANTILES: tuple[float, float] = (0.001, 0.999)
