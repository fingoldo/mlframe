"""Numerical-zero floors shared by the modules that guard a division or a standardisation."""

#: A standard deviation at or below this is treated as zero (constant column).
SD_FLOOR = 1e-12

#: A softmax temperature at or below this falls back to 1.0, otherwise the logits overflow and the weights degrade to uniform.
SOFTMAX_TEMP_FLOOR = 1e-12
