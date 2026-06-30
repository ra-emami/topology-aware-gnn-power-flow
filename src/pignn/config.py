"""Project-wide constants."""

# Buses hosting distributed energy resources (DERs) in the IEEE 33-bus feeder.
DER_BUSES = [10, 20, 30]

# Default reproducibility seed.
SEED = 42

# Load-scaling range applied per scenario.
LOAD_SCALE_RANGE = (0.5, 1.5)

# DER active / reactive dispatch ranges (MW / MVAr).
DER_P_RANGE = (0.0, 1.0)
DER_Q_RANGE = (-0.5, 0.5)
