"""Project-wide constants and per-network parameters."""

# Per-network parameters. DER buses are 0-indexed pandapower bus indices, spread
# across the main feeder and laterals of each system.
NETWORKS = {
    "case33bw": {"der_buses": [10, 20, 30]},
    "case69": {"der_buses": [15, 40, 60]},
}
DEFAULT_NETWORK = "case33bw"

# Buses hosting distributed energy resources (DERs) in the default feeder.
DER_BUSES = NETWORKS[DEFAULT_NETWORK]["der_buses"]

# Default reproducibility seed.
SEED = 42

# Load-scaling range applied per scenario.
LOAD_SCALE_RANGE = (0.5, 1.5)

# DER active / reactive dispatch ranges (MW / MVAr).
DER_P_RANGE = (0.0, 1.0)
DER_Q_RANGE = (-0.5, 0.5)


def der_buses_for(network=DEFAULT_NETWORK):
    """DER bus indices for a named network."""
    return NETWORKS[network]["der_buses"]
