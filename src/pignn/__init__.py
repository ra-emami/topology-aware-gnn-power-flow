"""Topology-aware, physics-informed GNN surrogate for AC power flow."""

__version__ = "0.1.0"

from .config import DER_BUSES, SEED, NETWORKS, DEFAULT_NETWORK, der_buses_for
from .topology import (
    base_case,
    build_topology,
    enumerate_radial_configs,
    apply_config,
    tie_lines,
    lindistflow_matrices,
    lindistflow_predict,
)
from .data import (
    generate_dataset,
    generate_multitopology_dataset,
    split_dataset,
    fit_scalers,
    standardize,
)
from .model import ProxySolverGNN
from .physics import build_ybus, make_physics_residual
from .train import train_supervised, finetune_physics
from .evaluate import (
    predict,
    evaluate_model,
    baseline_scores,
    undervoltage_screening,
    evaluate_on_topologies,
)

__all__ = [
    "DER_BUSES", "SEED", "NETWORKS", "DEFAULT_NETWORK", "der_buses_for",
    "base_case", "build_topology", "enumerate_radial_configs",
    "apply_config", "tie_lines", "lindistflow_matrices", "lindistflow_predict",
    "generate_dataset", "generate_multitopology_dataset", "split_dataset",
    "fit_scalers", "standardize", "ProxySolverGNN", "build_ybus",
    "make_physics_residual", "train_supervised", "finetune_physics", "predict",
    "evaluate_model", "baseline_scores", "undervoltage_screening",
    "evaluate_on_topologies",
]
