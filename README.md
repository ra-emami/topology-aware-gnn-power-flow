# Topology-Aware, Physics-Informed GNN for AC Power Flow

A Graph Neural Network that predicts bus voltage **magnitude and angle** for the
IEEE 33-bus distribution feeder (`case33bw`), trained against Newton–Raphson
solutions and refined with a **physics-informed** objective so its predictions
obey the AC power-balance law. The model is *topology-aware*: each branch carries
its per-unit series-admittance as an edge weight, so the network sees the
electrical structure of the grid and can be evaluated across reconfigurations.

## Why this is interesting

A learned surrogate for power flow is only useful if it is (a) more than a trivial
regressor, (b) physically consistent, and (c) able to handle the topology changes
that real feeders undergo. This project addresses all three: it benchmarks against
analytical baselines (including LinDistFlow), measures and reduces the AC
power-balance residual, and trains/evaluates across radial reconfigurations.

## Results (IEEE 33-bus, 5000 scenarios)

| Stage | Voltage R² | Voltage MAE | Angle R² | Power resid \|dP\| | Power resid \|dQ\| |
|-------|-----------:|------------:|---------:|------------------:|-------------------:|
| Supervised | 0.983 | 2.57 mV/pu | 0.928 | 0.266 MW | 0.219 MVAr |
| Physics-informed (λ=10) | 0.982 | 2.73 mV/pu | 0.926 | **0.099 MW** | **0.111 MVAr** |

Physics-informed fine-tuning cuts the power-balance violation by **~63% (active)**
and **~49% (reactive)** with negligible accuracy cost. On a single radial feeder
the linear **LinDistFlow** baseline is very strong; the GNN's advantage is its
differentiability, its consistency under the physics objective, and its ability to
generalize across topologies (where radial linearizations break).

## Repository structure

```
topology-aware-pignn/
├── src/pignn/
│   ├── config.py       # constants (DER buses, seeds, dispatch ranges)
│   ├── topology.py     # impedance-weighted graph, radial reconfiguration, LinDistFlow
│   ├── data.py         # scenario generation (single / multi-topology), splits, scaling
│   ├── model.py        # ProxySolverGNN (TAGConv)
│   ├── physics.py      # admittance matrix + differentiable power-balance residual
│   ├── train.py        # supervised training + physics-informed fine-tuning
│   └── evaluate.py     # metrics, baselines, undervoltage, topology generalization
├── scripts/
│   ├── train.py                   # supervised training -> checkpoints/gnn_powerflow.pt
│   ├── finetune_physics.py        # physics-informed fine-tuning -> ..._pinn.pt
│   ├── evaluate.py                # baselines + physics + topology generalization report
│   └── train_topology_general.py  # train across reconfigurations, test on held-out topologies
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .                                      # or: pip install -r requirements.txt
```

The scripts also add `src/` to the path, so they run from the repo root without
installation. A GPU is recommended for full-size runs.

## Usage

```bash
# 1) Train the supervised surrogate
python scripts/train.py --samples 5000 --epochs 300

# 2) Physics-informed fine-tuning (lambda controls accuracy/consistency balance)
python scripts/finetune_physics.py --ckpt checkpoints/gnn_powerflow.pt --lam 10

# 3) Full evaluation: baselines, physics consistency, undervoltage screening, topology generalization
python scripts/evaluate.py --ckpt checkpoints/gnn_powerflow_pinn.pt

# 4) Topology-general training (next research step)
python scripts/train_topology_general.py --samples 6000 --epochs 200
```

Every script accepts `--quick` for a fast smoke run.

## Method

- **Impedance-weighted graph.** Each in-service branch is weighted by the magnitude
  of its per-unit series admittance `|y| = 1/sqrt(R² + X²)`; normally-open tie
  switches are excluded, giving the true radial tree.
- **Scenarios.** Random load scaling and DER active/reactive dispatch, solved with
  Newton–Raphson. Node features `[P, Q, is_DER, is_slack]`; targets `[V, θ]`.
- **Physics-informed objective.** The admittance matrix **Y** yields a differentiable
  power-balance residual `S = V ⊙ conj(YV)` added to the loss as `MSE + λ · residual`.
- **Topology generalization.** Valid radial reconfigurations are enumerated by
  closing each tie switch and opening one line on the loop it creates; the model is
  trained on a subset and evaluated on held-out configurations.

## Roadmap

- **Ablations.** Sweep the TAGConv hop count `K` and toggle the impedance edge
  weights to quantify the contribution of multi-hop, impedance-aware propagation.
- **Additional feeders.** Extend to the 69-bus system and cross-feeder transfer.
- **Hard feasibility.** Replace the soft penalty with a differentiable
  Newton–Raphson correction layer, or use the surrogate to warm-start NR and
  measure iteration savings.

## Stack

PyTorch · PyTorch Geometric (`TAGConv`) · pandapower · scikit-learn · matplotlib
