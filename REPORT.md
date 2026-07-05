# Topology-Aware, Physics-Informed GNN for AC Power Flow

A graph neural network that predicts bus voltage magnitude and angle on the IEEE
33-bus distribution feeder, trained against Newton-Raphson solutions, refined with a
differentiable AC power-balance residual, and made topology-aware through
impedance-weighted edges so a single model transfers across radial reconfigurations.

## Problem and motivation

AC power flow is the workhorse computation of distribution system analysis: given
loads and generator dispatch, solve for the complex bus voltages that satisfy
Kirchhoff's laws. Newton-Raphson (NR) solves it accurately but must be re-run from
scratch for every scenario, which becomes the bottleneck in contingency screening,
hosting-capacity studies, and any inner loop that sweeps thousands of dispatch or
topology cases.

A learned surrogate is only worth building if it clears three bars: it must be more
than a trivial regressor, it must produce physically consistent voltages rather than
plausible-looking ones, and it must survive the topology changes that real feeders
undergo through switching. This project targets all three on the `case33bw` feeder
(pandapower). The graph encoding carries the feeder's electrical structure through
per-unit series-admittance edge weights; a differentiable power-balance residual pulls
predictions onto the AC manifold; and training and evaluation span the feeder's radial
reconfigurations. The payoff of a differentiable surrogate is not raw accuracy on one
feeder (a linear baseline is already strong there) but consistency, cross-topology
transfer, and the ability to warm-start NR.

## Method

**Impedance-weighted graph.** Each bus is a node with features `[P, Q, is_DER,
is_slack]`; each in-service branch is an undirected edge weighted by the magnitude of
its per-unit series admittance `|y| = 1/sqrt(R^2 + X^2)`. Normally-open tie switches are
excluded, so the base graph is the true radial spanning tree. The edge weight gives the
network a physical sense of electrical distance: stiff, low-impedance branches couple
their endpoints strongly, high-impedance branches weakly. The model is `ProxySolverGNN`
— three `TAGConv` layers (hidden widths 128/64/32, default hop counts `ks=(3,3,2)`) with
a linear head predicting standardized `[V, theta]`. `TAGConv` aggregates a K-hop
electrical neighborhood per layer, and the admittance weights modulate every message.

**Scenarios and NR supervision.** Training scenarios are drawn by randomly scaling loads
over `(0.5, 1.5)` and dispatching active/reactive power at the DER buses `[10, 20, 30]`,
then solving each with Newton-Raphson in pandapower to obtain ground-truth `[V (pu),
theta (rad)]`. The slack (ext_grid) bus anchors the angle reference. Features and targets
are standardized on the training split.

**Physics-informed power-balance residual.** The bus admittance matrix `Y` yields the
complex nodal power `S = V ⊙ conj(Y V)`. The mismatch between this and the scheduled
injections is a differentiable residual added to the supervised objective as
`MSE + lambda · residual`, so fine-tuning explicitly penalizes violations of the AC
power-balance law rather than only regression error against NR.

**Topology reconfiguration enumeration.** Valid radial reconfigurations are enumerated by
closing each tie switch and opening one line on the loop it creates, keeping only
configurations that remain connected radial spanning trees. This yields 59 valid radial
configurations for the feeder, which are partitioned into seen and held-out sets for
topology-general training and evaluation.

## Experiments and results

Headline metrics on a held-out test set (5000 scenarios, 300 epochs supervised; physics
fine-tuning at `lambda=10`):

| Stage | Voltage R2 | Voltage MAE | Angle R2 | Angle MAE | \|dP\| (MW) | \|dQ\| (MVAr) |
|---|---:|---:|---:|---:|---:|---:|
| Supervised | 0.983 | 2.57 mV/pu | 0.931 | 0.12 deg | 0.261 | 0.235 |
| Physics-informed (lambda=10) | 0.983 | 2.61 mV/pu | 0.927 | — | 0.107 | 0.110 |

![](figures/training_curve.png)

![](figures/parity.png)

The parity plot shows predicted versus NR voltages clustering on the diagonal across the
test scenarios. Against analytical baselines (flat 1.0 pu, per-bus mean, and LinDistFlow),
the GNN reaches roughly 2.6 mV/pu voltage MAE on the base feeder. LinDistFlow — the linear
DistFlow approximation — is itself very strong on a single radial feeder, so the GNN is
competitive rather than dominant here; its advantages are differentiability,
physics-consistency, and cross-topology transfer, developed below.

![](figures/baseline_comparison.png)

### Physics-informed fine-tuning

Fine-tuning with the power-balance residual at `lambda=10` roughly halves the AC
power-balance violation at negligible accuracy cost. Active-power mismatch `|dP|` drops
from 0.254 to 0.107 MW (about 58% lower) and reactive `|dQ|` from 0.235 to 0.110 MVAr
(about 53% lower), while voltage R2 moves 0.984 -> 0.983 (MAE 2.44 -> 2.61 mV/pu) and
angle R2 0.930 -> 0.927. The predictions become more physically self-consistent without
meaningfully sacrificing agreement with NR.

![](figures/physics_before_after.png)

### K / edge-weight ablation (single topology)

Sweeping the `TAGConv` hop count K from 1 to 4, with impedance edge weights on and off,
on the single base topology (held-out voltage MAE, mV/pu):

| K | edge weights ON | edge weights OFF |
|---|---:|---:|
| 1 | 4.52 | 4.53 |
| 2 | 3.28 | 3.28 |
| 3 | 2.46 | 2.31 |
| 4 | 1.98 | 1.46 |

Hop count K dominates single-topology accuracy — going from K=1 to K=4 more than halves
the MAE, consistent with voltage at a bus depending on injections several hops away along
the feeder. The best single-topology configuration is K=4 with edge weights off, at
1.46 mV/pu (R2 = 0.9916). On one fixed topology the impedance weights are close to neutral
and slightly better off at high K: when the graph never changes, the network can absorb
the fixed electrical structure into its learned weights without needing the explicit edge
signal.

![](figures/ablation_k.png)

### Topology-general training

Training a single model across 42 seen reconfigurations (6000 scenarios, 200 epochs) and
evaluating on 17 held-out configurations tests whether the surrogate learns switching-
invariant structure rather than memorizing one tree. The topology-general model reaches
mean voltage MAE 12.20 mV/pu on seen configurations and 18.18 mV/pu on held-out ones; a
single-topology model evaluated on the same held-out configurations reaches only
19.88 mV/pu. Training across reconfigurations transfers to unseen topologies
(18.18 < 19.88). Absolute errors are larger than in the single-topology setting because
the model must fit a whole family of feeders at once.

### Phase 1 edge-weight topology ablation

This is the ablation that reconciles the single-topology result above with the
"topology-aware" claim. Two otherwise identical multi-topology models are trained, differing
only in the impedance edge-weight switch (mean voltage MAE, mV/pu):

| Edge weights | Seen | Held-out | Generalization gap |
|---|---:|---:|---:|
| ON | 12.46 | 18.14 | +5.68 |
| OFF | 11.15 | 18.79 | +7.64 |

Turning the impedance weights on fits the seen topologies slightly *less* tightly (12.46
vs 11.15) yet generalizes *better* to unseen ones (18.14 vs 18.79, about 3.4% lower error)
and shrinks the generalization gap from 7.64 to 5.68. This resolves the tension with the
single-topology ablation, where the weights looked neutral: the impedance weighting is an
inductive bias whose value appears specifically when the topology changes. On a fixed graph
the network can bake in the structure; when the graph is reconfigured, the explicit
per-branch admittance tells the network how the new tree redistributes coupling, so it
extrapolates rather than overfits the seen wiring. This is precisely what justifies calling
the model topology-aware. The margin is modest (3.4%) and from a single seed, so it is
reported as directional rather than definitive.

![](figures/ablation_edgeweights_topology.png)

### Phase 2 NR warm-start

Beyond acting as a standalone predictor, the surrogate can seed Newton-Raphson. Over 500
scenarios on the base feeder, flat-start NR takes 3.26 iterations on average; initializing
NR from the GNN prediction reduces this to 3.00 (8.0% fewer). The warm start is never worse
than a flat start (100% of scenarios) and strictly faster in 26.2% of them; wall-clock
falls from 22.6 to 19.2 ms (x1.18). Both starts converge to an identical solution
(max `|V|` discrepancy about 7e-9 pu), so the acceleration comes at no cost to correctness.

The surrogate is thus a safe, measurable NR accelerator. Headroom is modest here because a
33-bus feeder is small and well-conditioned — NR needs only about 3.3 iterations from flat —
and it would grow on larger, stiffer systems and at tighter convergence tolerances. At this
size wall-clock is dominated by pandapower's per-call setup, so iteration count is the
scale-free metric to read.

![](figures/warmstart_nr.png)

## Honest findings and limitations

- **Single feeder, mostly single seed.** All results are on the 33-bus feeder, and the
  topology and warm-start experiments use a single seed. Multi-seed error bars are needed
  before treating small margins as settled.
- **Modest margins where it matters most.** The edge-weight generalization benefit (3.4%)
  and the warm-start iteration saving (8%) are real but small; they are reported as
  directional evidence for the topology-aware inductive bias, not as large effects.
- **A strong linear baseline.** LinDistFlow is very accurate on a single radial feeder, so
  the GNN's edge is differentiability, physics-consistency, and cross-topology transfer
  rather than raw single-feeder voltage accuracy.
- **Data generation is CPU-bound.** NR scenario generation runs in pandapower on CPU, so the
  practical scaling limit is scenario generation throughput, not GPU training.
- **Soft physics constraint.** The power-balance residual is a penalty, not a hard feasibility
  guarantee; predictions are consistent, not exactly feasible.

Future work: extend to the 69-bus feeder and cross-feeder transfer; add multi-seed error
bars; and replace the soft penalty with a differentiable NR correction layer or a
hard-feasibility projection.

## Reproducibility

Install (Python >= 3.9; `numpy<2.1`, `pandas==2.2.2`):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    POSIX: source .venv/bin/activate
pip install -e .            # add ".[test]" for the test suite
```

Run the test suite (5 tests: topology build structure; reconfigurations are radial spanning
trees; Ybus power-balance residual ~0 on a true NR solution; LinDistFlow zero-injection flat;
LinDistFlow tracks NR):

```bash
python -m pytest -q
```

Regenerate each result (every script also accepts `--quick` for a fast smoke run):

```bash
# Supervised surrogate  -> checkpoints/gnn_powerflow.pt, results/supervised.json
python scripts/train.py --samples 5000 --epochs 300

# Physics-informed fine-tuning  -> checkpoints/gnn_powerflow_pinn.pt, results/physics_finetune.json
python scripts/finetune_physics.py --ckpt checkpoints/gnn_powerflow.pt --lam 10

# Full evaluation: baselines, physics consistency, topology generalization (prints to console)
python scripts/evaluate.py --ckpt checkpoints/gnn_powerflow_pinn.pt

# K / edge-weight ablation (single topology)  -> results/ablation_k.csv
python scripts/ablate_k.py --ks 1 2 3 4

# Topology-general training  -> checkpoints/gnn_topogeneral.pt, results/topology_general.json
python scripts/train_topology_general.py --samples 6000 --epochs 200

# Phase 1 edge-weight topology ablation (mean +/- std over seeds)
#   -> results/ablation_edgeweights_topology.json
python scripts/ablate_edgeweights_topology.py --samples 6000 --epochs 200 --seeds 42 43 44

# Phase 2 NR warm-start  -> results/warmstart_nr.json, results/warmstart_nr_per_scenario.csv
python scripts/warmstart_nr.py --n 500

# All figures  -> figures/*.png  (reads results/ and checkpoints/)
python scripts/make_figures.py
```

Figures: `training_curve.png`, `parity.png`, `baseline_comparison.png`,
`physics_before_after.png`, `ablation_k.png`, `ablation_edgeweights_topology.png`,
`warmstart_nr.png` (all under `figures/`).

Repository: https://github.com/ra-emami/topology-aware-gnn-power-flow

Stack: PyTorch, PyTorch Geometric (`TAGConv`), pandapower, scikit-learn, matplotlib, numba.
Reported numbers are from full runs; a fresh run may vary slightly with seed and hardware.
