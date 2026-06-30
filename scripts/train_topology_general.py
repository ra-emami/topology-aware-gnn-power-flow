#!/usr/bin/env python
"""Topology-general training (next research step).

Trains the surrogate across many radial reconfigurations of the feeder and
evaluates on *held-out* configurations it never saw, measuring true cross-topology
generalization. If a single-topology checkpoint exists, it is evaluated on the same
held-out configurations for contrast.

Example:
    python scripts/train_topology_general.py --samples 6000 --epochs 200
    python scripts/train_topology_general.py --quick
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from pignn import (enumerate_radial_configs, generate_multitopology_dataset,
                   split_dataset, fit_scalers, standardize, ProxySolverGNN,
                   train_supervised, evaluate_on_topologies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.3, help="fraction of configs held out")
    ap.add_argument("--out", default="checkpoints/gnn_topogeneral.pt")
    ap.add_argument("--single", default="checkpoints/gnn_powerflow.pt",
                    help="single-topology checkpoint to contrast against (optional)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.samples, args.epochs = 1500, 40

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Split the configuration pool into seen (train) and held-out (test) topologies.
    configs = enumerate_radial_configs()
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(configs))
    n_hold = max(2, int(args.holdout * len(configs)))
    test_cfgs = [configs[i] for i in order[:n_hold]]
    train_cfgs = [configs[i] for i in order[n_hold:]]
    print(f"Topologies: {len(train_cfgs)} train / {len(test_cfgs)} held-out "
          f"(total {len(configs)})")

    # Train on scenarios drawn across the seen topologies only.
    dataset = generate_multitopology_dataset(args.samples, train_cfgs, seed=args.seed)
    tr, va, te = split_dataset(dataset, seed=args.seed)
    scalers = fit_scalers(tr)
    train_loader = DataLoader(standardize(tr, scalers), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(standardize(va, scalers), batch_size=args.batch)

    model = ProxySolverGNN().to(device)
    train_supervised(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), **scalers,
                "train_configs": [c["name"] for c in train_cfgs],
                "test_configs": [c["name"] for c in test_cfgs]}, args.out)
    print(f"Saved topology-general checkpoint -> {args.out}")

    # Cross-topology generalization.
    seen = evaluate_on_topologies(model, train_cfgs, scalers, device, n=60)
    unseen = evaluate_on_topologies(model, test_cfgs, scalers, device, n=60)
    print("\n=== Topology-general model ===")
    print(f"  seen topologies   : mean voltage MAE = {np.mean(list(seen.values())):.3f} mV/pu")
    print(f"  HELD-OUT topologies: mean voltage MAE = {np.mean(list(unseen.values())):.3f} mV/pu")

    # Contrast: how does a single-topology model do on the same held-out configs?
    if os.path.exists(args.single):
        sc_ck = torch.load(args.single, map_location=device)
        sc_scalers = {k: sc_ck[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
        single = ProxySolverGNN().to(device)
        single.load_state_dict(sc_ck["model"])
        unseen_single = evaluate_on_topologies(single, test_cfgs, sc_scalers, device, n=60)
        print("\n=== Single-topology model (for contrast) ===")
        print(f"  HELD-OUT topologies: mean voltage MAE = "
              f"{np.mean(list(unseen_single.values())):.3f} mV/pu")
        print("\nLower held-out error for the topology-general model indicates it transfers "
              "to unseen network configurations.")
    else:
        print(f"\n(No single-topology checkpoint at {args.single} to contrast against — "
              "run scripts/train.py first.)")


if __name__ == "__main__":
    main()
