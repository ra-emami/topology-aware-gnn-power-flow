#!/usr/bin/env python
"""Train the supervised GNN surrogate on a feeder.

Example:
    python scripts/train.py --samples 5000 --epochs 300
    python scripts/train.py --network case69 --samples 5000 --epochs 300
    python scripts/train.py --quick
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch_geometric.loader import DataLoader

from pignn import (base_case, generate_dataset, split_dataset, fit_scalers,
                   standardize, ProxySolverGNN, train_supervised, evaluate_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--network", default="case33bw", choices=["case33bw", "case69"],
                    help="feeder to train on")
    ap.add_argument("--out", default="checkpoints/gnn_powerflow.pt")
    ap.add_argument("--metrics", default="results/supervised.json",
                    help="where to save the training curve + held-out metrics")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()
    if args.quick:
        args.samples, args.epochs = 400, 20
    # Keep per-network artifacts apart when the defaults were not overridden.
    if args.network != "case33bw":
        if args.out == "checkpoints/gnn_powerflow.pt":
            args.out = f"checkpoints/gnn_powerflow_{args.network}.pt"
        if args.metrics == "results/supervised.json":
            args.metrics = f"results/supervised_{args.network}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | network: {args.network}")

    dataset = generate_dataset(args.samples, seed=args.seed, network=args.network)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    scalers = fit_scalers(train_set)
    print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    train_loader = DataLoader(standardize(train_set, scalers), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(standardize(val_set, scalers), batch_size=args.batch)

    model = ProxySolverGNN().to(device)
    history = train_supervised(model, train_loader, val_loader,
                               epochs=args.epochs, lr=args.lr, device=device)

    metrics = evaluate_model(model, test_set, scalers, device,
                             net=base_case(args.network))
    print(f"\nHeld-out test | V R2={metrics['V_r2']:.4f} MAE={metrics['V_mae']:.3f} mV/pu | "
          f"angle R2={metrics['th_r2']:.4f} MAE={metrics['th_mae']:.4f} deg | "
          f"|dP|={metrics['dP']:.4f} MW")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), **scalers, "network": args.network}, args.out)
    print(f"Saved checkpoint -> {args.out}")

    # Persist the training curve and held-out metrics for figures and the report.
    scalar_keys = ("V_r2", "V_mae", "V_max", "th_r2", "th_mae", "dP", "dQ")
    payload = dict(
        config=dict(samples=args.samples, epochs=args.epochs, batch=args.batch,
                    lr=args.lr, seed=args.seed, network=args.network,
                    quick=args.quick),
        history=dict(train=history["train"], val=history["val"],
                     best_val=history["best_val"]),
        test={k: float(metrics[k]) for k in scalar_keys},
        test_perbus_mae_mv=metrics["perbus"].tolist(),
    )
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics -> {args.metrics}")


if __name__ == "__main__":
    main()
