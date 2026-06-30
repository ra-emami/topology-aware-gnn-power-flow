#!/usr/bin/env python
"""Physics-informed fine-tuning of a trained checkpoint.

Adds the power-balance residual to the objective and reports before/after metrics.

Example:
    python scripts/finetune_physics.py --ckpt checkpoints/gnn_powerflow.pt --lam 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch_geometric.loader import DataLoader

from pignn import (base_case, generate_dataset, standardize, ProxySolverGNN,
                   make_physics_residual, finetune_physics, evaluate_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gnn_powerflow.pt")
    ap.add_argument("--out", default="checkpoints/gnn_powerflow_pinn.pt")
    ap.add_argument("--lam", type=float, default=10.0, help="physics penalty weight")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.samples, args.epochs = 600, 15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    scalers = {k: ckpt[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
    model = ProxySolverGNN().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.ckpt}")

    # Fine-tuning + evaluation data, standardized with the checkpoint's own scalers.
    train_set = generate_dataset(args.samples, seed=7)
    eval_set = generate_dataset(max(args.samples // 3, 200), seed=123)
    train_loader = DataLoader(standardize(train_set, scalers), batch_size=args.batch, shuffle=True)

    before = evaluate_model(model, eval_set, scalers, device)
    residual = make_physics_residual(base_case(), scalers, device)

    import copy
    model_pinn = copy.deepcopy(model).to(device)
    finetune_physics(model_pinn, train_loader, residual, lam=args.lam,
                     epochs=args.epochs, lr=args.lr, device=device)
    after = evaluate_model(model_pinn, eval_set, scalers, device)

    print(f"\n{'metric':<22}{'before':>12}{'after':>12}")
    for label, key in [("Voltage R2", "V_r2"), ("Voltage MAE (mV/pu)", "V_mae"),
                       ("Angle R2", "th_r2"), ("Angle MAE (deg)", "th_mae"),
                       ("|dP| (MW)", "dP"), ("|dQ| (MVAr)", "dQ")]:
        print(f"{label:<22}{before[key]:>12.4f}{after[key]:>12.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model_pinn.state_dict(), **scalers, "lambda_phys": args.lam}, args.out)
    print(f"\nSaved physics-informed checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
