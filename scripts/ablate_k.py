#!/usr/bin/env python
"""Ablation: TAGConv hop count K and impedance edge weights.

Sweeps the TAGConv ``K`` (applied to all three message-passing layers) and toggles
the per-unit admittance edge weights on/off, retraining a fresh surrogate for each
combination on a *shared* train/val/test split so that only the architecture
changes. Reports how each factor affects voltage accuracy (and angle / power
balance as secondary metrics), isolating the contribution of multi-hop,
impedance-aware propagation.

Example:
    python scripts/ablate_k.py --ks 1 2 3 4
    python scripts/ablate_k.py --quick
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch_geometric.loader import DataLoader

from pignn import (generate_dataset, split_dataset, fit_scalers, standardize,
                   ProxySolverGNN, train_supervised, evaluate_model)


def run_one(ks, use_edge_weights, train_loader, val_loader, test_set, scalers,
            args, device):
    """Train one configuration from scratch and return its held-out metrics."""
    torch.manual_seed(args.seed)  # same init regime across configs
    model = ProxySolverGNN(ks=ks, use_edge_weights=use_edge_weights).to(device)
    train_supervised(model, train_loader, val_loader, epochs=args.epochs,
                     lr=args.lr, device=device, log_every=0)
    return evaluate_model(model, test_set, scalers, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="hop counts to sweep (applied to all three TAGConv layers)")
    ap.add_argument("--out", default="results/ablation_k.csv")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()
    if args.quick:
        args.samples, args.epochs, args.ks = 400, 20, [1, 3]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # One dataset + split + scalers, shared by every configuration so the only
    # difference between runs is the model architecture.
    dataset = generate_dataset(args.samples, seed=args.seed)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    scalers = fit_scalers(train_set)
    print(f"train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    train_loader = DataLoader(standardize(train_set, scalers),
                              batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(standardize(val_set, scalers), batch_size=args.batch)

    rows = []
    n_runs = len(args.ks) * 2
    i = 0
    for use_ew in (True, False):
        for k in args.ks:
            i += 1
            ks = (k, k, k)
            tag = f"K={k} edge_weights={'on' if use_ew else 'off'}"
            print(f"\n[{i}/{n_runs}] training {tag} ...")
            m = run_one(ks, use_ew, train_loader, val_loader, test_set,
                        scalers, args, device)
            rows.append(dict(K=k, edge_weights="on" if use_ew else "off",
                             V_r2=m["V_r2"], V_mae=m["V_mae"], V_max=m["V_max"],
                             th_r2=m["th_r2"], dP=m["dP"]))
            print(f"      V R2={m['V_r2']:.4f}  V MAE={m['V_mae']:.3f} mV/pu  "
                  f"angle R2={m['th_r2']:.4f}  |dP|={m['dP']:.4f} MW")

    # Results table, grouped by edge-weight setting.
    print("\n=== K-ablation: held-out voltage accuracy ===")
    header = f"{'edge wts':<9}{'K':>3}{'V R2':>9}{'V MAE':>10}{'V max':>9}{'ang R2':>9}{'|dP|':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['edge_weights']:<9}{r['K']:>3}{r['V_r2']:>9.4f}"
              f"{r['V_mae']:>10.3f}{r['V_max']:>9.2f}{r['th_r2']:>9.4f}{r['dP']:>9.4f}")
    print("(V MAE / V max in mV/pu, |dP| in MW)")

    # Headline contrasts: best K, and the edge-weight effect at the best K.
    best = min(rows, key=lambda r: r["V_mae"])
    print(f"\nBest config: K={best['K']} edge_weights={best['edge_weights']} "
          f"-> V MAE={best['V_mae']:.3f} mV/pu (R2={best['V_r2']:.4f})")

    by = {(r["K"], r["edge_weights"]): r for r in rows}
    print("\nImpedance edge-weight effect (V MAE on -> off, mV/pu):")
    for k in args.ks:
        on, off = by.get((k, "on")), by.get((k, "off"))
        if on and off:
            d = off["V_mae"] - on["V_mae"]
            sign = "+" if d >= 0 else ""
            print(f"  K={k}: {on['V_mae']:.3f} -> {off['V_mae']:.3f}  "
                  f"(removing weights changes MAE by {sign}{d:.3f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "edge_weights", "V_r2", "V_mae",
                                          "V_max", "th_r2", "dP"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved ablation table -> {args.out}")


if __name__ == "__main__":
    main()
