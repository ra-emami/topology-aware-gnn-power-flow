#!/usr/bin/env python
"""Evaluate a checkpoint: baselines, physics consistency, undervoltage, topology generalization.

Example:
    python scripts/evaluate.py --ckpt checkpoints/gnn_powerflow_pinn.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from pignn import (generate_dataset, ProxySolverGNN, evaluate_model, baseline_scores,
                   undervoltage_screening, enumerate_radial_configs, evaluate_on_topologies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gnn_powerflow.pt")
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.samples = 300

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    scalers = {k: ckpt[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
    model = ProxySolverGNN().to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {args.ckpt} on {device}\n")

    test_set = generate_dataset(args.samples, seed=123)

    m = evaluate_model(model, test_set, scalers, device)
    print("Accuracy & physics:")
    print(f"  Voltage R2={m['V_r2']:.4f}  MAE={m['V_mae']:.3f} mV/pu  max={m['V_max']:.2f} mV/pu")
    print(f"  Angle   R2={m['th_r2']:.4f}  MAE={m['th_mae']:.4f} deg")
    print(f"  Power-balance |dP|={m['dP']:.4f} MW  |dQ|={m['dQ']:.4f} MVAr")
    worst = m["perbus"].argsort()[::-1][:5]
    print(f"  Highest-error buses: {worst.tolist()} -> {m['perbus'][worst].round(2).tolist()} mV/pu\n")

    print("Baselines (voltage):")
    for name, s in baseline_scores(model, test_set, scalers, device).items():
        print(f"  {name:<14} R2={s['R2']:.4f}  MAE={s['MAE_mV']:.3f} mV/pu  max={s['max_mV']:.2f} mV/pu")

    uv = undervoltage_screening(model, test_set, scalers, device)
    print(f"\nUndervoltage screening (V<0.95): precision={uv['precision']:.3f} "
          f"recall={uv['recall']:.3f} F1={uv['f1']:.3f}\n")

    print("Topology generalization (per-config voltage MAE, mV/pu):")
    configs = enumerate_radial_configs()
    res = evaluate_on_topologies(model, configs, scalers, device, n=60)
    for name, mae in list(res.items())[:8]:
        print(f"  {name:<18} {mae:.3f}")
    import numpy as np
    print(f"  mean over {len(res)} configs: {np.mean(list(res.values())):.3f} mV/pu")


if __name__ == "__main__":
    main()
