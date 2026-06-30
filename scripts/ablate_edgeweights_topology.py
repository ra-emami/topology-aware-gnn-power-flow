#!/usr/bin/env python
"""Phase 1 ablation: do impedance edge weights improve topology generalization?

Trains the surrogate WITH and WITHOUT per-unit admittance edge weights on the
multi-topology task -- scenarios drawn across a set of *seen* radial
reconfigurations -- and evaluates both models on *held-out* reconfigurations they
never saw. This is the experiment that tests the project's central "topology-aware"
claim: the impedance edge weights tell the network how a rewired feeder behaves, so
they should help the model transfer to unseen network configurations.

Both variants share one dataset, one train/val/test split, and one set of scalers;
the only difference between the two models is whether the admittance edge weights
modulate the messages. Reports seen vs held-out voltage MAE for each variant and
states the conclusion.

Example:
    python scripts/ablate_edgeweights_topology.py --samples 6000 --epochs 200
    python scripts/ablate_edgeweights_topology.py --quick
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from pignn import (enumerate_radial_configs, generate_multitopology_dataset,
                   split_dataset, fit_scalers, standardize, ProxySolverGNN,
                   train_supervised, evaluate_on_topologies)


def train_variant(use_ew, train_loader, val_loader, args, device):
    """Train one fresh model; identical recipe save for the edge-weight switch."""
    torch.manual_seed(args.seed)  # same initialization regime for both variants
    model = ProxySolverGNN(use_edge_weights=use_ew).to(device)
    train_supervised(model, train_loader, val_loader, epochs=args.epochs,
                     lr=args.lr, device=device, log_every=args.log_every)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of reconfigurations held out for the unseen test")
    ap.add_argument("--eval-n", type=int, default=60,
                    help="scenarios sampled per topology at evaluation")
    ap.add_argument("--log-every", type=int, default=0, help="0 silences epoch logs")
    ap.add_argument("--out", default="results/ablation_edgeweights_topology.json")
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()
    if args.quick:
        args.samples, args.epochs, args.eval_n = 1500, 40, 30

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Split the reconfiguration pool into seen (train) and held-out (test).
    configs = enumerate_radial_configs()
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(configs))
    n_hold = max(2, int(args.holdout * len(configs)))
    test_cfgs = [configs[i] for i in order[:n_hold]]
    train_cfgs = [configs[i] for i in order[n_hold:]]
    print(f"Topologies: {len(train_cfgs)} seen / {len(test_cfgs)} held-out "
          f"(total {len(configs)})")

    # One dataset + split + scalers, shared by both variants so the only difference
    # between the two trained models is the use of impedance edge weights.
    dataset = generate_multitopology_dataset(args.samples, train_cfgs, seed=args.seed)
    tr, va, te = split_dataset(dataset, seed=args.seed)
    scalers = fit_scalers(tr)
    train_loader = DataLoader(standardize(tr, scalers), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(standardize(va, scalers), batch_size=args.batch)

    results = {}
    for use_ew in (True, False):
        tag = "on" if use_ew else "off"
        print(f"\n=== Training edge_weights={tag} ===")
        model = train_variant(use_ew, train_loader, val_loader, args, device)
        seen = evaluate_on_topologies(model, train_cfgs, scalers, device, n=args.eval_n)
        unseen = evaluate_on_topologies(model, test_cfgs, scalers, device, n=args.eval_n)
        seen_mae = float(np.mean(list(seen.values())))
        unseen_mae = float(np.mean(list(unseen.values())))
        results[tag] = dict(seen_mae=seen_mae, unseen_mae=unseen_mae,
                            gen_gap=unseen_mae - seen_mae,
                            seen_per_config=seen, unseen_per_config=unseen)
        print(f"  seen     mean voltage MAE = {seen_mae:.3f} mV/pu")
        print(f"  HELD-OUT mean voltage MAE = {unseen_mae:.3f} mV/pu  "
              f"(gen. gap {unseen_mae - seen_mae:+.3f})")

    # Verdict: compare held-out error with vs without the edge weights.
    on, off = results["on"], results["off"]
    print("\n=== Edge-weight effect on topology generalization ===")
    print(f"{'':<14}{'seen MAE':>12}{'held-out MAE':>16}")
    print(f"{'edge wts on':<14}{on['seen_mae']:>12.3f}{on['unseen_mae']:>16.3f}")
    print(f"{'edge wts off':<14}{off['seen_mae']:>12.3f}{off['unseen_mae']:>16.3f}")
    delta = off["unseen_mae"] - on["unseen_mae"]  # >0 means weights help
    pct = 100.0 * delta / off["unseen_mae"] if off["unseen_mae"] else 0.0
    if delta > 0:
        verdict = (f"Impedance edge weights IMPROVE generalization to unseen "
                   f"topologies: held-out voltage MAE {on['unseen_mae']:.3f} vs "
                   f"{off['unseen_mae']:.3f} mV/pu ({pct:.1f}% lower error with "
                   f"weights). This supports the 'topology-aware' claim.")
    elif delta < 0:
        verdict = (f"Impedance edge weights do NOT improve generalization to unseen "
                   f"topologies in this setting: held-out voltage MAE "
                   f"{on['unseen_mae']:.3f} (on) vs {off['unseen_mae']:.3f} (off) "
                   f"mV/pu (weights {-pct:.1f}% worse).")
    else:
        verdict = ("Impedance edge weights have no measurable effect on held-out "
                   "generalization in this setting.")
    print("\n" + verdict)

    payload = dict(
        config=dict(samples=args.samples, epochs=args.epochs, seed=args.seed,
                    holdout=args.holdout, eval_n=args.eval_n,
                    n_seen_cfgs=len(train_cfgs), n_held_out_cfgs=len(test_cfgs),
                    quick=args.quick),
        results=results, verdict=verdict)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
