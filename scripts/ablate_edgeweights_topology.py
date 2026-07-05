#!/usr/bin/env python
"""Phase 1 ablation: do impedance edge weights improve topology generalization?

Trains the surrogate WITH and WITHOUT per-unit admittance edge weights on the
multi-topology task -- scenarios drawn across a set of *seen* radial
reconfigurations -- and evaluates both models on *held-out* reconfigurations they
never saw. This is the experiment that tests the project's central "topology-aware"
claim: the impedance edge weights tell the network how a rewired feeder behaves, so
they should help the model transfer to unseen network configurations.

Within each seed, both variants share one dataset, one train/val/test split, and one
set of scalers; the only difference between the two models is whether the admittance
edge weights modulate the messages. Multiple seeds (--seeds) vary the topology
holdout split, the scenario draw, and the initialization together, and the report
aggregates mean +/- std across seeds so the conclusion reflects run-to-run
variability rather than a single draw.

Example:
    python scripts/ablate_edgeweights_topology.py --samples 6000 --epochs 200 --seeds 42 43 44
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


def train_variant(use_ew, train_loader, val_loader, args, seed, device):
    """Train one fresh model; identical recipe save for the edge-weight switch."""
    torch.manual_seed(seed)  # same initialization regime for both variants
    model = ProxySolverGNN(use_edge_weights=use_ew).to(device)
    train_supervised(model, train_loader, val_loader, epochs=args.epochs,
                     lr=args.lr, device=device, log_every=args.log_every)
    return model


def run_seed(seed, configs, args, device):
    """One complete on/off comparison at a given seed; returns per-variant metrics."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(configs))
    n_hold = max(2, int(args.holdout * len(configs)))
    test_cfgs = [configs[i] for i in order[:n_hold]]
    train_cfgs = [configs[i] for i in order[n_hold:]]
    print(f"\n########## seed {seed}: {len(train_cfgs)} seen / {len(test_cfgs)} "
          f"held-out topologies ##########")

    # One dataset + split + scalers per seed, shared by both variants so the only
    # difference between the two trained models is the use of impedance edge weights.
    dataset = generate_multitopology_dataset(args.samples, train_cfgs, seed=seed)
    tr, va, te = split_dataset(dataset, seed=seed)
    scalers = fit_scalers(tr)
    train_loader = DataLoader(standardize(tr, scalers), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(standardize(va, scalers), batch_size=args.batch)

    out = {"seed": seed, "n_seen_cfgs": len(train_cfgs), "n_held_out_cfgs": len(test_cfgs)}
    for use_ew in (True, False):
        tag = "on" if use_ew else "off"
        print(f"=== seed {seed}: training edge_weights={tag} ===")
        model = train_variant(use_ew, train_loader, val_loader, args, seed, device)
        seen = evaluate_on_topologies(model, train_cfgs, scalers, device, n=args.eval_n)
        unseen = evaluate_on_topologies(model, test_cfgs, scalers, device, n=args.eval_n)
        seen_mae = float(np.mean(list(seen.values())))
        unseen_mae = float(np.mean(list(unseen.values())))
        out[tag] = dict(seen_mae=seen_mae, unseen_mae=unseen_mae,
                        gen_gap=unseen_mae - seen_mae)
        print(f"  seen     mean voltage MAE = {seen_mae:.3f} mV/pu")
        print(f"  HELD-OUT mean voltage MAE = {unseen_mae:.3f} mV/pu  "
              f"(gen. gap {unseen_mae - seen_mae:+.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                    help="seeds; each varies the holdout split, data draw, and init")
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
        args.seeds = args.seeds[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    configs = enumerate_radial_configs()
    print(f"Reconfiguration pool: {len(configs)} valid radial configs; "
          f"seeds: {args.seeds}")

    per_seed = [run_seed(s, configs, args, device) for s in args.seeds]

    # Aggregate across seeds: mean +/- std for each variant.
    results = {}
    for tag in ("on", "off"):
        seen = np.array([p[tag]["seen_mae"] for p in per_seed])
        unseen = np.array([p[tag]["unseen_mae"] for p in per_seed])
        gap = np.array([p[tag]["gen_gap"] for p in per_seed])
        results[tag] = dict(
            seen_mae=float(seen.mean()), seen_std=float(seen.std()),
            unseen_mae=float(unseen.mean()), unseen_std=float(unseen.std()),
            gen_gap=float(gap.mean()), gap_std=float(gap.std()))

    on, off = results["on"], results["off"]
    n = len(per_seed)
    print(f"\n=== Edge-weight effect on topology generalization "
          f"(mean +/- std over {n} seed{'s' if n > 1 else ''}) ===")
    print(f"{'':<14}{'seen MAE':>20}{'held-out MAE':>22}{'gen. gap':>18}")
    for tag in ("on", "off"):
        r = results[tag]
        print(f"{'edge wts ' + tag:<14}"
              f"{r['seen_mae']:>12.3f} +/-{r['seen_std']:<5.3f}"
              f"{r['unseen_mae']:>14.3f} +/-{r['unseen_std']:<5.3f}"
              f"{r['gen_gap']:>10.3f} +/-{r['gap_std']:<5.3f}")

    # Verdict: per-seed held-out wins, and the gap comparison, tell the story.
    unseen_wins = sum(p["on"]["unseen_mae"] < p["off"]["unseen_mae"] for p in per_seed)
    gap_wins = sum(p["on"]["gen_gap"] < p["off"]["gen_gap"] for p in per_seed)
    delta = off["unseen_mae"] - on["unseen_mae"]
    pct = 100.0 * delta / off["unseen_mae"] if off["unseen_mae"] else 0.0
    noise = abs(delta) <= (on["unseen_std"] + off["unseen_std"]) / max(np.sqrt(n), 1)
    verdict = (
        f"Across {n} seed(s): edge weights win on held-out voltage MAE in "
        f"{unseen_wins}/{n} seeds (mean {on['unseen_mae']:.3f} vs "
        f"{off['unseen_mae']:.3f} mV/pu, {pct:+.1f}%"
        f"{', within run-to-run noise' if noise else ''}); they yield the smaller "
        f"generalization gap in {gap_wins}/{n} seeds "
        f"(mean {on['gen_gap']:.3f} vs {off['gen_gap']:.3f} mV/pu). "
        + ("The consistent gap reduction supports the topology-aware inductive "
           "bias; the held-out MAE margin should be read as directional."
           if gap_wins > n / 2 else
           "Neither metric consistently favors the edge weights in this setting."))
    print("\n" + verdict)

    payload = dict(
        config=dict(samples=args.samples, epochs=args.epochs, seeds=args.seeds,
                    holdout=args.holdout, eval_n=args.eval_n, quick=args.quick),
        per_seed=per_seed, results=results, verdict=verdict)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
