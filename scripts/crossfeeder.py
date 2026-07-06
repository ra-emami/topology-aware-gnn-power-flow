#!/usr/bin/env python
"""Phase 4: cross-feeder evaluation between the 33-bus and 69-bus systems.

The GNN is size-agnostic — the same weights run on any graph — so a model trained
on one feeder can be applied zero-shot to another. This script evaluates, on a
fresh scenario set for each target feeder:

- the feeder's *native* model (trained on that feeder), when its checkpoint exists;
- the *other* feeder's model applied zero-shot (transfer);
- the LinDistFlow linear baseline, for context.

The comparison quantifies how much of the surrogate's skill is feeder-specific
versus transferable electrical structure.

Example:
    python scripts/crossfeeder.py --samples 500
    python scripts/crossfeeder.py --quick
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from sklearn.metrics import r2_score

from pignn import (base_case, generate_dataset, lindistflow_matrices,
                   lindistflow_predict, ProxySolverGNN, predict)

CKPTS = {
    "case33bw": "checkpoints/gnn_powerflow.pt",
    "case69": "checkpoints/gnn_powerflow_case69.pt",
}


def load_model(ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return None, None
    ck = torch.load(ckpt_path, map_location=device)
    scalers = {k: ck[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
    model = ProxySolverGNN().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, scalers


def score_model(model, scalers, eval_set, device):
    """Voltage MAE (mV/pu) and R2 of a model on a raw scenario set."""
    tv, pv = [], []
    for raw in eval_set:
        gv, _ = predict(model, raw, scalers, device)
        tv += raw.y[:, 0].tolist(); pv += gv.tolist()
    tv, pv = np.array(tv), np.array(pv)
    return dict(V_mae=float(np.abs(tv - pv).mean() * 1000),
                V_r2=float(r2_score(tv, pv)))


def score_lindistflow(network, eval_set):
    net = base_case(network)
    R, X = lindistflow_matrices(net)
    sn = net.sn_mva
    tv, pv = [], []
    for raw in eval_set:
        vl = lindistflow_predict(raw.x[:, 0].numpy(), raw.x[:, 1].numpy(), R, X, sn)
        tv += raw.y[:, 0].tolist(); pv += vl.tolist()
    tv, pv = np.array(tv), np.array(pv)
    return dict(V_mae=float(np.abs(tv - pv).mean() * 1000),
                V_r2=float(r2_score(tv, pv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=500,
                    help="evaluation scenarios per target feeder")
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--ckpt33", default=CKPTS["case33bw"])
    ap.add_argument("--ckpt69", default=CKPTS["case69"])
    ap.add_argument("--out", default="results/crossfeeder.json")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.samples = 60

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    models = {}
    for name, path in (("case33bw", args.ckpt33), ("case69", args.ckpt69)):
        m, s = load_model(path, device)
        models[name] = (m, s)
        print(f"model[{name}]: {'loaded ' + path if m else 'not found (' + path + ')'}")

    results = {}
    for target in ("case33bw", "case69"):
        print(f"\n=== target feeder: {target} ===")
        eval_set = generate_dataset(args.samples, seed=args.seed, network=target)
        entry = {"lindistflow": score_lindistflow(target, eval_set)}
        for source in ("case33bw", "case69"):
            m, s = models[source]
            if m is None:
                continue
            kind = "native" if source == target else "zero-shot transfer"
            entry[source] = dict(kind=kind, **score_model(m, s, eval_set, device))
        results[target] = entry
        for k, v in entry.items():
            tag = v.get("kind", "baseline")
            print(f"  {k:<12} ({tag:<18}) V MAE = {v['V_mae']:8.3f} mV/pu   "
                  f"R2 = {v['V_r2']:.4f}")

    payload = dict(config=dict(samples=args.samples, seed=args.seed,
                               quick=args.quick),
                   results=results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
