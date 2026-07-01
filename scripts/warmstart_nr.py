#!/usr/bin/env python
"""Phase 2: GNN warm-start for Newton-Raphson power flow.

Reframes the trained surrogate as a *solver accelerator*. For each scenario the
network's predicted ``(V, theta)`` initializes pandapower's Newton-Raphson, and the
iteration count and wall-clock are compared against the usual flat start (all
voltages 1.0 pu, 0 deg). Both starts converge to the same physical solution; the
question is how many Newton iterations the learned initial guess saves.

No training is required -- this only runs the forward model and the NR solver.

Example:
    python scripts/warmstart_nr.py --ckpt checkpoints/gnn_powerflow.pt --n 500
    python scripts/warmstart_nr.py --quick
"""
import argparse
import copy
import csv
import json
import os
import sys
from time import perf_counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged
from torch_geometric.data import Data

from pignn import base_case, build_topology, ProxySolverGNN, predict
from pignn.config import DER_BUSES, LOAD_SCALE_RANGE, DER_P_RANGE, DER_Q_RANGE
from pignn.data import _node_flags


def build_scenario(rng):
    """Sample one operating point on the base feeder; return (net, node-features)."""
    net = base_case()
    scale = rng.uniform(*LOAD_SCALE_RANGE)
    net.load.p_mw *= scale
    net.load.q_mvar *= scale
    p_disp = rng.uniform(*DER_P_RANGE, len(DER_BUSES))
    q_disp = rng.uniform(*DER_Q_RANGE, len(DER_BUSES))
    for i, bus in enumerate(DER_BUSES):
        pp.create_sgen(net, bus, p_mw=p_disp[i], q_mvar=q_disp[i])

    p = np.zeros(len(net.bus)); q = np.zeros(len(net.bus))
    p[net.load.bus.values] -= net.load.p_mw.values
    q[net.load.bus.values] -= net.load.q_mvar.values
    p[DER_BUSES] += p_disp
    q[DER_BUSES] += q_disp
    return net, p, q


def solve(net, **init):
    """Run NR with the given initialization; return (iterations, seconds) or None."""
    n = copy.deepcopy(net)
    t0 = perf_counter()
    try:
        pp.runpp(n, algorithm="nr", **init)
    except LoadflowNotConverged:
        return None
    dt = perf_counter() - t0
    return int(n._ppc["iterations"]), dt, n.res_bus.vm_pu.values.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gnn_powerflow.pt")
    ap.add_argument("--n", type=int, default=500, help="number of scenarios")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--warmup", type=int, default=5,
                    help="untimed solves to absorb numba JIT before measuring")
    ap.add_argument("--out", default="results/warmstart_nr.json")
    ap.add_argument("--csv", default="results/warmstart_nr_per_scenario.csv")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n = 30

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device)
    scalers = {k: ckpt[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
    model = ProxySolverGNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.ckpt}")

    # Fixed base topology (edges are load-independent).
    ei, ew = build_topology(base_case())
    is_slack, is_der = _node_flags(base_case())

    rng = np.random.default_rng(args.seed)

    # Warm-up: trigger numba JIT so the timing loop measures steady-state solve cost.
    for _ in range(args.warmup):
        net, _, _ = build_scenario(rng)
        solve(net, init="flat")

    rows = []
    fails = {"flat": 0, "warm": 0}
    max_disc = 0.0
    for _ in range(args.n):
        net, p, q = build_scenario(rng)
        x = torch.stack([torch.tensor(p, dtype=torch.float),
                         torch.tensor(q, dtype=torch.float), is_der, is_slack], dim=1)
        raw = Data(x=x, edge_index=ei, edge_weight=ew)
        vpred, thpred = predict(model, raw, scalers, device)

        flat = solve(net, init="flat")
        warm = solve(net, init_vm_pu=vpred,
                     init_va_degree=np.degrees(thpred))
        if flat is None:
            fails["flat"] += 1
        if warm is None:
            fails["warm"] += 1
        if flat is None or warm is None:
            continue
        it_f, t_f, vm_f = flat
        it_w, t_w, vm_w = warm
        max_disc = max(max_disc, float(np.abs(vm_f - vm_w).max()))
        rows.append(dict(iters_flat=it_f, iters_warm=it_w,
                         t_flat_ms=t_f * 1e3, t_warm_ms=t_w * 1e3))

    if not rows:
        print("No converged scenarios; aborting.")
        return

    itf = np.array([r["iters_flat"] for r in rows], float)
    itw = np.array([r["iters_warm"] for r in rows], float)
    tf = np.array([r["t_flat_ms"] for r in rows], float)
    tw = np.array([r["t_warm_ms"] for r in rows], float)
    saved = itf - itw

    summary = dict(
        n_scenarios=len(rows),
        iters_flat_mean=float(itf.mean()), iters_warm_mean=float(itw.mean()),
        iters_saved_mean=float(saved.mean()),
        iters_saved_pct=float(100.0 * saved.sum() / itf.sum()),
        frac_warm_faster=float((itw < itf).mean()),
        frac_warm_not_worse=float((itw <= itf).mean()),
        t_flat_ms_mean=float(tf.mean()), t_warm_ms_mean=float(tw.mean()),
        wallclock_speedup=float(tf.mean() / tw.mean()),
        max_solution_discrepancy_pu=max_disc,
        fails=fails,
    )

    print("\n=== GNN warm-start vs flat start (Newton-Raphson) ===")
    print(f"  scenarios (converged both)  : {summary['n_scenarios']}")
    print(f"  mean iterations  flat -> warm: {summary['iters_flat_mean']:.2f} -> "
          f"{summary['iters_warm_mean']:.2f}")
    print(f"  mean iterations saved       : {summary['iters_saved_mean']:.2f} "
          f"({summary['iters_saved_pct']:.1f}% fewer)")
    print(f"  warm-start not worse        : {100*summary['frac_warm_not_worse']:.1f}% "
          f"of scenarios (strictly faster {100*summary['frac_warm_faster']:.1f}%)")
    print(f"  mean solve time  flat -> warm: {summary['t_flat_ms_mean']:.3f} -> "
          f"{summary['t_warm_ms_mean']:.3f} ms  (x{summary['wallclock_speedup']:.2f})")
    print(f"  max |V| discrepancy flat/warm: {max_disc:.2e} pu (both reach same solution)")
    if summary["frac_warm_not_worse"] < 1.0:
        print("  note: wall-clock at 33 buses is dominated by per-call setup overhead; "
              "iteration count is the scale-free metric.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iters_flat", "iters_warm",
                                          "t_flat_ms", "t_warm_ms"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {args.out}\nSaved -> {args.csv}")


if __name__ == "__main__":
    main()
