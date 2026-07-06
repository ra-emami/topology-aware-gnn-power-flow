#!/usr/bin/env python
"""Phase 3: regenerate all portfolio figures into figures/.

Reads the metrics that the training/ablation/warm-start scripts persist under
results/ (JSON/CSV) and the trained checkpoints, and renders the project's key
plots: the supervised training curve, voltage/angle parity, the baseline
comparison, the physics-informed before/after residual and per-bus error, the
TAGConv K ablation, the topology-general edge-weight ablation (Phase 1), and the
Newton-Raphson warm-start (Phase 2).

Each figure is generated independently: a missing input file (e.g. a result that
has not been produced yet) skips only that figure with a note, so the script is
useful both after a full pipeline run and incrementally.

Example:
    python scripts/make_figures.py                 # full-resolution eval set
    python scripts/make_figures.py --quick         # tiny eval set, fast
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pignn import (base_case, generate_dataset, ProxySolverGNN, predict,
                   evaluate_model, baseline_scores)

RESULTS = "results"
FIGS = "figures"


def _load_json(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


def _load_model(ckpt, device):
    """Return (model, scalers) from a checkpoint, or (None, None) if absent."""
    if not os.path.exists(ckpt):
        return None, None
    ck = torch.load(ckpt, map_location=device)
    scalers = {k: ck[k] for k in ("x_mean", "x_std", "y_mean", "y_std")}
    model = ProxySolverGNN().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, scalers


# --------------------------------------------------------------------------- #
# Figures driven purely by saved result files
# --------------------------------------------------------------------------- #
def fig_training_curve():
    d = _load_json("supervised.json")
    if not d or "history" not in d:
        print("  [skip] training_curve: results/supervised.json not found")
        return
    tr, va = d["history"]["train"], d["history"]["val"]
    ep = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, tr, label="train", lw=1.8)
    ax.plot(ep, va, label="validation", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss (standardized)")
    ax.set_title("Supervised training curve")
    ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "training_curve.png")


def fig_ablation_k():
    path = os.path.join(RESULTS, "ablation_k.csv")
    if not os.path.exists(path):
        print("  [skip] ablation_k: results/ablation_k.csv not found")
        return
    rows = list(csv.DictReader(open(path)))
    fig, ax = plt.subplots(figsize=(6, 4))
    for ew, marker in (("on", "o"), ("off", "s")):
        sub = sorted([r for r in rows if r["edge_weights"] == ew],
                     key=lambda r: int(r["K"]))
        if sub:
            ks = [int(r["K"]) for r in sub]
            mae = [float(r["V_mae"]) for r in sub]
            ax.plot(ks, mae, marker=marker, lw=1.8, label=f"edge weights {ew}")
    ax.set_xlabel("TAGConv hop count K"); ax.set_ylabel("held-out voltage MAE (mV/pu)")
    ax.set_title("Ablation: hop count K and impedance edge weights")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xticks(sorted({int(r["K"]) for r in rows}))
    _save(fig, "ablation_k.png")


def fig_ablation_edgeweights_topology():
    d = _load_json("ablation_edgeweights_topology.json")
    if not d:
        print("  [skip] ablation_edgeweights_topology: result not found")
        return
    on, off = d["results"]["on"], d["results"]["off"]
    groups = ["seen topologies", "held-out topologies"]
    on_vals = [on["seen_mae"], on["unseen_mae"]]
    off_vals = [off["seen_mae"], off["unseen_mae"]]
    # Multi-seed results carry std across seeds; single-seed results have none.
    on_err = [on.get("seen_std", 0), on.get("unseen_std", 0)]
    off_err = [off.get("seen_std", 0), off.get("unseen_std", 0)]
    n_seeds = len(d.get("per_seed", [])) or 1
    x = np.arange(len(groups)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - w / 2, on_vals, w, yerr=on_err, capsize=4, label="edge weights on")
    b2 = ax.bar(x + w / 2, off_vals, w, yerr=off_err, capsize=4, label="edge weights off")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("mean voltage MAE (mV/pu)")
    title = "Edge weights and topology generalization"
    if n_seeds > 1:
        title += f" (mean ± std, {n_seeds} seeds)"
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, "ablation_edgeweights_topology.png")


def fig_crossfeeder():
    d = _load_json("crossfeeder.json")
    if not d:
        print("  [skip] crossfeeder: results/crossfeeder.json not found")
        return
    targets = [t for t in ("case33bw", "case69") if t in d["results"]]
    series = [("case33bw", "33-bus model"), ("case69", "69-bus model"),
              ("lindistflow", "LinDistFlow")]
    x = np.arange(len(targets)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (key, label) in enumerate(series):
        vals = [d["results"][t][key]["V_mae"] if key in d["results"][t] else np.nan
                for t in targets]
        b = ax.bar(x + (i - 1) * w, vals, w, label=label)
        ax.bar_label(b, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"target: {t}" for t in targets])
    ax.set_ylabel("voltage MAE (mV/pu)")
    ax.set_yscale("log")
    ax.set_title("Cross-feeder evaluation: native vs zero-shot vs linear baseline")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    _save(fig, "crossfeeder.png")


def fig_warmstart():
    d = _load_json("warmstart_nr.json")
    csv_path = os.path.join(RESULTS, "warmstart_nr_per_scenario.csv")
    if not d:
        print("  [skip] warmstart_nr: result not found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: iteration-count distribution, flat vs warm.
    if os.path.exists(csv_path):
        rows = list(csv.DictReader(open(csv_path)))
        itf = np.array([int(r["iters_flat"]) for r in rows])
        itw = np.array([int(r["iters_warm"]) for r in rows])
        lo, hi = min(itf.min(), itw.min()), max(itf.max(), itw.max())
        bins = np.arange(lo, hi + 2) - 0.5
        axes[0].hist([itf, itw], bins=bins, label=["flat start", "GNN warm-start"])
        axes[0].set_xlabel("Newton-Raphson iterations")
        axes[0].set_ylabel("scenarios")
        axes[0].set_title("Iteration count distribution")
        axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    # Right: mean iterations bar with % saved.
    means = [d["iters_flat_mean"], d["iters_warm_mean"]]
    b = axes[1].bar(["flat", "warm"], means, color=["#888", "#2a7"])
    axes[1].bar_label(b, fmt="%.2f", padding=2)
    axes[1].set_ylabel("mean NR iterations")
    axes[1].set_title(f"{d['iters_saved_pct']:.1f}% fewer iterations "
                      f"(never worse: {100*d['frac_warm_not_worse']:.0f}%)")
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Phase 2: GNN warm-start for Newton-Raphson")
    _save(fig, "warmstart_nr.png")


# --------------------------------------------------------------------------- #
# Figures that need a trained model + an evaluation set
# --------------------------------------------------------------------------- #
def fig_parity(model, scalers, eval_set, device):
    if model is None:
        print("  [skip] parity: checkpoints/gnn_powerflow.pt not found")
        return
    tv, pv, tt, pt = [], [], [], []
    for raw in eval_set:
        gv, gt = predict(model, raw, scalers, device)
        y = raw.y.numpy()
        tv += y[:, 0].tolist(); pv += gv.tolist()
        tt += np.degrees(y[:, 1]).tolist(); pt += np.degrees(gt).tolist()
    tv, pv, tt, pt = map(np.array, (tv, pv, tt, pt))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (t, p, name, unit) in zip(
            axes, [(tv, pv, "Voltage magnitude", "pu"),
                   (tt, pt, "Voltage angle", "deg")]):
        ax.scatter(t, p, s=4, alpha=0.25, edgecolors="none")
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"true ({unit})"); ax.set_ylabel(f"predicted ({unit})")
        ax.set_title(name); ax.grid(alpha=0.3)
    fig.suptitle("GNN prediction parity (held-out scenarios)")
    _save(fig, "parity.png")


def fig_baseline(model, scalers, eval_set, device):
    if model is None:
        print("  [skip] baseline: checkpoints/gnn_powerflow.pt not found")
        return
    scores = baseline_scores(model, eval_set, scalers, device)
    names = list(scores.keys())
    mae = [scores[n]["MAE_mV"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    b = ax.bar(names, mae, color=["#bbb", "#9ac", "#79b", "#2a7"])
    ax.bar_label(b, fmt="%.2f", padding=2, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("voltage MAE (mV/pu, log)")
    ax.set_title("Voltage accuracy vs baselines")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, "baseline_comparison.png")


def fig_physics(before_model, before_sc, after_model, after_sc, eval_set, device):
    """Prefer the saved physics_finetune.json; fall back to recomputing."""
    d = _load_json("physics_finetune.json")
    if d and "before_dP_list" in d:
        b_dP = np.array(d["before_dP_list"]); a_dP = np.array(d["after_dP_list"])
        b_pb = np.array(d["before_perbus_mae_mv"]); a_pb = np.array(d["after_perbus_mae_mv"])
    elif before_model is not None and after_model is not None:
        b = evaluate_model(before_model, eval_set, before_sc, device)
        a = evaluate_model(after_model, eval_set, after_sc, device)
        b_dP, a_dP = b["resid_list"], a["resid_list"]
        b_pb, a_pb = b["perbus"], a["perbus"]
    else:
        print("  [skip] physics: need physics_finetune.json or both checkpoints")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(b_dP, bins=30, alpha=0.6, label="supervised")
    axes[0].hist(a_dP, bins=30, alpha=0.6, label="physics-informed")
    axes[0].set_xlabel("per-scenario power residual |dP| (MW)")
    axes[0].set_ylabel("scenarios")
    axes[0].set_title("Power-balance violation")
    axes[0].legend(); axes[0].grid(alpha=0.3, axis="y")
    bus = np.arange(len(b_pb))
    axes[1].plot(bus, b_pb, label="supervised", lw=1.5)
    axes[1].plot(bus, a_pb, label="physics-informed", lw=1.5)
    axes[1].set_xlabel("bus index"); axes[1].set_ylabel("voltage MAE (mV/pu)")
    axes[1].set_title("Per-bus voltage error")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("Physics-informed fine-tuning: before vs after")
    _save(fig, "physics_before_after.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gnn_powerflow.pt")
    ap.add_argument("--pinn-ckpt", default="checkpoints/gnn_powerflow_pinn.pt")
    ap.add_argument("--eval-samples", type=int, default=750,
                    help="scenarios generated for parity/baseline/physics figures")
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.eval_samples = 60

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(FIGS, exist_ok=True)

    # Result-only figures (no model needed).
    print("Rendering result-driven figures:")
    fig_training_curve()
    fig_ablation_k()
    fig_ablation_edgeweights_topology()
    fig_warmstart()
    fig_crossfeeder()

    # Model-driven figures: build one evaluation set and reuse it.
    model, scalers = _load_model(args.ckpt, device)
    pinn, pinn_sc = _load_model(args.pinn_ckpt, device)
    need_eval = (model is not None) or (pinn is not None)
    eval_set = None
    if need_eval:
        print(f"Generating {args.eval_samples} evaluation scenarios ...")
        eval_set = generate_dataset(args.eval_samples, seed=args.seed)

    print("Rendering model-driven figures:")
    if eval_set is not None:
        fig_parity(model, scalers, eval_set, device)
        fig_baseline(model, scalers, eval_set, device)
        fig_physics(model, scalers, pinn, pinn_sc, eval_set, device)
    else:
        # Physics figure may still render from saved JSON without checkpoints.
        fig_physics(None, None, None, None, [], device)

    print("\nDone. Figures in figures/.")


if __name__ == "__main__":
    main()
