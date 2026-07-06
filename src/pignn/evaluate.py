"""Evaluation: accuracy + physics metrics, baselines, topology generalization."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics import r2_score

from .config import DEFAULT_NETWORK, der_buses_for
from .topology import (base_case, build_topology, apply_config,
                       lindistflow_matrices, lindistflow_predict)
from .physics import build_ybus
from .data import _node_flags, _solve_scenario


@torch.no_grad()
def predict(model, raw, scalers, device):
    """Run the model on a raw (unstandardized) Data; return (V_pu, theta_rad).

    Scaler tensors may live on any device (e.g. loaded from a CUDA checkpoint);
    all arithmetic happens on the graph tensors' device.
    """
    x = raw.x.clone()
    x[:, :2] = (x[:, :2] - scalers["x_mean"].to(x.device)) / scalers["x_std"].to(x.device)
    d = Data(x=x, edge_index=raw.edge_index, edge_weight=raw.edge_weight).to(device)
    model.eval()
    out = (model(d) * scalers["y_std"].to(device) + scalers["y_mean"].to(device)).cpu().numpy()
    return out[:, 0], out[:, 1]


def evaluate_model(model, test_set, scalers, device, net=None):
    """Voltage/angle accuracy plus power-balance violation on a held-out set."""
    net = net or base_case()
    Y = build_ybus(net)
    slack = int(net.ext_grid.bus.values[0])
    NB = len(net.bus)
    sn = net.sn_mva

    tv, pv, tt, pt, dP, dQ = [], [], [], [], [], []
    perbus = np.zeros(NB); cnt = 0
    for raw in test_set:
        t = raw.y.numpy()
        gv, gt = predict(model, raw, scalers, device)
        tv += t[:, 0].tolist(); pv += gv.tolist()
        tt += np.degrees(t[:, 1]).tolist(); pt += np.degrees(gt).tolist()
        perbus += np.abs(t[:, 0] - gv); cnt += 1
        V = gv * np.exp(1j * gt); S = V * np.conj(Y @ V)
        mm = np.ones(NB, bool); mm[slack] = False
        dP.append(np.abs((S.real - raw.x[:, 0].numpy() / sn)[mm]).mean() * sn)
        dQ.append(np.abs((S.imag - raw.x[:, 1].numpy() / sn)[mm]).mean() * sn)
    tv, pv, tt, pt = map(np.array, [tv, pv, tt, pt])
    return dict(
        V_r2=r2_score(tv, pv), V_mae=np.abs(tv - pv).mean() * 1000,
        V_max=np.abs(tv - pv).max() * 1000,
        th_r2=r2_score(tt, pt), th_mae=np.abs(tt - pt).mean(),
        dP=float(np.mean(dP)), dQ=float(np.mean(dQ)),
        perbus=perbus / cnt * 1000, resid_list=np.array(dP),
    )


def baseline_scores(model, test_set, scalers, device, net=None):
    """Compare the GNN against flat-voltage, per-bus-mean, and LinDistFlow."""
    net = net or base_case()
    R, X = lindistflow_matrices(net)
    sn = net.sn_mva
    bus_mean = torch.stack([d.y[:, 0] for d in test_set]).mean(0).numpy()

    tv, gnn, ldf, flat, mean = [], [], [], [], []
    for raw in test_set:
        t = raw.y[:, 0].numpy()
        gv, _ = predict(model, raw, scalers, device)
        vl = lindistflow_predict(raw.x[:, 0].numpy(), raw.x[:, 1].numpy(), R, X, sn)
        tv += t.tolist(); gnn += gv.tolist(); ldf += vl.tolist()
        flat += [1.0] * len(t); mean += bus_mean.tolist()
    tv, gnn, ldf, flat, mean = map(np.array, [tv, gnn, ldf, flat, mean])

    def sc(p):
        return dict(R2=r2_score(tv, p), MAE_mV=np.abs(tv - p).mean() * 1000,
                    max_mV=np.abs(tv - p).max() * 1000)
    return {"Flat 1.0": sc(flat), "Per-bus mean": sc(mean),
            "LinDistFlow": sc(ldf), "GNN": sc(gnn)}


def undervoltage_screening(model, test_set, scalers, device, thr=0.95):
    """Precision/recall of the surrogate as a V < thr violation detector."""
    T = np.array([d.y[:, 0].numpy() for d in test_set])
    P = np.array([predict(model, d, scalers, device)[0] for d in test_set])
    yt, yp = (T < thr), (P < thr)
    TP = int((yt & yp).sum()); FP = int((~yt & yp).sum())
    FN = int((yt & ~yp).sum()); TN = int((~yt & ~yp).sum())
    prec = TP / (TP + FP + 1e-9); rec = TP / (TP + FN + 1e-9)
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec + 1e-9),
                TP=TP, FP=FP, FN=FN, TN=TN)


def evaluate_on_topologies(model, configs, scalers, device, n=80, seed=0,
                           network=DEFAULT_NETWORK):
    """Mean voltage MAE (mV/pu) of the model on each topology configuration."""
    der_buses = der_buses_for(network)
    base = base_case(network)
    rng = np.random.default_rng(seed)
    results = {}
    for cfg in configs:
        net = apply_config(base, cfg)
        ei, ew = build_topology(net)
        is_slack, is_der = _node_flags(net, der_buses)
        errs = []
        for _ in range(n):
            d = _solve_scenario(net, ei, ew, is_der, is_slack, rng, der_buses)
            if d is None:
                continue
            gv, _ = predict(model, d, scalers, device)
            errs.append(np.abs(d.y[:, 0].numpy() - gv).mean())
        if errs:
            results[cfg["name"]] = float(np.mean(errs)) * 1000
    return results
