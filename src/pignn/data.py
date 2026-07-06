"""Scenario generation and dataset preparation.

Each scenario randomizes load level and DER active/reactive dispatch, solves the
AC power flow with Newton-Raphson, and stores the result as a PyG graph. Supports
single-topology and multi-topology (reconfiguration) datasets.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import pandapower as pp
from pandapower.powerflow import LoadflowNotConverged
from torch_geometric.data import Data

from .config import (DEFAULT_NETWORK, LOAD_SCALE_RANGE, DER_P_RANGE, DER_Q_RANGE,
                     der_buses_for)
from .topology import base_case, build_topology, apply_config


def _node_flags(net, der_buses=None):
    if der_buses is None:
        der_buses = der_buses_for()
    slack = int(net.ext_grid.bus.values[0])
    is_slack = torch.zeros(len(net.bus)); is_slack[slack] = 1.0
    is_der = torch.zeros(len(net.bus)); is_der[der_buses] = 1.0
    return is_slack, is_der


def _solve_scenario(net_template, edge_index, edge_weight, is_der, is_slack, rng,
                    der_buses=None):
    """Sample one operating point on a fixed topology and solve it. Returns Data or None."""
    if der_buses is None:
        der_buses = der_buses_for()
    net = copy.deepcopy(net_template)
    scale = rng.uniform(*LOAD_SCALE_RANGE)
    net.load.p_mw *= scale
    net.load.q_mvar *= scale

    p_disp = rng.uniform(*DER_P_RANGE, len(der_buses))
    q_disp = rng.uniform(*DER_Q_RANGE, len(der_buses))
    for i, bus in enumerate(der_buses):
        pp.create_sgen(net, bus, p_mw=p_disp[i], q_mvar=q_disp[i])

    try:
        pp.runpp(net, algorithm="nr")
    except LoadflowNotConverged:
        return None

    p = np.zeros(len(net.bus)); q = np.zeros(len(net.bus))
    p[net.load.bus.values] -= net.load.p_mw.values
    q[net.load.bus.values] -= net.load.q_mvar.values
    p[der_buses] += p_disp
    q[der_buses] += q_disp

    x = torch.stack([torch.tensor(p, dtype=torch.float),
                     torch.tensor(q, dtype=torch.float), is_der, is_slack], dim=1)
    v = torch.tensor(net.res_bus.vm_pu.values, dtype=torch.float)
    th = torch.tensor(np.radians(net.res_bus.va_degree.values), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                y=torch.stack([v, th], dim=1))


def generate_dataset(n_samples, config=None, seed=0, verbose=True,
                     network=DEFAULT_NETWORK):
    """Generate scenarios on a single topology (a base feeder or a reconfiguration)."""
    der_buses = der_buses_for(network)
    net = apply_config(base_case(network), config)
    ei, ew = build_topology(net)
    is_slack, is_der = _node_flags(net, der_buses)
    rng = np.random.default_rng(seed)
    out, fail = [], 0
    for _ in range(n_samples):
        d = _solve_scenario(net, ei, ew, is_der, is_slack, rng, der_buses)
        (out.append(d) if d is not None else None)
        fail += d is None
    if verbose:
        print(f"[data] {len(out)}/{n_samples} converged (failed {fail})")
    return out


def generate_multitopology_dataset(n_samples, configs, seed=0, verbose=True,
                                   network=DEFAULT_NETWORK):
    """Generate scenarios spread across several topologies (one random config each)."""
    der_buses = der_buses_for(network)
    base = base_case(network)
    rng = np.random.default_rng(seed)
    prep = []
    for cfg in configs:
        net = apply_config(base, cfg)
        ei, ew = build_topology(net)
        is_slack, is_der = _node_flags(net, der_buses)
        prep.append((net, ei, ew, is_der, is_slack))
    out, fail = [], 0
    for _ in range(n_samples):
        net, ei, ew, is_der, is_slack = prep[rng.integers(len(prep))]
        d = _solve_scenario(net, ei, ew, is_der, is_slack, rng, der_buses)
        (out.append(d) if d is not None else None)
        fail += d is None
    if verbose:
        print(f"[data] multitopology {len(out)}/{n_samples} converged "
              f"over {len(configs)} configs (failed {fail})")
    return out


def split_dataset(dataset, seed=42, fracs=(0.70, 0.85)):
    """Shuffle (seeded) and split into train / val / test."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    ds = [dataset[i] for i in perm]
    n = len(ds)
    a, b = int(fracs[0] * n), int(fracs[1] * n)
    return ds[:a], ds[a:b], ds[b:]


def fit_scalers(train_set):
    """z-score statistics from the training split (injections + targets)."""
    X = torch.cat([d.x for d in train_set])
    Y = torch.cat([d.y for d in train_set])
    return dict(x_mean=X[:, :2].mean(0), x_std=X[:, :2].std(0) + 1e-8,
                y_mean=Y.mean(0), y_std=Y.std(0) + 1e-8)


def standardize(ds, sc):
    """Apply scalers to a dataset (injections standardized; flags untouched).

    Scaler tensors may live on any device (e.g. loaded from a CUDA checkpoint);
    statistics are brought to the dataset tensors' device before use.
    """
    out = []
    for d in ds:
        xm, xs = sc["x_mean"].to(d.x.device), sc["x_std"].to(d.x.device)
        ym, ys = sc["y_mean"].to(d.y.device), sc["y_std"].to(d.y.device)
        x = d.x.clone()
        x[:, :2] = (x[:, :2] - xm) / xs
        out.append(Data(x=x, edge_index=d.edge_index, edge_weight=d.edge_weight,
                        y=(d.y - ym) / ys))
    return out
