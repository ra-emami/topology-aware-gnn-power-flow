"""Grid topology utilities.

Builds the graph the GNN consumes (impedance-weighted edges), enumerates valid
radial network reconfigurations for topology-generalization experiments, and
provides the LinDistFlow linear power-flow baseline.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import pandapower as pp
import pandapower.networks as nw


def base_case(network="case33bw"):
    """Return a fresh distribution feeder by name.

    ``case33bw`` comes from pandapower; ``case69`` is the IEEE 69-bus feeder
    constructed from the published Baran & Wu data (see :mod:`pignn.case69`).
    """
    if network == "case33bw":
        return nw.case33bw()
    if network == "case69":
        from .case69 import case69
        return case69()
    raise ValueError(f"unknown network '{network}' (choose 'case33bw' or 'case69')")


def build_topology(net):
    """Edges + per-unit |series admittance| edge weights for the in-service branches.

    Electrically close buses (short, low-impedance lines) receive larger weights.
    Returns ``(edge_index [2, 2E], edge_weight [2E])`` as undirected (both directions).
    """
    line = net.line[net.line.in_service].copy()
    fb, tb = line.from_bus.values, line.to_bus.values
    vn = net.bus.loc[fb, "vn_kv"].values
    zbase = vn ** 2 / net.sn_mva
    r = (line.r_ohm_per_km.values * line.length_km.values) / line.parallel.values / zbase
    x = (line.x_ohm_per_km.values * line.length_km.values) / line.parallel.values / zbase
    ymag = 1.0 / np.sqrt(r ** 2 + x ** 2)
    w = ymag / ymag.mean()  # scale ~O(1) relative to TAGConv self-loops

    fb_t, tb_t = torch.tensor(fb), torch.tensor(tb)
    edge_index = torch.stack([torch.cat([fb_t, tb_t]), torch.cat([tb_t, fb_t])]).long()
    edge_weight = torch.tensor(np.concatenate([w, w]), dtype=torch.float)
    return edge_index, edge_weight


# --------------------------------------------------------------------------- #
# Radial reconfiguration enumeration
# --------------------------------------------------------------------------- #
def _radial_tree(net):
    """BFS the in-service network from the slack; return parent / parent-line / depth."""
    line = net.line[net.line.in_service]
    N = len(net.bus)
    root = int(net.ext_grid.bus.values[0])
    adj = {i: [] for i in range(N)}
    for idx, f, t in zip(line.index, line.from_bus.values, line.to_bus.values):
        adj[f].append((t, idx))
        adj[t].append((f, idx))
    parent = [-1] * N
    parent_line = [-1] * N
    depth = [0] * N
    seen = [False] * N
    stack = [root]
    seen[root] = True
    while stack:
        u = stack.pop()
        for v, idx in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                parent_line[v] = idx
                depth[v] = depth[u] + 1
                stack.append(v)
    return parent, parent_line, depth


def _tree_path_lines(net, u, v):
    """Line indices on the in-service tree path between buses u and v."""
    parent, parent_line, depth = _radial_tree(net)
    a, b = u, v
    lines = []
    while depth[a] > depth[b]:
        lines.append(parent_line[a]); a = parent[a]
    while depth[b] > depth[a]:
        lines.append(parent_line[b]); b = parent[b]
    while a != b:
        lines.append(parent_line[a]); a = parent[a]
        lines.append(parent_line[b]); b = parent[b]
    return lines


def tie_lines(net):
    """Indices of the normally-open tie switches."""
    return net.line.index[~net.line.in_service].tolist()


def _converges(net):
    """True if a nominal power flow solves on this configuration."""
    try:
        pp.runpp(net, algorithm="nr")
        return True
    except Exception:
        return False


def enumerate_radial_configs(net=None, validate=True):
    """Enumerate valid radial configurations of the feeder.

    The base configuration plus, for each tie switch, every reconfiguration formed
    by closing that tie and opening one line on the loop it creates (which restores
    a radial spanning tree). Each config is ``{'close': tie|None, 'open': line|None,
    'name': str}``. With ``validate`` set, configurations whose nominal power flow
    does not converge (numerically stiff reroutings) are dropped.
    """
    net = net or base_case()
    configs = [{"close": None, "open": None, "name": "base"}]
    for tie in tie_lines(net):
        u = int(net.line.loc[tie, "from_bus"])
        v = int(net.line.loc[tie, "to_bus"])
        for line_idx in _tree_path_lines(net, u, v):
            configs.append({"close": tie, "open": int(line_idx),
                            "name": f"tie{tie}_open{int(line_idx)}"})
    if validate:
        configs = [c for c in configs if _converges(apply_config(net, c))]
    return configs


def apply_config(net, config):
    """Return a copy of ``net`` with a reconfiguration applied."""
    out = copy.deepcopy(net)
    if config is None:
        return out
    if config.get("close") is not None:
        out.line.loc[config["close"], "in_service"] = True
    if config.get("open") is not None:
        out.line.loc[config["open"], "in_service"] = False
    return out


# --------------------------------------------------------------------------- #
# LinDistFlow linear baseline
# --------------------------------------------------------------------------- #
def lindistflow_matrices(net):
    """Common-path resistance/reactance matrices (R, X) for the radial feeder."""
    line = net.line[net.line.in_service]
    vn = net.bus.loc[line.from_bus.values, "vn_kv"].values
    zb = vn ** 2 / net.sn_mva
    r = (line.r_ohm_per_km.values * line.length_km.values) / line.parallel.values / zb
    x = (line.x_ohm_per_km.values * line.length_km.values) / line.parallel.values / zb
    N = len(net.bus)
    root = int(net.ext_grid.bus.values[0])
    adj = {i: [] for i in range(N)}
    for k, (f, t) in enumerate(zip(line.from_bus.values, line.to_bus.values)):
        adj[f].append((t, r[k], x[k]))
        adj[t].append((f, r[k], x[k]))
    parent = [-1] * N; depth = [0] * N; cumR = [0.0] * N; cumX = [0.0] * N; seen = [False] * N
    stack = [root]; seen[root] = True
    while stack:
        u = stack.pop()
        for v, rr, xx in adj[u]:
            if not seen[v]:
                seen[v] = True; parent[v] = u; depth[v] = depth[u] + 1
                cumR[v] = cumR[u] + rr; cumX[v] = cumX[u] + xx; stack.append(v)

    def lca(a, b):
        while depth[a] > depth[b]: a = parent[a]
        while depth[b] > depth[a]: b = parent[b]
        while a != b: a = parent[a]; b = parent[b]
        return a

    R = np.zeros((N, N)); X = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            l = lca(i, j); R[i, j] = cumR[l]; X[i, j] = cumX[l]
    return R, X


def lindistflow_predict(p_inj_mw, q_inj_mvar, R, X, sn_mva):
    """Linearized voltage magnitudes from injections (MW/MVAr) via LinDistFlow."""
    p_pu = np.asarray(p_inj_mw) / sn_mva
    q_pu = np.asarray(q_inj_mvar) / sn_mva
    vsq = 1.0 + 2.0 * (R @ p_pu + X @ q_pu)
    return np.sqrt(np.clip(vsq, 1e-6, None))
