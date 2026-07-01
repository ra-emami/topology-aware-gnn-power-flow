"""Topology construction and reconfiguration validity."""
import numpy as np

from pignn import base_case, build_topology, enumerate_radial_configs, apply_config


def test_build_topology_structure():
    net = base_case()
    ei, ew = build_topology(net)
    n_lines = int(net.line.in_service.sum())

    # Undirected graph: two directed edges per in-service line.
    assert ei.shape == (2, 2 * n_lines)
    assert ew.shape[0] == 2 * n_lines

    # Every reverse edge is present (symmetry).
    edges = set(map(tuple, ei.t().tolist()))
    assert all((b, a) in edges for (a, b) in edges)

    # Admittance weights are positive and normalized to unit mean.
    assert bool((ew > 0).all())
    assert abs(float(ew.mean()) - 1.0) < 1e-4


def _is_spanning_tree(net):
    """A radial feeder's in-service lines must form a spanning tree of the buses."""
    line = net.line[net.line.in_service]
    N = len(net.bus)
    n_edges = len(line)
    parent = list(range(N))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    acyclic = True
    for f, t in zip(line.from_bus.values, line.to_bus.values):
        rf, rt = find(int(f)), find(int(t))
        if rf == rt:
            acyclic = False
        else:
            parent[rf] = rt
    connected = len({find(i) for i in range(N)}) == 1
    return n_edges == N - 1 and connected and acyclic


def test_reconfigurations_are_radial_trees():
    configs = enumerate_radial_configs(validate=False)
    # There is more than just the base configuration, and base is included.
    assert len(configs) > 1
    assert any(c["name"] == "base" for c in configs)

    # Every enumerated reconfiguration is a valid radial spanning tree.
    for c in configs:
        net = apply_config(base_case(), c)
        assert _is_spanning_tree(net), f"config {c['name']} is not a radial tree"
