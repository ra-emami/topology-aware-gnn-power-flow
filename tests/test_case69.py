"""The constructed IEEE 69-bus feeder must match its published load-flow results."""
import numpy as np
import pandapower as pp

from pignn import base_case, build_topology


def test_case69_structure():
    net = base_case("case69")
    assert len(net.bus) == 69
    assert int(net.line.in_service.sum()) == 68      # radial spanning tree
    assert int((~net.line.in_service).sum()) == 5    # normally-open tie switches

    # Published totals: 3.802 MW / 2.694 MVAr.
    assert abs(net.load.p_mw.sum() - 3.802) < 0.01
    assert abs(net.load.q_mvar.sum() - 2.694) < 0.01

    # The GNN graph builds cleanly: two directed edges per in-service line.
    ei, ew = build_topology(net)
    assert ei.shape == (2, 2 * 68)
    assert bool((ew > 0).all())


def test_case69_matches_published_power_flow():
    """Nominal NR solution: V_min ~ 0.9092 pu at published bus 65, ~225 kW losses."""
    net = base_case("case69")
    pp.runpp(net, algorithm="nr")
    vm = net.res_bus.vm_pu.values
    assert abs(vm.min() - 0.9092) < 0.002
    assert int(np.argmin(vm)) + 1 == 65             # published 1-indexed bus number
    assert abs(net.res_line.pl_mw.sum() * 1e3 - 225.0) < 8.0
