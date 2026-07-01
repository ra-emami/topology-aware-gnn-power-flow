"""LinDistFlow linear baseline sanity checks."""
import numpy as np
import pandapower as pp

from pignn import base_case, lindistflow_matrices, lindistflow_predict


def test_lindistflow_zero_injection_is_flat():
    """With no net injection anywhere, every voltage collapses to 1.0 pu."""
    net = base_case()
    R, X = lindistflow_matrices(net)
    NB = len(net.bus)
    v = lindistflow_predict(np.zeros(NB), np.zeros(NB), R, X, net.sn_mva)
    assert np.allclose(v, 1.0, atol=1e-9)


def test_lindistflow_tracks_newton_raphson():
    """On the base loads LinDistFlow approximates the NR solution closely."""
    net = base_case()
    pp.runpp(net, algorithm="nr")
    R, X = lindistflow_matrices(net)
    NB = len(net.bus)

    p = np.zeros(NB)
    q = np.zeros(NB)
    p[net.load.bus.values] -= net.load.p_mw.values
    q[net.load.bus.values] -= net.load.q_mvar.values

    v = lindistflow_predict(p, q, R, X, net.sn_mva)
    assert np.abs(v - net.res_bus.vm_pu.values).mean() < 0.01
