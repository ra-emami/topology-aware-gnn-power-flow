"""The admittance matrix must reproduce the AC power balance of a true solution."""
import numpy as np
import pandapower as pp

from pignn import base_case, build_ybus


def test_ybus_power_balance_on_true_solution():
    """On a converged Newton-Raphson solution the residual S - S_injected ~ 0.

    This validates that build_ybus reproduces pandapower's network model: the
    complex power S = V . conj(Y V) at each non-slack bus must equal that bus's
    net injection (here, minus the load) to solver precision.
    """
    net = base_case()
    pp.runpp(net, algorithm="nr")

    Y = build_ybus(net)
    NB = len(net.bus)
    slack = int(net.ext_grid.bus.values[0])
    sn = net.sn_mva

    V = net.res_bus.vm_pu.values * np.exp(1j * np.radians(net.res_bus.va_degree.values))
    S = V * np.conj(Y @ V)

    p = np.zeros(NB)
    q = np.zeros(NB)
    p[net.load.bus.values] -= net.load.p_mw.values
    q[net.load.bus.values] -= net.load.q_mvar.values

    mask = np.ones(NB, bool)
    mask[slack] = False
    assert np.abs(S.real - p / sn)[mask].max() < 1e-5
    assert np.abs(S.imag - q / sn)[mask].max() < 1e-5
