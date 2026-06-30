"""Physics operators: admittance matrix and differentiable power-balance residual.

The residual expresses the AC power mismatch ``S - S_injected`` as a differentiable
function of the model's ``[V, theta]`` output, enabling a physics-informed loss.
"""
from __future__ import annotations

import numpy as np
import torch


def build_ybus(net):
    """Assemble the per-unit nodal admittance matrix from in-service line parameters."""
    line = net.line[net.line.in_service]
    vn = net.bus.loc[line.from_bus.values, "vn_kv"].values
    zb = vn ** 2 / net.sn_mva
    r = (line.r_ohm_per_km.values * line.length_km.values) / line.parallel.values / zb
    x = (line.x_ohm_per_km.values * line.length_km.values) / line.parallel.values / zb
    N = len(net.bus)
    Y = np.zeros((N, N), dtype=complex)
    ys = 1.0 / (r + 1j * x)
    if "c_nf_per_km" in line.columns:
        bsh = 2 * np.pi * net.f_hz * line.c_nf_per_km.values * 1e-9 * line.length_km.values * zb
    else:
        bsh = np.zeros(len(line))
    for k, (f, t) in enumerate(zip(line.from_bus.values, line.to_bus.values)):
        Y[f, f] += ys[k] + 1j * bsh[k] / 2
        Y[t, t] += ys[k] + 1j * bsh[k] / 2
        Y[f, t] -= ys[k]
        Y[t, f] -= ys[k]
    return Y


def make_physics_residual(net, scalers, device):
    """Build a differentiable power-balance residual closure for a fixed topology.

    Returns ``residual(out_std, x_std)`` giving the mean per-(non-slack)-bus squared
    power mismatch in per-unit, differentiable w.r.t. the standardized model output.
    """
    Y = build_ybus(net)
    Yt = torch.tensor(Y, dtype=torch.cfloat, device=device)
    NB = len(net.bus)
    slack = int(net.ext_grid.bus.values[0])
    sn = net.sn_mva

    xm = scalers["x_mean"].to(device); xs = scalers["x_std"].to(device)
    ym = scalers["y_mean"].to(device); ys = scalers["y_std"].to(device)
    mask = torch.ones(NB, device=device); mask[slack] = 0.0

    def residual(out_std, x_std):
        B = out_std.shape[0] // NB
        out = out_std * ys + ym
        Vm = out[:, 0].view(B, NB).clamp(min=1e-3)
        th = out[:, 1].view(B, NB)
        Vc = torch.polar(Vm, th)
        S = Vc * torch.conj(Vc @ Yt.T)
        p_pu = ((x_std[:, 0] * xs[0] + xm[0]).view(B, NB)) / sn
        q_pu = ((x_std[:, 1] * xs[1] + xm[1]).view(B, NB)) / sn
        dP = S.real - p_pu
        dQ = S.imag - q_pu
        return ((dP ** 2 + dQ ** 2) * mask).sum() / (B * mask.sum())

    return residual
