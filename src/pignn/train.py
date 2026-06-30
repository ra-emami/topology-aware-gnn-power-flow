"""Training loops: supervised and physics-informed."""
from __future__ import annotations

import copy
import torch
import torch.nn.functional as F


def _run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train() if train else model.eval()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            loss = F.mse_loss(model(batch), batch.y)
            if train:
                loss.backward(); optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def train_supervised(model, train_loader, val_loader, epochs=300, lr=1e-3,
                     device="cpu", log_every=20):
    """Train with MSE on standardized targets, keeping the best-val checkpoint."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    history = {"train": [], "val": []}
    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, optimizer, device)
        va = _run_epoch(model, val_loader, None, device)
        scheduler.step(va)
        history["train"].append(tr); history["val"].append(va)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d} | train {tr:.6f} | val {va:.6f} | best {best_val:.6f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val"] = best_val
    return history


def finetune_physics(model, train_loader, physics_residual, lam=10.0,
                     epochs=80, lr=5e-4, device="cpu", log_every=20):
    """Fine-tune with ``MSE + lam * power-balance residual``. Operates in place."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"sup": [], "phys": []}
    for epoch in range(1, epochs + 1):
        model.train()
        sup = phys = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            l_sup = F.mse_loss(out, batch.y)
            l_phys = physics_residual(out, batch.x)
            (l_sup + lam * l_phys).backward()
            optimizer.step()
            sup += l_sup.item(); phys += l_phys.item()
        history["sup"].append(sup / len(train_loader))
        history["phys"].append(phys / len(train_loader))
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d} | sup {history['sup'][-1]:.5f} | "
                  f"phys {history['phys'][-1]:.6f}")
    return history
