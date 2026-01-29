from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    show_progress: bool = True


def train_regressor(
    model: "object",  # torch.nn.Module
    *,
    train_ds: "object",  # torch Dataset
    val_ds: "object",
    config: TrainConfig,
    seed: int = 1337,
) -> Dict[str, float]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state: Optional[dict] = None
    best_epoch = -1

    train_mse_history: list[float] = []
    val_mse_history: list[float] = []

    epoch_iter = tqdm(
        range(config.epochs), desc="epochs", disable=not config.show_progress
    )
    for epoch in epoch_iter:
        model.train()
        train_losses = []

        batch_iter = tqdm(
            train_loader,
            desc=f"train (epoch {epoch + 1}/{config.epochs})",
            leave=False,
            disable=not config.show_progress,
        )
        for xb, yb in batch_iter:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))
            if config.show_progress and train_losses:
                batch_iter.set_postfix(train_mse=float(np.mean(train_losses[-50:])))

        train_mse = float(np.mean(train_losses)) if train_losses else float("inf")
        train_mse_history.append(train_mse)

        model.eval()
        val_losses = []
        with torch.no_grad():
            val_iter = tqdm(
                val_loader,
                desc=f"val   (epoch {epoch + 1}/{config.epochs})",
                leave=False,
                disable=not config.show_progress,
            )
            for xb, yb in val_iter:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(float(loss_fn(pred, yb).detach().cpu().item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        val_mse_history.append(val_loss)

        if config.show_progress:
            epoch_iter.set_postfix(
                train_mse=train_mse, val_mse=val_loss, best_val=best_val
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "val_mse": float(best_val),
        "best_epoch": float(best_epoch),
        "train_mse_last": (
            float(train_mse_history[-1]) if train_mse_history else float("inf")
        ),
        "val_mse_last": float(val_mse_history[-1]) if val_mse_history else float("inf"),
        "train_mse_history": train_mse_history,
        "val_mse_history": val_mse_history,
    }


def predict_regressor(
    model: "object",
    *,
    x: np.ndarray,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    x_tensor = torch.from_numpy(x).float().unsqueeze(-1)
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev)
            out = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(out)

    if not preds:
        return np.asarray([], dtype=np.float64)

    return np.concatenate(preds).astype(np.float64)
