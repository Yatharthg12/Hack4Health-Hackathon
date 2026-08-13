"""Shared neural training loop, evaluation, and temperature calibration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for inputs, labels in loader:
        logits.append(model(inputs.to(device)).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a positive scalar temperature on validation logits."""

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)
    criterion = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits_tensor / log_temperature.exp(), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float = 1e-3,
) -> tuple[nn.Module, dict[str, list[float]], dict[str, Any]]:
    """Train with AdamW, ReduceLROnPlateau, clipping, macro-F1 selection, and early stopping."""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [], "val_macro_f1": []
    }
    best_state = deepcopy(model.state_dict())
    best_f1 = -1.0
    stale = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += len(labels)

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_count = 0
        predictions: list[int] = []
        truth: list[int] = []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                validation_loss += float(criterion(logits, labels).item()) * len(labels)
                predicted = logits.argmax(dim=1)
                validation_correct += int((predicted == labels).sum().item())
                validation_count += len(labels)
                predictions.extend(predicted.cpu().tolist())
                truth.extend(labels.cpu().tolist())
        macro_f1 = float(f1_score(truth, predictions, average="macro", zero_division=0))
        scheduler.step(macro_f1)
        history["train_loss"].append(running_loss / max(count, 1))
        history["val_loss"].append(validation_loss / max(validation_count, 1))
        history["train_accuracy"].append(correct / max(count, 1))
        history["val_accuracy"].append(validation_correct / max(validation_count, 1))
        history["val_macro_f1"].append(macro_f1)
        print(
            f"epoch={epoch + 1:02d} train_loss={history['train_loss'][-1]:.4f} "
            f"val_loss={history['val_loss'][-1]:.4f} val_f1={macro_f1:.4f}",
            flush=True,
        )
        if macro_f1 > best_f1 + 1e-4:
            best_f1 = macro_f1
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, {"best_validation_macro_f1": best_f1, "epochs_completed": len(history["train_loss"])}

