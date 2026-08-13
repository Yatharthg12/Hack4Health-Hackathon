"""Headless, reusable evaluation plot generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402


COLORS = ["#61d4b3", "#68a7ff", "#9d8cff", "#f3b562", "#eb7181", "#86c5da", "#8fa7a0", "#c58fce"]


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0b1518")
    plt.close(fig)


def confusion_matrix_plot(matrix: Sequence[Sequence[int]], labels: Sequence[str], path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(7, 6), facecolor="#0b1518")
    axis.set_facecolor("#0b1518")
    sns.heatmap(matrix, annot=True, fmt="d", cmap="mako", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set(title=title, xlabel="Predicted", ylabel="Actual")
    axis.tick_params(colors="#dbe9e5")
    axis.title.set_color("#f4fbf8")
    axis.xaxis.label.set_color("#dbe9e5")
    axis.yaxis.label.set_color("#dbe9e5")
    _finish(fig, path)


def training_history_plot(history: dict[str, list[float]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#0b1518")
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    for axis in axes:
        axis.set_facecolor("#102126")
        axis.tick_params(colors="#c8dad5")
        for spine in axis.spines.values():
            spine.set_color("#365057")
    axes[0].plot(epochs, history.get("train_loss", []), label="Train", color=COLORS[0])
    axes[0].plot(epochs, history.get("val_loss", []), label="Validation", color=COLORS[3])
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(epochs, history.get("train_accuracy", []), label="Train", color=COLORS[1])
    axes[1].plot(epochs, history.get("val_accuracy", []), label="Validation", color=COLORS[4])
    axes[1].set(title="Accuracy", xlabel="Epoch")
    for axis in axes:
        axis.legend(frameon=False, labelcolor="#e5f2ee")
        axis.title.set_color("#f4fbf8")
        axis.xaxis.label.set_color("#dbe9e5")
    fig.suptitle(title, color="#f4fbf8", fontsize=14)
    _finish(fig, path)


def class_distribution_plot(counts: dict[str, int], path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5), facecolor="#0b1518")
    axis.set_facecolor("#102126")
    labels, values = list(counts), list(counts.values())
    axis.bar(labels, values, color=COLORS[: len(labels)])
    axis.set_title(title, color="#f4fbf8")
    axis.tick_params(axis="x", labelrotation=25, colors="#dbe9e5")
    axis.tick_params(axis="y", colors="#dbe9e5")
    for spine in axis.spines.values():
        spine.set_color("#365057")
    _finish(fig, path)


def feature_importance_plot(items: Iterable[dict[str, float | str]], path: Path, title: str) -> None:
    rows = list(items)[::-1]
    fig, axis = plt.subplots(figsize=(8, 6), facecolor="#0b1518")
    axis.set_facecolor("#102126")
    axis.barh([str(row["feature"]) for row in rows], [float(row["importance"]) for row in rows], color="#61d4b3")
    axis.set_title(title, color="#f4fbf8")
    axis.tick_params(colors="#dbe9e5")
    for spine in axis.spines.values():
        spine.set_color("#365057")
    _finish(fig, path)


def regression_scatter_plot(truth: np.ndarray, predictions: np.ndarray, names: Sequence[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(13, 4), facecolor="#0b1518")
    for index, (axis, name) in enumerate(zip(np.atleast_1d(axes), names, strict=True)):
        axis.set_facecolor("#102126")
        axis.scatter(truth[:, index], predictions[:, index], alpha=0.55, s=16, color=COLORS[index])
        low = min(float(truth[:, index].min()), float(predictions[:, index].min()))
        high = max(float(truth[:, index].max()), float(predictions[:, index].max()))
        axis.plot([low, high], [low, high], linestyle="--", color="#dbe9e5", linewidth=1)
        axis.set(title=name.replace("_", " "), xlabel="Actual", ylabel="Predicted")
        axis.tick_params(colors="#dbe9e5")
        axis.title.set_color("#f4fbf8")
        axis.xaxis.label.set_color("#dbe9e5")
        axis.yaxis.label.set_color("#dbe9e5")
        for spine in axis.spines.values():
            spine.set_color("#365057")
    _finish(fig, path)

