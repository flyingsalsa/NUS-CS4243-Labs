"""Reusable visualisations for intermediate quantities and reports."""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt


def show_gabor_bank(bank, columns: int = 6):
    rows = math.ceil(len(bank) / columns); fig, axes = plt.subplots(rows, columns, figsize=(2*columns, 2*rows))
    for ax, item in zip(np.asarray(axes).ravel(), bank):
        kernel, meta = item; ax.imshow(kernel, cmap="coolwarm"); ax.set_title(f"f={meta['frequency']:.2f}, o={meta['orientation']:.2f}\n{meta['phase']}"); ax.axis("off")
    for ax in np.asarray(axes).ravel()[len(bank):]: ax.axis("off")
    fig.tight_layout(); return fig


def show_feature_maps(image, maps, names, max_maps: int = 16):
    count = min(max_maps, maps.shape[-1]); columns = 4; rows = math.ceil((count + 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns, 3.4*rows)); axes = np.asarray(axes).ravel()
    axes[0].imshow(image, cmap="gray" if np.asarray(image).ndim == 2 else None); axes[0].set_title("input")
    for i in range(count): axes[i+1].imshow(maps[..., i], cmap="magma"); axes[i+1].set_title(names[i])
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); return fig


def show_anomaly_result(image, score, mask, truth=None):
    panels = 4 if truth is not None else 3
    fig, axes = plt.subplots(1, panels, figsize=(4*panels, 4))
    axes[0].imshow(image); axes[0].set_title("image")
    heat = axes[1].imshow(score, cmap="inferno"); axes[1].set_title("anomaly score"); fig.colorbar(heat, ax=axes[1], fraction=.046)
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title("predicted mask")
    if truth is not None: axes[3].imshow(truth, cmap="gray"); axes[3].set_title("ground truth")
    for ax in axes: ax.axis("off")
    fig.tight_layout(); return fig


def plot_confusion(matrix, labels):
    fig, ax = plt.subplots(figsize=(6, 5)); image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="predicted", ylabel="true", title="Material confusion matrix")
    for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout(); return fig


def plot_reliability(y_true, probabilities, classes, bins: int = 10):
    classes = np.asarray(classes); conf = probabilities.max(1); pred = classes[probabilities.argmax(1)]
    centres, acc, observed = [], [], []
    for low in np.linspace(0, 1, bins, endpoint=False):
        mask = (conf >= low) & (conf < low + 1/bins)
        if mask.any(): centres.append(conf[mask].mean()); acc.append((pred[mask] == y_true[mask]).mean()); observed.append(mask.sum())
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0,1],[0,1], "--", color="gray"); ax.scatter(centres, acc, s=np.asarray(observed)*10)
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="mean confidence", ylabel="accuracy", title="Reliability diagram")
    return fig
