"""Grad-CAM and input-gradient visualizations returned as embeddable PNG data URLs."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402
from torch import nn  # noqa: E402

from mindfuse.models.face_hog import extract_hog_from_image


def _figure_data_url(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=130, bbox_inches="tight", facecolor="#0b1518")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def face_gradcam_image(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original: Image.Image,
    class_index: int,
) -> str:
    """Compute Grad-CAM from the model's final residual convolution and overlay it."""

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    forward_handle = model.target_layer.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output.detach())
    )
    backward_handle = model.target_layer.register_full_backward_hook(
        lambda _module, _grad_input, grad_output: gradients.append(grad_output[0].detach())
    )
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        logits[0, class_index].backward()
    finally:
        forward_handle.remove()
        backward_handle.remove()
    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    heatmap = torch.relu((weights * activations[0]).sum(dim=1))[0].cpu().numpy()
    if float(heatmap.max()) > 0:
        heatmap /= float(heatmap.max())
    base = np.asarray(original.convert("RGB").resize((240, 240)), dtype=float) / 255.0
    heatmap = zoom(heatmap, (240 / heatmap.shape[0], 240 / heatmap.shape[1]), order=1)[:240, :240]
    colored = plt.get_cmap("magma")(heatmap)[..., :3]
    alpha = (0.12 + 0.48 * heatmap)[..., None]
    overlay = np.clip(base * (1 - alpha) + colored * alpha, 0, 1)
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), facecolor="#0b1518")
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(base)
    axes[0].set_title("Validated input", color="#e7f4f0")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM evidence", color="#e7f4f0")
    return _figure_data_url(figure)


def face_occlusion_image(
    estimator,
    original: Image.Image,
    class_index: int,
    patch_size: int = 8,
    stride: int = 4,
) -> str:
    """Map predicted-class probability drops after local mean occlusion."""

    grayscale = original.convert("L").resize((48, 48))
    base = np.asarray(grayscale, dtype=np.float32) / 255.0
    baseline_probability = float(estimator.predict_proba(extract_hog_from_image(grayscale)[None, :])[0, class_index])
    heatmap = np.zeros((48, 48), dtype=np.float32)
    counts = np.zeros((48, 48), dtype=np.float32)
    fill = float(base.mean())
    for row in range(0, 48 - patch_size + 1, stride):
        for column in range(0, 48 - patch_size + 1, stride):
            occluded = base.copy()
            occluded[row : row + patch_size, column : column + patch_size] = fill
            image = Image.fromarray(np.uint8(np.clip(occluded, 0, 1) * 255), mode="L")
            probability = float(estimator.predict_proba(extract_hog_from_image(image)[None, :])[0, class_index])
            drop = max(0.0, baseline_probability - probability)
            heatmap[row : row + patch_size, column : column + patch_size] += drop
            counts[row : row + patch_size, column : column + patch_size] += 1
    heatmap /= np.maximum(counts, 1)
    if float(heatmap.max()) > 0:
        heatmap /= float(heatmap.max())
    color_base = np.asarray(original.convert("RGB").resize((240, 240)), dtype=float) / 255.0
    heatmap = zoom(heatmap, (5, 5), order=1)
    colored = plt.get_cmap("magma")(heatmap)[..., :3]
    alpha = (0.1 + 0.5 * heatmap)[..., None]
    overlay = np.clip(color_base * (1 - alpha) + colored * alpha, 0, 1)
    figure, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), facecolor="#0b1518")
    for axis in axes:
        axis.axis("off")
    axes[0].imshow(color_base)
    axes[0].set_title("Validated input", color="#e7f4f0")
    axes[1].imshow(overlay)
    axes[1].set_title("Occlusion sensitivity", color="#e7f4f0")
    return _figure_data_url(figure)


def audio_explanation_image(
    waveform: np.ndarray,
    sample_rate: int,
    spectrogram: np.ndarray,
    saliency: np.ndarray,
) -> str:
    """Plot waveform, log-Mel representation, and genuine input-gradient saliency."""

    duration = np.arange(waveform.size) / sample_rate
    saliency = np.asarray(saliency, dtype=float)
    if float(saliency.max()) > 0:
        saliency /= float(saliency.max())
    figure, axes = plt.subplots(3, 1, figsize=(8, 6.3), facecolor="#0b1518")
    for axis in axes:
        axis.set_facecolor("#102126")
        axis.tick_params(colors="#bed1cc", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#365057")
    axes[0].plot(duration, waveform, color="#61d4b3", linewidth=0.7)
    axes[0].set(title="Waveform", xlabel="Time (s)")
    # Use a Matplotlib-native map. ``mako`` only exists when Seaborn happens to
    # have been imported first, which made normal Flask startup fail here.
    axes[1].imshow(spectrogram, origin="lower", aspect="auto", cmap="viridis")
    axes[1].set(title="Standardized 64-bin log-Mel spectrogram", ylabel="Mel bin")
    axes[2].imshow(saliency, origin="lower", aspect="auto", cmap="magma")
    axes[2].set(title="Input-gradient saliency for the predicted emotion", xlabel="Frame", ylabel="Mel bin")
    for axis in axes:
        axis.title.set_color("#e7f4f0")
        axis.xaxis.label.set_color("#bed1cc")
        axis.yaxis.label.set_color("#bed1cc")
    figure.tight_layout()
    return _figure_data_url(figure)
