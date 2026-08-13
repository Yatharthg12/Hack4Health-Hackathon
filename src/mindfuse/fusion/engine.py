"""Confidence-aware late fusion for scientifically unpaired training datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from mindfuse.constants import STRESS_CLASSES


def normalize_probabilities(probabilities: Sequence[float]) -> np.ndarray:
    """Return a finite, non-negative probability vector summing to one."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(STRESS_CLASSES),):
        raise ValueError(f"Expected {len(STRESS_CLASSES)} stress probabilities")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Probabilities must be finite and non-negative")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one probability must be positive")
    return values / total


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy scaled to [0, 1]."""

    probs = normalize_probabilities(probabilities)
    positive = probs[probs > 0]
    return float(-np.sum(positive * np.log(positive)) / np.log(len(probs)))


def aggregate_emotion_probabilities(
    probabilities: Sequence[float],
    emotion_classes: Sequence[str],
    emotion_to_stress: Mapping[str, str],
) -> dict[str, float]:
    """Sum a complete emotion distribution into the four stress categories."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(emotion_classes),):
        raise ValueError("Emotion probability count does not match class count")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Invalid emotion probability distribution")
    values = values / values.sum()
    stress = dict.fromkeys(STRESS_CLASSES, 0.0)
    for emotion, probability in zip(emotion_classes, values, strict=True):
        if emotion not in emotion_to_stress:
            raise ValueError(f"No stress mapping configured for emotion: {emotion}")
        stress[emotion_to_stress[emotion]] += float(probability)
    return stress


def _pairwise_agreement(vectors: list[np.ndarray], weights: np.ndarray) -> float:
    if len(vectors) == 1:
        return 1.0
    similarities: list[float] = []
    pair_weights: list[float] = []
    for first in range(len(vectors)):
        for second in range(first + 1, len(vectors)):
            # One minus total-variation distance is a bounded distribution agreement score.
            similarities.append(1.0 - 0.5 * float(np.abs(vectors[first] - vectors[second]).sum()))
            pair_weights.append(float(weights[first] * weights[second]))
    return float(np.average(similarities, weights=pair_weights))


def fuse_stress_probabilities(
    modality_probabilities: Mapping[str, Sequence[float]],
    reliability_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Fuse any non-empty subset of modalities using a confidence-aware linear pool.

    A modality's effective weight is its held-out reliability multiplied by a bounded
    sample confidence factor, ``0.5 + 0.5 * (1 - normalized_entropy)``. The 0.5 floor
    prevents an uncertain distribution from being discarded without evidence.
    """

    if not modality_probabilities:
        raise ValueError("At least one modality is required for fusion")

    reliabilities = dict(reliability_weights or {})
    names = list(modality_probabilities)
    vectors = [normalize_probabilities(modality_probabilities[name]) for name in names]
    entropies = np.asarray([normalized_entropy(vector) for vector in vectors])
    base = np.asarray([max(float(reliabilities.get(name, 1.0)), 1e-6) for name in names])
    effective = base * (0.5 + 0.5 * (1.0 - entropies))
    weights = effective / effective.sum()
    fused = np.sum(np.stack(vectors) * weights[:, None], axis=0)
    fused = normalize_probabilities(fused)
    agreement = _pairwise_agreement(vectors, weights)
    uncertainty = normalized_entropy(fused)
    top_indices = [int(vector.argmax()) for vector in vectors]
    confident_disagreement = len(set(top_indices)) > 1 and max(1.0 - entropies) >= 0.45
    conflict = bool((agreement < 0.6) or confident_disagreement)

    contributions: dict[str, dict[str, object]] = {}
    for idx, name in enumerate(names):
        contributions[name] = {
            "weight": float(weights[idx]),
            "base_reliability": float(base[idx]),
            "sample_confidence": float(1.0 - entropies[idx]),
            "predicted_class": STRESS_CLASSES[top_indices[idx]],
            "probabilities": {
                label: float(value) for label, value in zip(STRESS_CLASSES, vectors[idx], strict=True)
            },
        }

    winner = int(fused.argmax())
    return {
        "final_class": STRESS_CLASSES[winner],
        "confidence": float(fused[winner]),
        "probabilities": {label: float(value) for label, value in zip(STRESS_CLASSES, fused, strict=True)},
        "contributions": contributions,
        "agreement_score": agreement,
        "uncertainty_score": uncertainty,
        "conflict": conflict,
        "method": "confidence-aware weighted linear opinion pool",
    }

