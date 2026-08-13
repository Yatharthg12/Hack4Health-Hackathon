from __future__ import annotations

import math

import pytest

from mindfuse.constants import (
    FACE_EMOTIONS,
    FACE_EMOTION_TO_STRESS,
    SPEECH_EMOTIONS,
    SPEECH_EMOTION_TO_STRESS,
    STRESS_CLASSES,
)
from mindfuse.fusion.engine import (
    aggregate_emotion_probabilities,
    fuse_stress_probabilities,
    normalize_probabilities,
)


def one_hot(classes: list[str], selected: str) -> list[float]:
    return [1.0 if name == selected else 0.0 for name in classes]


def test_modality_specific_angry_and_disgust_mappings_are_preserved() -> None:
    face_angry = aggregate_emotion_probabilities(one_hot(FACE_EMOTIONS, "Angry"), FACE_EMOTIONS, FACE_EMOTION_TO_STRESS)
    speech_angry = aggregate_emotion_probabilities(one_hot(SPEECH_EMOTIONS, "Angry"), SPEECH_EMOTIONS, SPEECH_EMOTION_TO_STRESS)
    face_disgust = aggregate_emotion_probabilities(one_hot(FACE_EMOTIONS, "Disgust"), FACE_EMOTIONS, FACE_EMOTION_TO_STRESS)
    speech_disgust = aggregate_emotion_probabilities(one_hot(SPEECH_EMOTIONS, "Disgust"), SPEECH_EMOTIONS, SPEECH_EMOTION_TO_STRESS)
    assert face_angry["Severe_Stress"] == 1
    assert speech_angry["Moderate_Stress"] == 1
    assert face_disgust["Moderate_Stress"] == 1
    assert speech_disgust["Severe_Stress"] == 1


def test_full_distribution_is_aggregated_not_only_argmax() -> None:
    probabilities = [0.1, 0.1, 0.1, 0.2, 0.15, 0.1, 0.15, 0.1]
    stress = aggregate_emotion_probabilities(probabilities, SPEECH_EMOTIONS, SPEECH_EMOTION_TO_STRESS)
    assert stress == pytest.approx({
        "Healthy": 0.3, "Mild_Stress": 0.3, "Moderate_Stress": 0.25, "Severe_Stress": 0.15
    })


def test_fusion_normalizes_weights_and_probabilities() -> None:
    result = fuse_stress_probabilities(
        {"face": [0.7, 0.2, 0.1, 0.0], "audio": [0.1, 0.2, 0.6, 0.1]},
        {"face": 0.8, "audio": 0.6},
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert sum(item["weight"] for item in result["contributions"].values()) == pytest.approx(1.0)
    assert 0 <= result["confidence"] <= 1
    assert 0 <= result["agreement_score"] <= 1
    assert 0 <= result["uncertainty_score"] <= 1


def test_missing_modality_fusion_uses_available_branch_only() -> None:
    supplied = [0.05, 0.15, 0.7, 0.1]
    result = fuse_stress_probabilities({"numerical": supplied}, {"numerical": 0.81})
    assert list(result["probabilities"].values()) == pytest.approx(supplied)
    assert result["contributions"]["numerical"]["weight"] == pytest.approx(1.0)
    assert result["agreement_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("values", ([0, 0, 0, 0], [1, -1, 0, 0], [1, math.nan, 0, 0], [1, 2]))
def test_invalid_probability_vectors_are_rejected(values) -> None:
    with pytest.raises(ValueError):
        normalize_probabilities(values)

