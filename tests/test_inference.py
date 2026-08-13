from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mindfuse.constants import FEATURE_METADATA, NUMERICAL_FEATURES, REGRESSION_TARGETS, SCORE_RANGES, STRESS_CLASSES
from mindfuse.inference.service import ModelRegistry, ModelUnavailableError
from mindfuse.models.neural import FaceEmotionCNN, SpeechEmotionCNN


class ConstantClassifier:
    def predict_proba(self, frame):
        return np.tile(np.asarray([[0.1, 0.2, 0.6, 0.1]]), (len(frame), 1))


class ConstantRegressor:
    def predict(self, frame):
        return np.tile(np.asarray([[80.0, -5.0, 20.0]]), (len(frame), 1))


def tabular_registry() -> ModelRegistry:
    registry = ModelRegistry.__new__(ModelRegistry)
    registry.device = torch.device("cpu")
    registry.face_model = None
    registry.face_hog = None
    registry.audio_model = None
    registry.numerical_classifier = {
        "pipeline": ConstantClassifier(),
        "background": {"median": {feature: float(FEATURE_METADATA[feature]["demo"]) for feature in NUMERICAL_FEATURES}},
    }
    registry.regressor = {"pipeline": ConstantRegressor()}
    registry.load_errors = {}
    return registry


def test_numerical_prediction_shapes_and_regression_clamps() -> None:
    registry = tabular_registry()
    result = registry.predict_numerical({feature: FEATURE_METADATA[feature]["demo"] for feature in NUMERICAL_FEATURES})
    assert list(result["stress_probabilities"]) == STRESS_CLASSES
    assert len(result["local_explanation"]) == len(NUMERICAL_FEATURES)
    assert list(result["scores"]) == REGRESSION_TARGETS
    assert result["scores"]["Depression_Score"]["value"] == SCORE_RANGES["Depression_Score"][1]
    assert result["scores"]["Anxiety_Score"]["value"] == SCORE_RANGES["Anxiety_Score"][0]


def test_missing_model_raises_actionable_error() -> None:
    registry = tabular_registry(); registry.numerical_classifier = None
    with pytest.raises(ModelUnavailableError, match="train_all"):
        registry.predict_numerical({})


def test_registry_fusion_rejects_incomplete_distribution() -> None:
    registry = tabular_registry()
    with pytest.raises(ValueError, match="missing stress probabilities"):
        registry.fuse({"face": {"Healthy": 1.0}})


def test_corrupt_image_and_audio_are_rejected(tmp_path: Path) -> None:
    registry = tabular_registry()
    registry.face_model = FaceEmotionCNN().eval(); registry.face_temperature = 1.0
    registry.audio_model = SpeechEmotionCNN().eval(); registry.audio_temperature = 1.0
    image = tmp_path / "broken.png"; image.write_bytes(b"not an image")
    audio = tmp_path / "broken.wav"; audio.write_bytes(b"RIFFbroken")
    with pytest.raises(ValueError, match="unreadable|corrupt"):
        registry.predict_face(image)
    with pytest.raises(ValueError, match="unreadable|corrupt"):
        registry.predict_audio(audio)
