from __future__ import annotations

import io

import pytest

from app import create_app
from mindfuse.fusion.engine import fuse_stress_probabilities
from mindfuse.inference.service import ModelUnavailableError


class FakeRegistry:
    status = {"ready": True, "models": {"face": True, "audio": True, "numerical": True, "regression": True}, "errors": {}}
    reliability_weights = {"face": 0.7, "audio": 0.65, "numerical": 0.8}

    def predict_face(self, path, explain=True):
        if path.read_bytes().startswith(b"bad"):
            raise ValueError("The uploaded image is unreadable or corrupt")
        return {"stress_probabilities": {"Healthy": 0.7, "Mild_Stress": 0.2, "Moderate_Stress": 0.1, "Severe_Stress": 0.0}}

    def predict_audio(self, path, explain=True):
        raise ValueError("The WAV file is unreadable or corrupt")

    def predict_numerical(self, payload):
        if not payload:
            raise ValueError("Missing numerical fields")
        return {"stress_probabilities": {"Healthy": 0.2, "Mild_Stress": 0.5, "Moderate_Stress": 0.2, "Severe_Stress": 0.1}, "scores": None}

    def fuse(self, modalities):
        return fuse_stress_probabilities(
            {name: list(probabilities.values()) for name, probabilities in modalities.items()},
            self.reliability_weights,
        )

    def predict_batch(self, frame):
        result = frame.copy(); result["Predicted_Mental_Health_Status"] = "Healthy"; return result


@pytest.fixture()
def client():
    application = create_app({"TESTING": True}, registry=FakeRegistry())
    return application.test_client()


def test_health_endpoint(client) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["model_status"]["ready"] is True


def test_invalid_image_extension_and_corrupt_image(client) -> None:
    invalid = client.post("/api/predict/face", data={"image": (io.BytesIO(b"x"), "face.exe")})
    assert invalid.status_code == 400
    corrupt = client.post("/api/predict/face", data={"image": (io.BytesIO(b"bad bytes"), "face.png")})
    assert corrupt.status_code == 400
    assert corrupt.get_json()["error"] == "validation_error"


def test_zero_byte_and_malformed_audio(client) -> None:
    empty = client.post("/api/predict/audio", data={"audio": (io.BytesIO(b""), "clip.wav")})
    assert empty.status_code == 400
    malformed = client.post("/api/predict/audio", data={"audio": (io.BytesIO(b"RIFFbad"), "clip.wav")})
    assert malformed.status_code == 400


def test_numerical_endpoint_validation(client) -> None:
    response = client.post("/api/predict/numerical", json={})
    assert response.status_code == 400
    assert response.get_json()["message"] == "Missing numerical fields"


def test_multimodal_and_missing_modality_behavior(client) -> None:
    probabilities = {"Healthy": 0.1, "Mild_Stress": 0.2, "Moderate_Stress": 0.6, "Severe_Stress": 0.1}
    response = client.post("/api/predict/multimodal", json={"modalities": {"audio": probabilities}})
    assert response.status_code == 200
    assert response.get_json()["final_class"] == "Moderate_Stress"
    missing = client.post("/api/predict/multimodal", json={})
    assert missing.status_code == 400


def test_all_pages_render_without_template_errors(client) -> None:
    for path in ("/", "/assessment", "/explainability", "/performance", "/methodology", "/batch"):
        response = client.get(path)
        assert response.status_code == 200
        assert b"MindFuse" in response.data

