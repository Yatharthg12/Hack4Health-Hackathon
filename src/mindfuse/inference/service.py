"""Artifact-backed, thread-safe-enough read-only inference services."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from mindfuse.config import METRICS_DIR, MODEL_DIR
from mindfuse.constants import (
    FACE_EMOTIONS,
    FACE_EMOTION_TO_STRESS,
    NUMERICAL_FEATURES,
    REGRESSION_TARGETS,
    SCORE_RANGES,
    SPEECH_EMOTIONS,
    SPEECH_EMOTION_TO_STRESS,
    STRESS_CLASSES,
)
from mindfuse.data.face import face_transform
from mindfuse.data.numerical import validate_numerical_payload
from mindfuse.data.speech import extract_audio_features
from mindfuse.explainability.visuals import audio_explanation_image, face_gradcam_image
from mindfuse.fusion.engine import aggregate_emotion_probabilities, fuse_stress_probabilities
from mindfuse.models.neural import FaceEmotionCNN, SpeechEmotionCNN
from mindfuse.models.face_hog import extract_hog_from_image
from mindfuse.explainability.visuals import face_occlusion_image
from mindfuse.utils.json_io import read_json

LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when an endpoint requires an artifact that has not been trained."""


class AudioInferenceError(RuntimeError):
    """A safe audio-inference failure carrying a stable machine-readable code."""

    def __init__(self, code: str, public_message: str, *, stage: str, detail: str | None = None) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.stage = stage
        self.detail = detail


class ModelRegistry:
    """Load all local artifacts once and provide consistent response contracts."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model: FaceEmotionCNN | None = None
        self.face_hog: dict[str, Any] | None = None
        self.face_temperature = 1.0
        self.audio_model: SpeechEmotionCNN | None = None
        self.audio_temperature = 1.0
        self.audio_classes = list(SPEECH_EMOTIONS)
        self.numerical_classifier: dict[str, Any] | None = None
        self.regressor: dict[str, Any] | None = None
        self.load_errors: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self.load_errors = {}
        hog_path = MODEL_DIR / "face_hog.joblib"
        try:
            self.face_hog = joblib.load(hog_path) if hog_path.exists() else None
        except Exception as exc:
            self.face_hog = None
            self.load_errors["face_hog"] = str(exc)
        face_path = MODEL_DIR / "face_model.pt"
        if face_path.exists():
            try:
                checkpoint = torch.load(face_path, map_location=self.device, weights_only=True)
                model = FaceEmotionCNN(len(checkpoint["class_names"]))
                model.load_state_dict(checkpoint["state_dict"])
                self.face_model = model.to(self.device).eval()
                self.face_temperature = float(checkpoint.get("temperature", 1.0))
            except Exception as exc:
                self.face_model = None
                self.load_errors["face_cnn"] = str(exc)
        else:
            self.face_model = None
            if self.face_hog is None:
                self.load_errors["face"] = "Run python scripts/train_all.py"
        audio_path = MODEL_DIR / "audio_model.pt"
        if audio_path.exists():
            try:
                checkpoint = torch.load(audio_path, map_location=self.device, weights_only=True)
                model = SpeechEmotionCNN(len(checkpoint["class_names"]))
                model.load_state_dict(checkpoint["state_dict"])
                self.audio_model = model.to(self.device).eval()
                self.audio_temperature = float(checkpoint.get("temperature", 1.0))
                self.audio_classes = list(checkpoint["class_names"])
            except Exception as exc:
                self.audio_model = None
                self.load_errors["audio"] = str(exc)
        else:
            self.audio_model = None
            self.load_errors["audio"] = "Run python scripts/train_all.py"
        for name, filename in (("numerical", "numerical_classifier.joblib"), ("regression", "das_regressor.joblib")):
            path = MODEL_DIR / filename
            try:
                artifact = joblib.load(path) if path.exists() else None
                if name == "numerical":
                    self.numerical_classifier = artifact
                else:
                    self.regressor = artifact
                if artifact is None:
                    self.load_errors[name] = "Run python scripts/train_all.py"
            except Exception as exc:
                if name == "numerical":
                    self.numerical_classifier = None
                else:
                    self.regressor = None
                self.load_errors[name] = str(exc)

    @property
    def status(self) -> dict[str, Any]:
        models = {
            "face": self.face_hog is not None or self.face_model is not None,
            "audio": self.audio_model is not None,
            "numerical": self.numerical_classifier is not None,
            "regression": self.regressor is not None,
        }
        return {"ready": all(models.values()), "models": models, "errors": self.load_errors}

    @property
    def reliability_weights(self) -> dict[str, float]:
        payload = read_json(METRICS_DIR / "fusion_weights.json", {})
        return {key: float(value) for key, value in payload.get("weights", {}).items()}

    def predict_face(self, path: Path, explain: bool = True) -> dict[str, Any]:
        if self.face_hog is None and self.face_model is None:
            raise ModelUnavailableError("Facial model is unavailable; run python scripts/train_all.py")
        try:
            with Image.open(path) as loaded:
                loaded.verify()
            with Image.open(path) as loaded:
                image = loaded.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("The uploaded image is unreadable or corrupt") from exc
        if image.width < 24 or image.height < 24 or image.width * image.height > 20_000_000:
            raise ValueError("Image dimensions are unsupported")
        tensor = None
        if self.face_hog is not None:
            features = extract_hog_from_image(image)[None, :]
            probabilities = self.face_hog["pipeline"].predict_proba(features)[0]
            backend = str(self.face_hog.get("display_name", "HOG small-data model"))
        else:
            assert self.face_model is not None
            tensor = face_transform(False)(image.convert("L")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.face_model(tensor) / self.face_temperature
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
            backend = "residual CNN"
        predicted = int(probabilities.argmax())
        stress = aggregate_emotion_probabilities(probabilities, FACE_EMOTIONS, FACE_EMOTION_TO_STRESS)
        result: dict[str, Any] = {
            "emotion": FACE_EMOTIONS[predicted],
            "confidence": float(probabilities[predicted]),
            "emotion_probabilities": {name: float(value) for name, value in zip(FACE_EMOTIONS, probabilities, strict=True)},
            "stress_probabilities": stress,
            "stress_class": max(stress, key=stress.get),
            "mapping": "facial-emotion-specific",
            "model": backend,
        }
        if explain:
            if self.face_hog is not None:
                result["face_explanation"] = face_occlusion_image(
                    self.face_hog["pipeline"], image, predicted
                )
                result["explanation_method"] = "predicted-class occlusion sensitivity"
            else:
                assert self.face_model is not None and tensor is not None
                result["face_explanation"] = face_gradcam_image(self.face_model, tensor, image, predicted)
                result["explanation_method"] = "Grad-CAM"
        return result

    def predict_audio(self, path: Path, explain: bool = True) -> dict[str, Any]:
        if self.audio_model is None:
            raise ModelUnavailableError("Speech model is unavailable; run python scripts/train_all.py")
        spectrogram, waveform, sample_rate, source_metadata = extract_audio_features(
            path, include_metadata=True
        )
        try:
            tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).to(self.device)
            tensor.requires_grad_(explain)
            with torch.set_grad_enabled(explain):
                logits = self.audio_model(tensor) / self.audio_temperature
                probabilities = torch.softmax(logits, dim=1)[0]
            predicted = int(probabilities.argmax().item())
            probability_values = probabilities.detach().cpu().numpy()
            if (
                probability_values.shape != (len(self.audio_classes),)
                or not np.all(np.isfinite(probability_values))
                or not np.isclose(float(probability_values.sum()), 1.0, atol=1e-5)
            ):
                raise RuntimeError("model returned an invalid probability vector")
        except Exception as exc:
            raise AudioInferenceError(
                "AUDIO_INFERENCE_FAILED",
                "The recording was decoded, but the speech model could not complete the analysis.",
                stage="model_inference",
                detail=f"{type(exc).__name__}:{exc}",
            ) from exc
        stress = aggregate_emotion_probabilities(probability_values, self.audio_classes, SPEECH_EMOTION_TO_STRESS)
        result: dict[str, Any] = {
            "emotion": self.audio_classes[predicted],
            "confidence": float(probability_values[predicted]),
            "emotion_probabilities": {name: float(value) for name, value in zip(self.audio_classes, probability_values, strict=True)},
            "stress_probabilities": stress,
            "stress_class": max(stress, key=stress.get),
            "mapping": "speech-emotion-specific",
            "metadata": {
                **source_metadata,
                "sample_rate": sample_rate,
                "duration_seconds": float(len(waveform) / sample_rate),
                "model_input_shape": list(spectrogram.shape),
            },
        }
        if explain:
            try:
                self.audio_model.zero_grad(set_to_none=True)
                logits[0, predicted].backward()
                if tensor.grad is None:
                    raise RuntimeError("input gradient was not produced")
                saliency = tensor.grad.detach().abs()[0, 0].cpu().numpy()
                result["audio_explanation"] = audio_explanation_image(
                    waveform, sample_rate, spectrogram, saliency
                )
                result["explanation"] = {
                    "available": True,
                    "method": "predicted-class input-gradient saliency",
                }
            except Exception as exc:
                LOGGER.exception(
                    "Audio explanation failed after successful model inference (%s)",
                    type(exc).__name__,
                )
                result["audio_explanation"] = None
                result["explanation"] = {
                    "available": False,
                    "method": "predicted-class input-gradient saliency",
                    "message": "The prediction completed, but its audio visualization is unavailable.",
                }
        return result

    def _local_numerical_explanation(self, frame: pd.DataFrame, probabilities: np.ndarray) -> list[dict[str, Any]]:
        assert self.numerical_classifier is not None
        estimator = self.numerical_classifier["pipeline"]
        medians = self.numerical_classifier["background"]["median"]
        severity = np.arange(len(STRESS_CLASSES), dtype=float)
        current = float(probabilities @ severity)
        influences: list[dict[str, Any]] = []
        for feature in NUMERICAL_FEATURES:
            counterfactual = frame.copy()
            counterfactual.loc[counterfactual.index[0], feature] = medians[feature]
            reference = float(estimator.predict_proba(counterfactual)[0] @ severity)
            delta = current - reference
            influences.append({
                "feature": feature,
                "value": float(frame.iloc[0][feature]),
                "reference": float(medians[feature]),
                "influence": delta,
                "direction": "risk-increasing" if delta > 0 else "protective/lower-risk" if delta < 0 else "neutral",
            })
        return sorted(influences, key=lambda item: abs(float(item["influence"])), reverse=True)

    def predict_numerical(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.numerical_classifier is None:
            raise ModelUnavailableError("Numerical model is unavailable; run python scripts/train_all.py")
        clean = validate_numerical_payload(payload)
        frame = pd.DataFrame([clean], columns=NUMERICAL_FEATURES)
        probabilities = self.numerical_classifier["pipeline"].predict_proba(frame)[0]
        predicted = int(probabilities.argmax())
        result: dict[str, Any] = {
            "stress_class": STRESS_CLASSES[predicted],
            "confidence": float(probabilities[predicted]),
            "stress_probabilities": {name: float(value) for name, value in zip(STRESS_CLASSES, probabilities, strict=True)},
            "local_explanation": self._local_numerical_explanation(frame, probabilities),
        }
        if self.regressor is not None:
            raw = np.asarray(self.regressor["pipeline"].predict(frame))[0]
            result["scores"] = {
                target: {
                    "value": float(np.clip(value, *SCORE_RANGES[target])),
                    "raw_value": float(value),
                    "range": list(SCORE_RANGES[target]),
                }
                for target, value in zip(REGRESSION_TARGETS, raw, strict=True)
            }
        else:
            result["scores"] = None
            result["scores_note"] = "Regression artifact is unavailable"
        return result

    def predict_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.numerical_classifier is None or self.regressor is None:
            raise ModelUnavailableError("Both tabular artifacts are required for batch prediction")
        probabilities = self.numerical_classifier["pipeline"].predict_proba(frame)
        predictions = probabilities.argmax(axis=1)
        scores = np.asarray(self.regressor["pipeline"].predict(frame))
        result = frame.copy()
        result["Predicted_Mental_Health_Status"] = [STRESS_CLASSES[index] for index in predictions]
        result["Prediction_Confidence"] = probabilities.max(axis=1)
        for index, label in enumerate(STRESS_CLASSES):
            result[f"Probability_{label}"] = probabilities[:, index]
        for index, target in enumerate(REGRESSION_TARGETS):
            result[f"Estimated_{target}"] = np.clip(scores[:, index], *SCORE_RANGES[target])
        return result

    def fuse(self, modalities: dict[str, dict[str, float]]) -> dict[str, Any]:
        allowed = {"face", "audio", "numerical"}
        unknown = set(modalities) - allowed
        if unknown:
            raise ValueError(f"Unknown modalities: {', '.join(sorted(unknown))}")
        ordered: dict[str, list[float]] = {}
        for name, probabilities in modalities.items():
            missing = [label for label in STRESS_CLASSES if label not in probabilities]
            if missing:
                raise ValueError(f"{name} is missing stress probabilities: {', '.join(missing)}")
            ordered[name] = [probabilities[label] for label in STRESS_CLASSES]
        return fuse_stress_probabilities(ordered, self.reliability_weights)
