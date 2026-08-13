"""MindFuse XAI Flask application and validated inference API."""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mindfuse import __version__  # noqa: E402
from mindfuse.config import (  # noqa: E402
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    FIGURES_DIR,
    MAX_UPLOAD_BYTES,
    METRICS_DIR,
    UPLOAD_DIR,
    ensure_runtime_directories,
)
from mindfuse.constants import FEATURE_METADATA, NUMERICAL_FEATURES, STRESS_CLASSES  # noqa: E402
from mindfuse.data.numerical import validate_batch_frame  # noqa: E402
from mindfuse.data.speech import AudioProcessingError  # noqa: E402
from mindfuse.inference.service import AudioInferenceError, ModelRegistry, ModelUnavailableError  # noqa: E402
from mindfuse.utils.json_io import read_json  # noqa: E402

LOGGER = logging.getLogger("mindfuse")


class UploadValidationError(ValueError):
    """An upload validation problem with a stable API error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message


@contextmanager
def temporary_upload(file_storage, allowed_extensions: set[str]) -> Iterator[Path]:
    """Persist an upload under an opaque server filename and always remove it."""

    if file_storage is None or not file_storage.filename:
        raise UploadValidationError("MISSING_UPLOAD", "No file was uploaded.")
    original = secure_filename(file_storage.filename)
    suffix = Path(original).suffix.lower()
    if suffix not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise UploadValidationError("UNSUPPORTED_FILE_TYPE", f"Unsupported file type. Allowed: {allowed}")
    destination = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    file_storage.save(destination)
    try:
        if destination.stat().st_size == 0:
            raise UploadValidationError("EMPTY_UPLOAD", "The uploaded file is empty.")
        yield destination
    finally:
        destination.unlink(missing_ok=True)


def create_app(test_config: dict | None = None, registry: ModelRegistry | None = None) -> Flask:
    ensure_runtime_directories()
    application = Flask(__name__)
    application.config.update(
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        JSON_SORT_KEYS=False,
        SECRET_KEY="development-only-not-used-for-auth",
    )
    if test_config:
        application.config.update(test_config)
    application.extensions["model_registry"] = registry or ModelRegistry()

    def models() -> ModelRegistry:
        return application.extensions["model_registry"]

    @application.after_request
    def security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

    @application.get("/")
    def overview():
        return render_template("overview.html", page="overview")

    @application.get("/assessment")
    def assessment():
        return render_template(
            "assessment.html", page="assessment", feature_metadata=FEATURE_METADATA,
            numerical_features=NUMERICAL_FEATURES,
        )

    @application.get("/explainability")
    def explainability():
        return render_template("explainability.html", page="explainability")

    @application.get("/performance")
    def performance():
        return render_template("performance.html", page="performance")

    @application.get("/methodology")
    def methodology():
        return render_template("methodology.html", page="methodology")

    @application.get("/batch")
    def batch():
        return render_template("batch.html", page="batch", numerical_features=NUMERICAL_FEATURES)

    @application.get("/generated/<path:filename>")
    def generated_figure(filename: str):
        return send_from_directory(FIGURES_DIR, filename)

    @application.get("/favicon.ico")
    def favicon():
        return send_from_directory(ROOT / "static", "assets/favicon.svg", mimetype="image/svg+xml")

    @application.get("/api/health")
    def api_health():
        audit = read_json(METRICS_DIR / "dataset_audit.json", {})
        return jsonify({
            "status": "ready" if models().status["ready"] else "degraded",
            "version": __version__,
            "model_status": models().status,
            "dataset": {
                "face_files": audit.get("face", {}).get("files"),
                "audio_files": audit.get("speech", {}).get("files"),
                "numerical_rows": audit.get("numerical", {}).get("rows"),
            },
        })

    @application.get("/api/metrics")
    def api_metrics():
        names = ["summary", "dataset_audit", "face", "audio", "numerical_classification", "regression", "fusion_weights"]
        return jsonify({name: read_json(METRICS_DIR / f"{name}.json") for name in names})

    @application.get("/api/config")
    def api_config():
        fallback_profile = {
            feature: FEATURE_METADATA[feature]["demo"] for feature in NUMERICAL_FEATURES
        }
        demo_profiles = read_json(METRICS_DIR / "demo_profiles.json", {})
        if not demo_profiles:
            demo_profiles = {
                "profiles": {
                    "good": {"label": "Good", "values": fallback_profile},
                    "typical": {"label": "Typical", "values": fallback_profile},
                    "high_strain": {"label": "High strain", "values": fallback_profile},
                },
                "derivation": "Fallback UI defaults; regenerate artifacts for training-only profiles.",
            }
        return jsonify({
            "stress_classes": STRESS_CLASSES,
            "features": FEATURE_METADATA,
            "demo_profiles": demo_profiles,
            "demo_profile": demo_profiles.get("profiles", {}).get("typical", {}).get(
                "values", fallback_profile
            ),
        })

    @application.post("/api/predict/face")
    def api_predict_face():
        with temporary_upload(request.files.get("image"), ALLOWED_IMAGE_EXTENSIONS) as path:
            return jsonify(models().predict_face(path, explain=True))

    @application.post("/api/predict/audio")
    def api_predict_audio():
        uploaded = request.files.get("audio")
        original = secure_filename(uploaded.filename) if uploaded and uploaded.filename else ""
        mime_type = uploaded.mimetype if uploaded else None
        try:
            with temporary_upload(uploaded, ALLOWED_AUDIO_EXTENSIONS) as path:
                size_bytes = path.stat().st_size
                LOGGER.info(
                    "Audio upload accepted name=%r size_bytes=%d mime=%r upload_id=%s",
                    original, size_bytes, mime_type, path.stem,
                )
                result = models().predict_audio(path, explain=True)
                LOGGER.info(
                    "Audio prediction completed name=%r decoder=%s input_shape=%s explanation=%s",
                    original,
                    result.get("metadata", {}).get("decoder"),
                    result.get("metadata", {}).get("model_input_shape"),
                    result.get("explanation", {}).get("available"),
                )
                return jsonify({"ok": True, **result})
        except (UploadValidationError, AudioProcessingError) as error:
            LOGGER.warning(
                "Audio request rejected name=%r size_bytes=%s mime=%r code=%s stage=%s decoder=%s detail=%s",
                original,
                locals().get("size_bytes"),
                mime_type,
                getattr(error, "code", "INVALID_AUDIO"),
                getattr(error, "stage", "upload_validation"),
                getattr(error, "decoder", None),
                getattr(error, "detail", None),
            )
            return jsonify({
                "ok": False,
                "error": {
                    "code": getattr(error, "code", "INVALID_AUDIO"),
                    "message": getattr(error, "public_message", str(error)),
                },
            }), 400
        except AudioInferenceError as error:
            LOGGER.error(
                "Audio inference failed name=%r size_bytes=%s mime=%r code=%s stage=%s detail=%s",
                original, locals().get("size_bytes"), mime_type, error.code, error.stage, error.detail,
                exc_info=True,
            )
            return jsonify({
                "ok": False,
                "error": {"code": error.code, "message": error.public_message},
            }), 500
        except ModelUnavailableError as error:
            LOGGER.error(
                "Audio model unavailable name=%r size_bytes=%s mime=%r",
                original, locals().get("size_bytes"), mime_type,
            )
            return jsonify({
                "ok": False,
                "error": {"code": "MODEL_UNAVAILABLE", "message": str(error)},
            }), 503
        except ValueError as error:
            # Preserve compatibility with alternate registries that expose
            # decoded-file validation as a plain ValueError.
            LOGGER.warning(
                "Audio validation failed name=%r size_bytes=%s mime=%r exception=%s",
                original, locals().get("size_bytes"), mime_type, type(error).__name__,
                exc_info=True,
            )
            return jsonify({
                "ok": False,
                "error": {"code": "INVALID_AUDIO", "message": str(error)},
            }), 400
        except Exception as error:
            LOGGER.exception(
                "Unexpected audio endpoint failure name=%r size_bytes=%s mime=%r exception=%s",
                original, locals().get("size_bytes"), mime_type, type(error).__name__,
            )
            return jsonify({
                "ok": False,
                "error": {
                    "code": "INTERNAL_AUDIO_ERROR",
                    "message": "The audio request could not be completed safely. Please retry the recording.",
                },
            }), 500

    @application.post("/api/predict/numerical")
    def api_predict_numerical():
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValueError("Expected a JSON numerical profile")
        return jsonify(models().predict_numerical(payload))

    @application.post("/api/predict/multimodal")
    def api_predict_multimodal():
        modality_results: dict[str, dict] = {}
        scores = None
        evidence: dict[str, object] = {}
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            supplied = payload.get("modalities", {})
            if not isinstance(supplied, dict):
                raise ValueError("modalities must be an object")
            for name in ("face", "audio", "numerical"):
                if name in supplied:
                    probabilities = supplied[name]
                    if not isinstance(probabilities, dict):
                        raise ValueError(f"{name} probabilities must be an object")
                    modality_results[name] = probabilities
            if "numerical" in payload:
                numerical = models().predict_numerical(payload["numerical"])
                modality_results["numerical"] = numerical["stress_probabilities"]
                scores = numerical.get("scores")
                evidence["numerical"] = numerical
        else:
            face_file = request.files.get("image")
            if face_file and face_file.filename:
                with temporary_upload(face_file, ALLOWED_IMAGE_EXTENSIONS) as path:
                    face = models().predict_face(path, explain=True)
                modality_results["face"] = face["stress_probabilities"]
                evidence["face"] = face
            audio_file = request.files.get("audio")
            if audio_file and audio_file.filename:
                with temporary_upload(audio_file, ALLOWED_AUDIO_EXTENSIONS) as path:
                    audio = models().predict_audio(path, explain=True)
                modality_results["audio"] = audio["stress_probabilities"]
                evidence["audio"] = audio
            numerical_text = request.form.get("numerical")
            if numerical_text:
                try:
                    numerical_payload = json.loads(numerical_text)
                except json.JSONDecodeError as exc:
                    raise ValueError("numerical must contain valid JSON") from exc
                numerical = models().predict_numerical(numerical_payload)
                modality_results["numerical"] = numerical["stress_probabilities"]
                scores = numerical.get("scores")
                evidence["numerical"] = numerical
        if not modality_results:
            raise ValueError("Provide at least one available modality")
        result = models().fuse(modality_results)
        result["session_id"] = f"MFX-{uuid.uuid4().hex[:12].upper()}"
        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["scores"] = scores
        result["evidence"] = evidence
        result["responsible_use"] = "Experimental decision support only; not a diagnosis or replacement for qualified clinical assessment."
        return jsonify(result)

    @application.post("/api/predict/batch")
    def api_predict_batch():
        file_storage = request.files.get("csv")
        if file_storage is None or not file_storage.filename:
            raise ValueError("No CSV file was uploaded")
        if Path(secure_filename(file_storage.filename)).suffix.lower() != ".csv":
            raise ValueError("Batch input must be a CSV file")
        try:
            frame = pd.read_csv(file_storage.stream)
        except Exception as exc:
            raise ValueError("The uploaded CSV is malformed or unreadable") from exc
        validated = validate_batch_frame(frame)
        result = models().predict_batch(validated)
        buffer = io.StringIO()
        result.to_csv(buffer, index=False, quoting=csv.QUOTE_MINIMAL)
        return Response(
            buffer.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=mindfuse_batch_results.csv"},
        )

    @application.errorhandler(ValueError)
    def handle_validation_error(error: ValueError):
        return jsonify({"error": "validation_error", "message": str(error)}), 400

    @application.errorhandler(ModelUnavailableError)
    def handle_model_error(error: ModelUnavailableError):
        return jsonify({"error": "model_unavailable", "message": str(error)}), 503

    @application.errorhandler(RequestEntityTooLarge)
    def handle_too_large(_error: RequestEntityTooLarge):
        if request.path == "/api/predict/audio":
            return jsonify({
                "ok": False,
                "error": {
                    "code": "FILE_TOO_LARGE",
                    "message": f"Audio uploads are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                },
            }), 413
        return jsonify({"error": "file_too_large", "message": f"Uploads are limited to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}), 413

    @application.errorhandler(404)
    def handle_not_found(_error):
        if request.path.startswith("/api/"):
            return jsonify({"error": "not_found", "message": "API route not found"}), 404
        return render_template("404.html", page=""), 404

    @application.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        LOGGER.exception("Unhandled request failure", exc_info=error)
        return jsonify({"error": "internal_error", "message": "The request could not be completed safely"}), 500

    return application


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.run(host="127.0.0.1", port=5000, debug=False)
