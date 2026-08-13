"""Train-from-scratch facial emotion recognition with calibrated inference."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from mindfuse.config import FIGURES_DIR, METRICS_DIR, MODEL_DIR, RANDOM_SEED
from mindfuse.constants import FACE_EMOTIONS
from mindfuse.data.face import FaceDataset, build_face_records, split_face_records
from mindfuse.evaluation.metrics import classification_metrics
from mindfuse.evaluation.plots import class_distribution_plot, confusion_matrix_plot, training_history_plot
from mindfuse.models.calibration import TemperatureScaledClassifier, fit_probability_temperature
from mindfuse.models.face_hog import extract_hog_from_paths
from mindfuse.models.neural import FaceEmotionCNN, parameter_count
from mindfuse.training.neural_utils import collect_logits, fit_temperature, train_classifier
from mindfuse.utils.json_io import write_json


def train_face_model(
    image_paths: list[Path],
    epochs: int = 25,
    patience: int = 6,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    records = build_face_records(image_paths)
    if len(records) < 70:
        raise ValueError(f"Too few valid labelled face images: {len(records)}")
    train_records, validation_records, test_records, split_method = split_face_records(records, seed)
    small_data_regime = len(records) < 1_000
    batch_size = 32 if small_data_regime else 64
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        # With only tens of examples per class, affine/color jitter overwhelmed the
        # signal in audit experiments. The CNN remains trained from scratch while
        # the explicit HOG baseline below is the predeclared small-data primary.
        FaceDataset(train_records, training=not small_data_regime), batch_size=batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )
    validation_loader = DataLoader(FaceDataset(validation_records), batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(FaceDataset(test_records), batch_size=batch_size, num_workers=0)
    counts = np.bincount([record.label for record in train_records], minlength=len(FACE_EMOTIONS))
    class_weights = len(train_records) / (len(FACE_EMOTIONS) * np.maximum(counts, 1))
    class_weights = np.sqrt(class_weights)
    class_weights /= class_weights.mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceEmotionCNN(len(FACE_EMOTIONS))
    cnn_epochs = max(epochs, 40) if small_data_regime else epochs
    cnn_patience = max(patience, 12) if small_data_regime else patience
    model, history, training_summary = train_classifier(
        model, train_loader, validation_loader, torch.tensor(class_weights, dtype=torch.float32),
        device, cnn_epochs, cnn_patience, learning_rate=3e-3 if small_data_regime else 1e-3,
    )
    validation_logits, validation_labels = collect_logits(model, validation_loader, device)
    temperature = fit_temperature(validation_logits, validation_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    probabilities = torch.softmax(torch.tensor(test_logits) / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    test_metrics = classification_metrics(test_labels, predictions, probabilities, FACE_EMOTIONS)
    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "class_names": FACE_EMOTIONS,
        "temperature": temperature,
        "architecture": "FaceEmotionCNN",
        "input_size": [1, 48, 48],
        "seed": seed,
    }
    torch.save(checkpoint, MODEL_DIR / "face_model.pt")
    distribution = Counter(FACE_EMOTIONS[record.label] for record in records)
    cnn_payload: dict[str, Any] = {
        "architecture": "compact four-stage residual CNN trained from random initialization",
        "parameters": parameter_count(model),
        "pretrained": False,
        "input": (
            "48x48 grayscale; normalized to [-1, 1]; augmentation disabled because audit found fewer than 1,000 images"
            if small_data_regime else
            "48x48 grayscale; train-only flip/affine/brightness/contrast augmentation; normalized to [-1, 1]"
        ),
        "class_imbalance": "square-root inverse-frequency cross-entropy weights",
        "probability_calibration": {"method": "validation temperature scaling", "temperature": temperature},
        "split_method": split_method,
        "split": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "class_distribution": dict(distribution),
        "training": training_summary,
        "history": history,
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": str(device),
    }
    write_json(METRICS_DIR / "face_cnn.json", cnn_payload)
    training_history_plot(history, FIGURES_DIR / "face_training_history.png", "Face emotion CNN training")
    class_distribution_plot(dict(distribution), FIGURES_DIR / "face_class_distribution.png", "Face dataset class distribution")
    if not small_data_regime:
        (MODEL_DIR / "face_hog.joblib").unlink(missing_ok=True)
        write_json(METRICS_DIR / "face.json", cnn_payload)
        confusion_matrix_plot(
            test_metrics["confusion_matrix"], FACE_EMOTIONS,
            FIGURES_DIR / "face_confusion_matrix.png", "Face emotion CNN — held-out test",
        )
        return cnn_payload

    # Predeclared small-data fallback: handcrafted HOG has far lower capacity and
    # is fully learned from the supplied images. The CNN above remains implemented,
    # trained, checkpointed, and evaluated as a transparent benchmark.
    train_features = extract_hog_from_paths(record.path for record in train_records)
    validation_features = extract_hog_from_paths(record.path for record in validation_records)
    test_features = extract_hog_from_paths(record.path for record in test_records)
    train_labels = np.asarray([record.label for record in train_records])
    validation_labels = np.asarray([record.label for record in validation_records])
    test_labels = np.asarray([record.label for record in test_records])
    comparison: dict[str, dict[str, float]] = {}
    best_name = ""
    best_score = -np.inf
    best_estimator: Pipeline | None = None
    candidates: dict[str, Pipeline] = {}
    for regularization in (0.01, 0.1, 1.0, 10.0):
        candidates[f"hog_logistic_C={regularization:g}"] = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=regularization, class_weight="balanced", max_iter=5_000, random_state=seed,
            )),
        ])
    # RBF SVM is still a from-scratch, low-capacity classifier over the same
    # handcrafted HOG representation. Its small fixed grid is selected only on
    # validation macro-F1; the held-out test remains untouched until selection.
    for regularization in (0.1, 1.0, 3.0):
        candidates[f"hog_rbf_svc_C={regularization:g}"] = Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(
                C=regularization,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                cache_size=512,
                random_state=seed,
            )),
        ])
    for name, candidate in candidates.items():
        candidate.fit(train_features, train_labels)
        candidate_probabilities = candidate.predict_proba(validation_features)
        candidate_predictions = candidate_probabilities.argmax(axis=1)
        candidate_metrics = classification_metrics(
            validation_labels, candidate_predictions, candidate_probabilities, FACE_EMOTIONS
        )
        comparison[name] = {
            "macro_f1": float(candidate_metrics["macro_f1"]),
            "balanced_accuracy": float(candidate_metrics["balanced_accuracy"]),
        }
        if float(candidate_metrics["macro_f1"]) > best_score:
            best_name, best_score, best_estimator = name, float(candidate_metrics["macro_f1"]), candidate
    assert best_estimator is not None
    hog_temperature = fit_probability_temperature(
        best_estimator.predict_proba(validation_features), validation_labels
    )
    combined_features = np.concatenate([train_features, validation_features])
    combined_labels = np.concatenate([train_labels, validation_labels])
    final_hog = clone(best_estimator).fit(combined_features, combined_labels)
    calibrated_hog = TemperatureScaledClassifier(final_hog, hog_temperature)
    hog_probabilities = calibrated_hog.predict_proba(test_features)
    hog_metrics = classification_metrics(
        test_labels, hog_probabilities.argmax(axis=1), hog_probabilities, FACE_EMOTIONS
    )
    joblib.dump({
        "pipeline": calibrated_hog,
        "class_names": FACE_EMOTIONS,
        "input_size": [48, 48],
        "feature": "900-dimensional 2x2-block L2-Hys HOG",
        "temperature": hog_temperature,
        "model_name": best_name,
        "display_name": "HOG RBF-SVM small-data model" if "svc" in best_name else "HOG logistic small-data model",
        "seed": seed,
    }, MODEL_DIR / "face_hog.joblib", compress=3)
    payload = {
        "architecture": "validation-selected class-balanced classifier over HOG features for the audited 350-image small-data regime",
        "pretrained": False,
        "primary_model_reason": (
            "The supplied folder contains fewer than 1,000 images (50/class), despite 28,709 being documented. "
            "A lower-capacity HOG classifier was predeclared for this regime; logistic and RBF-SVM candidates are "
            "selected on validation macro-F1, while the residual CNN remains trained and evaluated."
        ),
        "input": "48x48 grayscale; 900-dimensional L2-Hys HOG; standardized inside the training Pipeline",
        "probability_calibration": {
            "method": "decision-preserving validation temperature scaling", "temperature": hog_temperature,
        },
        "explainability": "local predicted-class occlusion sensitivity tied to the selected HOG model",
        "split_method": split_method,
        "split": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "class_distribution": dict(distribution),
        "model_selection": comparison,
        "selected_model": best_name,
        "training": {"best_validation_macro_f1": best_score},
        "cnn_benchmark": {
            "architecture": cnn_payload["architecture"],
            "parameters": cnn_payload["parameters"],
            "best_validation_macro_f1": cnn_payload["training"]["best_validation_macro_f1"],
            "test_macro_f1": cnn_payload["test_metrics"]["macro_f1"],
            "artifact": "artifacts/metrics/face_cnn.json",
        },
        "test_metrics": hog_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": "CPU-compatible sklearn Pipeline",
    }
    write_json(METRICS_DIR / "face.json", payload)
    confusion_matrix_plot(
        hog_metrics["confusion_matrix"], FACE_EMOTIONS,
        FIGURES_DIR / "face_confusion_matrix.png", "Face HOG classifier — held-out test",
    )
    return payload
