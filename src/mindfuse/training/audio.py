"""Actor-disjoint speech-emotion CNN training and calibrated evaluation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mindfuse.config import DATA_ROOT, FIGURES_DIR, METRICS_DIR, MODEL_DIR, RANDOM_SEED
from mindfuse.constants import SPEECH_EMOTIONS
from mindfuse.data.audio_dataset import (
    AudioFeatureDataset,
    actor_disjoint_split,
    build_audio_records,
    build_or_load_feature_cache,
)
from mindfuse.evaluation.metrics import classification_metrics
from mindfuse.evaluation.plots import class_distribution_plot, confusion_matrix_plot, training_history_plot
from mindfuse.models.neural import SpeechEmotionCNN, parameter_count
from mindfuse.training.neural_utils import collect_logits, fit_temperature, train_classifier
from mindfuse.utils.json_io import write_json


def train_audio_model(
    audio_paths: list[Path],
    epochs: int = 25,
    patience: int = 6,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    records = build_audio_records(audio_paths)
    if len(records) < 80:
        raise ValueError(f"Too few valid speech files: {len(records)}")
    features = build_or_load_feature_cache(records, DATA_ROOT / "cache" / "audio_features.npz")
    global_labels = np.asarray([record.label for record in records], dtype=int)
    observed_global_indices = sorted(set(global_labels.tolist()))
    observed_emotions = [SPEECH_EMOTIONS[index] for index in observed_global_indices]
    global_to_local = {global_index: local_index for local_index, global_index in enumerate(observed_global_indices)}
    labels = np.asarray([global_to_local[index] for index in global_labels], dtype=int)
    train_indices, validation_indices, test_indices = actor_disjoint_split(records, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        AudioFeatureDataset(features, labels, train_indices, training=True),
        batch_size=48, shuffle=True, num_workers=0, generator=generator,
    )
    validation_loader = DataLoader(
        AudioFeatureDataset(features, labels, validation_indices), batch_size=64, num_workers=0,
    )
    test_loader = DataLoader(AudioFeatureDataset(features, labels, test_indices), batch_size=64, num_workers=0)
    counts = np.bincount(labels[train_indices], minlength=len(observed_emotions))
    weights = len(train_indices) / (len(observed_emotions) * np.maximum(counts, 1))
    weights = np.sqrt(weights)
    weights /= weights.mean()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpeechEmotionCNN(len(observed_emotions))
    model, history, training_summary = train_classifier(
        model, train_loader, validation_loader, torch.tensor(weights, dtype=torch.float32),
        device, epochs, patience,
    )
    validation_logits, validation_labels = collect_logits(model, validation_loader, device)
    temperature = fit_temperature(validation_logits, validation_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    probabilities = torch.softmax(torch.tensor(test_logits) / temperature, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    test_metrics = classification_metrics(test_labels, predictions, probabilities, observed_emotions)
    torch.save({
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "class_names": observed_emotions,
        "temperature": temperature,
        "architecture": "SpeechEmotionCNN",
        "feature_config": {"sample_rate": 16000, "duration_seconds": 4.0, "n_fft": 512, "hop_length": 256, "n_mels": 64},
        "seed": seed,
    }, MODEL_DIR / "audio_model.pt")
    actors = lambda indices: sorted({records[index].actor for index in indices})
    distribution = Counter(SPEECH_EMOTIONS[record.label] for record in records)
    payload: dict[str, Any] = {
        "architecture": "compact residual 2D CNN trained from random initialization",
        "parameters": parameter_count(model),
        "pretrained": False,
        "input": "mono 16 kHz, peak-normalized, fixed 4 s, 64-bin standardized log-Mel spectrogram",
        "augmentation": "train-only Gaussian noise plus time/frequency masking",
        "split_method": "actor-disjoint GroupShuffleSplit; no actor occurs in multiple partitions",
        "split": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test_indices)},
        "actors": {"train": actors(train_indices), "validation": actors(validation_indices), "test": actors(test_indices)},
        "class_distribution": dict(distribution),
        "documented_classes": SPEECH_EMOTIONS,
        "observed_classes": observed_emotions,
        "missing_documented_classes": [name for name in SPEECH_EMOTIONS if name not in observed_emotions],
        "probability_calibration": {"method": "validation temperature scaling", "temperature": temperature},
        "training": training_summary,
        "history": history,
        "test_metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "device": str(device),
    }
    write_json(METRICS_DIR / "audio.json", payload)
    confusion_matrix_plot(
        test_metrics["confusion_matrix"], observed_emotions,
        FIGURES_DIR / "audio_confusion_matrix.png", "Speech emotion CNN — actor-disjoint test",
    )
    training_history_plot(history, FIGURES_DIR / "audio_training_history.png", "Speech emotion CNN training")
    class_distribution_plot(dict(distribution), FIGURES_DIR / "audio_class_distribution.png", "Speech dataset class distribution")
    return payload
