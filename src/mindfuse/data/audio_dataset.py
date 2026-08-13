"""Cached audio feature dataset and actor-disjoint split construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from mindfuse.constants import SPEECH_EMOTIONS
from mindfuse.data.speech import extract_audio_features, parse_speech_filename


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    label: int
    actor: int


def build_audio_records(paths: list[Path]) -> list[AudioRecord]:
    label_to_index = {label: index for index, label in enumerate(SPEECH_EMOTIONS)}
    records: list[AudioRecord] = []
    for path in paths:
        metadata = parse_speech_filename(path)
        records.append(AudioRecord(path, label_to_index[str(metadata["emotion"])], int(metadata["actor"])))
    return records


def actor_disjoint_split(records: list[AudioRecord], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(records))
    groups = np.asarray([record.actor for record in records])
    first = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_validation, test = next(first.split(indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    relative_train, relative_validation = next(
        second.split(train_validation, groups=groups[train_validation])
    )
    train = train_validation[relative_train]
    validation = train_validation[relative_validation]
    observed_labels = {record.label for record in records}
    for partition in (train, validation, test):
        if {records[index].label for index in partition} != observed_labels:
            raise ValueError("Actor split unexpectedly omitted one or more emotion classes")
    return np.sort(train), np.sort(validation), np.sort(test)


def build_or_load_feature_cache(records: list[AudioRecord], cache_path: Path) -> np.ndarray:
    fingerprints = np.asarray([
        f"{record.path.resolve()}|{record.path.stat().st_size}|{record.path.stat().st_mtime_ns}"
        for record in records
    ])
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if np.array_equal(cached["fingerprints"], fingerprints):
                return cached["features"]
        except Exception:
            pass
    features: list[np.ndarray] = []
    for index, record in enumerate(records):
        spectrogram, _, _ = extract_audio_features(record.path)
        features.append(spectrogram)
        if (index + 1) % 100 == 0:
            print(f"audio_features={index + 1}/{len(records)}", flush=True)
    stacked = np.stack(features).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=stacked, fingerprints=fingerprints)
    return stacked


class AudioFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, indices: np.ndarray, training: bool = False) -> None:
        self.features = features
        self.labels = labels
        self.indices = indices
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        feature = torch.from_numpy(self.features[self.indices[item]]).unsqueeze(0).clone()
        if self.training:
            if torch.rand(()) < 0.45:
                feature += torch.randn_like(feature) * 0.025
            if torch.rand(()) < 0.35:
                width = int(torch.randint(4, 18, (1,)).item())
                start = int(torch.randint(0, max(feature.shape[-1] - width, 1), (1,)).item())
                feature[..., start : start + width] = 0
            if torch.rand(()) < 0.25:
                height = int(torch.randint(3, 9, (1,)).item())
                start = int(torch.randint(0, max(feature.shape[-2] - height, 1), (1,)).item())
                feature[..., start : start + height, :] = 0
        return feature, int(self.labels[self.indices[item]])
