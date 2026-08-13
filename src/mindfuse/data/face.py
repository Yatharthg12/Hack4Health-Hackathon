"""Facial-expression records, transforms, and reproducible split logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from mindfuse.constants import FACE_EMOTIONS
from mindfuse.data.audit import infer_face_label


@dataclass(frozen=True)
class FaceRecord:
    path: Path
    label: int


def build_face_records(paths: list[Path]) -> list[FaceRecord]:
    label_to_index = {label: index for index, label in enumerate(FACE_EMOTIONS)}
    records = []
    for path in paths:
        label = infer_face_label(path)
        if label in label_to_index:
            records.append(FaceRecord(path=path, label=label_to_index[label]))
    return records


def split_face_records(
    records: list[FaceRecord],
    seed: int,
) -> tuple[list[FaceRecord], list[FaceRecord], list[FaceRecord], str]:
    """Honor official train/test folders when present; otherwise stratify 70/15/15."""

    official_test = [record for record in records if "test" in {part.lower() for part in record.path.parts}]
    official_train = [record for record in records if "train" in {part.lower() for part in record.path.parts}]
    if official_test and official_train:
        train, validation = train_test_split(
            official_train,
            test_size=0.15,
            random_state=seed,
            stratify=[record.label for record in official_train],
        )
        return list(train), list(validation), official_test, "official train/test; stratified 15% of train used for validation"
    train_validation, test = train_test_split(
        records,
        test_size=0.15,
        random_state=seed,
        stratify=[record.label for record in records],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.1764705882,
        random_state=seed,
        stratify=[record.label for record in train_validation],
    )
    return list(train), list(validation), list(test), "reproducible stratified 70/15/15"


def face_transform(training: bool = False) -> transforms.Compose:
    operations: list[object] = [transforms.Resize((48, 48))]
    if training:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.94, 1.06)),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
        ])
    operations.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(operations)


class FaceDataset(Dataset):
    def __init__(self, records: list[FaceRecord], training: bool = False) -> None:
        self.records = records
        self.transform = face_transform(training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("L"))
        return tensor, record.label

