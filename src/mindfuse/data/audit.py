"""Filesystem-first dataset discovery and integrity auditing."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from mindfuse.constants import CLASSIFICATION_TARGET, FACE_EMOTIONS, NUMERICAL_FEATURES
from mindfuse.data.numerical import canonicalize_columns, load_numerical_dataset
from mindfuse.data.speech import load_waveform, parse_speech_filename


_FACE_ALIASES = {
    "angry": "Angry", "0": "Angry",
    "disgust": "Disgust", "1": "Disgust",
    "fear": "Fear", "fearful": "Fear", "2": "Fear",
    "happy": "Happy", "3": "Happy",
    "sad": "Sad", "4": "Sad",
    "surprise": "Surprise", "surprised": "Surprise", "5": "Surprise",
    "neutral": "Neutral", "6": "Neutral",
}


def infer_face_label(path: Path) -> str | None:
    """Infer a class from the nearest parent directory, supporting names or IDs."""

    for parent in path.parents:
        label = _FACE_ALIASES.get(parent.name.strip().lower())
        if label:
            return label
    return None


def discover_datasets(root: str | Path) -> dict[str, Any]:
    """Discover actual files by content/schema rather than assumed directory names."""

    root = Path(root)
    discovered_audio = sorted(root.rglob("*.wav")) if root.exists() else []
    # The organizer folder can contain repeated copies of the same RAVDESS tree.
    # Filenames are specified as globally unique sample identifiers, so retain one
    # shortest-path copy per identifier and report the extra paths in the audit.
    audio_by_identifier: dict[str, Path] = {}
    for path in sorted(discovered_audio, key=lambda item: (len(item.parts), str(item))):
        audio_by_identifier.setdefault(path.name.lower(), path)
    audio_files = sorted(audio_by_identifier.values())
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions and infer_face_label(path)
    ] if root.exists() else []
    csv_candidates: list[Path] = []
    for path in sorted(root.rglob("*.csv")) if root.exists() else []:
        try:
            columns = canonicalize_columns(pd.read_csv(path, nrows=3)).columns
            if all(feature in columns for feature in NUMERICAL_FEATURES):
                csv_candidates.append(path)
        except Exception:
            continue
    return {
        "root": root,
        "audio_files": audio_files,
        "audio_paths_discovered": discovered_audio,
        "image_files": sorted(image_files),
        "numerical_csv": csv_candidates[0] if csv_candidates else None,
        "all_csv_candidates": csv_candidates,
    }


def audit_datasets(root: str | Path, validate_media: bool = True) -> dict[str, Any]:
    """Create a serializable audit covering structure, labels, corrupt files, and CSV quality."""

    found = discover_datasets(root)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(Path(root).resolve()),
        "speech": {
            "files": len(found["audio_files"]),
            "paths_discovered": len(found["audio_paths_discovered"]),
            "duplicate_filename_paths": len(found["audio_paths_discovered"]) - len(found["audio_files"]),
            "deduplication_key": "globally unique seven-field RAVDESS filename",
            "class_distribution": {}, "actors": [], "corrupt": [],
        },
        "face": {"files": len(found["image_files"]), "class_distribution": {}, "corrupt": [], "dimensions": {}},
        "numerical": {"path": str(found["numerical_csv"]) if found["numerical_csv"] else None},
    }

    audio_counts: Counter[str] = Counter()
    actors: set[int] = set()
    for path in found["audio_files"]:
        try:
            metadata = parse_speech_filename(path)
            audio_counts[str(metadata["emotion"])] += 1
            actors.add(int(metadata["actor"]))
            if validate_media:
                load_waveform(path)
        except Exception as exc:
            report["speech"]["corrupt"].append({"path": str(path), "error": str(exc)})
    report["speech"]["class_distribution"] = dict(sorted(audio_counts.items()))
    report["speech"]["actors"] = sorted(actors)

    image_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    for path in found["image_files"]:
        label = infer_face_label(path)
        if label:
            image_counts[label] += 1
        if validate_media:
            try:
                with Image.open(path) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                report["face"]["corrupt"].append({"path": str(path), "error": str(exc)})
    report["face"]["class_distribution"] = {
        label: image_counts.get(label, 0) for label in FACE_EMOTIONS
    }
    report["face"]["dimensions"] = dict(dimensions.most_common())

    csv_path = found["numerical_csv"]
    if csv_path:
        raw = canonicalize_columns(pd.read_csv(csv_path))
        report["numerical"].update({
            "rows": int(len(raw)),
            "columns": list(raw.columns),
            "duplicate_rows": int(raw.duplicated().sum()),
            "missing_values": {column: int(value) for column, value in raw.isna().sum().items() if value},
        })
        try:
            clean = load_numerical_dataset(csv_path)
            report["numerical"].update({
                "valid": True,
                "class_distribution": clean[CLASSIFICATION_TARGET].value_counts().to_dict(),
                "feature_ranges": {
                    feature: {"min": float(np.nanmin(clean[feature])), "max": float(np.nanmax(clean[feature]))}
                    for feature in NUMERICAL_FEATURES
                },
            })
        except ValueError as exc:
            report["numerical"].update({"valid": False, "error": str(exc)})
    else:
        report["numerical"].update({"valid": False, "error": "No CSV matching the required schema was found"})
    return report
