"""Numerical dataset and API-schema validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mindfuse.constants import (
    CLASSIFICATION_TARGET,
    FEATURE_METADATA,
    NUMERICAL_FEATURES,
    REGRESSION_TARGETS,
    SCORE_RANGES,
    STRESS_CLASSES,
)


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


_CANONICAL_COLUMNS = {
    _key(column): column
    for column in [*NUMERICAL_FEATURES, CLASSIFICATION_TARGET, *REGRESSION_TARGETS]
}


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt benign spacing/case/punctuation differences to the documented schema."""

    renames: dict[str, str] = {}
    for column in frame.columns:
        canonical = _CANONICAL_COLUMNS.get(_key(column))
        if canonical:
            if canonical in renames.values():
                raise ValueError(f"Multiple columns resolve to {canonical}")
            renames[column] = canonical
    return frame.rename(columns=renames)


def normalize_status(value: Any) -> str:
    normalized = _key(str(value))
    matches = {_key(label): label for label in STRESS_CLASSES}
    if normalized not in matches:
        raise ValueError(f"Unknown mental-health status: {value}")
    return matches[normalized]


def load_numerical_dataset(path: str | Path) -> pd.DataFrame:
    """Load, canonicalize, and validate the complete labelled numerical dataset."""

    try:
        frame = canonicalize_columns(pd.read_csv(path))
    except Exception as exc:
        raise ValueError(f"Unable to read numerical CSV: {exc}") from exc
    required = [*NUMERICAL_FEATURES, CLASSIFICATION_TARGET, *REGRESSION_TARGETS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    frame = frame.loc[:, required].copy()
    for column in [*NUMERICAL_FEATURES, *REGRESSION_TARGETS]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[CLASSIFICATION_TARGET].isna().any():
        raise ValueError("Mental_Health_Status contains missing values")
    frame[CLASSIFICATION_TARGET] = frame[CLASSIFICATION_TARGET].map(normalize_status)
    for target, (minimum, maximum) in SCORE_RANGES.items():
        invalid = frame[target].notna() & ~frame[target].between(minimum, maximum)
        if invalid.any():
            raise ValueError(f"{target} contains values outside [{minimum}, {maximum}]")
    if frame[REGRESSION_TARGETS].isna().any().any():
        raise ValueError("Regression targets contain missing or non-numeric values")
    return frame


def validate_numerical_payload(payload: Any, strict_ranges: bool = True) -> dict[str, float]:
    """Validate a single JSON/form profile and return ordered finite floats."""

    if not isinstance(payload, dict):
        raise ValueError("Numerical input must be a JSON object")
    missing = [feature for feature in NUMERICAL_FEATURES if feature not in payload]
    if missing:
        raise ValueError(f"Missing numerical fields: {', '.join(missing)}")
    clean: dict[str, float] = {}
    for feature in NUMERICAL_FEATURES:
        value = payload[feature]
        if isinstance(value, bool):
            raise ValueError(f"{feature} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric") from exc
        if not np.isfinite(number):
            raise ValueError(f"{feature} must be finite")
        metadata = FEATURE_METADATA[feature]
        if strict_ranges and not float(metadata["min"]) <= number <= float(metadata["max"]):
            raise ValueError(
                f"{feature} must be between {metadata['min']} and {metadata['max']}"
            )
        clean[feature] = number
    return clean


def validate_batch_frame(frame: pd.DataFrame, max_rows: int = 10_000) -> pd.DataFrame:
    """Validate an unlabelled numerical batch without silently accepting bad rows."""

    if frame.empty:
        raise ValueError("The uploaded CSV is empty")
    if len(frame) > max_rows:
        raise ValueError(f"Batch uploads are limited to {max_rows:,} rows")
    frame = canonicalize_columns(frame)
    missing = [feature for feature in NUMERICAL_FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"Missing numerical fields: {', '.join(missing)}")
    result = frame.loc[:, NUMERICAL_FEATURES].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        bad_rows = np.where(~np.isfinite(result.to_numpy(dtype=float)).all(axis=1))[0][:10]
        raise ValueError(f"Non-numeric or missing values in rows: {', '.join(map(str, bad_rows + 2))}")
    for feature in NUMERICAL_FEATURES:
        metadata = FEATURE_METADATA[feature]
        invalid = ~result[feature].between(float(metadata["min"]), float(metadata["max"]))
        if invalid.any():
            raise ValueError(f"{feature} contains values outside accepted input limits")
    return result

