"""Central configuration with repository-relative, environment-overridable paths."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("MINDFUSE_DATA_ROOT", PROJECT_ROOT / "data"))
RAW_DATA_DIR = Path(os.getenv("MINDFUSE_RAW_DATA_DIR", DATA_ROOT / "raw"))
ARTIFACTS_DIR = Path(os.getenv("MINDFUSE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
UPLOAD_DIR = Path(os.getenv("MINDFUSE_UPLOAD_DIR", PROJECT_ROOT / "instance" / "uploads"))

RANDOM_SEED = int(os.getenv("MINDFUSE_SEED", "42"))
MAX_UPLOAD_BYTES = int(os.getenv("MINDFUSE_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTENSIONS = {".wav"}


def ensure_runtime_directories() -> None:
    """Create only application-owned runtime/artifact directories."""

    for path in (MODEL_DIR, METRICS_DIR, FIGURES_DIR, UPLOAD_DIR):
        path.mkdir(parents=True, exist_ok=True)

