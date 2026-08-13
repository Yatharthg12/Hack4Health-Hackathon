"""Atomic JSON artifact helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    """Atomically write a JSON artifact to avoid half-written metrics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON or return a caller-provided default when absent."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

