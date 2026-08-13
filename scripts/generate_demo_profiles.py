"""Regenerate UI demo profiles from the persisted training partition only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindfuse.config import METRICS_DIR, RAW_DATA_DIR  # noqa: E402
from mindfuse.data.audit import discover_datasets  # noqa: E402
from mindfuse.data.numerical import load_numerical_dataset  # noqa: E402
from mindfuse.training.tabular import generate_demo_profiles  # noqa: E402
from mindfuse.utils.json_io import read_json, write_json  # noqa: E402


def main() -> int:
    discovered = discover_datasets(RAW_DATA_DIR)
    csv_path = discovered.get("numerical_csv")
    if not csv_path:
        raise SystemExit("No organizer numerical CSV was discovered")
    split = read_json(METRICS_DIR / "tabular_split.json", {})
    train_indices = split.get("train_indices")
    if not train_indices:
        raise SystemExit("Training split artifact is missing or empty")
    frame = load_numerical_dataset(Path(csv_path))
    payload = generate_demo_profiles(frame, train_indices, int(split.get("seed", 42)))
    write_json(METRICS_DIR / "demo_profiles.json", payload)
    write_json(METRICS_DIR / "demo_profile.json", payload["profiles"]["typical"]["values"])
    print(
        f"Wrote {len(payload['profiles'])} training-only profiles with "
        f"{payload['feature_count']} features each"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
