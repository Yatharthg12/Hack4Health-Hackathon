"""Audit the supplied datasets and save a machine-readable report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindfuse.config import METRICS_DIR, RAW_DATA_DIR  # noqa: E402
from mindfuse.data.audit import audit_datasets  # noqa: E402
from mindfuse.utils.json_io import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--fast", action="store_true", help="Skip opening every media file")
    args = parser.parse_args()
    report = audit_datasets(args.data_root, validate_media=not args.fast)
    destination = METRICS_DIR / "dataset_audit.json"
    write_json(destination, report)
    print(json.dumps(report, indent=2))
    print(f"Saved audit: {destination}")
    return 0 if report["numerical"].get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())

