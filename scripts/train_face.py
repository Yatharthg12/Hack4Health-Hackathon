"""Train and evaluate the facial-expression model from scratch."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindfuse.config import METRICS_DIR, RANDOM_SEED, RAW_DATA_DIR  # noqa: E402
from mindfuse.data.audit import discover_datasets  # noqa: E402
from mindfuse.training.face import train_face_model  # noqa: E402
from mindfuse.utils.json_io import read_json, write_json  # noqa: E402
from mindfuse.utils.reproducibility import seed_everything  # noqa: E402


def refresh_dependent_metrics(result: dict) -> None:
    """Keep summary and deployed fusion reliability aligned after face-only training."""

    validation_f1 = float(result["training"]["best_validation_macro_f1"])
    test_f1 = float(result["test_metrics"]["macro_f1"])
    timestamp = datetime.now(timezone.utc).isoformat()
    fusion = read_json(METRICS_DIR / "fusion_weights.json", {})
    fusion.setdefault("weights", {})["face"] = validation_f1
    fusion["generated_at"] = timestamp
    fusion["last_updated_by"] = "scripts/train_face.py"
    write_json(METRICS_DIR / "fusion_weights.json", fusion)
    summary = read_json(METRICS_DIR / "summary.json", {})
    summary.setdefault("metrics", {})["face_macro_f1"] = test_f1
    summary.setdefault("fusion_weights", {})["face"] = validation_f1
    summary["last_updated_at"] = timestamp
    write_json(METRICS_DIR / "summary.json", summary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    seed_everything(RANDOM_SEED)
    paths = discover_datasets(args.data_root)["image_files"]
    result = train_face_model(paths, epochs=args.epochs)
    refresh_dependent_metrics(result)
    print(f"Face training complete: test macro-F1={result['test_metrics']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
