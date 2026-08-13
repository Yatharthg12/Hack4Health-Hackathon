"""Audit, train all four learning components, evaluate, and derive fusion weights."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindfuse.config import METRICS_DIR, RANDOM_SEED, RAW_DATA_DIR, ensure_runtime_directories  # noqa: E402
from mindfuse.data.audit import audit_datasets, discover_datasets  # noqa: E402
from mindfuse.training.audio import train_audio_model  # noqa: E402
from mindfuse.training.face import train_face_model  # noqa: E402
from mindfuse.training.tabular import train_tabular_models  # noqa: E402
from mindfuse.utils.json_io import write_json  # noqa: E402
from mindfuse.utils.reproducibility import seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--face-epochs", type=int, default=25)
    parser.add_argument("--audio-epochs", type=int, default=25)
    args = parser.parse_args()
    ensure_runtime_directories()
    seed_everything(RANDOM_SEED)
    print("[1/6] Auditing datasets", flush=True)
    audit = audit_datasets(args.data_root, validate_media=True)
    write_json(METRICS_DIR / "dataset_audit.json", audit)
    found = discover_datasets(args.data_root)
    if not found["numerical_csv"] or not found["image_files"] or not found["audio_files"]:
        raise SystemExit("Required datasets are missing; see artifacts/metrics/dataset_audit.json")
    print("[2/6] Training numerical classifier and D/A/S regressor", flush=True)
    tabular = train_tabular_models(Path(found["numerical_csv"]))
    print("[3/6] Training facial emotion CNN", flush=True)
    face = train_face_model(found["image_files"], epochs=args.face_epochs)
    print("[4/6] Training actor-disjoint speech emotion CNN", flush=True)
    audio = train_audio_model(found["audio_files"], epochs=args.audio_epochs)
    weights = {
        "face": float(face["training"]["best_validation_macro_f1"]),
        "audio": float(audio["training"]["best_validation_macro_f1"]),
        "numerical": float(tabular["classification"]["model_selection"][tabular["classification"]["model"]]["macro_f1"]),
    }
    print("[5/6] Deriving validation-reliability fusion weights", flush=True)
    write_json(METRICS_DIR / "fusion_weights.json", {
        "weights": weights,
        "basis": "modality-specific held-out validation macro-F1 before test evaluation",
        "normalization": "Weights are re-normalized per request after entropy confidence adjustment",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    })
    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "dataset": {"face": audit["face"]["files"], "audio": audit["speech"]["files"], "numerical": audit["numerical"].get("rows")},
        "metrics": {
            "face_macro_f1": face["test_metrics"]["macro_f1"],
            "audio_macro_f1": audio["test_metrics"]["macro_f1"],
            "numerical_macro_f1": tabular["classification"]["test_metrics"]["macro_f1"],
            "regression_mean_rmse": tabular["regression"]["test_metrics"]["aggregate"]["rmse"],
        },
        "fusion_weights": weights,
    }
    write_json(METRICS_DIR / "summary.json", summary)
    print("[6/6] Complete", flush=True)
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

