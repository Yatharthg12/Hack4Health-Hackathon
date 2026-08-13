"""Train and evaluate the actor-disjoint speech-emotion model from scratch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindfuse.config import RANDOM_SEED, RAW_DATA_DIR  # noqa: E402
from mindfuse.data.audit import discover_datasets  # noqa: E402
from mindfuse.training.audio import train_audio_model  # noqa: E402
from mindfuse.utils.reproducibility import seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    seed_everything(RANDOM_SEED)
    paths = discover_datasets(args.data_root)["audio_files"]
    result = train_audio_model(paths, epochs=args.epochs)
    print(f"Audio training complete: test macro-F1={result['test_metrics']['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

