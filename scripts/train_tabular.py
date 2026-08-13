"""Train and evaluate the numerical classifier and D/A/S regressor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mindfuse.config import RANDOM_SEED, RAW_DATA_DIR  # noqa: E402
from mindfuse.data.audit import discover_datasets  # noqa: E402
from mindfuse.training.tabular import train_tabular_models  # noqa: E402
from mindfuse.utils.reproducibility import seed_everything  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_DIR)
    args = parser.parse_args()
    seed_everything(RANDOM_SEED)
    csv_path = args.csv or discover_datasets(args.data_root)["numerical_csv"]
    if not csv_path:
        raise SystemExit("No CSV matching the required numerical schema was found")
    results = train_tabular_models(Path(csv_path))
    print(
        "Tabular training complete: "
        f"classifier={results['classification']['model']}, "
        f"regressor={results['regression']['model']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

