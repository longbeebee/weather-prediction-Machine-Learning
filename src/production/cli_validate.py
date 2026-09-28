from __future__ import annotations

import argparse
from pathlib import Path

from src.production.validation import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    print(validate_dataset(args.data_path, args.report_path))


if __name__ == "__main__":
    main()
