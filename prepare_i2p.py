from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


EXPECTED_SIZE = 4703


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the I2P benchmark and freeze the "
            "prompt/seed columns into a local CSV."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/i2p.csv"),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="AIML-TUDA/i2p",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=(
            str(args.cache_dir)
            if args.cache_dir is not None
            else None
        ),
    )

    required_columns = {
        "prompt",
        "sd_seed",
    }

    missing = required_columns - set(
        dataset.column_names
    )

    if missing:
        raise RuntimeError(
            "I2P dataset is missing required columns: "
            f"{sorted(missing)}. "
            f"Available columns: {dataset.column_names}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "i2p_index",
                "prompt",
                "sd_seed",
            ],
        )

        writer.writeheader()

        for index, row in enumerate(dataset):
            writer.writerow(
                {
                    "i2p_index": index,
                    "prompt": str(row["prompt"]),
                    "sd_seed": int(row["sd_seed"]),
                }
            )

    print(f"Saved {len(dataset)} rows to {args.output}")

    if len(dataset) != EXPECTED_SIZE:
        print(
            "WARNING: the paper reports "
            f"{EXPECTED_SIZE} I2P prompts, but the "
            f"current dataset contains {len(dataset)}."
        )


if __name__ == "__main__":
    main()
