from __future__ import annotations

import argparse
import csv
import os

from pathlib import Path

from evaluate_table1_i2p import ImageRecord, load_records


REVIEW_FIELDS = [
    "case_name",
    "seed",
    "prompt",
    "schedule",
    "token_strength",
    "variant_id",
    "baseline_path",
    "shifted_path",
    "concept_removed",
    "subject_preserved",
    "pose_or_composition_preserved",
    "image_coherent",
    "empty_or_unrelated",
    "acceptable",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-destructive CSV template for paired human review of "
            "completed I2P intervention images. Use 1 for yes, 0 for no, and "
            "leave fields blank when they are not applicable."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination CSV. Defaults to "
            "<root>/evaluation/preservation_review.csv."
        ),
    )
    parser.add_argument(
        "--schedule",
        action="append",
        dest="schedules",
        help="Schedule to include. Repeat to select multiple schedules.",
    )
    parser.add_argument(
        "--strength",
        action="append",
        type=float,
        dest="strengths",
        help="Token strength to include. Repeat to select multiple strengths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing template, including any annotations in it.",
    )
    return parser.parse_args()


def build_review_rows(
    records: list[ImageRecord],
    schedules: set[str] | None = None,
    strengths: set[float] | None = None,
    path_base: Path | None = None,
) -> list[dict[str, object]]:
    baselines = {
        record.pair_key: record
        for record in records
        if record.schedule == "baseline"
    }
    if not baselines:
        raise RuntimeError("Cannot prepare paired review without baselines.")

    shifted = [
        record
        for record in records
        if record.schedule != "baseline"
        and (schedules is None or record.schedule in schedules)
        and (strengths is None or record.strength in strengths)
    ]
    shifted.sort(
        key=lambda record: (
            record.case_name,
            record.seed,
            record.schedule,
            record.strength,
            record.variant_id,
        )
    )
    if not shifted:
        raise RuntimeError("The review filters selected no intervention images.")

    rows: list[dict[str, object]] = []
    for record in shifted:
        baseline = baselines.get(record.pair_key)
        if baseline is None:
            raise RuntimeError(
                "No matched baseline for "
                f"case={record.case_name}, seed={record.seed}."
            )
        rows.append(
            {
                "case_name": record.case_name,
                "seed": record.seed,
                "prompt": record.prompt,
                "schedule": record.schedule,
                "token_strength": record.strength,
                "variant_id": record.variant_id,
                "baseline_path": (
                    os.path.relpath(baseline.path.resolve(), path_base.resolve())
                    if path_base is not None
                    else str(baseline.path)
                ),
                "shifted_path": (
                    os.path.relpath(record.path.resolve(), path_base.resolve())
                    if path_base is not None
                    else str(record.path)
                ),
                "concept_removed": "",
                "subject_preserved": "",
                "pose_or_composition_preserved": "",
                "image_coherent": "",
                "empty_or_unrelated": "",
                "acceptable": "",
                "notes": "",
            }
        )
    return rows


def write_review_csv(
    path: Path,
    rows: list[dict[str, object]],
    force: bool = False,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"Review file already exists: {path}. "
            "Refusing to overwrite annotations; pass --force intentionally."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root
    output = args.output or root / "evaluation" / "preservation_review.csv"
    rows = build_review_rows(
        records=load_records(root),
        schedules=set(args.schedules) if args.schedules else None,
        strengths=set(args.strengths) if args.strengths else None,
        path_base=output.parent,
    )
    write_review_csv(output, rows, force=args.force)
    print(f"Review rows: {len(rows)}")
    print(f"Review template: {output}")


if __name__ == "__main__":
    main()
