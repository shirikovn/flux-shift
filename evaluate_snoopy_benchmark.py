from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from cleanfid import fid
from omegaconf import OmegaConf
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
)


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

EXPECTED_TEMPLATES = 80
EXPECTED_SEEDS = 9
EXPECTED_IMAGES_PER_CONCEPT = (
    EXPECTED_TEMPLATES * EXPECTED_SEEDS
)


PAPER_RESULTS = {
    "snoopy": {
        "base_clip": 28.01,
        "shift_clip": 18.57,
        "shift_fid": 136.20,
    },
    "mickey": {
        "base_clip": 26.72,
        "shift_clip": 26.06,
        "shift_fid": 55.56,
    },
    "spongebob": {
        "base_clip": 27.94,
        "shift_clip": 27.35,
        "shift_fid": 63.27,
    },
    "pikachu": {
        "base_clip": 27.15,
        "shift_clip": 26.25,
        "shift_fid": 74.24,
    },
    "dog": {
        "base_clip": 24.62,
        "shift_clip": 24.18,
        "shift_fid": 56.43,
    },
    "legislator": {
        "base_clip": 21.89,
        "shift_clip": 21.77,
        "shift_fid": 47.08,
    },
}


@dataclass(frozen=True)
class ImageRecord:
    concept: str
    case_name: str
    prompt: str
    seed: int
    schedule: str
    strength: float
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/snoopy_table3"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("metrics/snoopy_table3"),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
    )

    parser.add_argument(
        "--skip-fid",
        action="store_true",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    config = OmegaConf.load(path)

    value = OmegaConf.to_container(
        config,
        resolve=True,
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Record is not a mapping: {path}"
        )

    return value


def concept_from_case_name(
    case_name: str,
) -> str:
    marker = "__t"

    if marker not in case_name:
        raise ValueError(
            f"Unexpected benchmark case name: {case_name}"
        )

    return case_name.split(marker, 1)[0]


def load_records(
    root: Path,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    paths = sorted(
        root.rglob("benchmark/records/*.yaml")
    )

    if not paths:
        raise FileNotFoundError(
            f"No benchmark records found below {root}"
        )

    seen: set[
        tuple[str, str, int, str, float]
    ] = set()

    for record_path in paths:
        record = load_yaml(record_path)

        if record.get("status") != "completed":
            continue

        case_name = str(record["case_name"])
        concept = concept_from_case_name(
            case_name
        )

        image_metadata = record.get("image")

        if not isinstance(image_metadata, dict):
            raise RuntimeError(
                f"Record has no image metadata: "
                f"{record_path}"
            )

        relative_path = Path(
            str(image_metadata["relative_path"])
        )

        # .../task_X/benchmark/records/foo.yaml
        benchmark_dir = record_path.parent.parent
        image_path = (
            benchmark_dir / relative_path
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing image referenced by "
                f"{record_path}: {image_path}"
            )

        item = ImageRecord(
            concept=concept,
            case_name=case_name,
            prompt=str(record["prompt"]),
            seed=int(record["seed"]),
            schedule=str(record["schedule"]),
            strength=float(
                record["base_strength"]
            ),
            path=image_path,
        )

        identity = (
            item.concept,
            item.case_name,
            item.seed,
            item.schedule,
            item.strength,
        )

        if identity in seen:
            raise RuntimeError(
                f"Duplicate run encountered: {identity}"
            )

        seen.add(identity)
        records.append(item)

    return records


def group_records(
    records: list[ImageRecord],
) -> tuple[
    dict[str, list[ImageRecord]],
    dict[
        tuple[str, float],
        list[ImageRecord],
    ],
]:
    baseline: dict[
        str,
        list[ImageRecord],
    ] = {}

    shifted: dict[
        tuple[str, float],
        list[ImageRecord],
    ] = {}

    for record in records:
        if record.schedule == "baseline":
            baseline.setdefault(
                record.concept,
                [],
            ).append(record)
            continue

        if record.schedule != "full_shift":
            continue

        shifted.setdefault(
            (
                record.concept,
                record.strength,
            ),
            [],
        ).append(record)

    return baseline, shifted


def validate_count(
    label: str,
    records: list[ImageRecord],
    allow_incomplete: bool,
) -> None:
    actual = len(records)

    if actual == EXPECTED_IMAGES_PER_CONCEPT:
        return

    message = (
        f"{label}: expected "
        f"{EXPECTED_IMAGES_PER_CONCEPT} images, "
        f"found {actual}."
    )

    if allow_incomplete:
        print(f"WARNING: {message}")
        return

    raise RuntimeError(
        message
        + " Use --allow-incomplete for a partial "
        + "smoke-test evaluation."
    )


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class CLIPEvaluator:
    def __init__(
        self,
        device: str,
        batch_size: int,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

        print(
            f"Loading CLIP {CLIP_MODEL_ID} "
            f"on {self.device}..."
        )

        self.model = (
            CLIPModel.from_pretrained(
                CLIP_MODEL_ID
            )
            .to(self.device)
            .eval()
        )

        self.processor = (
            CLIPProcessor.from_pretrained(
                CLIP_MODEL_ID
            )
        )

    @torch.inference_mode()
    def score(
        self,
        records: list[ImageRecord],
    ) -> float:
        if not records:
            raise ValueError(
                "Cannot compute CLIP on empty records."
            )

        score_sum = 0.0
        count = 0

        for start in range(
            0,
            len(records),
            self.batch_size,
        ):
            batch = records[
                start : start + self.batch_size
            ]

            images = [
                load_rgb(record.path)
                for record in batch
            ]

            texts = [
                record.prompt
                for record in batch
            ]

            inputs = self.processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            outputs = self.model(**inputs)

            image_features = (
                outputs.image_embeds
                / outputs.image_embeds.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            text_features = (
                outputs.text_embeds
                / outputs.text_embeds.norm(
                    dim=-1,
                    keepdim=True,
                )
            )

            similarities = (
                image_features
                * text_features
            ).sum(dim=-1)

            # Authors report cosine similarity * 100.
            similarities = similarities * 100.0

            score_sum += float(
                similarities.sum().item()
            )
            count += len(batch)

            for image in images:
                image.close()

        return score_sum / count


def link_images(
    records: list[ImageRecord],
    destination: Path,
) -> None:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, record in enumerate(records):
        target = (
            destination
            / f"{index:05d}.png"
        )

        source = record.path.resolve()

        try:
            os.link(
                source,
                target,
            )
        except OSError:
            # Works even when source and temp dir are
            # on different filesystems.
            shutil.copy2(
                source,
                target,
            )


def compute_fid(
    baseline: list[ImageRecord],
    shifted: list[ImageRecord],
    device: str,
    batch_size: int,
    temp_parent: Path,
) -> float:
    temp_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix="fid_",
        dir=temp_parent,
    ) as temporary_directory:
        temporary_root = Path(
            temporary_directory
        )

        baseline_dir = (
            temporary_root / "baseline"
        )
        shifted_dir = (
            temporary_root / "shifted"
        )

        link_images(
            baseline,
            baseline_dir,
        )

        link_images(
            shifted,
            shifted_dir,
        )

        return float(
            fid.compute_fid(
                str(baseline_dir),
                str(shifted_dir),
                device=torch.device(device),
                batch_size=batch_size,
            )
        )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_records(args.root)

    baseline_by_concept, shifted_groups = (
        group_records(records)
    )

    concepts = list(PAPER_RESULTS)

    for concept in concepts:
        baseline = baseline_by_concept.get(
            concept,
            [],
        )

        validate_count(
            f"{concept}/baseline",
            baseline,
            args.allow_incomplete,
        )

    if not shifted_groups:
        raise RuntimeError(
            "No full_shift records were found."
        )

    for (
        concept,
        strength,
    ), group in shifted_groups.items():
        validate_count(
            f"{concept}/full_shift/{strength:g}",
            group,
            args.allow_incomplete,
        )

    clip_evaluator = CLIPEvaluator(
        device=args.device,
        batch_size=args.clip_batch_size,
    )

    baseline_clip: dict[str, float] = {}

    print("\nComputing baseline CLIP...")

    for concept in concepts:
        records_for_concept = (
            baseline_by_concept.get(
                concept,
                [],
            )
        )

        if not records_for_concept:
            continue

        baseline_clip[concept] = (
            clip_evaluator.score(
                records_for_concept
            )
        )

        print(
            f"  {concept:12s}: "
            f"{baseline_clip[concept]:.2f}"
        )

    result_rows: list[
        dict[str, Any]
    ] = []

    for (
        concept,
        strength,
    ) in sorted(shifted_groups):
        shifted = shifted_groups[
            (concept, strength)
        ]

        baseline = baseline_by_concept.get(
            concept,
            [],
        )

        if not baseline:
            continue

        print(
            f"\nEvaluating "
            f"{concept}, strength={strength:g}"
        )

        shift_clip = clip_evaluator.score(
            shifted
        )

        if args.skip_fid:
            fid_score = float("nan")
        else:
            fid_score = compute_fid(
                baseline=baseline,
                shifted=shifted,
                device=args.device,
                batch_size=args.fid_batch_size,
                temp_parent=args.output_dir,
            )

        paper = PAPER_RESULTS[concept]

        result_rows.append(
            {
                "concept": concept,
                "strength": strength,
                "n_baseline": len(baseline),
                "n_shift": len(shifted),
                "base_clip": baseline_clip[
                    concept
                ],
                "shift_clip": shift_clip,
                "fid": fid_score,
                "paper_base_clip": (
                    paper["base_clip"]
                ),
                "paper_shift_clip": (
                    paper["shift_clip"]
                ),
                "paper_fid": (
                    paper["shift_fid"]
                ),
                "delta_base_clip": (
                    baseline_clip[concept]
                    - paper["base_clip"]
                ),
                "delta_shift_clip": (
                    shift_clip
                    - paper["shift_clip"]
                ),
                "delta_fid": (
                    fid_score
                    - paper["shift_fid"]
                    if not args.skip_fid
                    else float("nan")
                ),
            }
        )

    output_path = (
        args.output_dir / "summary.csv"
    )

    if not result_rows:
        raise RuntimeError(
            "No evaluation rows were produced."
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(result_rows)

    print("\n" + "=" * 100)

    print(
        f"{'concept':12s} "
        f"{'str':>6s} "
        f"{'base':>8s} "
        f"{'paper':>8s} "
        f"{'shift':>8s} "
        f"{'paper':>8s} "
        f"{'FID':>8s} "
        f"{'paper':>8s}"
    )

    print("-" * 100)

    for row in result_rows:
        fid_text = (
            "-"
            if args.skip_fid
            else f"{row['fid']:.2f}"
        )

        print(
            f"{row['concept']:12s} "
            f"{row['strength']:6.0f} "
            f"{row['base_clip']:8.2f} "
            f"{row['paper_base_clip']:8.2f} "
            f"{row['shift_clip']:8.2f} "
            f"{row['paper_shift_clip']:8.2f} "
            f"{fid_text:>8s} "
            f"{row['paper_fid']:8.2f}"
        )

    print("=" * 100)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
