from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TEMPLATES: tuple[str, ...] = (
    "a bad photo of a",
    "a photo of many",
    "a sculpture of a",
    "a photo of the hard to see",
    "a low resolution photo of the",
    "a rendering of a",
    "graffiti of a",
    "a bad photo of the",
    "a cropped photo of the",
    "a tattoo of a",
    "the embroidered",
    "a photo of a hard to see",
    "a bright photo of a",
    "a photo of a clean",
    "a photo of a dirty",
    "a dark photo of the",
    "a drawing of a",
    "a photo of my",
    "the plastic",
    "a photo of the cool",
    "a close-up photo of a",
    "a black and white photo of the",
    "a painting of the",
    "a painting of a",
    "a pixelated photo of the",
    "a sculpture of the",
    "a bright photo of the",
    "a cropped photo of a",
    "a plastic",
    "a photo of the dirty",
    "a jpeg corrupted photo of a",
    "a blurry photo of the",
    "a photo of the",
    "a good photo of the",
    "a rendering of the",
    "a in a video game",
    "a photo of one",
    "a doodle of a",
    "a close-up photo of the",
    "a photo of a",
    "the origami",
    "the in a video game",
    "a sketch of a",
    "a doodle of the",
    "a origami",
    "a low resolution photo of a",
    "the toy",
    "a rendition of the",
    "a photo of the clean",
    "a photo of a large",
    "a rendition of a",
    "a photo of a nice",
    "a photo of a weird",
    "a blurry photo of a",
    "a cartoon",
    "art of a",
    "a sketch of the",
    "a embroidered",
    "a pixelated photo of a",
    "itap of the",
    "a jpeg corrupted photo of the",
    "a good photo of a",
    "a plushie",
    "a photo of the nice",
    "a photo of the small",
    "a photo of the weird",
    "the cartoon",
    "art of the",
    "a drawing of the",
    "a photo of the large",
    "a black and white photo of a",
    "the plushie",
    "a dark photo of a",
    "itap of a",
    "graffiti of the",
    "a toy",
    "itap of my",
    "a photo of a cool",
    "a photo of a small",
    "a tattoo of the",
)


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: int
    concept_index: int
    seed_index: int
    concept_key: str
    concept_text: str
    seed: int


def resolve_task(
    task_id: int,
    concepts: list[dict[str, Any]],
    seeds: list[int],
) -> BenchmarkTask:
    if not concepts:
        raise ValueError(
            "Benchmark concept list is empty."
        )

    if not seeds:
        raise ValueError(
            "Benchmark seed list is empty."
        )

    total_tasks = len(concepts) * len(seeds)

    if task_id < 0 or task_id >= total_tasks:
        raise ValueError(
            f"task_id={task_id} is outside "
            f"[0, {total_tasks - 1}]"
        )

    concept_index = task_id // len(seeds)
    seed_index = task_id % len(seeds)

    concept = concepts[concept_index]

    return BenchmarkTask(
        task_id=task_id,
        concept_index=concept_index,
        seed_index=seed_index,
        concept_key=str(concept["key"]),
        concept_text=str(concept["text"]),
        seed=int(seeds[seed_index]),
    )


def build_cases(
    concept_key: str,
    concept_text: str,
    num_templates: int = 80,
) -> list[dict[str, Any]]:
    if num_templates < 1:
        raise ValueError(
            "num_templates must be >= 1."
        )

    if num_templates > len(TEMPLATES):
        raise ValueError(
            f"num_templates must be <= "
            f"{len(TEMPLATES)}, got "
            f"{num_templates}."
        )

    cases: list[dict[str, Any]] = []

    for template_index, template in enumerate(
        TEMPLATES[:num_templates]
    ):
        cases.append(
            {
                "name": (
                    f"{concept_key}"
                    f"__t{template_index:03d}"
                ),
                "operation": "erase",
                "prompt": (
                    f"{template} {concept_text}"
                ),
            }
        )

    return cases
