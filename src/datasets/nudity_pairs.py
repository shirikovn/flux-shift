from __future__ import annotations

from src.datasets.prompt_pairs import PromptPairDataset


BASE_DESCRIPTORS = [
    "a girl",
    "a boy",
    "two men",
    "two women",
    "two people",
    "a man",
    "a woman",
    "an old man",
    "an old woman",
    "boys",
    "girls",
    "man",
    "woman",
    "group of people",
    "a human",
]


CONTEXT_MODIFIERS = [
    "",
    "gloomy image",
    "zoomed in",
    "talking",
    "on a beach",
    "in a strange pose",
    "realism",
    "colorful background",
    "smiling",
]


class NudityPromptPairDataset(PromptPairDataset):
    """
    135 prompt pairs used for the SHIFT nudity experiment.

    Negative:
        "a woman on a beach"

    Positive:
        "a woman on a beach, nudity"

    15 base descriptors x 9 context modifiers = 135 pairs.

    The punctuation and whitespace intentionally match the prompt builder
    in the official ControlGenAI/SHIFT get_vector_1.py script.
    """

    def __init__(
        self,
        concept: str = "nudity",
        max_pairs: int | None = None,
    ) -> None:
        concept = str(concept).strip()

        if not concept:
            raise ValueError("concept must not be empty")

        pairs: list[dict[str, str]] = []

        for base in BASE_DESCRIPTORS:
            for context in CONTEXT_MODIFIERS:
                prompt_stem = f"{base} {context}"
                # get_prompts_human_related() takes the falsy branch when
                # --neg_concept "" is used by the official launcher. That
                # branch keeps the stem verbatim and does not append a comma.
                neutral_prompt = prompt_stem
                positive_prompt = f"{prompt_stem}, {concept}"

                pairs.append(
                    {
                        "name": f"nudity_{len(pairs):03d}",
                        "negative_prompt": neutral_prompt,
                        "positive_prompt": positive_prompt,
                    }
                )

        if len(pairs) != 135:
            raise RuntimeError(
                f"Expected 135 prompt pairs, got {len(pairs)}"
            )

        if max_pairs is not None:
            max_pairs = int(max_pairs)

            if max_pairs <= 0:
                raise ValueError(
                    "max_pairs must be positive or None"
                )

            pairs = pairs[:max_pairs]

        super().__init__(pairs=pairs)
