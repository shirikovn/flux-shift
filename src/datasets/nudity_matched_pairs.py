from __future__ import annotations

from src.datasets.prompt_pairs import PromptPairDataset


ADULT_SUBJECTS = [
    "one adult woman",
    "one adult man",
    "two adult women",
    "two adult men",
    "one adult woman and one adult man",
]


MATCHED_POSES = [
    "standing front-facing with arms relaxed",
    "standing in side profile with arms relaxed",
    "standing back-facing with arms relaxed",
]


MATCHED_CONTEXTS = [
    "in a softly lit photography studio",
    "against a plain gray studio background",
    "on a secluded beach",
    "in a modern bathroom",
    "beside a private swimming pool",
    "in a bedroom with neutral decor",
    "in a forest clearing",
    "on a rooftop at sunset",
    "in a figure drawing studio",
]


class MatchedNudityPromptPairDataset(PromptPairDataset):
    """Adult nude/clothed counterfactual pairs for cleaner directions.

    Every pair holds subject, count, viewpoint, pose, setting, framing, and
    photographic style fixed.  Only the clothing state changes:

        negative: fully dressed
        positive: fully naked

    The 5 x 3 x 9 Cartesian product deliberately contains 135 pairs, making
    it directly comparable in size with the paper's nudity prompt set.
    """

    # Keep the syntax and word count aligned as closely as possible. The
    # longer "clothed in ordinary opaque clothing" phrase shifts every later
    # token and contaminates a position-wise activation difference.
    NEGATIVE_STATE = "fully dressed"
    POSITIVE_STATE = "fully naked"

    def __init__(self, max_pairs: int | None = None) -> None:
        pairs: list[dict[str, str]] = []

        for subject in ADULT_SUBJECTS:
            for pose in MATCHED_POSES:
                for context in MATCHED_CONTEXTS:
                    prefix = (
                        "A photorealistic full-body photograph of "
                        f"{subject}, {pose}, "
                    )
                    suffix = f", {context}."

                    pairs.append(
                        {
                            "name": f"nudity_matched_{len(pairs):03d}",
                            "negative_prompt": (
                                f"{prefix}{self.NEGATIVE_STATE}{suffix}"
                            ),
                            "positive_prompt": (
                                f"{prefix}{self.POSITIVE_STATE}{suffix}"
                            ),
                        }
                    )

        if len(pairs) != 135:
            raise RuntimeError(
                f"Expected 135 matched prompt pairs, got {len(pairs)}"
            )

        if max_pairs is not None:
            max_pairs = int(max_pairs)
            if max_pairs <= 0:
                raise ValueError("max_pairs must be positive or None")
            pairs = pairs[:max_pairs]

        super().__init__(pairs=pairs)
