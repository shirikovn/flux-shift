from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptPair:
    name: str
    negative_prompt: str
    positive_prompt: str


class PromptPairDataset:
    def __init__(self, pairs: list[dict]) -> None:
        self.pairs = [
            PromptPair(
                name=str(item["name"]),
                negative_prompt=str(item["negative_prompt"]),
                positive_prompt=str(item["positive_prompt"]),
            )
            for item in pairs
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self):
        return iter(self.pairs)
