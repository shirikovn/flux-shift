from __future__ import annotations

import unittest

import torch

from src.datasets.nudity_pairs import NudityPromptPairDataset
from src.shift.hooks import ShiftTextBlockOutputHook
from src.shift.state import ShiftRuntimeState


class RecordingCollector:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, **kwargs) -> None:
        self.calls.append(kwargs)


class AddOneController:
    def apply(
        self,
        block_index: int,
        step_index: int,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        del block_index, step_index
        return activation + 1


class ShiftBlockOutputTests(unittest.TestCase):
    def test_collects_text_branch_from_complete_block_output(self) -> None:
        state = ShiftRuntimeState(mode="collect")
        state.set_active_locations(blocks=[3], steps=[0])
        state.begin_prompt_run("pair", "positive")
        state.advance_step(torch.tensor([999.0]))
        collector = RecordingCollector()
        hook = ShiftTextBlockOutputHook(3, state, collector=collector)

        text_tokens = torch.randn(1, 8, 6)
        image_tokens = torch.randn(1, 16, 6)
        result = hook(torch.nn.Identity(), (), (text_tokens, image_tokens))

        self.assertIsNone(result)
        self.assertEqual(len(collector.calls), 1)
        self.assertIs(collector.calls[0]["activation"], text_tokens)
        self.assertEqual(collector.calls[0]["step_index"], 0)

    def test_replaces_only_text_branch_when_steering(self) -> None:
        state = ShiftRuntimeState(mode="steer")
        state.set_active_locations(blocks=[2], steps=[0])
        state.begin_steering_run("case")
        state.advance_step(torch.tensor([999.0]))
        hook = ShiftTextBlockOutputHook(2, state, controller=AddOneController())

        text_tokens = torch.zeros(1, 8, 6)
        image_tokens = torch.randn(1, 16, 6)
        result = hook(torch.nn.Identity(), (), (text_tokens, image_tokens))

        self.assertIsNotNone(result)
        assert result is not None
        torch.testing.assert_close(result[0], torch.ones_like(text_tokens))
        self.assertIs(result[1], image_tokens)

    def test_nudity_prompts_match_released_builder(self) -> None:
        dataset = NudityPromptPairDataset()
        first = next(iter(dataset))

        self.assertEqual(first.negative_prompt, "a girl , ")
        self.assertEqual(first.positive_prompt, "a girl , nudity")
        self.assertEqual(len(dataset), 135)


if __name__ == "__main__":
    unittest.main()
