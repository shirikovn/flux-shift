from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.classifiers.svm_trainer import LinearSVMTrainer
from src.datasets.nudity_matched_pairs import MatchedNudityPromptPairDataset
from src.datasets.nudity_pairs import NudityPromptPairDataset
from src.shift.collector import MeanDifferenceCollector
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

    def test_nudity_prompts_match_official_builder(self) -> None:
        dataset = NudityPromptPairDataset()
        first = next(iter(dataset))

        self.assertEqual(first.negative_prompt, "a girl ")
        self.assertEqual(first.positive_prompt, "a girl , nudity")
        self.assertEqual(len(dataset), 135)

    def test_matched_nudity_pairs_change_only_clothing_state(self) -> None:
        dataset = MatchedNudityPromptPairDataset()

        self.assertEqual(len(dataset), 135)
        for pair in dataset:
            self.assertIn("adult", pair.negative_prompt.lower())
            self.assertIn("adult", pair.positive_prompt.lower())
            self.assertIn(
                MatchedNudityPromptPairDataset.NEGATIVE_STATE,
                pair.negative_prompt,
            )
            self.assertIn(
                MatchedNudityPromptPairDataset.POSITIVE_STATE,
                pair.positive_prompt,
            )
            negative_frame = pair.negative_prompt.replace(
                MatchedNudityPromptPairDataset.NEGATIVE_STATE,
                "<state>",
            )
            positive_frame = pair.positive_prompt.replace(
                MatchedNudityPromptPairDataset.POSITIVE_STATE,
                "<state>",
            )
            self.assertEqual(negative_frame, positive_frame)

    def test_consistent_vector_retains_cross_pair_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            collector = MeanDifferenceCollector(
                save_dir=temporary_directory,
                tensor_dtype="float32",
            )
            negative = torch.zeros(1, 2, 2)
            positives = (
                torch.tensor([[[2.0, 0.0], [1.0, 0.0]]]),
                torch.tensor([[[1.0, 0.0], [-3.0, 0.0]]]),
            )

            for index, positive in enumerate(positives):
                pair_name = f"pair_{index}"
                collector.add(
                    pair_name,
                    "negative",
                    0,
                    0,
                    1.0,
                    negative,
                )
                collector.add(
                    pair_name,
                    "positive",
                    0,
                    0,
                    1.0,
                    positive,
                )

            collector.save()
            root = Path(temporary_directory) / "block_00"
            standard = torch.load(
                root / "step_00_vector.pt",
                map_location="cpu",
                weights_only=True,
            )
            consistent = torch.load(
                root / "step_00_consistent_vector.pt",
                map_location="cpu",
                weights_only=True,
            )

            torch.testing.assert_close(standard[0], torch.tensor([1.0, 0.0]))
            torch.testing.assert_close(standard[1], torch.tensor([-1.0, 0.0]))
            torch.testing.assert_close(consistent[0], torch.tensor([1.0, 0.0]))
            torch.testing.assert_close(consistent[1], torch.zeros(2))

    def test_grouped_svm_split_keeps_replicas_together(self) -> None:
        trainer = object.__new__(LinearSVMTrainer)
        trainer.split_by_pair = True
        trainer.validation_fraction = 0.4

        samples = []
        labels = []
        for pair_index in range(10):
            for replica_index in range(2):
                name = f"pair_{pair_index:02d}__replica_{replica_index:02d}"
                for role, label in (("positive", 1), ("negative", 0)):
                    samples.append(
                        {
                            "pair_name": name,
                            "prompt_role": role,
                            "label": label,
                        }
                    )
                    labels.append(label)

        train_indices, validation_indices = trainer._split_indices(
            labels=np.asarray(labels),
            samples=samples,
            random_seed=42,
        )
        train_groups = {
            trainer._pair_group_name(samples[index]["pair_name"])
            for index in train_indices
        }
        validation_groups = {
            trainer._pair_group_name(samples[index]["pair_name"])
            for index in validation_indices
        }

        self.assertFalse(train_groups.intersection(validation_groups))
        self.assertEqual(len(train_groups), 6)
        self.assertEqual(len(validation_groups), 4)

    def test_grouped_svm_can_validate_then_refit_all_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = root / "dataset"
            block_dir = dataset_dir / "block_00"
            block_dir.mkdir(parents=True)

            features = []
            labels = []
            samples = []
            for pair_index in range(12):
                nuisance = pair_index / 100.0
                for role, label, feature in (
                    ("negative", 0, [1.0, nuisance, 0.0, 0.0]),
                    ("positive", 1, [0.0, nuisance, 1.0, 0.0]),
                ):
                    row_index = len(features)
                    features.append(feature)
                    labels.append(label)
                    samples.append(
                        {
                            "row_index": row_index,
                            "pair_name": f"pair_{pair_index:02d}",
                            "prompt_role": role,
                            "label": label,
                            "block_index": 0,
                            "step_index": 0,
                            "timestep": 1.0,
                        }
                    )

            torch.save(
                torch.tensor(features, dtype=torch.float32),
                block_dir / "step_00_features.pt",
            )
            torch.save(
                torch.tensor(labels, dtype=torch.long),
                block_dir / "step_00_labels.pt",
            )
            OmegaConf.save(
                OmegaConf.create({"samples": samples}),
                block_dir / "step_00_samples.yaml",
            )
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "activation_location": (
                            "transformer_block_output_text"
                        )
                    }
                ),
                dataset_dir / "metadata.yaml",
            )

            output_dir = root / "classifiers"
            trainer = LinearSVMTrainer(
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                block_indices=[0],
                step_indices=[0],
                split_by_pair=True,
                refit_full_after_validation=True,
            )
            trainer.run()

            metadata = OmegaConf.load(output_dir / "metadata.yaml")
            split = OmegaConf.load(
                output_dir / "block_00" / "step_00_split.yaml"
            )
            metrics = OmegaConf.load(
                output_dir / "block_00" / "step_00_metrics.yaml"
            )

            self.assertEqual(metadata.split_strategy, "grouped_prompt_pairs")
            self.assertTrue(metadata.refit_full_after_validation)
            self.assertEqual(metrics.saved_model, "ensemble_refit_on_all_samples")
            self.assertTrue(
                all(len(member.pair_overlap) == 0 for member in split.members)
            )
            self.assertGreaterEqual(
                float(metrics.validation.expected_calibration_error),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
